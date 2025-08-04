#!/usr/bin/env python3
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#from gr00t.eval.robot import RobotInferenceClient
from Isaac_GR00T.gr00t.eval.robot import RobotInferenceClient

from rtde_control import RTDEControlInterface  # 机械臂控制[1](@ref)
from rtde_receive import RTDEReceiveInterface 
import rtde_control
import rtde_receive
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
import numpy as np
import rerun as rr
import random
import threading 
# from rtde_control import RTDEControlInterface  # 机械臂控制[1](@ref)
# from rtde_receive import RTDEReceiveInterface  # 状态读取[1](@ref)
# from rtde_io import RTDEIOInterface  # 数字IO控制
import cv2  # 摄像头采集
#from lerobot.common.datasets.push_dataset_to_hub import push_to_hub  # 数据集上传[5](@ref)
import threading  # 添加线程支持
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations
import argparse
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from typing import Optional
from lerobot.common.robots.UR5e_follower import URRobotLeRobot  # noqa: F401
from lerobot.common.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.robots.UR5e_follower import URRobotLeRobot
from lerobot.common.teleoperators.VR import VR_leader  # noqa: F401
from lerobot.common.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
    VR_leader,
)
from lerobot.common.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.common.utils.visualization_utils import _init_rerun
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from demo_can import hand_control,hand_start,hand_read,write6
import socket

def is_port_in_use(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        result = s.connect_ex((ip, port))
        s.close()
        # 0 表示端口已被占用（有服务监听），111/10061等表示未监听
        return result == 0
    except Exception as e:
        print(f"Port check error: {e}")
        return False

@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 60
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 20
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RecordConfig:
    srobot: RobotConfig
    #robot = make_robot_from_config(cfg.robot)
    #robot: Optional[RobotConfig] = None
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | None = None,
    policy: PreTrainedPolicy | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    # if policy is given it needs cleaning up
    # if policy is not None:
    #     policy.reset()
    old_control = [0,0,0,0,0,0,0,0]
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break
        #start_time = time.perf_counter()
        observation = robot.get_observation()
        #print(f"获取观察数据耗时: {time.perf_counter() - start_time:.4f} 秒")
        #hand_pose = [observation[f"hand_{i+1}.pos"] for i in range(6)]
        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        if policy is not None:
            action_values = predict_action(
                observation_frame,
                policy,
                get_safe_torch_device(policy.config.device),
                policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
        elif policy is None:
            #start_time = time.perf_counter()
            action = robot.get_action(events,old_control)
            #print(f"获取action耗时: {time.perf_counter() - start_time:.4f} 秒")
            old_control = events["control"]
            if action is None:
                logging.info("Teleoperator returned None action, skipping this loop iteration.")
                continue
        else:
            logging.info(
                "No policy or teleoperator provided, skipping action generation."
                "This is likely to happen when resetting the environment without a teleop device."
                "The robot won't be at its rest position at the start of the next episode."
            )
            continue

        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset.
        #start_time = time.perf_counter()
        sent_action = robot.send_action(action)
        #print(f"发送动作耗时: {time.perf_counter() - start_time:.4f} 秒")
        #start_time = time.perf_counter()
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
            frame = {**observation_frame, **action_frame}
            dataset.add_frame(frame, task=single_task)
        #print(f"保存数据耗时: {time.perf_counter() - start_time:.4f} 秒")
        if display_data:
            for obs, val in observation.items():
                if isinstance(val, float):
                    rr.log(f"observation.{obs}", rr.Scalar(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"observation.{obs}", rr.Image(val), static=True)
            for act, val in action.items():
                if isinstance(val, float):
                    rr.log(f"action.{act}", rr.Scalar(val))

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: RecordConfig,robot2,listener, events) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="recording")

    #robot = make_robot_from_config(cfg.robot)
    #robot = URRobotLeRobot()
    robot = robot2
    #teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    teleop = None
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    # Load pretrained policy
    policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)

    robot.connect()
    # if teleop is not None:
    #     teleop.connect()

    

    recorded_episodes = 0
    while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {dataset.num_episodes}", cfg.play_sounds)
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop=teleop,
            policy=policy,
            dataset=dataset,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.single_task,
            display_data=cfg.display_data,
        )

        # Execute a few seconds without recording to give time to manually reset the environment
        # Skip reset for the last episode to be recorded
        if not events["stop_recording"] and (
            (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment", cfg.play_sounds)
            print("Reset the environment9999999999999999999999999999999999999999999")
            events["control"] = [0, 0, 0, 0, 0, 0, 0, 0]
            joint_pos = [-0.7, -1.8, -2.1, -2.37, -0.7, -0.1]
            robot.robot1.moveJ(joint_pos, 0.3, 0.5, False)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop=teleop,
                control_time_s=cfg.dataset.reset_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    # if teleop is not None:
    #     teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    # if cfg.dataset.push_to_hub:
    #     dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    log_say("Exiting", cfg.play_sounds)
    return dataset


if __name__ == "__main__":
    robot1= URRobotLeRobot()
    #teleoperator = VuerTeleop('/home/hpx/vla/src/lerobot/lerobot/inspire_hand.yml')
    #robot1 = teleoperator.robot2
    # keyboard_listener = KeyboardListener()
    # keyboard_listener.start_listening()
    listener, events = init_keyboard_listener()
    policy_client = RobotInferenceClient(host="localhost", port=5555)
    # robot_ip = "192.168.31.2"
    # rtde_port = 30004
    # if is_port_in_use(robot_ip, rtde_port):
    #     print(f"警告：{robot_ip}:{rtde_port} 端口已被占用，RTDE 可能已被其他服务占用！")
    # else:
    #     print(f"{robot_ip}:{rtde_port} 端口未被占用，可以尝试连接。")
    #robot1= UR5e_arm()
    start_flag = 0
    try:
        while True:
            if events["stop_recording"]:
                break


            if events["restart_arm"]:
                print("机器人已重置！")
                # 重置机器人到初始位置
                joint_pos, tcp_pose = robot1.read()
                print(f"当前关节角度: {joint_pos}")
                print(f"当前TCP位姿: {tcp_pose}")
                arm_pose = tcp_pose.copy()
                arm_pose[2] -= 0.2
                #joint_pos[3] +=  -0.4
                #joint_pos = [0, -2.1, -2, -2.1, 0, 0]
                joint_pos = [-0.6, -1.8, -2.1, -2.37, -0.6, -0.1]
                robot1.robot1.moveJ(joint_pos, 0.3, 0.5, False)
                #robot.move_safety(arm_pose)
                #time.sleep(1)
                # hand_targets = [1000, 1000, 1000, 1000, 1000, 500]  # 手部目标位置
                # result = hand_control(self.ser1,hand_targets)
                #break
            
            if events["start_record"] and start_flag == 0:
                start_flag = 1
            
            else:
                # joint_pos, tcp_pose = robot1.read()
                # print(f"当前关节角度: {joint_pos}")
                # print(f"当前TCP位姿: {tcp_pose}")
                # tcp = [0.45, -0.23, 0.03, 1.5692098537490884, -0.06472226082896908, 0.06965037143778885]
                # robot1.robot1.moveL(tcp, speed=0.1, acceleration=0.5, asynchronous=False)
                joint_pos = [-0.6, -1.8, -2.1, -2.37, -0.6, -0.1]
                robot1.robot1.moveJ(joint_pos, 0.3, 0.5, False)
                #print("机器人未启动，等待按 'z' 键...")
            # 获取当前帧的头部和手部数据
            if start_flag == 0:
                joint_pos, tcp_pose = robot1.read()
                # print(f"当前关节角度: {joint_pos}")
                # print(f"当前TCP位姿: {tcp_pose}")
                observation = robot1.get_observation()
                print(f"当前观察数据: {observation}")

                # 准备观测数据


# 获取action
                #action = policy_client.get_action(obs)
                #print(events["control"])
                # head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
                # print(f'Right Hand Pose: {right_pose}')
                # print(f'计数数据：{events["w_count"]}')
            elif start_flag == 1:
                time.sleep(2)  # 等待一段时间以确保数据稳定
                start_flag = 2
                #head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
               # global old_pose = right_pose.copy()
            elif start_flag == 2:
                #record(teleop=teleoperator,listener = listener, events=events)
                #record(robot2 = robot1,listener = listener, events=events)
                break
            #time.sleep(0.5)
    # keyboard_listener = KeyboardListener()
    # keyboard_listener.start_listening()
    # robot1_ip = "192.168.31.2"
    # while True:
    #     try:
    #         print(f"尝试连接 RTDE 控制端口（{robot1_ip}:30004）...")
    #         robot1 = rtde_control.RTDEControlInterface(robot1_ip)
    #         robot1.endFreedriveMode()
    #         print("连接成功！")
    #         break
    #     except Exception as e:
    #         print(e)
    #         print(robot1_ip)
    #         print("Failed to connect to the robot. Will retry in 1 seconds...")
    #         time.sleep(1)
    # r_inter = rtde_receive.RTDEReceiveInterface(robot1_ip)

    except KeyboardInterrupt:
        # 退出时释放资源
        exit(0)
