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

########################################################################################
# Utilities
########################################################################################


import logging
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
import threading 
import numpy as np
import torch
from deepdiff import DeepDiff
from termcolor import colored
import time
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import DEFAULT_FEATURES
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import Robot


######键盘相关的文件。

def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)

@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action

def handle_key_increment(events, key_index, key_last_time, key_lock, w_key_interval, increment=1):
    now = time.monotonic()
    with key_lock:
        if now - key_last_time[0] >= w_key_interval:
            events["control"][key_index] += increment
            key_last_time[0] = now


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    # wasd控制前后左右，s是x+，d是y+，qe控制上和下，z是开始录制，r是开始手部抓取，f是手部放开, x是重置
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["start_hand"] = False
    events["start_record"] = False
    events["restart_arm"] = False
    events["quit"] = False
    events["auto_mode"] = False
    events["control"] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]   
    key_last_time = [0.0]  # 使用列表以便闭包可修改
    w_key_interval = 0.005  # 0.1秒
    key_lock = threading.Lock()
    start_flag = False
    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif hasattr(key, 'char'):
                char = key.char.lower()

                if char == 'x':
                    print("X key pressed. Restart arm...")
                    events["restart_arm"] = True
                    events["control"] = [0] * len(events["control"])

                elif char == 'z':
                    print("Z key pressed. Start recording...")
                    events["start_record"] = True

                elif char == 'w':
                    handle_key_increment(events, 0, key_last_time, key_lock, w_key_interval)
                elif char == 's':
                    handle_key_increment(events, 1, key_last_time, key_lock, w_key_interval)
                elif char == 'a':
                    handle_key_increment(events, 2, key_last_time, key_lock, w_key_interval)
                elif char == 'd':
                    handle_key_increment(events, 3, key_last_time, key_lock, w_key_interval)
                elif char == 'q':
                    handle_key_increment(events, 4, key_last_time, key_lock, w_key_interval)
                elif char == 'e':
                    handle_key_increment(events, 5, key_last_time, key_lock, w_key_interval)
                elif char == 'r':
                    handle_key_increment(events, 6, key_last_time, key_lock, w_key_interval)
                elif char == 'f':
                    handle_key_increment(events, 7, key_last_time, key_lock, w_key_interval)
                elif char == 't':
                    handle_key_increment(events, 8, key_last_time, key_lock, w_key_interval)
                elif char == 'g':
                    handle_key_increment(events, 9, key_last_time, key_lock, w_key_interval)
                elif char == 'y':
                    handle_key_increment(events, 10, key_last_time, key_lock, w_key_interval)
                elif char == 'h':
                    handle_key_increment(events, 11, key_last_time, key_lock, w_key_interval)
                elif char == 'u':
                    handle_key_increment(events, 12, key_last_time, key_lock, w_key_interval)
                elif char == 'j':
                    handle_key_increment(events, 13, key_last_time, key_lock, w_key_interval)
                elif char == 'c':
                    print("C key pressed. Start recording (auto mode)...")
                    events["start_record"] = True
                    events["auto_mode"] = True

        except Exception as e:
            print(f"Error handling key press: {e}")


    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
