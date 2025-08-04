#!/usr/bin/env python3
# from isaacgym import gymapi
# from isaacgym import gymutil
# from isaacgym import gymtorch
from ur_ikfast import ur_kinematics
import time
import numpy as np
from rtde_control import RTDEControlInterface  # 机械臂控制[1](@ref)
from rtde_receive import RTDEReceiveInterface  # 状态读取[1](@ref)
from rtde_io import RTDEIOInterface  # 数字IO控制
import cv2  # 摄像头采集
#from lerobot.common.datasets.push_dataset_to_hub import push_to_hub  # 数据集上传[5](@ref)
import threading  # 添加线程支持
import sys 
import math
import torch
import random

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
import pyrealsense2 as rs

class KeyboardListener:
    def __init__(self):
        self.robot_enabled = False
        self.running = True
        self.robot_reset = False
    def listen_for_keys(self):
        """监听键盘输入的线程函数"""
        print("键盘监听已启动！按 'a' 键启动机器人，按 'q' 键退出程序")
        while self.running:
            try:
                key = input().strip().lower()
                if key == 'a':
                    self.robot_enabled = True
                    print("机器人已启动！")
                elif key == 'r':    
                    self.robot_reset = True
                    print("机器人已重置！")
                elif key == 'q':
                    self.running = False
                    print("程序即将退出...")
                    break
            except KeyboardInterrupt:
                break
                
    def start_listening(self):
        """启动键盘监听线程"""
        self.listener_thread = threading.Thread(target=self.listen_for_keys, daemon=True)
        self.listener_thread.start()
        
    def is_robot_enabled(self):
        """检查机器人是否被启用"""
        return self.robot_enabled
    
    def is_robot_reset(self):
        """检查机器人是否被重置"""
        # 这里可以添加具体的重置逻辑
        return self.robot_reset
        
    def should_exit(self):
        """检查是否应该退出程序"""
        return not self.running
    
class VuerTeleop:
    """
    负责远程操作手部数据的采集与预处理，包括图像共享内存、预处理、重定向配置等。
    """
    def __init__(self, config_file_path):
        # 原始分辨率
        self.resolution = (720, 1280)
        # 裁剪宽高（此处为0，未裁剪）
        self.crop_size_w = 0
        self.crop_size_h = 0
        # 裁剪后的分辨率
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        # 图像shape：(高, 2*宽, 3)，左右拼接
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        # 创建共享内存用于图像传递
        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()

        # 初始化视频采集与预处理
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=False)
        self.processor = VuerPreprocessor()

        # 设置重定向配置
        RetargetingConfig.set_default_urdf_dir('/home/hpx/vla/src/lerobot/lerobot/assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        """
        处理一帧数据，输出头部旋转矩阵、左右手位姿和关节角。
        """
        # 处理视频流，获得头部和手部的位姿矩阵
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        # 提取头部旋转矩阵
        head_rmat = head_mat[:3, :3]

        # 计算左手和右手的位姿（位置+四元数），并做平移修正
        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        # 通过重定向获得左右手的关节角，并按指定顺序排列
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos
    def step_right(self):
        """
        处理一帧数据，输出头部旋转矩阵、左右手位姿和关节角。
        """
        # 处理视频流，获得头部和手部的位姿矩阵
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        # 提取头部旋转矩阵
        head_rmat = head_mat[:3, :3]

        # 计算左手和右手的位姿（位置+四元数），并做平移修正
        #left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
        #                            rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        # 通过重定向获得左右手的关节角，并按指定顺序排列
        #left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        return head_rmat, right_pose, right_qpos

class Sim:
    """
    负责仿真环境的初始化、手部模型加载、相机设置以及仿真步进。
    """
    def __init__(self,
                 print_freq=False):
        self.print_freq = print_freq

        # 初始化gym
        self.gym = gymapi.acquire_gym()

        # 配置仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        # 创建仿真
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # 添加地面
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # 加载桌子模型
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # 加载方块模型
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        # 加载左右手模型
        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.dof = self.gym.get_asset_dof_count(left_asset)

        # 设置环境网格（这里只创建一个环境）
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # 添加桌子到环境
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # 添加方块到环境
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # 添加左手到环境
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # 添加右手到环境
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        # 获取根状态张量，用于后续控制
        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        # 创建默认观察器（viewer）
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # 相机相关参数
        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        # 创建左手第一视角相机
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # 创建右手第一视角相机
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):
        """
        仿真步进：更新手部位姿、关节角，刷新相机视角，返回左右相机图像。
        """
        if self.print_freq:
            start = time.time()

        # 更新左右手根状态（位姿）
        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        # 更新左手关节角
        left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        left_states['pos'] = left_qpos
        self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

        # 更新右手关节角
        right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        right_states['pos'] = right_qpos
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)

        # 仿真步进
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # 根据头部旋转矩阵调整相机朝向
        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        # 设置左右相机位置和朝向
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        # 获取左右相机图像
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        # 渲染与同步
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        return left_image, right_image

    def end(self):
        """
        释放仿真资源。
        """
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
# ===== 1. 初始化UR机械臂连接 =====
class UR5e_arm:
    def __init__(self, robot_ip: str = "192.168.31.2"):
        import rtde_control
        import rtde_receive

        try:
            self.robot1 = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        [print("connect") for _ in range(4)]

        #self._free_drive = False
        #self.robot.endFreedriveMode()
        
    def read(self):
        """Get the current state of the UR robot.

        Returns:
            T: The current state of the UR robot.
        """
        joint_pos = self.r_inter.getActualQ()
        tcp_pose = self.r_inter.getActualTCPPose()
        # print(f"当前关节角度: {joint_pos}")
        # print(f"当前TCP位姿: {tcp_pose}")
        return joint_pos, tcp_pose
    
    def move_to_tcp(self, target_tcp, speed=0.1, acceleration=0.5):
        """Move the robot to a target TCP pose.

        Args:
            target_tcp (list): Target TCP pose as a list of 6 elements [x, y, z, rx, ry, rz].
            speed (float): Speed of the movement.
            acceleration (float): Acceleration of the movement.
        """
        self.robot1.moveL(target_tcp, speed=speed, acceleration=acceleration, asynchronous=False)

    def move_safety(self, target_tcp):
        joint_pos, tcp_pose = self.read()
        delta = np.abs(np.array(tcp_pose[:3]) - np.array(target_tcp[:3]))
        over_limit = delta > 0.3
        axis = ['x', 'y', 'z']
        if np.any(over_limit):
            for i, flag in enumerate(over_limit):
                if flag:
                    print(f"{axis[i]} 方向移动超过0.3米，无法移动")
            return
        delta1 = np.abs(np.array(tcp_pose[-3:]) - np.array(target_tcp[-3:]))
        over_limit = delta1 > 2
        axis = ['x', 'y', 'z']
        if np.any(over_limit):
            for i, flag in enumerate(over_limit):
                if flag:
                    print(f"{axis[i]} 旋转方向角度移动过大，无法移动")
            return
        self.move_to_tcp(target_tcp)
        time.sleep(0.1)  # 等待机械臂运动完成
def axis_angle_to_quaternion(rx, ry, rz):
    """
    将轴角表示 (rx, ry, rz) 转换为四元数 (qx, qy, qz, qw)。

    Args:
        rx (float): 旋转轴的 x 分量。
        ry (float): 旋转轴的 y 分量。
        rz (float): 旋转轴的 z 分量。

    Returns:
        list: 四元数 [qx, qy, qz, qw]。
    """
    # 计算旋转角度 theta
    theta = np.linalg.norm([rx, ry, rz])  # ||[rx, ry, rz]||

    # 如果旋转角度接近 0，返回单位四元数
    if theta < 1e-6:
        return [0.0, 0.0, 0.0, 1.0]

    # 计算旋转轴 [ux, uy, uz]
    ux, uy, uz = rx / theta, ry / theta, rz / theta

    # 计算四元数
    qx = ux * np.sin(theta / 2)
    qy = uy * np.sin(theta / 2)
    qz = uz * np.sin(theta / 2)
    qw = np.cos(theta / 2)
    return [qw, qx, qy, qz]
def main():
    # 创建 RealSense 管道
    pipeline = rs.pipeline()

    # 配置流
    config = rs.config()
    #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 深度流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # RGB 流

    # 启动管道
    pipeline.start(config)

    try:
        while True:
            # 等待一帧数据
            frames = pipeline.wait_for_frames()

            # 获取颜色帧
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 将颜色帧转换为 NumPy 数组
            color_image = np.asanyarray(color_frame.get_data())

            # 裁剪图像：左右各减少 200 个像素
            cropped_image = color_image[:, 220:-340, :] # 高度不变，宽度裁剪

            # 显示裁剪后的图像
            cv2.imshow('Original RealSense', color_image)
            cv2.imshow('Cropped RealSense', cropped_image)

            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 主流程：初始化远程操作与仿真，循环处理与同步图像
    '''
    keyboard_listener = KeyboardListener()
    keyboard_listener.start_listening()
    teleoperator = VuerTeleop('/home/hpx/vla/src/lerobot/lerobot/inspire_hand.yml')
    robot= UR5e_arm()
    start_flag = 0
    ur5e_arm = ur_kinematics.URKinematics('ur5e')
    #simulator = Sim()
    # joint_pos, tcp_pose = robot.read()
    # joint_pos[4]= joint_pos[4] + 1
    # robot.robot.moveJ(joint_pos,0.1,0.5, False)
    # print(f'Current Joint Pose:{joint_pos}')
    try:
        while True:
            if keyboard_listener.should_exit():
                break
            joint_pos, tcp_pose = robot.read()
            #joint_pos[3] = joint_pos[3] + 0.1
            # joint_pos[3] = -2.3
            # joint_pos[4] = 0.2
            # robot.robot1.moveJ(joint_pos, 0.3, 0.5, False)
            # print(f'Current Joint Pose:{joint_pos}')
            # break
            # # 获取当前帧的头部和手部数据
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            #head_rmat, right_pose, right_qpos = teleoperator.step_right()
            #right_pose = left_pose.copy()
            if start_flag == 0:
                print(f'Right Hand Pose: {right_pose}')
            # print(f'Right Hand Qpos: {right_qpos}')
            joint_pos, tcp_pose = robot.read()
            #print(f'Current TCP Pose: {tcp_pose}')
            tcp_pose = list(tcp_pose)
            # tcp_pose[2] -= 0.15
            # robot.move_safety(tcp_pose)
            # break
            # 创建新的目标位姿，基于当前TCP位姿和手部位置变化

            if keyboard_listener.is_robot_reset():
                print("机器人已重置！")
                # 重置机器人到初始位置
                joint_pos, tcp_pose = robot.read()
                print(f"当前关节角度: {joint_pos}")
                print(f"当前TCP位姿: {tcp_pose}")
                arm_pose = tcp_pose.copy()
                arm_pose[2] -= 0.2
                #joint_pos[3] +=  -0.4
                joint_pos = [0, -2.1, -2.1, -2.1, 0, 0]
                #joint_pos = [0, -2.1, -2.1, -3.6, 0, 0]
                robot.robot1.moveJ(joint_pos, 0.3, 0.5, False)
                #robot.move_safety(arm_pose)
                #time.sleep(1)
                break
            
            if keyboard_listener.is_robot_enabled() and start_flag == 0:
                start_flag = 1
                
            else:
                print("机器人未启动，等待按 'a' 键...")
            if start_flag == 1:
                time.sleep(2)  # 等待一段时间以确保数据稳定
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
                old_pose = right_pose.copy()
                old_tcp_pose = list(tcp_pose) 
                quat = right_pose[3:7]  # x, y, z, w
                R = rotations.matrix_from_quaternion([quat[3], quat[0], quat[1], quat[2]])  # w, x, y, z
                mirror = np.diag([-1, -1, 1])
                R_new = mirror @ R @ mirror
                theta = np.pi / 2  # 90度
                Rz = rotations.matrix_from_axis_angle([0, 0, 1, theta])
                R_new = Rz @ R_new
                axis_angle = rotations.axis_angle_from_matrix(R_new)
                rx, ry, rz = axis_angle[:3] * axis_angle[3]
                old_rx = rx
                old_ry = ry
                old_rz = rz
                start_flag = 2

            if start_flag == 2:
                # 转换为列表以便修改
                joint_pos, tcp_pose = robot.read()
                print(f"当前关节角度: {joint_pos}")
                print(f"当前TCP位姿33333333333: {tcp_pose}")
                new_tcp_pose = list(tcp_pose)
                new_tcp_pose[0] = old_tcp_pose[0] + (old_pose[0] - right_pose[0]) *3   # x方向
                new_tcp_pose[1] = old_tcp_pose[1] + (old_pose[1] - right_pose[1]) *3 # y方向
                if new_tcp_pose[1] > (old_tcp_pose[1] + 0.2):
                    new_tcp_pose[1] = old_tcp_pose[1] + 0.1
                if new_tcp_pose[1] < (old_tcp_pose[1] - 1):
                    new_tcp_pose[1] = old_tcp_pose[1] - 1     
                if new_tcp_pose[0] > (old_tcp_pose[0] + 0.6):
                    new_tcp_pose[0] = old_tcp_pose[0] + 0.6
                if new_tcp_pose[0] < (old_tcp_pose[0] - 0.6):
                    new_tcp_pose[0] = old_tcp_pose[0] - 0.6             
                new_tcp_pose[2] = old_tcp_pose[2] - (old_pose[2] - right_pose[2])  # z方向
                if new_tcp_pose[2] < 0.05:
                    new_tcp_pose[2] = 0.06
                
                # new_tcp_pose[0] = old_tcp_pose[0] - (old_pose[0] - right_pose[0])    # x方向
                # new_tcp_pose[1] = old_tcp_pose[1] - (old_pose[1] - right_pose[1])  # y方向
                # new_tcp_pose[2] = old_tcp_pose[2] - (old_pose[2] - right_pose[2])  # z方向
                
                quat = right_pose[3:7]  # x, y, z, w
                R = rotations.matrix_from_quaternion([quat[3], quat[0], quat[1], quat[2]])  # w, x, y, z
                mirror = np.diag([-1, -1, 1])
                R_new = mirror @ R @ mirror
                theta = np.pi / 2  # 90度
                Rz = rotations.matrix_from_axis_angle([0, 0, 1, theta])

                # 3. 先左乘Rz，再乘R
                R_new = Rz @ R_new
                axis_angle = rotations.axis_angle_from_matrix(R_new)
                rx, ry, rz = axis_angle[:3] * axis_angle[3]
                # new_tcp_pose[3] = rx
                # new_tcp_pose[4] = ry
                # new_tcp_pose[5] = rz
                new_tcp_pose[3] = old_tcp_pose[3]+ round(random.uniform(-0.1, 0.1), 3) * 0.5
                new_tcp_pose[4] = old_tcp_pose[4]+ round(random.uniform(-0.1, 0.1), 3) * 0.5
                new_tcp_pose[5] = old_tcp_pose[5]+ round(random.uniform(-0.1, 0.1), 3) * 0.5
                final_joint = robot.robot1.getInverseKinematics(
                    new_tcp_pose,          # 第一个参数必须是目标位姿列表，不可用target_pose=
                    joint_pos,              # 第二个参数为q_near列表，不可为None
                    max_position_error=10, 
                    max_orientation_error=10
                )
                print(f'New TCP Pose: {new_tcp_pose}')
                print(f'Old joint Pose22222: {joint_pos}')
                print(f'New Joint Angles1111111111111111111111: {final_joint}')
                if final_joint[3] > -0.5:
                    final_joint[3] = -0.6
                elif final_joint[3] < -3.6:
                    final_joint[3] = -3.5
                #for i in range(5)
                velocity = 0.2
                acceleration = 0.5
                robot.robot1.moveJ(final_joint, velocity, acceleration, False)
                #robot.move_safety(new_tcp_pose)
                # target_pose_quat = []
                # target_pose_quat.extend(new_tcp_pose[:3])
                # quat = right_pose[3:7] # x, y, z, w
                # target_pose_quat.append(quat[3])
                # target_pose_quat.extend(quat[:3])
                # target_pose_quat[4] = - target_pose_quat[4]
                # target_pose_quat[5] = - target_pose_quat[5]
                # print(f'New target TCP Pose22222222222222: {new_tcp_pose}')
                # tcp_pose = list(tcp_pose)
                # for i in range(3):
                #     tcp_pose[i+3] = tcp_pose[i+3] + round(random.uniform(-0.2, 0.2), 3)
                # quat1 = axis_angle_to_quaternion(tcp_pose[3], tcp_pose[4], tcp_pose[5])

                # target_pose_quat.extend(quat1)
                # joint_angles = ur5e_arm.inverse(target_pose_quat, False , q_guess=joint_pos)
                # print(f'New Joint Angles1111111111111: {joint_angles}')
                #robot.robot1.moveJ(joint_angles, 0.1, 0.5, False)
            time.sleep(0.05)
    except KeyboardInterrupt:
        # 退出时释放资源
        exit(0)        '''
    main()

