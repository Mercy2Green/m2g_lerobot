#!/usr/bin/env python3
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
#from control_hand import hand_control,hand_start,hand_read,write6
from demo_can import hand_control1,hand_start,hand_read,write6
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

# ===== 1. 初始化UR机械臂连接 =====
class UR5e_arm:
    def __init__(self, robot_ip: str = "192.168.31.2"):
        import rtde_control
        import rtde_receive

        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(e)
            print(robot_ip)

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        [print("connect") for _ in range(4)]

        self._free_drive = False
        self.robot.endFreedriveMode()
        
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
        self.robot.moveL(target_tcp, speed=speed, acceleration=acceleration, asynchronous=False)

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
        self.move_to_tcp(target_tcp)
        time.sleep(1)  # 等待机械臂运动完成
def control_hand(ser,right_qpos):
    """控制手部动作"""
    # 这里可以添加手部控制的具体实现
    selected = [
        right_qpos[0],   # 1st
        right_qpos[2],   # 3rd
        right_qpos[4],   # 5th
        right_qpos[6],   # 7th
        right_qpos[8],   # 9th
        right_qpos[10]   # 11th
    ]
    
    # 映射前4个元素 (1.0-1.7 → 0-1000)
    mapped_first_part = []
    for value in selected[:4]:
        # 线性映射公式：y = (x - min) * (1000/(max-min))
        mapped_value = 1000 - (value - 1.0) * (1000 / 0.7)
        # 确保结果在[0,1000]范围内
        mapped_value = max(0, min(1000, mapped_value))
        mapped_first_part.append(mapped_value)
    
    # 映射后2个元素 (0-0.8 → 0-1000)
    mapped_second_part = []
    value = selected[4]  # 5th
    mapped_value = 1000 - value * (1000 / 1.3)
    mapped_value = max(0, min(1000, mapped_value))
    mapped_second_part.append(mapped_value)
    value = selected[5]  # 5th
    mapped_value = 1000 - value * (1000 / 0.8)
    mapped_value = max(0, min(1000, mapped_value))
    mapped_second_part.append(mapped_value)
    result = [
     int(mapped_first_part[2]),
     int(mapped_first_part[3]),
     int(mapped_first_part[1]),
     int(mapped_first_part[0]),
     int(mapped_second_part[1]),
     int(mapped_second_part[0])
    ]

    print(f'映射后的手部动作: {result}')
    write6(ser, 1, 'angleSet', result)
    #time.sleep(0.01) 
    # 组合结果
    return result
if __name__ == '__main__':
    # 主流程：初始化远程操作与仿真，循环处理与同步图像
    keyboard_listener = KeyboardListener()
    keyboard_listener.start_listening()
    teleoperator = VuerTeleop('/home/hpx/vla/src/lerobot/lerobot/inspire_hand.yml')
    #robot= UR5e_arm()
    start_flag = 0
    # joint_pos, tcp_pose = robot.read()
    # joint_pos[4]= joint_pos[4] + 1
    # robot.robot.moveJ(joint_pos,0.1,0.5, False)
    # print(f'Current Joint Pose:{joint_pos}')
    try:
        while True:
            if keyboard_listener.should_exit():
                break
            # 获取当前帧的头部和手部数据
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            #head_rmat, right_pose, right_qpos = teleoperator.step_right()
            #right_pose = left_pose.copy()
            #print(f'Right Hand Pose: {right_pose}')
            #print(f'HEAD: {head_rmat}')
            #print(f'Left Hand Qpos: {right_qpos[8:12]}')
            print(f'Right Hand Qpos!!!!!!!!!: {right_qpos}')
            # print(f'Right Hand Qpos: {right_qpos}')
            #joint_pos, tcp_pose = robot.read()
            #print(f'Current TCP Pose: {tcp_pose}')
            #tcp_pose = list(tcp_pose)
            # tcp_pose[2] -= 0.15
            # robot.move_safety(tcp_pose)
            # break
            # 创建新的目标位姿，基于当前TCP位姿和手部位置变化
            #ser = hand_start()

            #hand_control(ser)
            #time.sleep(0.1)  # 等待手部控制器准备就绪
            #hand_read(ser)
            #break
            
            if keyboard_listener.is_robot_enabled() and start_flag == 0:
                start_flag = 1
                
            else:
                print("机器人未启动，等待按 'a' 键...")
            if start_flag == 1:
                #time.sleep(2)  # 等待一段时间以确保数据稳定
                head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
                print(f'Right Hand Pose: {right_pose}')
                print(f'888888 Qpos: {right_qpos}')
                ser = hand_start()
                result = hand_control1(ser, right_qpos)
                print(f'控制手部动作: {result}')

            #time.sleep(0.1)
    except KeyboardInterrupt:
        # 退出时释放资源
        exit(0)
 
