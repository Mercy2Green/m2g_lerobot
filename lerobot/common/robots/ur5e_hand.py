import numpy as np
import pyrealsense2 as rs
from pyorbbecsdk import *
import cv2
import numpy as np
from typing import Union, Any, Optional
from demo_modbus import hand_control,hand_start,hand_read
import random
from scipy.spatial.transform import Rotation as R
import time
import rtde_control
import rtde_receive
import cv2
from pymodbus.client.sync import ModbusTcpClient  # pip install pymodbus==2.5.3
from lerobot.common.robots.dex_hand import DexHandClient


class UR5eHand:
    """
    LeRobot兼容的UR机械臂Robot类（6关节+1相机，无gripper）。
    """

    name = "ur5e"
    robot_type = "ur5e"

    # 6个关节动作
    @property
    def action_features(self):
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "tcp_1.pos": float,  # 假设TCP动作
            "tcp_2.pos": float,  # 假设TCP动作
            "tcp_3.pos": float,  # 假设TCP动作
            "tcp_4.pos": float,  # 假设TCP动作
            "tcp_5.pos": float,  # 假设TCP动作
            "tcp_6.pos": float,  # 假设TCP动作
            "hand_1.pos": float,  # 假设手指1的动作
            "hand_2.pos": float,  # 假设手指2的动作
            "hand_3.pos": float,  # 假设手指3的动作
            "hand_4.pos": float,  # 假设手指4的动作
            "hand_5.pos": float,  # 假设手指5的动作
            "hand_6.pos": float,  # 假设手指6的动作
        }

    # 观测：6关节+相机
    @property
    def observation_features(self):
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,
            "tcp_1.pos": float,  # 假设TCP动作
            "tcp_2.pos": float,  # 假设TCP动作
            "tcp_3.pos": float,  # 假设TCP动作
            "tcp_4.pos": float,  # 假设TCP动作
            "tcp_5.pos": float,  # 假设TCP动作
            "tcp_6.pos": float,  # 假设TCP动作
            "hand_angle_1.pos": float,  # 假设手指1的动作
            "hand_angle_2.pos": float,  # 假设手指2的动作
            "hand_angle_3.pos": float,  # 假设手指3的动作
            "hand_angle_4.pos": float,  # 假设手指4的动作
            "hand_angle_5.pos": float,  # 假设手指5的动作
            "hand_angle_6.pos": float,  # 假设手指6的动作
            "hand_force_1.pos":float,
            "hand_force_2.pos":float,
            "hand_force_3.pos":float,
            "hand_force_4.pos":float,
            "hand_force_5.pos":float,
            "hand_force_6.pos":float,
            "head_camera": (720,720,3),  # 假设相机分辨率为480x640
            "wrist_camera":(720,720,3),
            "hand_tactile": dict[str, np.ndarray],  # 假设触觉数据为字典
        }

    def __init__(
            self, 
            robot_ip="192.168.31.2", 
            hand_ip="192.168.11.210",
            hand_port=6000,
            init_force_values=[100, 100, 100, 100, 100, 100],
            init_speed_values=[500, 500, 500, 500, 500, 500],
            init_arm_pose = [-1.2, -1.6716, -1.5113, -3.71581, -1.29335, -2.890]
            ):

        self.robot1 = rtde_control.RTDEControlInterface(robot_ip)
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)

        self.init_arm_pose = init_arm_pose  # 初始化机械臂姿态

        self.hand = DexHandClient(ip=hand_ip, port=hand_port)  # 初始化Dex手客户端
        self.hand.connect()  # 连接Dex手
        self.hand.init_hand(
            force_values= init_force_values,
            speed_values= init_speed_values
        )  # 初始化Dex手,包括设置力和速度参数，并且reset力感受

        #'''
        self.config1 = Config()
        self.pipeline1 = Pipeline()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        try:
            # 获取摄像头的流配置
            profile_list = self.pipeline1.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            try:
                # 设置分辨率为 1280x720，格式为 RGB，帧率为 30
                color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
            except OBError as e:
                print(e)
                color_profile = profile_list.get_default_video_stream_profile()
                print("使用默认的 color profile: ", color_profile)
            self.config1.enable_stream(color_profile)
        except Exception as e:
            print("配置头部摄像头失败:", e)
            return

        # 启动摄像头
        self.pipeline1.start(self.config1)
        # self.ser1 = hand_start()
        #'''
        self.old_hand_pose = [1000,1000,1000,1000,1000,0] # 初始化手部姿态
        self.old_hand_observation = [1000,1000,1000,1000,1000,0]  # 初始化手部观测
        self.old_arm_pose = []
        self.start_flag1 = 0


        ###### For auto mode ######
        # ----- 自动 episode 状态机配置 -----
        self.auto_state = "idle"  # 初始状态
        self.auto_cfg = {
            "z_down": 0.10,          # 相对下降距离(米)
            "z_lift": 0.10,          # 从抓取高度向上抬升距离(米)
            "wait_s": 3.0,           # 抬起后的等待时间(秒)
            "down_dwell_s": 2,    # 到达抓取高度后的驻留时间
            "grasp_dwell_s": 3,   # 抓取动作后的驻留时间
            "down2_dwell_s": 2,   # 第二次下降后的驻留时间
            "release_dwell_s": 3, # 放开动作后的驻留时间
            "rise_dwell_s": 0.5,    # 回到悬停高度后的驻留时间
            "grasp":   [400,400,400,400,700,0],      # 抓取时的手指角度
            "release": [1000,1000,1000,1000,1000,0],  # 放开时的手指角度
            "max_z": 0.5,           # Z 轴最大高度限制
            "min_z": 0.1,           # Z 轴最小高度限制
        }
        self.auto_cache = {
            "start_ref_z": None  # 自动模式首次进入时记录一次参考悬停高度
        }


    def connect(self):
        pass

    def disconnect(self):
        pass
    def read(self):
        """Get the current state of the UR robot.

        Returns:
            T: The current state of the UR robot.
        """
        joint_pos = self.r_inter.getActualQ()
        tcp_pose = self.r_inter.getActualTCPPose()

        return joint_pos, tcp_pose

    def frame_to_bgr_image(self,frame: VideoFrame) -> Union[Optional[np.array], Any]:
        width = frame.get_width()
        height = frame.get_height()
        color_format = frame.get_format()
        data = np.asanyarray(frame.get_data())
        image = np.zeros((height, width, 3), dtype=np.uint8)
        if color_format == OBFormat.RGB:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif color_format == OBFormat.BGR:
            image = np.resize(data, (height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif color_format == OBFormat.YUYV:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
        elif color_format == OBFormat.MJPG:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif color_format == OBFormat.I420:
            image = i420_to_bgr(data, width, height)
            return image
        elif color_format == OBFormat.NV12:
            image = nv12_to_bgr(data, width, height)
            return image
        elif color_format == OBFormat.NV21:
            image = nv21_to_bgr(data, width, height)
            return image
        elif color_format == OBFormat.UYVY:
            image = np.resize(data, (height, width, 2))
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
        else:
            print("Unsupported color format: {}".format(color_format))
            return None
        return image 
    

    def get_observation(self):

        #### UR机械臂状态读取 ####
        joints, tcp_pose = self.read()
        joint_dict = {f"joint_{i+1}.pos": float(joints[i]) for i in range(6)}
        tcp_dict = {f"tcp_{i+1}.pos": float(tcp_pose[i]) for i in range(6)}

        #### Dex手状态读取 ####
        hand_data_dict = self.hand.read_force_angle_tactile()
        obs = {
            **joint_dict,
            **tcp_dict,
            **{f"hand_angle_{i+1}.pos": int(val) for i, val in enumerate(hand_data_dict["angles"])},
            **{f"hand_force_{i+1}.pos": int(val) for i, val in enumerate(hand_data_dict["forces"])},
            "hand_tactile": hand_data_dict["tactile"]
        }

        #### 相机数据读取 ####
        frames1 = self.pipeline.wait_for_frames()
        color_frame = frames1.get_color_frame()
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            color_image = color_image[:, 240:-320, :]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        else:
            color_image = np.zeros((720, 720, 3), dtype=np.uint8)
        obs["wrist_camera"] = color_image
        frames2 = self.pipeline1.wait_for_frames(1)
        if frames2:
            color_frame = frames2.get_color_frame()
            color_image = self.frame_to_bgr_image(color_frame)
            color_image = color_image[:, 240:-320, :]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        else:
            color_image = np.zeros((720, 720, 3), dtype=np.uint8)
        obs["head_camera"]= color_image

        return obs 

    def send_action(self, action: dict):
        # action: dict, key为joint_x.pos，value为角度（度）
        joint_targets = [action[f"joint_{i+1}.pos"] for i in range(6)]
        tcp_targets = [action[f"tcp_{i+1}.pos"] for i in range(6)]
        hand_targets = [action[f"hand_{i+1}.pos"] for i in range(6)]
        #joint_targets_rad = np.radians(joint_targets)
        velocity = 0.2
        acceleration = 0.5
        self.robot1.moveL(tcp_targets, velocity, acceleration, False)
        #self.robot1.moveJ(joint_targets, velocity, acceleration, False)

        # result = hand_control(self.hand,hand_targets)  # 控制手指动作
        self.hand.set_hand_angle(hand_targets)

        # 返回实际执行的动作
        return action

    @property
    def cameras(self):
        # LeRobot用于判断有无相机
        return {"webcam": self.pipeline}
    def get_action(self, events):

        # --- 自动模式：覆盖手动按键 ---
        if events.get("auto_mode", False):
            return self._auto_action(events)

        if self.start_flag1 == 0:
            self.old_joint_pose, self.old_tcp_pose = self.get_old_action()
        joint_pos, tcp_pose = self.read()
        new_control = [0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
        for i in range(14):
            if events["control"][i] > 0:
                new_control[i] = events["control"][i] * 0.007
                events["control"][i] = 0
        new = new_control.copy()
        '''
        new[0] = new_control[3]
        new[1] = new_control[2]
        new[2] = new_control[0]
        new[3] = new_control[1]
        '''
        new_control = new

        # 统一的位移缩放系数（米）
        MOVE_SCALE = 0.01  #越大越快，越小越精细

        # 各方向正反向增益
        GAIN_X_POS = 20    # X正向（new_control[0]）
        GAIN_X_NEG = 20     # X反向（new_control[1]）

        GAIN_Y_POS = 20    # Y正向（new_control[2]）
        GAIN_Y_NEG = 20     # Y反向（new_control[3]）

        GAIN_Z_UP   = 20   # Z上升（new_control[4]）
        GAIN_Z_DOWN = 20    # Z下降（new_control[5]）

        # 计算新的 TCP 位置
        new_tcp_pose = list(tcp_pose)
        new_tcp_pose[0] = tcp_pose[0] - (new_control[0] * GAIN_X_POS - new_control[1] * GAIN_X_NEG) * MOVE_SCALE
        new_tcp_pose[1] = tcp_pose[1] - (new_control[2] * GAIN_Y_POS - new_control[3] * GAIN_Y_NEG) * MOVE_SCALE
        new_tcp_pose[2] = tcp_pose[2] + (new_control[4] * GAIN_Z_UP  - new_control[5] * GAIN_Z_DOWN) * MOVE_SCALE

        r_current = R.from_rotvec(tcp_pose[3:6])
        theta_x = (new_control[8]*5 - new_control[9]*5) * 0.01 
        theta_y = (new_control[10]*5 - new_control[11]*5) * 0.01 
        theta_z = (new_control[12]*5 - new_control[13]*5) * 0.01 
        #第一位上下，第三位左右
        r_z90 = R.from_rotvec([theta_x, theta_y, theta_z])  # 绕 Z 轴旋转
        # 3. 组合旋转矩阵
        r_combined = r_z90 * r_current  # 注意旋转的顺序
        # 4. 将组合后的旋转矩阵转换回旋转向量
        combined_rotvec = r_combined.as_rotvec()
        new_tcp_pose[3] = combined_rotvec[0]
        new_tcp_pose[4] = combined_rotvec[1]
        new_tcp_pose[5] = combined_rotvec[2]        
        if new_tcp_pose[2] > 0.15:
            new_tcp_pose[0] = tcp_pose[0] - (new_control[0]*30 - new_control[1]*5) * 0.01
        if new_tcp_pose[1] > (self.old_tcp_pose[1] + 0.15):
            new_tcp_pose[1] = self.old_tcp_pose[1] + 0.15
            print(f"new_tcp_pose[1] 超过0.5，已调整为 {new_tcp_pose[1]}")
        if new_tcp_pose[1] < (self.old_tcp_pose[1] - 0.16):
            new_tcp_pose[1] = self.old_tcp_pose[1] - 0.16     
            print(f"new_tcp_pose[1] 超过-0.5，已调整为 {new_tcp_pose[1]}")
        if new_tcp_pose[0] > (self.old_tcp_pose[0] + 0.4):
            new_tcp_pose[0] = self.old_tcp_pose[0] + 0.4
            print(f"new_tcp_pose[0] 超过0.4，已调整为 {new_tcp_pose[0]}")
        if new_tcp_pose[0] < (self.old_tcp_pose[0] - 0.4):
            new_tcp_pose[0] = self.old_tcp_pose[0] - 0.4  
            print(f"new_tcp_pose[0] 超过-0.4，已调整为 {new_tcp_pose[0]}")
        if (new_tcp_pose[0] < (self.old_tcp_pose[0] - 0.05)) and (new_tcp_pose[1] > (self.old_tcp_pose[1] + 0.12)):
            new_tcp_pose[1] = (self.old_tcp_pose[1] + 0.12)
            print(f"new_tcp_pose[0] 和 new_tcp_pose[1] 超过限制，已调整为, {new_tcp_pose[1]}")

        max_z = 0.45
        min_z = 0.03
        
        if new_tcp_pose[2] < min_z:
            new_tcp_pose[2] = min_z
        if new_tcp_pose[2] > max_z:
            new_tcp_pose[2] = max_z

        current_q = joint_pos  # 返回当前6个关节角的列表
        final_joint = self.robot1.getInverseKinematics(
            new_tcp_pose,          # 第一个参数必须是目标位姿列表，不可用target_pose=
            current_q,              # 第二个参数为q_near列表，不可为None
            max_position_error=10, 
            max_orientation_error=10
        )
        if final_joint[3] > -0.5:
            final_joint[3] = -0.6
        elif final_joint[3] < -3.6:
            final_joint[3] = -3.5

        if events["control"][6] > 0:
            print("手部抓取动作开始")
            time.sleep(0.01)
            hand_pose = [400,400,400,400,700,0]
            events["control"][6] = 0
            self.old_hand_pose = hand_pose
            events["control"][6] = 0
        elif events["control"][7] > 0:
            print("手部放开动作开始")
            time.sleep(0.01)
            hand_pose = [1000,1000,1000,1000,1000,0]
            events["control"][7] = 0      
            self.old_hand_pose = hand_pose   
            events["control"][6] = 0
        else:
            hand_pose = self.old_hand_pose
             
        action = {
            "joint_1.pos": final_joint[0],
            "joint_2.pos": final_joint[1],
            "joint_3.pos": final_joint[2],
            "joint_4.pos": final_joint[3],
            "joint_5.pos": final_joint[4],
            "joint_6.pos": final_joint[5],
            "tcp_1.pos": new_tcp_pose[0],  
            "tcp_2.pos": new_tcp_pose[1], 
            "tcp_3.pos": new_tcp_pose[2],  
            "tcp_4.pos": new_tcp_pose[3],  
            "tcp_5.pos": new_tcp_pose[4],  
            "tcp_6.pos": new_tcp_pose[5],  
            "hand_1.pos": hand_pose[0],
            "hand_2.pos": hand_pose[1],
            "hand_3.pos": hand_pose[2],
            "hand_4.pos": hand_pose[3],
            "hand_5.pos": hand_pose[4],
            "hand_6.pos": hand_pose[5],
        }
        return action
    def get_old_action(self):
        if self.start_flag1 == 0:
            joint_pos, old_tcp_pose = self.read()
            self.start_flag1 = 1
            return joint_pos, old_tcp_pose
        
    def _auto_action(self, events):
        joint_pos, tcp_pose = self.read()
        current_tcp = list(tcp_pose)
        hand_pose = self.old_hand_pose

        max_z = self.auto_cfg["max_z"]
        min_z = self.auto_cfg["min_z"]


        # ---------- 初始化 ----------
        if self.auto_state in ("idle", "done"):
            self.auto_state = "down"

            # 只在自动模式的第一集记录一次悬停高度
            if self.auto_cache.get("start_ref_z") is None:
                self.auto_cache["start_ref_z"] = current_tcp[2]

            start_ref_z = float(np.clip(self.auto_cache["start_ref_z"], min_z, max_z))
            grasp_z = max(min_z, start_ref_z - self.auto_cfg["z_down"])
            # 抬起高度回到悬停高度（若 z_lift != z_down，可以改为 grasp_z + z_lift 的 min/max 裁剪）
            lift_z  = start_ref_z

            self.auto_cache.update({
                "grasp_z": grasp_z,
                "lift_z":  lift_z,
                "t0": None,             # wait用
                "t_enter": time.time(), # 状态进入时间
            })

            hand_pose = self.auto_cfg["release"]
            self.old_hand_pose = hand_pose

        # def step_to_z(z_target, z_now, step=0.003):
        #     if abs(z_target - z_now) <= step:
        #         return z_target, True
        #     return (z_now + step, False) if z_target > z_now else (z_now - step, False)
        def step_to_z(z_target, z_now, base_step=0.003, max_step=0.015):
            dist = abs(z_target - z_now)
            step = min(max_step, max(base_step, dist / 2))  # 远时快，近时慢
            if dist <= step:
                return z_target, True
            return (z_now + step, False) if z_target > z_now else (z_now - step, False)

        state = self.auto_state

        # ------- DOWN -------
        if state == "down":
            new_z, reached = step_to_z(self.auto_cache["grasp_z"], current_tcp[2])
            target_tcp = current_tcp[:]
            target_tcp[2] = float(np.clip(new_z, min_z, max_z))
            hand_pose = self.auto_cfg["release"]

            if reached:
                if time.time() - self.auto_cache["t_enter"] >= self.auto_cfg.get("down_dwell_s", 0.0):
                    self.auto_state = "grasp"
                    self.auto_cache["t_enter"] = time.time()
            else:
                self.auto_cache["t_enter"] = time.time()

            return self._build_action_from_tcp_and_hand(target_tcp, hand_pose, joint_pos)

        # ------- GRASP -------
        elif state == "grasp":
            hand_pose = self.auto_cfg["grasp"]
            self.old_hand_pose = hand_pose
            target_tcp = current_tcp[:]
            if time.time() - self.auto_cache["t_enter"] >= self.auto_cfg.get("grasp_dwell_s", 0.0):
                self.auto_state = "lift"
                self.auto_cache["t_enter"] = time.time()
            return self._build_action_from_tcp_and_hand(target_tcp, hand_pose, joint_pos)

        # ------- LIFT -------
        elif state == "lift":
            new_z, reached = step_to_z(self.auto_cache["lift_z"], current_tcp[2])
            target_tcp = current_tcp[:]
            target_tcp[2] = float(np.clip(new_z, min_z, max_z))
            hand_pose = self.auto_cfg["grasp"]

            if reached:
                self.auto_state = "wait"
                self.auto_cache["t0"] = time.time()
                self.auto_cache["t_enter"] = time.time()
            return self._build_action_from_tcp_and_hand(target_tcp, hand_pose, joint_pos)

        # ------- WAIT -------
        elif state == "wait":
            hand_pose = self.auto_cfg["grasp"]
            target_tcp = current_tcp[:]
            if time.time() - self.auto_cache["t0"] >= self.auto_cfg["wait_s"]:
                self.auto_state = "down2"
                self.auto_cache["t_enter"] = time.time()
            return self._build_action_from_tcp_and_hand(target_tcp, hand_pose, joint_pos)

        # ------- DOWN2 -------
        elif state == "down2":
            new_z, reached = step_to_z(self.auto_cache["grasp_z"], current_tcp[2])
            target_tcp = current_tcp[:]
            target_tcp[2] = float(np.clip(new_z, min_z, max_z))
            hand_pose = self.auto_cfg["grasp"]

            if reached:
                if time.time() - self.auto_cache["t_enter"] >= self.auto_cfg.get("down2_dwell_s", 0.0):
                    self.auto_state = "release"
                    self.auto_cache["t_enter"] = time.time()
            else:
                self.auto_cache["t_enter"] = time.time()

            return self._build_action_from_tcp_and_hand(target_tcp, hand_pose, joint_pos)

        # ------- RELEASE -------
        elif state == "release":
            hand_pose = self.auto_cfg["release"]
            self.old_hand_pose = hand_pose
            target_tcp = current_tcp[:]

            if time.time() - self.auto_cache["t_enter"] >= self.auto_cfg.get("release_dwell_s", 0.0):
                # 新增：放开后回到悬停高度，再结束
                self.auto_state = "rise"
                self.auto_cache["t_enter"] = time.time()
            return self._build_action_from_tcp_and_hand(target_tcp, hand_pose, joint_pos)

        # ------- RISE（新增）：回到悬停高度 -------
        elif state == "rise":
            start_ref_z = float(np.clip(self.auto_cache["start_ref_z"], min_z, max_z))
            new_z, reached = step_to_z(start_ref_z, current_tcp[2])
            target_tcp = current_tcp[:]
            target_tcp[2] = float(np.clip(new_z, min_z, max_z))
            hand_pose = self.auto_cfg["release"]

            if reached and (time.time() - self.auto_cache["t_enter"] >= self.auto_cfg.get("rise_dwell_s", 0.0)):
                self.auto_state = "done"
                events["exit_early"] = True
            return self._build_action_from_tcp_and_hand(target_tcp, hand_pose, joint_pos)

        # ------- fallback -------
        else:
            self.auto_state = "idle"
            target_tcp = current_tcp[:]
            return self._build_action_from_tcp_and_hand(target_tcp, self.old_hand_pose, joint_pos)

        
    def _build_action_from_tcp_and_hand(self, target_tcp, hand_pose, current_q):
        # 复制你当前 get_action 里 IK 与限幅那段逻辑（保持一致）
        final_joint = self.robot1.getInverseKinematics(
            target_tcp, current_q, max_position_error=10, max_orientation_error=10
        )
        if final_joint[3] > -0.5:
            final_joint[3] = -0.6
        elif final_joint[3] < -3.6:
            final_joint[3] = -3.5

        action = {
            "joint_1.pos": final_joint[0],
            "joint_2.pos": final_joint[1],
            "joint_3.pos": final_joint[2],
            "joint_4.pos": final_joint[3],
            "joint_5.pos": final_joint[4],
            "joint_6.pos": final_joint[5],
            "tcp_1.pos": target_tcp[0],
            "tcp_2.pos": target_tcp[1],
            "tcp_3.pos": target_tcp[2],
            "tcp_4.pos": target_tcp[3],
            "tcp_5.pos": target_tcp[4],
            "tcp_6.pos": target_tcp[5],
            "hand_1.pos": hand_pose[0],
            "hand_2.pos": hand_pose[1],
            "hand_3.pos": hand_pose[2],
            "hand_4.pos": hand_pose[3],
            "hand_5.pos": hand_pose[4],
            "hand_6.pos": hand_pose[5],
        }
        return action


if __name__ == "__main__":
    robot = UR5eHand()
    while True:
        obs = robot.get_observation()