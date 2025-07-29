import numpy as np
import pyrealsense2 as rs
import cv2
from demo_can import hand_control,hand_start,hand_read,write6
import random
import time
class URRobotLeRobot:
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
            "hand_1.pos": float,  # 假设手指1的动作
            "hand_2.pos": float,  # 假设手指2的动作
            "hand_3.pos": float,  # 假设手指3的动作
            "hand_4.pos": float,  # 假设手指4的动作
            "hand_5.pos": float,  # 假设手指5的动作
            "hand_6.pos": float,  # 假设手指6的动作
            "webcam": (720, 720, 3),  # 假设相机分辨率为480x640
        }

    def __init__(self, robot_ip="192.168.31.2", camera_id=0):
        import rtde_control
        import rtde_receive
        import cv2

        self.robot1 = rtde_control.RTDEControlInterface(robot_ip)
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.camera = cv2.VideoCapture(camera_id)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.ser1 = hand_start()
        self.old_hand_pose = [1000,1000,1000,1000,1000,0] # 初始化手部姿态
        self.old_hand_observation = [1000,1000,1000,1000,1000,0]  # 初始化手部观测
        self.old_arm_pose = []
        self.start_flag1 = 0
    def connect(self):
        pass

    def disconnect(self):
        #self.pipeline.release()
        pass
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
        
    def get_observation(self):
        # 获取关节角度
        joints, tcp_pose = self.read()
        #joints = np.rad2deg(joints)
        #start_time = time.perf_counter()
        #time.sleep(0.015)  # 等待0.01秒，确保数据稳定
        hands = hand_read(self.ser1)
        #print(f"获取手部数据: {hands}")
        #print(f"获取手部数据耗时: {time.perf_counter() - start_time:.4f} 秒")
        joint_dict = {f"joint_{i+1}.pos": float(joints[i]) for i in range(6)}
        tcp_dict = {f"tcp_{i+1}.pos": float(tcp_pose[i]) for i in range(6)}
        #print(f"tcp数据: {tcp_dict}")
        if len(hands) != 6:
            hands = self.old_hand_observation  # 如果手部数据异常，使用默认值
            #print("手部数据异常，使用默认值 [0, 0, 0, 0, 0, 0]")
        else:
            for i in range(6):
                if hands[i] > 1000:
                    hands = self.old_hand_observation
            self.old_hand_observation = hands
        
        hand_dict = {f"hand_{i+1}.pos": int(hands[i]) for i in range(6)}
        #print(f"手部数据: {hands}")
        obs = {**joint_dict, **tcp_dict, **hand_dict}
        # 获取相机图像
        #start_time = time.perf_counter()
        frames = self.pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        #print(f"获取图像数据耗时: {time.perf_counter() - start_time:.4f} 秒")
        #if depth_frame and color_frame:
        if color_frame:
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            cropped_image = color_image[:, 240:-320, :]
            color_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
        else:
            #depth_image = np.zeros((480, 640), dtype=np.uint16)
            color_image = np.zeros((720, 720, 3), dtype=np.uint8)
        obs["webcam"] = color_image
        #obs["depth"] = depth_image
        return obs 
    # def get_observation(self):
    #     # 获取关节角度
        
    #     joints, tcp_pose = self.read()
    #     obs = {}
    #     #joints = np.rad2deg(joints)
    #     #start_time = time.perf_counter()
    #     #time.sleep(0.015)  # 等待0.01秒，确保数据稳定
    #     time.sleep(0.02)  # 等待0.01秒，确保数据稳定
    #     hands = hand_read(self.ser1)
    #     print(f"获取手部数据: {hands}")
    #     #print(f"获取手部数据: {hands}")
    #     #print(f"获取手部数据耗时: {time.perf_counter() - start_time:.4f} 秒")
    #     #joint_dict = {f"joint_{i+1}.pos": float(joints[i]) for i in range(6)}
    #     #tcp_dict = {f"tcp_{i+1}.pos": float(tcp_pose[i]) for i in range(6)}
    #     if len(hands) != 6:
    #         #hands = self.old_hand_observation  # 如果手部数据异常，使用默认值
    #         #print("手部数据异常，使用默认值 [0, 0, 0, 0, 0, 0]")
    #         obs["state.gripper"] = np.array([[0.0]])
    #     else:
    #         for i in range(6):
    #             if hands[i] > 1000:
    #                 obs["state.gripper"] = np.array([[0.0]])
    #         #self.old_hand_observation = obs["state.gripper"]
    #     if hands[0] > 800:
    #         obs["state.gripper"] = np.array([[0.0]])
    #     else:
    #         obs["state.gripper"] = np.array([[1.0]])
    #     #print(f"手部数据: {hands}")

    #     #obs = {**joint_dict,**hand_dict}
    #     # 获取相机图像
    #     #start_time = time.perf_counter()
    #     frames = self.pipeline.wait_for_frames()
    #     #depth_frame = frames.get_depth_frame()
    #     color_frame = frames.get_color_frame()
    #     #print(f"获取图像数据耗时: {time.perf_counter() - start_time:.4f} 秒")
    #     #if depth_frame and color_frame:
    #     if color_frame:
    #         #depth_image = np.asanyarray(depth_frame.get_data())
    #         color_image = np.asanyarray(color_frame.get_data())
    #         #cropped_image = color_image[50:, 240:-370, :]
    #         color_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
    #     else:
    #         #depth_image = np.zeros((480, 640), dtype=np.uint16)
    #         color_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    #     # obs = {
    #     #             # 摄像头数据 - 单帧图像 (1, height, width, 3)
    #     #             "video.webcam": np.random.randint(0, 255, (1, 670, 670, 3), dtype=np.uint8),
                    
    #     #             # 状态数据 - 当前机器人状态 (1, 7)
    #     #             "state.single_arm": np.array([[x, y, z]]),  # 末端位置
    #     #             "state.eef_rotation": np.array([[rx, ry, rz]]),  # 末端旋转
    #     #             "state.gripper": np.array([[gripper_pos]]),  # gripper状态
                    
    #     #             # 任务描述
    #     #             "annotation.human.task_description": ["Grab the pink cube and put it in the box"]
    #     #         }

    #     obs["video.webcam"] = color_image
    #     obs["state.single_arm"] = np.array(joints)  # 末端位置
    #     obs["annotation.human.task_description"] = ["Grab the pink cube and put it in the box"]
    #     #obs["webcam"] = color_image
    #     #obs["depth"] = depth_image
    #     return obs
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
        result = hand_control(self.ser1,hand_targets)  # 控制手指动作
        # 返回实际执行的动作
        return action

    @property
    def cameras(self):
        # LeRobot用于判断有无相机
        return {"webcam": self.pipeline}
    def get_action(self, events, old_control):

        if self.start_flag1 == 0:
            self.old_joint_pose, self.old_tcp_pose = self.get_old_action()
        joint_pos, tcp_pose = self.read()
        #tcp_pose = [0.45, -0.23, 0.10, 1.55, -0.08, 0.045]
        #time.sleep(0.01)
        new_control = [0,0,0,0,0,0,0,0] 
        for i in range(8):
            if events["control"][i] > 0:
                new_control[i] = events["control"][i] * 0.007
                events["control"][i] = 0
        new = new_control.copy()
        new[0] = new_control[3]
        new[1] = new_control[2]
        new[2] = new_control[0]
        new[3] = new_control[1]
        new_control = new
        #print(events["control"][0] - old_control[0])
        #print(f'control signal:66666666666666: {new_control}')           
        new_tcp_pose = list(tcp_pose)
        new_tcp_pose[0] = tcp_pose[0] + (new_control[0]*5 - new_control[1]*5) * 0.01 # x方向
        new_tcp_pose[1] = tcp_pose[1] + (new_control[2]*5 - new_control[3]*5) * 0.01
        new_tcp_pose[2] = tcp_pose[2] + (new_control[4]*40 - new_control[5]*4) * 0.01
        if new_tcp_pose[2] > 0.15:
            new_tcp_pose[0] = tcp_pose[0] + (new_control[0]*30 - new_control[1]*5) * 0.01
        # for i in range(2):
        #     if new_tcp_pose[i] - tcp_pose[i] > 0.5:
        #         new_tcp_pose[i] = tcp_pose[i] + 0.5
        #         print(f"new_tcp_pose[{i}] 超过0.5，已调整为 {new_tcp_pose[i]}")
        #     elif new_tcp_pose[i] - tcp_pose[i] < -0.5:
        #         new_tcp_pose[i] = tcp_pose[i] - 0.5
        #         print(f"new_tcp_pose[{i}] 超过-0.5，已调整为 {new_tcp_pose[i]}")
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
        if new_tcp_pose[2] < 0.03:
            new_tcp_pose[2] = 0.03
        if new_tcp_pose[2] > 0.25:
            new_tcp_pose[2] = 0.25


        # new_tcp_pose[3] = self.old_tcp_pose[3]+ round(random.uniform(-0.1, 0.1), 3) * 0.25
        # new_tcp_pose[4] = self.old_tcp_pose[4]+ round(random.uniform(-0.1, 0.1), 3) * 0.25
        # new_tcp_pose[5] = self.old_tcp_pose[5]+ round(random.uniform(-0.1, 0.1), 3) * 0.25
        current_q = joint_pos  # 返回当前6个关节角的列表
        velocity = 0.2
        acceleration = 0.5
        # self.robot2.robot1.moveL(new_tcp_pose, speed = velocity, acceleration=acceleration, asynchronous=False)
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
        #final_joint = current_q
        #print(f'Final Joint: {final_joint}')
        #[200, 200, 200, 200, 500, 0]  抓
        #[1000, 1000, 1000, 1000, 1000, 1000]  放
        #hand_pose = [0,0,0,0,500,500]
        if events["control"][6] >= 1:
            print("手部抓取动作开始")
            time.sleep(0.01)
            hand_pose = [400,400,400,400,700,0]
            time.sleep(0.01)
            hand_pose = [400,400,400,400,700,0]
            events["control"][6] = 0
            self.old_hand_pose = hand_pose
            events["control"][6] = 0
        elif events["control"][7] >= 1:
            print("手部放开动作开始")
            time.sleep(0.01)
            hand_pose = [1000,1000,1000,1000,1000,0]
            time.sleep(0.01)
            hand_pose = [1000,1000,1000,1000,1000,0]
            events["control"][7] = 0      
            self.old_hand_pose = hand_pose   
            events["control"][6] = 0
        else:
            hand_pose = self.old_hand_pose
        #print     
        #print(f'control signal:66666666666666: {new_tcp_pose}')                
        action = {
            "joint_1.pos": final_joint[0],
            "joint_2.pos": final_joint[1],
            "joint_3.pos": final_joint[2],
            "joint_4.pos": final_joint[3],
            "joint_5.pos": final_joint[4],
            "joint_6.pos": final_joint[5],
            "tcp_1.pos": new_tcp_pose[0],  # 假设TCP动作
            "tcp_2.pos": new_tcp_pose[1],  # 假设TCP动作
            "tcp_3.pos": new_tcp_pose[2],  # 假设TCP动作
            "tcp_4.pos": new_tcp_pose[3],  # 假设TCP动作
            "tcp_5.pos": new_tcp_pose[4],  # 假设TCP动作
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