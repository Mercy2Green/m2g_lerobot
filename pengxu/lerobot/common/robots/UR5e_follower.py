import numpy as np
import pyrealsense2 as rs
from pyorbbecsdk import *
import cv2
import numpy as np
from typing import Union, Any, Optional
#from demo_can import hand_control,hand_start,hand_read,write6
from demo_modbus import hand_control,hand_start,hand_read
import random
from scipy.spatial.transform import Rotation as R
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
            "tac_1.pos":float,
            "tac_2.pos":float,
            "tac_3.pos":float,
            "tac_4.pos":float,
            "tac_5.pos":float,
            "tac_6.pos":float,
            "webcam": (720, 720, 3),  # 假设相机分辨率为480x640
            "wrist_camera":(720,720,3),
        }

    def __init__(self, robot_ip="192.168.31.2", camera_id=0):
        import rtde_control
        import rtde_receive
        import cv2

        self.robot1 = rtde_control.RTDEControlInterface(robot_ip)
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        #self.camera = cv2.VideoCapture(camera_id)

        
        self.config1 = Config()
        self.pipeline1 = Pipeline()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
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
        # 获取关节角度
        joints, tcp_pose = self.read()
        #joints = np.rad2deg(joints)
        #start_time = time.perf_counter()
        #time.sleep(0.015)  # 等待0.01秒，确保数据稳定
        hands = hand_read(self.ser1)
        #print(hands)
        #print(f"获取手部数据: {hands}")
        #print(f"获取手部数据耗时: {time.perf_counter() - start_time:.4f} 秒")
        joint_dict = {f"joint_{i+1}.pos": float(joints[i]) for i in range(6)}
        tcp_dict = {f"tcp_{i+1}.pos": float(tcp_pose[i]) for i in range(6)}
        #print(f"tcp数据: {tcp_dict}")
        # if len(hands) != 6:
        #     hands = self.old_hand_observation  # 如果手部数据异常，使用默认值
        #     #print("手部数据异常，使用默认值 [0, 0, 0, 0, 0, 0]")
        # else:
        #     for i in range(6):
        #         if hands[i] > 1000:
        #             hands = self.old_hand_observation
        #     self.old_hand_observation = hands
        
        hand_dict = {f"hand_{i+1}.pos": int(hands[i]) for i in range(6)}
        tac_dict = {f"tac_{i+1}.pos": int(hands[i+6]) for i in range(6)}
        #print(f"手部数据: {hands}")
        obs = {**joint_dict, **tcp_dict, **hand_dict, **tac_dict}
        # 获取相机图像
        #start_time = time.perf_counter()
        #frames = self.pipeline.wait_for_frames()
        #depth_frame = frames.get_depth_frame()
        frames1 = self.pipeline.wait_for_frames()
        color_frame = frames1.get_color_frame()

        #print(f"获取图像数据耗时: {time.perf_counter() - start_time:.4f} 秒")
        #if depth_frame and color_frame:
        if color_frame:
            #depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            color_image = color_image[:, 240:-320, :]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            # image_filename = f"saved_image_1.png"
            # cv2.imwrite(image_filename, color_image)
        else:
            #depth_image = np.zeros((480, 640), dtype=np.uint16)
            color_image = np.zeros((720, 720, 3), dtype=np.uint8)
        #start_time = time.time()
        obs["wrist_camera"] = color_image
        #obs["depth"] = depth_image
        
        frames2 = self.pipeline1.wait_for_frames(1)
        if frames2:
            color_frame = frames2.get_color_frame()
        # 转换为 RGB 图像

            color_image = self.frame_to_bgr_image(color_frame)
            #color_image = np.array(color_frame, dtype=np.uint8)
            color_image = color_image[:, 240:-320, :]
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        else:
            color_image = np.zeros((720, 720, 3), dtype=np.uint8)
        #end_time = time.time()
        # image_filename = f"saved_image_2.png"
        # cv2.imwrite(image_filename, color_image)
        obs["webcam"]= color_image
        #elapsed_time = end_time - start_time
        #print(f"等待10帧所用时间: {elapsed_time:.4f} 秒")
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
        result = hand_control(self.ser1,hand_targets)  # 控制手指动作
        # 返回实际执行的动作
        return action

    @property
    def cameras(self):
        # LeRobot用于判断有无相机
        return {"webcam": self.pipeline}
    def get_action(self, events):

        if self.start_flag1 == 0:
            self.old_joint_pose, self.old_tcp_pose = self.get_old_action()
        joint_pos, tcp_pose = self.read()
        #tcp_pose = [0.45, -0.23, 0.10, 1.55, -0.08, 0.045]
        #time.sleep(0.01)
        new_control = [0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
        #print(new)
        if len(events["control"]) == 14:
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
        #print(new_control)
        #print(events["control"][0] - old_control[0])
        #print(f'control signal:66666666666666: {new_control}')           
        new_tcp_pose = list(tcp_pose)
        new_tcp_pose[0] = tcp_pose[0] - (new_control[0]*5 - new_control[1]*5) * 0.005 # x方向
        new_tcp_pose[1] = tcp_pose[1] - (new_control[2]*5 - new_control[3]*5) * 0.005
        new_tcp_pose[2] = tcp_pose[2] + (new_control[4]*5 - new_control[5]*5) * 0.005
        #new_tcp_pose[5] = tcp_pose[5] + (new_control[8]*5 - new_control[9]*5) * 0.01
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
        # if new_tcp_pose[2] > 0.15:
        #     new_tcp_pose[0] = tcp_pose[0] - (new_control[0]*30 - new_control[1]*5) * 0.01
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
        if new_tcp_pose[2] < 0.10:
            new_tcp_pose[2] = 0.10
        if new_tcp_pose[2] > 0.25:
            new_tcp_pose[2] = 0.25


        # new_tcp_pose[3] = self.old_tcp_pose[3]+ round(random.uniform(-0.1, 0.1), 3) * 0.25
        # new_tcp_pose[4] = self.old_tcp_pose[4]+ round(random.uniform(-0.1, 0.1), 3) * 0.25
        # new_tcp_pose[5] = self.old_tcp_pose[5]+ round(random.uniform(-0.1, 0.1), 3) * 0.25
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


if __name__ == "__main__":
    robot = URRobotLeRobot()
    while True:
        obs = robot.get_observation()
    #print(obs["tac_5.pos"])
    # s = time.time()
    # for i in range(10):
    #     obs = robot.get_observation()
    #     image_filename = f"saved_image_{i}.png"
    #     cv2.imwrite(image_filename, obs["wrist_camera"] )
    # e = time.time()
    # print(e-s)
        #obs["wrist_camera"] = color_image
    #print(obs)
    # try:
    #     while True:
    #         try:
    #             # 等待帧数据
    #             frames: FrameSet = robot.pipeline1.wait_for_frames(1)
    #             if frames is None:
    #                 continue
    #             color_frame = frames.get_color_frame()
    #             if color_frame is None:
    #                 continue

    #             # 转换为 RGB 图像
    #             color_image = robot.frame_to_bgr_image(color_frame)
    #             #color_image = np.array(color_frame, dtype=np.uint8)
    #             if color_image is None:
    #                 print("无法将帧转换为图像")
    #                 continue

    #             # 显示图像
    #             cv2.imshow("Orbbec 336L RGB Stream", color_image)
    #             key = cv2.waitKey(1)
    #             if key == ord('q') or key == ESC_KEY:
    #                 break
    #         except KeyboardInterrupt:
    #             break
    # finally:
    #     # 停止摄像头并释放资源
    #     cv2.destroyAllWindows()
    #     robot.pipeline1.stop()
    #     print("摄像头已停止")

    # while True:
    #     try:
    #         # 等待帧数据
    #         frames: FrameSet = robot.pipeline1.wait_for_frames(100)
    #         if frames is None:
    #             continue
    #         color_frame = frames.get_color_frame()
    #         if color_frame is None:
    #             continue

    #         # 转换为 RGB 图像
    #         color_image = robot.frame_to_bgr_image(color_frame)
    #         #color_image = np.array(color_frame, dtype=np.uint8)
    #         if color_image is None:
    #             print("无法将帧转换为图像")
    #             continue

    #         # 显示图像
    #         cv2.imshow("Orbbec 336L RGB Stream", color_image)
    #         key = cv2.waitKey(1)
    #         if key == ord('q') or key == ESC_KEY:
    #             break
    #     except KeyboardInterrupt:
    #         break
    #     finally:
    #         # 停止摄像头并释放资源
    #         cv2.destroyAllWindows()
    #         robot.pipeline1.stop()
    #         print("摄像头已停止")