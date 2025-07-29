import numpy as np

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
            "front": (480, 640, 3),  # 假设相机分辨率为480x640
        }

    def __init__(self, robot_ip="192.168.31.2", camera_id=0):
        import rtde_control
        import rtde_receive
        import cv2

        self.robot = rtde_control.RTDEControlInterface(robot_ip)
        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.camera = cv2.VideoCapture(camera_id)

    def connect(self):
        pass

    def disconnect(self):
        self.camera.release()

    def get_observation(self):
        # 获取关节角度
        joints = self.r_inter.getActualQ()
        joints = np.rad2deg(joints)
        obs = {
            f"joint_{i+1}.pos": float(joints[i]) for i in range(6)
        }
        # 获取相机图像
        ret, frame = self.camera.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        obs["front"] = frame
        return obs

    def send_action(self, action: dict):
        # action: dict, key为joint_x.pos，value为角度（度）
        joint_targets = [action[f"joint_{i+1}.pos"] for i in range(6)]
        joint_targets_rad = np.radians(joint_targets)
        velocity = 0.5
        acceleration = 0.5
        self.robot.moveJ(joint_targets_rad, velocity, acceleration, False)
        # 返回实际执行的动作
        return action

    @property
    def cameras(self):
        # LeRobot用于判断有无相机
        return {"front": self.camera}