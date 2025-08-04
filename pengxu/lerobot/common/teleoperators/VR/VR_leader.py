class VR_leader:
    """VR_leader teleoperator for UR5e Follower Robot."""
    def __init__(self):
        # 初始化你的teleop设备
        pass

    def connect(self):
        # 可选：连接设备
        pass

    def disconnect(self):
        # 可选：断开设备
        pass

    def get_action(self):
        # 返回一个 dict，key与robot.action_features一致
        # 这里举例返回全0
        return {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "joint_3.pos": 0.0,
            "joint_4.pos": 0.0,
            "joint_5.pos": 0.0,
            "joint_6.pos": 0.0,
        }