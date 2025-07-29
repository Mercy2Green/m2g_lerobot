import numpy as np
from pytransform3d import rotations

class VR_leader:
    """VR_leader teleoperator for UR5e Follower Robot."""
    def __init__(self, vr_input=None):
        # vr_input: 你自定义的VR输入采集类，需有get_pose()方法
        self.vr_input = vr_input

        # 初始化参考位姿
        self._init_pose = None
        self._init_tcp = None
        self._init_flag = False

    def connect(self):
        if self.vr_input is not None and hasattr(self.vr_input, "connect"):
            self.vr_input.connect()

    def disconnect(self):
        if self.vr_input is not None and hasattr(self.vr_input, "disconnect"):
            self.vr_input.disconnect()

    def get_action(self, current_tcp_pose):
        """
        current_tcp_pose: 当前机械臂的tcp_pose, shape=(6,)
        返回: new_tcp_pose, shape=(6,)
        """
        pose = self.vr_input.get_pose()  # [x, y, z, qx, qy, qz, qw]
        if pose is None:
            return current_tcp_pose  # 无输入时不动

        pos = pose[:3]
        quat = pose[3:7]  # x, y, z, w

        # test.py同款姿态变换
        R = rotations.matrix_from_quaternion([quat[3], quat[0], quat[1], quat[2]])  # w, x, y, z
        mirror = np.diag([-1, -1, 1])
        R_new = mirror @ R @ mirror
        theta = np.pi / 2  # 90度
        Rz = rotations.matrix_from_axis_angle([0, 0, 1, theta])
        R_new = Rz @ R_new
        axis_angle = rotations.axis_angle_from_matrix(R_new)
        rx, ry, rz = axis_angle[:3] * axis_angle[3]

        # 初始化参考
        if not self._init_flag:
            self._init_pose = pos.copy()
            self._init_tcp = current_tcp_pose.copy()
            self._init_flag = True
            return current_tcp_pose

        # 差分控制
        new_tcp_pose = list(self._init_tcp)
        new_tcp_pose[0] = self._init_tcp[0] + (self._init_pose[0] - pos[0])
        new_tcp_pose[1] = self._init_tcp[1] + (self._init_pose[1] - pos[1])
        new_tcp_pose[2] = self._init_tcp[2] - (self._init_pose[2] - pos[2])
        new_tcp_pose[3] = rx
        new_tcp_pose[4] = ry
        new_tcp_pose[5] = rz
        action = {
            "joint_1.pos": new_tcp_pose[0],
            "joint_2.pos": new_tcp_pose[1],
            "joint_3.pos": new_tcp_pose[2],
            "joint_4.pos": new_tcp_pose[3],
            "joint_5.pos": new_tcp_pose[4],
            "joint_6.pos": new_tcp_pose[5],
        }
        return action

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