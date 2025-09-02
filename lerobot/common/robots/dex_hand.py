from pymodbus.client.sync import ModbusTcpClient  # pip install pymodbus==2.5.3
import time
import numpy as np

# ===== 宏变量: 触觉键名 =====
FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb"]
METRICS = [
    "normal_force",
    "normal_force_delta",
    "tangential_force",
    "tangential_force_delta",
    "tangential_force_direction",
    "approach_delta",
]
TACTILE_KEYS = [f"{f}_{m}" for f in FINGER_NAMES for m in METRICS]

# ===== 手指对应的起始地址 =====
FINGER_ADDR_MAP = {
    "pinky": 3000,
    "ring": 3058,
    "middle": 3116,
    "index": 3174,
    "thumb": 3232,
}


class DexHandClient:
    regdict = {
        'ID': 1000,
        'baudrate': 1001,
        'clearErr': 1004,
        'forceClb': 1009,
        'angleSet': 1486,
        'forceSet': 1498,
        'speedSet': 1522,
        'angleAct': 1546,
        'forceAct': 1582,
        'errCode': 1606,
        'statusCode': 1612,
        'temp': 1618,
        'actionSeq': 2320,
        'actionRun': 2322
    }

    ### For 电阻手
    # tactile_parts = {
    #     "pinky_tip":      (0,    3,  3),
    #     "pinky_finger":   (9,   12,  8),
    #     "pinky_middle":   (105, 10,  8),
    #     "ring_tip":       (185, 3,   3),
    #     "ring_finger":    (194, 12,  8),
    #     "ring_middle":    (290, 10,  8),
    #     "middle_tip":     (370, 3,   3),
    #     "middle_finger":  (379, 12,  8),
    #     "middle_middle":  (475, 10,  8),
    #     "index_tip":      (555, 3,   3),
    #     "index_finger":   (564, 12,  8),
    #     "index_middle":   (660, 10,  8),
    #     "thumb_tip":      (740, 3,   3),
    #     "thumb_finger":   (749, 12,  8),
    #     "thumb_middle":   (845, 3,   3),
    #     "thumb_palm":     (854, 12,  8),
    #     "palm":           (950, 8,  14),  # 行倒序，可翻转
    # }

    def __init__(self, ip: str = "192.168.11.210", port: int = 6000):
        self.ip = ip
        self.port = port
        self.client = ModbusTcpClient(ip, port)


        # 每个手指的寄存器起始地址 (用户手册 §2.6.20)
        self.finger_addr_map = {
            "pinky": 3000,
            "ring": 3058,
            "middle": 3116,
            "index": 3174,
            "thumb": 3232,
        }
        self.byte_per_finger = 58
        self.reg_per_finger = self.byte_per_finger // 2  # 1寄存器=2字节


    def connect(self):
        return self.client.connect()

    def close(self):
        self.client.close()

    def write_register(self, address, values):
        self.client.write_registers(address, values)

    def read_register(self, address, count):
        response = self.client.read_holding_registers(address, count)
        return response.registers if not response.isError() else []

    def write6(self, reg_name, val):
        if reg_name in ['angleSet', 'forceSet', 'speedSet']:
            val_reg = [v & 0xFFFF for v in val]
            self.write_register(self.regdict[reg_name], val_reg)
        else:
            raise ValueError(f"[write6] 错误寄存器名称：{reg_name}")

    def read6(self, reg_name):
        if reg_name in ['angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct']:
            val = self.read_register(self.regdict[reg_name], 6)
            if len(val) < 6:
                raise RuntimeError(f"[read6] {reg_name} 无法获取完整数据")
            return val
        elif reg_name in ['errCode', 'statusCode', 'temp']:
            val = self.read_register(self.regdict[reg_name], 3)
            if len(val) < 3:
                raise RuntimeError(f"[read6] {reg_name} 无法获取完整状态")
            results = []
            for v in val:
                results.append(v & 0xFF)
                results.append((v >> 8) & 0xFF)
            return results
        else:
            raise ValueError(f"[read6] 不支持的寄存器：{reg_name}")

    def read_force_feedback(self):
        raw = self.read_register(self.regdict['forceAct'], 6)
        if not raw or len(raw) != 6:
            raise RuntimeError("读取受力值失败或数据不完整")
        return [(v - 65536) if v > 32767 else v for v in raw]
    
    def init_hand(self, force_values=[100, 100, 100, 100, 100, 100], speed_values=[500, 500, 500, 500, 500, 500]):

        self.set_force(force_values)
        self.set_speed(speed_values)
        print("手部初始化完成，已设置受力和速度")

        self.reset_force_feedback()
        print("手部受力反馈已重置")
        print("手部已初始化")


    def reset_force_feedback(self):

        wait_time = 2

        print("[1] 读取当前角度...")
        current_angles = self.read_register(self.regdict['angleAct'], 6)
        print(f"当前角度: {current_angles}")

        print("[2] 张开所有手指...")
        open_angles = [1000, 1000, 1000, 1000, 1000, 0]
        self.write_register(self.regdict['angleSet'], open_angles)
        time.sleep(wait_time)

        print("[3] 执行校准...")
        self.write_register(self.regdict['forceClb'], [1])
        print("校准命令已发送")

        print("[4] 恢复原始姿态...")
        self.write_register(self.regdict['angleSet'], current_angles)
        time.sleep(wait_time)
        print("已恢复")
    
    def set_force(self, force_values = [100,100,100,100,100,100]):
        if len(force_values) != 6:
            raise ValueError("force_values 必须是长度为6的列表")
        self.write6('forceSet', force_values)
        print(f"已设置受力值: {force_values}")

    def set_speed(self, speed_values = [500, 500, 500, 500, 500, 500]):
        if len(speed_values) != 6:
            raise ValueError("speed_values 必须是长度为6的列表")
        self.write6('speedSet', speed_values)
        print(f"已设置速度值: {speed_values}")

    def _regs_to_bytes(self, regs):
        """寄存器列表 -> bytes，低位在前"""
        byte_list = []
        for reg in regs:
            lo = reg & 0xFF
            hi = (reg >> 8) & 0xFF
            byte_list.extend([lo, hi])
        return bytes(byte_list)

    def read_all_tactile_5finger(self) -> dict[str, str]:
        """
        读取 5 指触觉手的法向力、法向力变化值、切向力、切向力变化值、切向力方向、接近变化值
        返回字典: key 为 finger_metric，value 为字符串
        """
        byte_per_finger = 58
        reg_per_finger = byte_per_finger // 2  # 1寄存器 = 2字节

        result = {}

        for finger, base_addr in FINGER_ADDR_MAP.items():
            regs = self.read_register(base_addr, reg_per_finger) or [0] * reg_per_finger
            raw_bytes = self._regs_to_bytes(regs)

            offset = 32  # 跳过原始值（8通道×4字节）

            normal_force = str(np.frombuffer(raw_bytes[offset:offset+4], dtype='<f4')[0]); offset += 4
            normal_force_delta = str(np.frombuffer(raw_bytes[offset:offset+4], dtype='<f4')[0]); offset += 4
            tangential_force = str(np.frombuffer(raw_bytes[offset:offset+4], dtype='<f4')[0]); offset += 4
            tangential_force_delta = str(np.frombuffer(raw_bytes[offset:offset+4], dtype='<f4')[0]); offset += 4
            tangential_force_direction = str(np.frombuffer(raw_bytes[offset:offset+2], dtype='<i2')[0]); offset += 2
            approach_delta = str(np.frombuffer(raw_bytes[offset:offset+4], dtype='<f4')[0]); offset += 4

            # 用下划线连接 finger 和 metric
            result[f"{finger}_normal_force"] = normal_force
            result[f"{finger}_normal_force_delta"] = normal_force_delta
            result[f"{finger}_tangential_force"] = tangential_force
            result[f"{finger}_tangential_force_delta"] = tangential_force_delta
            result[f"{finger}_tangential_force_direction"] = tangential_force_direction
            result[f"{finger}_approach_delta"] = approach_delta
        return result
    
    def read_all_tactile_full(self, signed: bool = True) -> dict[str, np.ndarray]:
        """
        按块读取触觉数据（分段读取，每段不足则补零，不等待）。
        :param signed: 是否转为 int16（带符号）
        """
        tactile_data = {}
        MAX_REGISTERS_PER_READ = 125  # Modbus 每次最多读取的寄存器数

        blocks = [
            (3000, 370, [
                ("pinky_tip",    0,   3, 3, False),
                ("pinky_finger", 6,  12, 8, False),
                ("pinky_middle", 102, 10, 8, False),
            ]),
            (3370, 370, [
                ("ring_tip",     0,   3, 3, False),
                ("ring_finger",  6,  12, 8, False),
                ("ring_middle",  102, 10, 8, False),
            ]),
            (3740, 370, [
                ("middle_tip",    0,   3, 3, False),
                ("middle_finger", 6,  12, 8, False),
                ("middle_middle", 102, 10, 8, False),
            ]),
            (4110, 370, [
                ("index_tip",    0,   3, 3, False),
                ("index_finger", 6,  12, 8, False),
                ("index_middle", 102, 10, 8, False),
            ]),
            (4480, 420, [
                ("thumb_tip",    0,   3, 3, False),
                ("thumb_finger", 6,  12, 8, False),
                ("thumb_middle", 102,  3, 3, False),
                ("thumb_palm",   108, 12, 8, False),
            ]),
            (4900, 224, [
                ("palm", 0, 8, 14, True),  # 翻转
            ]),
        ]

        def read_register_range(start_addr, total_regs):
            """分段读取一个块的数据"""
            all_regs = []
            for addr in range(start_addr, start_addr + total_regs, MAX_REGISTERS_PER_READ):
                count = min(MAX_REGISTERS_PER_READ, start_addr + total_regs - addr)
                regs = self.read_register(addr, count) or []
                if len(regs) < count:
                    regs.extend([0] * (count - len(regs)))  # 补 0
                all_regs.extend(regs)
            return all_regs

        for base_addr, total_bytes, subparts in blocks:
            reg_count = total_bytes // 2
            regs = read_register_range(base_addr, reg_count)
            byte_data = self._regs_to_bytes(regs)
            arr = np.frombuffer(byte_data, dtype=np.int16 if signed else np.uint16)

            for name, byte_offset, rows, cols, flip in subparts:
                start_idx = byte_offset // 2
                matrix = arr[start_idx:start_idx + rows * cols].reshape((rows, cols))
                if flip:
                    matrix = matrix[::-1]
                tactile_data[name] = matrix

        return tactile_data

    
    def read_force_angle_tactile(self) -> dict:
        """
        同步读取角度、受力和触觉数据
        - 角度和受力通过一次 read_register 调用分段处理
        - 触觉调用已有 read_all_tactile_sync() 函数

        返回:
            dict，包括 "timestamp", "angles", "forces", "tactile"
        """
        timestamp = time.time()

        # Step 1: 读取角度和受力的寄存器
        angle_addr = self.regdict['angleAct']
        force_addr = self.regdict['forceAct']

        angles = self.read_register(angle_addr, 6)
        forces_raw = self.read_register(force_addr, 6)
        tactile = self.read_all_tactile_5finger()

        if len(angles) != 6 or len(forces_raw) != 6:
            raise RuntimeError("角度或受力寄存器读取失败")

        # Step 2: 力转换为有符号整数
        forces = [(v - 65536) if v > 32767 else v for v in forces_raw]

        return {
            "timestamp": timestamp,
            "angles": angles,      # 原始角度值
            "forces": forces,      # 有符号力值
            "tactile": tactile     # dict[str, np.ndarray]
        }
    
    def set_hand_angle(self, angles):
        """
        设置手部角度
        :param angles: 长度为6的列表或数组，表示每个关节的角度
        """
        if len(angles) != 6:
            raise ValueError("angles 必须是长度为6的列表")
        self.write6('angleSet', angles)
        # print(f"已设置手部角度: {angles}")

    
if __name__ == '__main__':
    dex = DexHandClient("192.168.11.210", 6000)

    if dex.connect():
        print("连接成功")

        # 设置速度
        dex.write6('speedSet', [100, 100, 100, 100, 100, 100])
        time.sleep(1)

        # 读取受力
        forces = dex.read_force_feedback()
        print("受力反馈:", forces)

        # 校准
        dex.reset_force_feedback()

        # 读取触觉
        tactile = dex.read_force_angle_tactile()
        # print("pinky_tip matrix:\n", tactile["pinky_tip"])

        dex.close()
    else:
        print("连接失败")
