from pymodbus.client.sync import ModbusTcpClient #pip3 install pymodbus==2.5.3
import time
import numpy as np

# 寄存器字典
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

def open_modbus(ip, port):
    client = ModbusTcpClient(ip, port)
    client.connect()
    return client

def read_force_feedback(client):
    """
    Read 6 force values from FORCE_ACT(m) and convert to signed integers.

    Returns:
        A list of 6 integers in the range [-4000, 4000], unit is gram (g).
    """
    start_addr = 1582  # FORCE_ACT(m)
    raw = read_register(client, address=start_addr, count=6)

    if not raw or len(raw) != 6:
        raise RuntimeError("读取受力值失败或数据不完整")

    return [(v - 65536) if v > 32767 else v for v in raw]

def reset_force_feedback(client):
    """
    Reset and calibrate the force sensors by:
    1) Reading current hand pose,
    2) Opening all fingers,
    3) Performing calibration,
    4) Restoring the original pose.
    """
    # -------- Step 1: Read current hand angles --------
    print("[1] 读取当前手部姿态中...")
    angle_addr = 1546  # ANGLE_ACT(m)
    current_angles = read_register(client, address=angle_addr, count=6)
    if not current_angles:
        raise RuntimeError("读取当前角度失败")
    print(f"当前角度: {current_angles}")

    # -------- Step 2: Open all fingers --------
    print("[2] 张开所有手指...")
    open_angles = [1000, 1000, 1000, 1000, 1000, 0]  # 大拇指旋转为0，其它为1000
    angle_set_addr = 1486  # ANGLE_SET(m)
    write_register(client, address=angle_set_addr, values=open_angles)
    print(f"发送张开指令: {open_angles}")
    time.sleep(2.0)

    # -------- Step 3: Calibrate force sensor --------
    print("[3] 执行受力传感器校准...")
    write_register(client, address=1009, values=[1])  # GESTURE_FORCE_CLB
    print("校准命令已发送")

    # -------- Step 4: Restore original pose --------
    print("[4] 恢复原始姿态...")
    write_register(client, address=angle_set_addr, values=current_angles)
    time.sleep(2.0)
    print(f"姿态已恢复: {current_angles}")

def read_all_tactile_sync(client) -> dict[str, np.ndarray]:
    """
    Synchronously read all tactile sensor matrices from the dexterous hand
    using client.read_holding_registers().

    Returns:
        A dictionary of {sensor_part_name: matrix}.
    """
    start_address = 3000
    total_registers = 562  # 1124字节，共562个16位寄存器

    # 读取全部触觉寄存器
    all_data = read_register(client, address=start_address, count=total_registers)
    if not all_data or len(all_data) < total_registers:
        raise RuntimeError("触觉数据读取失败或不完整")

    tactile_parts = {
        "pinky_tip":      (0,    3,  3),
        "pinky_finger":   (9,   12,  8),
        "pinky_middle":   (105, 10,  8),
        "ring_tip":       (185, 3,   3),
        "ring_finger":    (194, 12,  8),
        "ring_middle":    (290, 10,  8),
        "middle_tip":     (370, 3,   3),
        "middle_finger":  (379, 12,  8),
        "middle_middle":  (475, 10,  8),
        "index_tip":      (555, 3,   3),
        "index_finger":   (564, 12,  8),
        "index_middle":   (660, 10,  8),
        "thumb_tip":      (740, 3,   3),
        "thumb_finger":   (749, 12,  8),
        "thumb_middle":   (845, 3,   3),
        "thumb_palm":     (854, 12,  8),
        "palm":           (950, 8,  14),  # 行倒序，可在可视化时翻转
    }

    tactile= {}
    for name, (offset, rows, cols) in tactile_parts.items():
        segment = all_data[offset:offset + rows * cols]
        matrix = np.array(segment, dtype=np.uint16).reshape((rows, cols))
        if name == "palm":
            matrix = matrix[::-1]  # 翻转为正序

        tactile[name] = matrix

    return tactile

def write_register(client, address, values):
    # Modbus 写入寄存器，传入寄存器地址和要写入的值列表
    client.write_registers(address, values)

def read_register(client, address, count):
    # Modbus 读取寄存器
    response = client.read_holding_registers(address, count)
    return response.registers if response.isError() is False else []

def write6(client, reg_name, val):
    if reg_name in ['angleSet', 'forceSet', 'speedSet']:
        val_reg = []
        for i in range(6):
            val_reg.append(val[i] & 0xFFFF)  # 取低16位
        write_register(client, regdict[reg_name], val_reg)
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'，val为长度为6的list，值为0~1000，允许使用-1作为占位符')

def read6(client, reg_name):
    # 检查寄存器名称是否在允许的范围内
    if reg_name in ['angleSet', 'forceSet', 'speedSet', 'angleAct', 'forceAct']:
        # 直接读取与reg_name对应的寄存器，读取的数量为6
        val = read_register(client, regdict[reg_name], 6)
        if len(val) < 6:
            print('没有读到数据')
            return
        print('读到的值依次为：', end='')
        for v in val:
            print(v, end=' ')
        print()
    
    elif reg_name in ['errCode', 'statusCode', 'temp']:
        # 读取错误代码、状态代码或温度，每次读取3个寄存器
        val_act = read_register(client, regdict[reg_name], 3)
        if len(val_act) < 3:
            print('没有读到数据')
            return
            
        # 初始化存储高低位的数组
        results = []
        
        # 将每个寄存器的高位和低位分开存储
        for i in range(len(val_act)):
            # 读取当前寄存器和下一个寄存器
            low_byte = val_act[i] & 0xFF            # 低八位
            high_byte = (val_act[i] >> 8) & 0xFF     # 高八位
        
            results.append(low_byte)  # 存储低八位
            results.append(high_byte)  # 存储高八位

        print('读到的值依次为：', end='')
        for v in results:
            print(v, end=' ')
        print()
    
    else:
        print('函数调用错误，正确方式：str的值为\'angleSet\'/\'forceSet\'/\'speedSet\'/\'angleAct\'/\'forceAct\'/\'errCode\'/\'statusCode\'/\'temp\'')

if __name__ == '__main__':
    ip_address = '192.168.11.210'
    port = 6000
    print('打开Modbus TCP连接！')
    client = open_modbus(ip_address, port)
    
    print('设置灵巧手运动速度参数，-1为不设置该运动速度！')
    write6(client, 'speedSet', [100, 100, 100, 100, 100, 100])
    time.sleep(2) 
    # 关闭 Modbus TCP 连接
    client.close()

