import cv2
from pyorbbecsdk import *
import numpy as np
from typing import Union, Any, Optional
#from utils import frame_to_bgr_image

ESC_KEY = 27
def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
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
def main():
    config = Config()
    pipeline = Pipeline()
    try:
        # 获取摄像头的流配置
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            # 设置分辨率为 1280x720，格式为 RGB，帧率为 30
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1280, 720, OBFormat.RGB, 30)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("使用默认的 color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print("配置摄像头失败:", e)
        return

    # 启动摄像头
    pipeline.start(config)
    print("摄像头已启动，按 'q' 或 'ESC' 键退出")

    try:
        while True:
            try:
                # 等待帧数据
                frames: FrameSet = pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                # 转换为 RGB 图像
                color_image = frame_to_bgr_image(color_frame)
                #color_image = np.array(color_frame, dtype=np.uint8)
                if color_image is None:
                    print("无法将帧转换为图像")
                    continue

                # 显示图像
                color_image = color_image[:, 240:-320, :]
                cv2.imshow("Orbbec 336L RGB Stream", color_image)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ESC_KEY:
                    break
            except KeyboardInterrupt:
                break
    finally:
        # 停止摄像头并释放资源
        cv2.destroyAllWindows()
        pipeline.stop()
        print("摄像头已停止")

if __name__ == "__main__":
    main()