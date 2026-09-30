"""摄像头打开的唯一定位：全项目共用一份逻辑（原 5 份副本合并而来）。

各脚本保留自己的 CAMERA_INDEX / WIDTH / HEIGHT 常量，作为参数传进来。
"""
import cv2


def open_camera(index: int = 0, width: int = 960, height: int = 540) -> cv2.VideoCapture:
    """打开摄像头：先 DirectShow（Windows 成功率高），失败回退默认后端；
    设置分辨率；打不开时报一句中文提示（含排查方向）。"""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(
            "摄像头无法打开。请拔插 USB 摄像头；关闭占用摄像头的软件；"
            f"或把 CAMERA_INDEX 从 {index} 改为 1。"
        )
    return cap
