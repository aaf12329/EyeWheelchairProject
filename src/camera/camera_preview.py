"""第 1 周：仅摄像头基线测试。禁止连接轮椅、Arduino 或电机驱动。"""
from datetime import datetime
from pathlib import Path
import time

import cv2

CAMERA_INDEX = 0
WIDTH = 1280
HEIGHT = 720
WINDOW_NAME = "EyeWheelchair Camera | Q quit | R record | S snapshot"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = PROJECT_ROOT / "data" / "raw_videos"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots"


# ---------- 工具函数 ----------

def timestamp() -> str:
    """返回当前时间字符串，用于文件名。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    """确保录制和截图目录存在。"""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 摄像头 ----------

def open_camera() -> cv2.VideoCapture:
    """打开摄像头，设置分辨率，失败则抛异常。"""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError(
            "摄像头无法打开。请拔插 USB 摄像头；关闭占用摄像头的软件；"
            "或把 CAMERA_INDEX 从 0 改为 1。"
        )
    return cap


def read_frame(cap: cv2.VideoCapture):
    """读一帧，失败则抛异常。"""
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败，请检查 USB 连接。")
    return frame


# ---------- 显示 ----------

def draw_status(frame, status: str, fps: float) -> None:
    """在画面上绘制状态和 FPS。"""
    color = (0, 0, 255) if status == "RECORDING" else (0, 180, 0)
    cv2.putText(frame, f"{status} | {fps:.1f} FPS", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, "Q quit | R record | S snapshot", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def show(frame) -> None:
    """显示画面。"""
    cv2.imshow(WINDOW_NAME, frame)


# ---------- 录制与截图 ----------

def save_snapshot(frame) -> None:
    """保存当前帧为截图。"""
    path = SNAPSHOT_DIR / f"snapshot_{timestamp()}.jpg"
    cv2.imwrite(str(path), frame)
    print(f"截图已保存：{path}")


def start_recording(cap: cv2.VideoCapture):
    """开始录制，返回 (writer, 文件路径)。"""
    path = VIDEO_DIR / f"camera_test_{timestamp()}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    print(f"开始录制：{path}")
    return writer, path


def stop_recording(writer, path) -> None:
    """停止录制并释放 writer。"""
    writer.release()
    print(f"录制已停止：{path}")


# ---------- 按键 ----------

def handle_key(key: int, frame, cap, writer, recording_path, key_is_held: bool):
    """处理按键，返回更新后的 (writer, recording_path, key_is_held, quit)。

    quit=True 表示要退出主循环。
    """
    if key == -1:
        return writer, recording_path, False, False
    if key_is_held:
        return writer, recording_path, True, False

    if key in (ord("q"), ord("Q")):
        return writer, recording_path, True, True

    if key in (ord("s"), ord("S")):
        save_snapshot(frame)
        return writer, recording_path, True, False

    if key in (ord("r"), ord("R")):
        if writer:
            stop_recording(writer, recording_path)
            return None, None, True, False
        else:
            new_writer, new_path = start_recording(cap)
            return new_writer, new_path, True, False

    return writer, recording_path, True, False


# ---------- 主流程 ----------

def run_loop(cap: cv2.VideoCapture) -> None:
    """主循环：读帧、画 HUD、处理按键。"""
    writer = None
    recording_path = None
    frame_count = 0
    start = time.perf_counter()
    key_is_held = False

    while True:
        frame = read_frame(cap)
        frame_count += 1
        elapsed = max(time.perf_counter() - start, 0.001)
        fps = frame_count / elapsed

        display = frame.copy()
        draw_status(display, "RECORDING" if writer else "PREVIEW", fps)
        show(display)

        if writer:
            writer.write(frame)

        raw_key = cv2.waitKey(1)
        key = raw_key & 0xFF if raw_key != -1 else -1
        writer, recording_path, key_is_held, quit_now = handle_key(
            key, frame, cap, writer, recording_path, key_is_held
        )
        if quit_now:
            break

    # 循环结束前，如果还在录制，停掉
    if writer:
        stop_recording(writer, recording_path)


def main() -> None:
    ensure_dirs()
    cap = open_camera()
    print("摄像头已打开：按 Q 退出，按 R 录制，按 S 截图。")
    try:
        run_loop(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()