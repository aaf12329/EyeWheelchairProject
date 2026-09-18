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
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def open_camera() -> cv2.VideoCapture:
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


def main() -> None:
    cap = open_camera()
    writer = None
    recording_path = None
    frame_count = 0
    start = time.perf_counter()
    key_is_held = False

    print("摄像头已打开：按 Q 退出，按 R 录制，按 S 截图。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败，请检查 USB 连接。")
            frame_count += 1
            elapsed = max(time.perf_counter() - start, 0.001)
            fps = frame_count / elapsed

            display = frame.copy()
            status = "RECORDING" if writer else "PREVIEW"
            color = (0, 0, 255) if writer else (0, 180, 0)
            cv2.putText(display, f"{status} | {fps:.1f} FPS", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(display, "Q quit | R record | S snapshot", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, display)

            if writer:
                writer.write(frame)

            raw_key = cv2.waitKey(1)
            if raw_key == -1:
                key_is_held = False
                continue
            key = raw_key & 0xFF
            if key_is_held:
                continue
            key_is_held = True

            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                path = SNAPSHOT_DIR / f"snapshot_{timestamp()}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"截图已保存：{path}")
            if key in (ord("r"), ord("R")):
                if writer:
                    writer.release()
                    writer = None
                    print(f"录制已停止：{recording_path}")
                else:
                    recording_path = VIDEO_DIR / f"camera_test_{timestamp()}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    writer = cv2.VideoWriter(str(recording_path), fourcc, actual_fps, (w, h))
                    print(f"开始录制：{recording_path}")
    finally:
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
