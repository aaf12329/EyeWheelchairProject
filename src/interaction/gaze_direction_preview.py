"""第 4 步：单摄像头视线方向选择预览（左/中/右）。

先看正前方完成校准，再把视线移到屏幕左侧、中间、右侧。
本程序只显示候选方向，不发送确认或任何硬件控制指令。

按 Q 退出；按 C 重新校准；按 I 反转左右方向（若左右显示与实际相反）。
"""
from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0
SIDE_THRESHOLD = 0.075      # 相对于正视基准的水平偏移
STABLE_SECONDS = 0.70       # 同方向稳定停留多久才成为候选
SMOOTHING_FRAMES = 5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
WINDOW_NAME = "视线方向选择预览｜Q 退出｜C 校准｜I 反转左右｜仅视觉测试"

# Face Landmarker 中的虹膜点及眼角点。
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_CORNERS = (33, 133)
RIGHT_CORNERS = (362, 263)


def create_detector():
    if not FACE_MODEL.exists():
        raise FileNotFoundError(f"缺少模型文件：{FACE_MODEL}")
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def iris_score(landmarks) -> float | None:
    """返回虹膜在双眼水平方向的平均归一化位置；0 左、1 右。"""
    def average_x(indices):
        return sum(landmarks[i].x for i in indices) / len(indices)

    ratios = []
    for iris, corners in ((LEFT_IRIS, LEFT_CORNERS), (RIGHT_IRIS, RIGHT_CORNERS)):
        a, b = landmarks[corners[0]].x, landmarks[corners[1]].x
        span = abs(b - a)
        if span < 1e-5:
            continue
        ratios.append((average_x(iris) - min(a, b)) / span)
    return sum(ratios) / len(ratios) if ratios else None


def draw_text(frame, text, xy, size, color):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(xy, text, font=ImageFont.truetype(str(FONT_PATH), size), fill=color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_cards(frame, candidate: str):
    h, w = frame.shape[:2]
    cards = [("左", 35, 390, 280, 515), ("中", 340, 390, 585, 515), ("右", 645, 390, 890, 515)]
    for label, x1, y1, x2, y2 in cards:
        active = label == candidate
        color = (0, 170, 0) if active else (70, 70, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1 if active else 2)
        frame = draw_text(frame, label, (x1 + 98, y1 + 35), 48, (255, 255, 255))
    return frame


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("摄像头无法打开，请关闭占用摄像头的软件后重试。")

    detector = create_detector()
    program_started = time.perf_counter()
    calibration_started = program_started
    calibration_values = []
    center_score = None
    score_history = deque(maxlen=SMOOTHING_FRAMES)
    raw_direction = "中"
    candidate = "无"
    direction_since = None
    invert_direction = False
    message = "请看屏幕正中间，正在校准"
    key_is_held = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败。")
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, int((now - program_started) * 1000))

            current_score = None
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                current_score = iris_score(landmarks)
                if current_score is not None:
                    score_history.append(current_score)
                    current_score = sum(score_history) / len(score_history)
                h, w = frame.shape[:2]
                for idx in LEFT_IRIS + RIGHT_IRIS:
                    lm = landmarks[idx]
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 255, 255), -1)

                if center_score is None and current_score is not None:
                    calibration_values.append(current_score)
                    elapsed = now - calibration_started
                    message = f"校准中：请看屏幕正中间 {max(0, CALIBRATION_SECONDS-elapsed):.1f} 秒"
                    if elapsed >= CALIBRATION_SECONDS and len(calibration_values) >= 10:
                        calibration_values.sort()
                        trim = max(1, len(calibration_values)//10)
                        values = calibration_values[trim:-trim] or calibration_values
                        center_score = sum(values)/len(values)
                        message = "校准完成：依次看屏幕左、中、右区域"
                elif center_score is not None and current_score is not None:
                    offset = current_score - center_score
                    if invert_direction:
                        offset = -offset
                    if offset < -SIDE_THRESHOLD:
                        new_direction = "左"
                    elif offset > SIDE_THRESHOLD:
                        new_direction = "右"
                    else:
                        new_direction = "中"
                    if new_direction != raw_direction:
                        raw_direction = new_direction
                        direction_since = now
                        candidate = "无"
                    elif direction_since is not None and now - direction_since >= STABLE_SECONDS:
                        candidate = raw_direction
                    message = "只是在选择候选方向；本程序不会移动轮椅"
            else:
                message = "未检测到人脸：请正对摄像头"
                candidate = "无"
                direction_since = None

            cv2.rectangle(frame, (0, 0), (650, 105), (0, 0, 0), -1)
            score_text = "--" if current_score is None else f"{current_score:.3f}"
            base_text = "--" if center_score is None else f"{center_score:.3f}"
            frame = draw_text(frame, f"虹膜位置：{score_text}｜正视基准：{base_text}", (14, 10), 20, (0, 255, 0))
            frame = draw_text(frame, f"当前方向：{raw_direction}｜稳定候选：{candidate}", (14, 38), 28, (255, 255, 0))
            frame = draw_text(frame, message, (14, 74), 18, (255, 255, 255))
            frame = draw_cards(frame, candidate)
            cv2.imshow(WINDOW_NAME, frame)

            raw_key = cv2.waitKey(1)
            if raw_key == -1:
                key_is_held = False
                continue
            if key_is_held:
                continue
            key_is_held = True
            key = raw_key & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('c'), ord('C')):
                calibration_started = now
                calibration_values, center_score = [], None
                score_history.clear()
                raw_direction, candidate, direction_since = "中", "无", None
                message = "重新校准：请看屏幕正中间"
            if key in (ord('i'), ord('I')):
                invert_direction = not invert_direction
                message = "左右方向已反转" if invert_direction else "左右方向已恢复"
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
