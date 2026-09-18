"""第 3 步：眨眼校准与计数，仅用于视觉交互验证。

启动后保持睁眼看向摄像头约 3 秒。完成校准后，每完成一次自然眨眼，
屏幕上的 BLINK COUNT 会加一。此程序不会输出任何轮椅或 Arduino 指令。
按 Q 退出，按 C 重新校准。
"""
from collections import deque
from pathlib import Path
import math
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0
CLOSE_RATIO = 0.78          # 放宽：低于睁眼基线的 78% 即进入闭眼候选
REOPEN_RATIO = 0.88         # 高于睁眼基线的 88% 即认为重新睁眼
SMOOTHING_FRAMES = 3        # 轻度平滑：保留自然眨眼的快速变化
MIN_CLOSED_SECONDS = 0.04   # 过滤极短噪声
MAX_CLOSED_SECONDS = 0.80   # 允许较慢/较完整的自然眨眼
MIN_OPEN_SECONDS = 0.10     # 再次计数前只需短暂稳定睁眼
REFRACTORY_SECONDS = 0.30   # 防止一次眨眼多计，但不妨碍连续眨眼

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
WINDOW_NAME = "眨眼检测预览｜Q 退出｜C 重新校准｜仅视觉测试"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")

# 每只眼睛：外眼角、上外、上内、内眼角、下内、下外。
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(landmarks, indices) -> float:
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in indices]
    horizontal = distance(p0, p3)
    if horizontal < 1e-6:
        return 0.0
    return (distance(p1, p5) + distance(p2, p4)) / (2.0 * horizontal)


def face_detector():
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


def draw_chinese_status(frame, lines) -> object:
    """用 Windows 中文字体绘制状态栏，避免 OpenCV 英文字体无法显示中文。"""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_large = ImageFont.truetype(str(FONT_PATH), 29)
    font_small = ImageFont.truetype(str(FONT_PATH), 20)
    draw.text((14, 12), lines[0], font=font_small, fill=(0, 255, 0))
    draw.text((14, 39), lines[1], font=font_large, fill=(255, 255, 0))
    draw.text((14, 76), lines[2], font=font_small, fill=(255, 255, 255))
    draw.text((14, 103), lines[3], font=font_small, fill=(255, 210, 80))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def reset_calibration():
    return [], None, None, 0


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("摄像头无法打开，请关闭占用摄像头的软件后重试。")

    detector = face_detector()
    program_started_at = time.perf_counter()
    calibration_started_at = program_started_at
    calibration_values, baseline, close_threshold, reopen_threshold, blink_count = [], None, None, None, 0
    is_closed = False
    closed_since = None
    open_since = None
    last_blink_at = float("-inf")
    ear_history = deque(maxlen=SMOOTHING_FRAMES)
    blink_state = "正在校准"
    recent_message = "请睁眼正视摄像头，正在校准"
    key_is_held = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败。")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.perf_counter()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, int((now - program_started_at) * 1000))

            ear = None
            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                raw_ear = (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2.0
                ear_history.append(raw_ear)
                ear = sum(ear_history) / len(ear_history)
                h, w = frame.shape[:2]
                for index in LEFT_EYE + RIGHT_EYE:
                    lm = landmarks[index]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1)

                if baseline is None:
                    calibration_values.append(ear)
                    elapsed = now - calibration_started_at
                    remaining = max(0.0, CALIBRATION_SECONDS - elapsed)
                    recent_message = f"校准中：请保持睁眼 {remaining:.1f}s"
                    if elapsed >= CALIBRATION_SECONDS and len(calibration_values) >= 10:
                        calibration_values.sort()
                        # 用中间 80% 的平均值，降低偶发眨眼影响。
                        trim = max(1, len(calibration_values) // 10)
                        values = calibration_values[trim:-trim] or calibration_values
                        baseline = sum(values) / len(values)
                        close_threshold = baseline * CLOSE_RATIO
                        reopen_threshold = baseline * REOPEN_RATIO
                        recent_message = "校准完成：现在请自然眨眼"
                        blink_state = "已准备好"
                else:
                    # 必须先记录“闭眼前是否已稳定睁眼”，再处理当前闭眼帧。
                    # 否则一闭眼就会先清空 open_since，导致永远无法进入计数。
                    was_stably_open = (
                        open_since is not None
                        and now - open_since >= MIN_OPEN_SECONDS
                        and now - last_blink_at >= REFRACTORY_SECONDS
                    )

                    if is_closed:
                        if ear >= reopen_threshold:
                            duration = now - closed_since if closed_since else 0.0
                            is_closed = False
                            closed_since = None
                            open_since = now
                            if MIN_CLOSED_SECONDS <= duration <= MAX_CLOSED_SECONDS:
                                blink_count += 1
                                last_blink_at = now
                                recent_message = f"检测到一次眨眼（闭眼 {duration:.2f}s）"
                                blink_state = "已确认"
                            else:
                                recent_message = f"忽略非自然眨眼（{duration:.2f}s）"
                                blink_state = "已准备好"
                        else:
                            blink_state = "闭眼候选中"
                    elif ear >= reopen_threshold:
                        if open_since is None:
                            open_since = now
                        blink_state = "已准备好" if was_stably_open else "等待稳定睁眼"
                    elif ear < close_threshold and was_stably_open:
                        is_closed = True
                        closed_since = now
                        open_since = None
                        blink_state = "闭眼候选中"
                    else:
                        open_since = None
                        blink_state = "等待稳定睁眼"
            else:
                recent_message = "未检测到人脸：请正对摄像头"
                blink_state = "未检测到人脸"
                is_closed = False
                closed_since = None
                open_since = None
                ear_history.clear()

            panel_color = (0, 0, 0)
            cv2.rectangle(frame, (0, 0), (560, 125), panel_color, -1)
            ear_text = "--" if ear is None else f"{ear:.3f}"
            base_text = "--" if baseline is None else f"{baseline:.3f}"
            close_text = "--" if close_threshold is None else f"{close_threshold:.3f}"
            frame = draw_chinese_status(frame, [
                f"眼睛开合值：{ear_text}｜睁眼基线：{base_text}｜闭眼线：{close_text}",
                f"眨眼次数：{blink_count}｜状态：{blink_state}",
                recent_message,
                "按 Q 退出｜按 C 重新校准｜仅视觉测试，不控制任何硬件",
            ])
            cv2.imshow(WINDOW_NAME, frame)

            raw_key = cv2.waitKey(1)
            if raw_key == -1:
                key_is_held = False
                continue
            if key_is_held:
                continue
            key_is_held = True
            key = raw_key & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                calibration_values, baseline, close_threshold, reopen_threshold, blink_count = [], None, None, None, 0
                calibration_started_at = time.perf_counter()
                is_closed = False
                closed_since = None
                open_since = None
                last_blink_at = float("-inf")
                ear_history.clear()
                recent_message = "重新校准：请睁眼正视摄像头"
                blink_state = "正在校准"
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
