"""第 5 步：视线选方向 + 眨眼确认（纯屏幕演示）。

流程：看正中校准 3 秒 → 稳定注视 左转/前进/右转 → 自然眨眼确认。
确认结果仅显示在屏幕上，不会向 Arduino、驱动板或轮椅发送任何指令。
按 Q 退出；C 重新校准；I 反转左右方向。
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
SIDE_THRESHOLD = 0.075
SELECT_STABLE_SECONDS = 0.70
CLOSE_RATIO, REOPEN_RATIO = 0.78, 0.88
MIN_CLOSED_SECONDS, MAX_CLOSED_SECONDS = 0.04, 0.80
MIN_OPEN_SECONDS, REFRACTORY_SECONDS = 0.10, 0.30

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "face_landmarker.task"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
WINDOW = "视线选择 + 眨眼确认｜Q 退出｜C 校准｜I 反转左右｜仅模拟演示"
LEFT_IRIS, RIGHT_IRIS = [468, 469, 470, 471, 472], [473, 474, 475, 476, 477]
LEFT_CORNERS, RIGHT_CORNERS = (33, 133), (362, 263)
LEFT_EYE, RIGHT_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]


def make_detector():
    if not MODEL.exists():
        raise FileNotFoundError(f"缺少模型：{MODEL}")
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def ear(landmarks, eye):
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in eye]
    horizontal = distance(p0, p3)
    return 0.0 if horizontal < 1e-6 else (distance(p1, p5) + distance(p2, p4)) / (2 * horizontal)


def gaze_score(landmarks):
    ratios = []
    for iris, corners in ((LEFT_IRIS, LEFT_CORNERS), (RIGHT_IRIS, RIGHT_CORNERS)):
        iris_x = sum(landmarks[i].x for i in iris) / len(iris)
        a, b = landmarks[corners[0]].x, landmarks[corners[1]].x
        if abs(a - b) > 1e-5:
            ratios.append((iris_x - min(a, b)) / abs(a - b))
    return sum(ratios) / len(ratios) if ratios else None


def chinese_overlay(frame, lines):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    small, large = ImageFont.truetype(str(FONT), 20), ImageFont.truetype(str(FONT), 29)
    positions = [(14, 10, small, (0, 255, 0)), (14, 37, large, (255, 255, 0)),
                 (14, 76, small, (255, 255, 255)), (14, 103, small, (255, 210, 80))]
    for text, (x, y, font, color) in zip(lines, positions):
        draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cards(frame, active):
    options = [("左转", 35, 390, 280, 515), ("前进", 340, 390, 585, 515), ("右转", 645, 390, 890, 515)]
    for label, x1, y1, x2, y2 in options:
        color = (0, 170, 0) if label == active else (70, 70, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1 if label == active else 2)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text((x1 + 64, y1 + 38), label, font=ImageFont.truetype(str(FONT), 36), fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frame


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened(): raise RuntimeError("摄像头无法打开。")
    detector = make_detector()
    started = calibration_started = time.perf_counter()
    gaze_values, ear_values = [], []
    center = close_line = reopen_line = None
    gaze_history, ear_history = deque(maxlen=5), deque(maxlen=3)
    state, raw_choice, stable_choice, pending = "校准", "前进", None, None
    direction_since = open_since = closed_since = confirm_started = None
    is_closed, invert, key_held = False, False, False
    last_blink = float("-inf")
    message = "请睁眼看屏幕正中间，正在校准"

    def reset(now):
        nonlocal calibration_started, gaze_values, ear_values, center, close_line, reopen_line
        nonlocal state, raw_choice, stable_choice, pending, direction_since, open_since, closed_since
        nonlocal is_closed, last_blink, confirm_started, message
        calibration_started = now; gaze_values, ear_values = [], []
        center = close_line = reopen_line = None
        gaze_history.clear(); ear_history.clear()
        state, raw_choice, stable_choice, pending = "校准", "前进", None, None
        direction_since = open_since = closed_since = confirm_started = None
        is_closed, last_blink = False, float("-inf")
        message = "请睁眼看屏幕正中间，正在校准"

    try:
        while True:
            ok, frame = cap.read()
            if not ok: raise RuntimeError("摄像头读取失败。")
            frame = cv2.flip(frame, 1); now = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now-started)*1000))
            score = current_ear = None
            blink_confirmed = False
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                score = gaze_score(lm)
                current_ear = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2
                if score is not None: gaze_history.append(score); score = sum(gaze_history)/len(gaze_history)
                ear_history.append(current_ear); current_ear = sum(ear_history)/len(ear_history)
                h, w = frame.shape[:2]
                for idx in LEFT_IRIS + RIGHT_IRIS + LEFT_EYE + RIGHT_EYE:
                    p = lm[idx]; cv2.circle(frame, (int(p.x*w), int(p.y*h)), 2, (0,255,255), -1)

                if state == "校准" and score is not None:
                    gaze_values.append(score); ear_values.append(current_ear)
                    elapsed = now-calibration_started
                    message = f"校准中：请看正中间 {max(0, CALIBRATION_SECONDS-elapsed):.1f} 秒"
                    if elapsed >= CALIBRATION_SECONDS and len(gaze_values) >= 10:
                        center = sum(gaze_values)/len(gaze_values); base = sum(ear_values)/len(ear_values)
                        close_line, reopen_line = base*CLOSE_RATIO, base*REOPEN_RATIO
                        state, message = "选择方向", "校准完成：看左转、前进或右转，稳定后等待眨眼"

                elif state in ("选择方向", "等待眨眼") and center is not None:
                    offset = score-center if score is not None else 0.0
                    if invert: offset = -offset
                    new_choice = "左转" if offset < -SIDE_THRESHOLD else "右转" if offset > SIDE_THRESHOLD else "前进"
                    if state == "选择方向":
                        if new_choice != raw_choice:
                            raw_choice, direction_since, stable_choice = new_choice, now, None
                        elif direction_since is not None and now-direction_since >= SELECT_STABLE_SECONDS:
                            stable_choice, pending, state = raw_choice, raw_choice, "等待眨眼"
                            message = f"已选择“{pending}”，请自然眨眼确认"

                    # 眨眼状态机：仅在“等待眨眼”时才把一次眨眼当确认。
                    stable_open = open_since is not None and now-open_since >= MIN_OPEN_SECONDS and now-last_blink >= REFRACTORY_SECONDS
                    if is_closed:
                        if current_ear >= reopen_line:
                            duration = now-closed_since if closed_since else 0.0
                            is_closed, closed_since, open_since = False, None, now
                            if MIN_CLOSED_SECONDS <= duration <= MAX_CLOSED_SECONDS:
                                last_blink, blink_confirmed = now, True
                        # 否则持续闭眼，等待睁开。
                    elif current_ear >= reopen_line:
                        if open_since is None: open_since = now
                    elif current_ear < close_line and stable_open:
                        is_closed, closed_since, open_since = True, now, None
                    else:
                        open_since = None

                    if state == "等待眨眼" and blink_confirmed:
                        state, confirm_started = "确认成功", now
                        message = f"已确认“{pending}”——这是模拟指令，不会移动轮椅"
                elif state == "确认成功" and confirm_started and now-confirm_started >= 1.6:
                    state, stable_choice, pending, direction_since = "选择方向", None, None, now
                    message = "请继续选择下一项方向"
            else:
                message = "未检测到人脸：已暂停，重新正对摄像头"
                if state != "校准": state, stable_choice, pending = "选择方向", None, None

            cv2.rectangle(frame, (0,0), (780,130), (0,0,0), -1)
            score_t = "--" if score is None else f"{score:.3f}"; ear_t = "--" if current_ear is None else f"{current_ear:.3f}"
            active = pending if state in ("等待眨眼", "确认成功") else stable_choice
            frame = chinese_overlay(frame, [f"虹膜位置：{score_t}｜眼睛开合值：{ear_t}", f"当前状态：{state}", message, "Q 退出｜C 重新校准｜I 反转左右｜仅屏幕模拟，不控制硬件"])
            frame = cards(frame, active)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1)
            if key == -1: key_held = False; continue
            if key_held: continue
            key_held = True; key &= 0xFF
            if key in (ord('q'),ord('Q')): break
            if key in (ord('c'),ord('C')): reset(now)
            if key in (ord('i'),ord('I')):
                invert = not invert; message = "左右方向已反转" if invert else "左右方向已恢复"
    finally:
        detector.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
