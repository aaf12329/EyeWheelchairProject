"""第 2 周：MediaPipe 人脸、眼睛和手部关键点可视化。

本程序仅把识别结果绘制到屏幕，绝不会向 Arduino 或轮椅发送命令。
按 Q 退出；按 S 保存当前带关键点的截图。
"""
from datetime import datetime
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
WINDOW_NAME = "MediaPipe Landmarks | Q quit | S snapshot"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "landmark_snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def point(lm, width: int, height: int) -> tuple[int, int]:
    return int(lm.x * width), int(lm.y * height)


def draw_face(frame, face_landmarks) -> None:
    """绘制全部脸部点，并突出眼睛轮廓。"""
    h, w = frame.shape[:2]
    for lm in face_landmarks:
        cv2.circle(frame, point(lm, w, h), 1, (0, 255, 0), -1)

    # MediaPipe Face Landmarker 的眼睛轮廓关键点索引。
    eye_loops = [
        [33, 160, 158, 133, 153, 144],       # 左眼
        [362, 385, 387, 263, 373, 380],       # 右眼
    ]
    for loop in eye_loops:
        pts = [point(face_landmarks[i], w, h) for i in loop]
        cv2.polylines(frame, [np.array(pts, dtype=np.int32)], True, (0, 255, 255), 1)


def draw_hand(frame, landmarks, label: str) -> None:
    """绘制 21 个手部点与基础骨架连接。"""
    h, w = frame.shape[:2]
    links = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    ]
    pts = [point(lm, w, h) for lm in landmarks]
    for a, b in links:
        cv2.line(frame, pts[a], pts[b], (255, 100, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    cv2.putText(frame, label, (pts[0][0] + 8, pts[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)


def create_detectors():
    if not FACE_MODEL.exists() or not HAND_MODEL.exists():
        missing = [str(p.name) for p in (FACE_MODEL, HAND_MODEL) if not p.exists()]
        raise FileNotFoundError(f"缺少模型：{', '.join(missing)}")

    vision = mp.tasks.vision
    mode = vision.RunningMode.VIDEO
    face_options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(face_options), vision.HandLandmarker.create_from_options(hand_options)


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("摄像头无法打开：请关闭占用摄像头的软件，或将 CAMERA_INDEX 改为 1。")

    face_detector, hand_detector = create_detectors()
    start = time.perf_counter()
    frame_count = 0
    key_is_held = False
    print("关键点预览已启动：Q 退出，S 保存截图。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("摄像头读取失败。")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)

            face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
            for landmarks in face_result.face_landmarks:
                draw_face(frame, landmarks)
            for index, landmarks in enumerate(hand_result.hand_landmarks):
                label = "HAND"
                if index < len(hand_result.handedness) and hand_result.handedness[index]:
                    label = hand_result.handedness[index][0].category_name.upper()
                draw_hand(frame, landmarks, label)

            frame_count += 1
            elapsed = max(time.perf_counter() - start, 0.001)
            fps = frame_count / elapsed
            faces = len(face_result.face_landmarks)
            hands = len(hand_result.hand_landmarks)
            cv2.rectangle(frame, (0, 0), (410, 82), (0, 0, 0), -1)
            cv2.putText(frame, f"FACE: {faces} | HAND: {hands} | {fps:.1f} FPS", (14, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Q quit | S snapshot | VISUAL ONLY", (14, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
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
            if key in (ord("s"), ord("S")):
                path = SNAPSHOT_DIR / f"landmarks_{stamp()}.jpg"
                cv2.imwrite(str(path), frame)
                print(f"截图已保存：{path}")
    finally:
        face_detector.close()
        hand_detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
