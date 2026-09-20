"""无摄像头的冒烟测试：不校验识别准不准，只保证整条链路不崩。

覆盖两件事：
  1. 自造空白画面 → 模型能加载、能推理、没人脸时返回空结果（不依赖任何素材）
  2. 拿 data/raw_videos 里录好的 mp4 当假摄像头逐帧跑（文件不可用时跳过；
     仓库里现存的那批是 0 字节空文件，属于当时录像没写进去的残留）
"""
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
mp = pytest.importorskip("mediapipe")

from blink_preview import FACE_MODEL, average_ear, face_detector  # noqa: E402

VIDEO_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_videos"


def _require_model() -> None:
    if not FACE_MODEL.exists():
        pytest.skip(f"缺少模型文件：{FACE_MODEL}")


def test_detector_runs_on_synthetic_blank_frame():
    """全黑画面走一遍完整推理：验证模型能加载、时间戳递增可用、空画面没有关键点。"""
    _require_model()
    detector = face_detector()
    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    # 不调用 detector.close()：实测它要卡约 42 秒（MediaPipe 1.0.1 / Windows）
    for index in range(3):
        rgb = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO 模式要求时间戳严格递增
        result = detector.detect_for_video(image, index * 33)
        assert not result.face_landmarks, "全黑画面不该检出人脸"


def test_pipeline_runs_on_recorded_video():
    _require_model()
    # 先按大小过滤：仓库里现存的那批 mp4 是 0 字节空壳，直接交给 OpenCV 会刷一屏 ffmpeg 报错
    videos = [p for p in sorted(VIDEO_DIR.glob("*.mp4")) if p.stat().st_size > 0]
    if not videos:
        pytest.skip("data/raw_videos 里没有可用的 mp4（现存的是 0 字节空文件），跳过")

    cap = None
    for path in videos:
        candidate = cv2.VideoCapture(str(path))
        if candidate.isOpened():
            cap = candidate
            break
        candidate.release()
    if cap is None:
        pytest.skip("data/raw_videos 里的 mp4 全都打不开，跳过")

    detector = face_detector()
    frames_read = 0
    frames_with_face = 0
    # 同样不调用 detector.close()：见上面那条说明
    for index in range(30):
        ok, frame = cap.read()
        if not ok:
            break
        frames_read += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(image, index * 33)
        if result.face_landmarks:
            frames_with_face += 1
            ear = average_ear(result.face_landmarks[0])
            assert 0.0 <= ear < 5.0, f"开合值不合理：{ear}"
    cap.release()

    assert frames_read > 0, "视频一帧都没读出来"
    print(f"读了 {frames_read} 帧，其中 {frames_with_face} 帧检出人脸")
