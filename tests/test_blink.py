"""眨眼判定的行为测试：不需要摄像头，用合成数据按时间轴驱动 BlinkDetector。

锁定的是重构前的行为（阈值一个都没改）：
  1. 校准：3 秒 + 至少 10 个样本 → 基线 = 中间 80% 均值，两条线 = 基线 × 0.78 / 0.88
  2. 一次自然眨眼（闭眼约 0.13 秒）计数 +1
  3. 只闭 1 帧（噪声）或闭 1.2 秒（眯眼/闭目）都不计数
  4. 不应期（0.3 秒）内不重复计数
  5. 人脸丢失：未完成的状态清零，但计数和基线保留
  6. 眼睛开合比（EAR）的几何正确性
"""
import pytest

from blink_preview import (
    LEFT_EYE,
    RIGHT_EYE,
    BlinkDetector,
    average_ear,
    eye_aspect_ratio,
)

FRAME = 1.0 / 30.0  # 模拟 30 FPS


def frames(seconds: float) -> int:
    """把秒数换算成帧数（至少 1 帧）。"""
    return max(1, int(round(seconds / FRAME)))


def feed(detector, now: float, ear_values):
    """按帧喂数据，返回（结束时间, 最后一个状态）。ear_values 里可以是 None（没人脸）。"""
    status = None
    for value in ear_values:
        status = detector.update(now, value)
        now += FRAME
    return now, status


def calibrated(now: float = 0.0):
    """造一个已完成校准（基线 0.50）的检测器，返回（检测器, 当前时间）。"""
    detector = BlinkDetector(now)
    now, status = feed(detector, now, [0.50] * frames(3.2))
    assert status.is_calibrating is False
    return detector, now


# ---------------------------- 校准 ----------------------------


def test_calibration_sets_baseline_and_thresholds():
    detector = BlinkDetector(0.0)
    _, status = feed(detector, 0.0, [0.50] * frames(3.2))
    assert status.is_calibrating is False
    assert status.baseline == pytest.approx(0.50)
    assert status.close_threshold == pytest.approx(0.50 * 0.78)
    assert status.blink_count == 0


def test_calibration_waits_for_enough_samples():
    """帧率过低时，光够 3 秒不算，样本数也得够。"""
    detector = BlinkDetector(0.0)
    now = 0.0
    status = None
    for _ in range(7):  # 走到 3.0 秒，但只有 7 个样本
        status = detector.update(now, 0.50)
        now += 0.5
    assert status.is_calibrating is True

    for _ in range(4):  # 样本补到 10 个以上才完成
        status = detector.update(now, 0.50)
        now += 0.5
    assert status.is_calibrating is False
    assert status.baseline == pytest.approx(0.50)


# ---------------------------- 计数 ----------------------------


def test_normal_blink_counts_once():
    detector, now = calibrated()
    now, _ = feed(detector, now, [0.50] * frames(0.3))  # 先稳定睁眼
    now, _ = feed(detector, now, [0.10] * frames(0.15))  # 闭眼约 0.13 秒
    now, status = feed(detector, now, [0.50] * frames(0.2))  # 睁开
    assert status.blink_count == 1
    assert "检测到一次眨眼" in status.message


def test_one_frame_dip_counts_as_short_blink_after_smoothing():
    """只闭 1 帧也会被算成一次眨眼——因为 3 帧平滑会把它拉长成约 0.10 秒。

    这是重构前就有的行为：MIN_CLOSED_SECONDS=0.04 在 30fps 下基本不会触发，
    真正过滤噪声的是平滑窗口本身。这里把它锁住，将来改平滑参数时能立刻发现。
    """
    detector, now = calibrated()
    now, _ = feed(detector, now, [0.50] * frames(0.3))
    now, _ = feed(detector, now, [0.10])  # 只闭 1 帧
    now, status = feed(detector, now, [0.50] * frames(0.2))
    assert status.blink_count == 1
    assert "检测到一次眨眼" in status.message


def test_closure_below_min_closed_seconds_is_ignored():
    """闭眼时长短于下限时不计数的分支（用调大的下限来触发）。"""
    detector = BlinkDetector(0.0, min_closed_seconds=0.5)
    now, status = feed(detector, 0.0, [0.50] * frames(3.2))
    assert status.is_calibrating is False
    now, _ = feed(detector, now, [0.50] * frames(0.3))
    now, _ = feed(detector, now, [0.10] * frames(0.15))  # 只闭约 0.2 秒（含平滑拖尾）
    now, status = feed(detector, now, [0.50] * frames(0.2))
    assert status.blink_count == 0
    assert "忽略" in status.message


def test_long_closure_is_not_a_blink():
    detector, now = calibrated()
    now, _ = feed(detector, now, [0.50] * frames(0.3))
    now, _ = feed(detector, now, [0.10] * frames(1.2))  # 闭了 1.2 秒
    now, status = feed(detector, now, [0.50] * frames(0.2))
    assert status.blink_count == 0
    assert "忽略" in status.message


def test_refractory_blocks_second_count():
    detector, now = calibrated()
    now, _ = feed(detector, now, [0.50] * frames(0.3))
    now, _ = feed(detector, now, [0.10] * frames(0.15))
    now, status = feed(detector, now, [0.50] * frames(0.2))
    assert status.blink_count == 1

    # 睁眼 0.15 秒后又要眨：睁眼时长够了，但还在 0.3 秒不应期内 → 不计数
    now, _ = feed(detector, now, [0.50] * frames(0.15))
    now, _ = feed(detector, now, [0.10] * frames(0.15))
    now, status = feed(detector, now, [0.50] * frames(0.2))
    assert status.blink_count == 1

    # 稳定睁眼足够久之后，下一次眨眼才重新计数
    now, _ = feed(detector, now, [0.50] * frames(0.5))
    now, _ = feed(detector, now, [0.10] * frames(0.15))
    now, status = feed(detector, now, [0.50] * frames(0.2))
    assert status.blink_count == 2


def test_face_lost_clears_in_flight_state():
    detector, now = calibrated()
    now, _ = feed(detector, now, [0.50] * frames(0.3))
    now, _ = feed(detector, now, [0.10] * frames(0.15))
    now, status = feed(detector, now, [0.50] * frames(0.2))
    assert status.blink_count == 1

    now, status = feed(detector, now, [None] * 3)  # 人脸丢了
    assert status.ear is None
    assert status.state == "未检测到人脸"
    assert status.blink_count == 1  # 已经数到的眨眼不清零
    assert status.baseline is not None  # 也不用重新校准

    now, status = feed(detector, now, [0.50])  # 人脸回来
    assert status.state == "等待稳定睁眼"  # 未完成的状态被清空，从头攒稳定睁眼


# ---------------------------- EAR 几何 ----------------------------


class FakeLandmark:
    """假的单个关键点：EAR 只读 x / y。"""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def make_landmarks(vertical_gap: float = 0.05, eye_width: float = 0.06):
    """造一副 478 点的假人脸：只摆两只眼睛的 12 个点，其余点放画面中央。"""
    landmarks = [FakeLandmark(0.5, 0.5) for _ in range(478)]
    for shift, indices in ((0.0, LEFT_EYE), (0.2, RIGHT_EYE)):
        cx, cy = 0.35 + shift, 0.5
        landmarks[indices[0]] = FakeLandmark(cx, cy)  # 外眼角
        landmarks[indices[1]] = FakeLandmark(cx + eye_width * 0.25, cy - vertical_gap / 2)  # 上外
        landmarks[indices[2]] = FakeLandmark(cx + eye_width * 0.75, cy - vertical_gap / 2)  # 上内
        landmarks[indices[3]] = FakeLandmark(cx + eye_width, cy)  # 内眼角
        landmarks[indices[4]] = FakeLandmark(cx + eye_width * 0.75, cy + vertical_gap / 2)  # 下内
        landmarks[indices[5]] = FakeLandmark(cx + eye_width * 0.25, cy + vertical_gap / 2)  # 下外
    return landmarks


def test_eye_aspect_ratio_open_bigger_than_closed():
    open_eye = make_landmarks(vertical_gap=0.05, eye_width=0.06)
    assert eye_aspect_ratio(open_eye, LEFT_EYE) == pytest.approx(0.05 / 0.06)
    assert average_ear(open_eye) == pytest.approx(0.05 / 0.06)

    closed_eye = make_landmarks(vertical_gap=0.0, eye_width=0.06)
    assert eye_aspect_ratio(closed_eye, LEFT_EYE) == pytest.approx(0.0)


def test_eye_aspect_ratio_degenerate_width_returns_zero():
    landmarks = make_landmarks()
    for index in LEFT_EYE:
        landmarks[index] = FakeLandmark(0.4, 0.5)  # 眼角重合 → 宽度为 0
    assert eye_aspect_ratio(landmarks, LEFT_EYE) == 0.0
