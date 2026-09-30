"""YOLO 后端的测试：纯信号逻辑不依赖任何模型，推理冒烟只在装好 ultralytics 时跑。

  1. 纯逻辑（合成数据，必跑）：双眼合并 / 睁眼置信度换算 / 注视信号映射 / 镜像坐标翻转
  2. 推理冒烟（可选）：ultralytics 与权重都在时，全黑画面走一遍完整链路不崩、
     无人脸返回空结果 —— 与旧 MediaPipe 冒烟测试同款定位：不校验准不准，只保证不崩
"""
import numpy as np
import pytest

from vision.yolo_backend import (
    DEFAULT_EYE_WEIGHTS,
    DEFAULT_GAZE_WEIGHTS,
    EyeGazeResult,
    YoloEyeGaze,
    combine,
    eye_open_confidence,
    flip_box,
    gaze_signal,
)

# ---------------------------- 纯逻辑 ----------------------------


def test_combine_both_eyes_agree_takes_min_confidence():
    """两眼一致 → 取该类别，置信度取较低者（保守，与 Yolo_model 的 predict 同款）。"""
    label, conf = combine([("open_eye", 0.90), ("open_eye", 0.70)])
    assert label == "open_eye"
    assert conf == pytest.approx(0.70)


def test_combine_disagreement_takes_higher_confidence():
    """两眼不一致 → 取置信度高的一方。"""
    label, conf = combine([("open_eye", 0.60), ("closed_eye", 0.85)])
    assert label == "closed_eye"
    assert conf == pytest.approx(0.85)


def test_combine_skips_unavailable_and_handles_empty():
    """单眼可用时直接用它；全不可用返回 (None, 0)。"""
    label, conf = combine([(None, 0.0), ("look_center", 0.80)])
    assert label == "look_center"
    assert conf == pytest.approx(0.80)
    assert combine([]) == (None, 0.0)


def test_eye_open_confidence_translates_labels():
    """睁眼 → 置信度本身；闭眼 → 1 - 置信度；不可用 → None。"""
    assert eye_open_confidence("open_eye", 0.90) == pytest.approx(0.90)
    assert eye_open_confidence("closed_eye", 0.80) == pytest.approx(0.20)
    assert eye_open_confidence(None, 0.0) is None


def test_gaze_signal_maps_labels():
    """水平注视信号：左 0 / 中 0.5 / 右 1；上下两态在水平轴上归中；不可用 → None。"""
    assert gaze_signal("look_left") == pytest.approx(0.0)
    assert gaze_signal("look_center") == pytest.approx(0.5)
    assert gaze_signal("look_right") == pytest.approx(1.0)
    assert gaze_signal("look_up") == pytest.approx(0.5)
    assert gaze_signal("look_down") == pytest.approx(0.5)
    assert gaze_signal(None) is None


def test_flip_box_mirrors_x_only():
    """显示层画框用：x 轴翻转，y 不动。"""
    assert flip_box((100, 50, 200, 120), width=960) == (760, 50, 860, 120)


def test_result_is_immutable_data_bag():
    """EyeGazeResult 是 frozen 数据袋：字段可读、不可改。"""
    result = EyeGazeResult(face_found=True, open_conf=0.9, gaze_label="look_center",
                           gaze_score=0.5, eye_boxes=((1, 2, 3, 4),), per_eye=())
    assert result.open_conf == pytest.approx(0.9)
    with pytest.raises(Exception):
        result.open_conf = 0.1


# ---------------------------- 推理冒烟（可选） ----------------------------


def _backend_or_skip():
    pytest.importorskip("ultralytics")  # 环境没装 ultralytics 时跳过，状态机测试不受影响
    for weights in (DEFAULT_EYE_WEIGHTS, DEFAULT_GAZE_WEIGHTS):
        if not weights.exists():
            pytest.skip(f"缺少权重文件：{weights}")
    return YoloEyeGaze()


def test_backend_runs_on_synthetic_blank_frame():
    """全黑画面走一遍完整链路：YuNet 找不到脸 → 返回空结果，不抛异常。"""
    backend = _backend_or_skip()
    blank = np.zeros((540, 960, 3), dtype=np.uint8)
    result = backend.analyze(blank)
    assert isinstance(result, EyeGazeResult)
    assert result.face_found is False
    assert result.open_conf is None
    assert result.gaze_score is None
    assert result.eye_boxes == ()


def test_backend_result_is_dense_on_noise_frame():
    """随机噪声画面同样不崩（YuNet 可能误检，但结果结构必须完整可用）。"""
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 255, size=(540, 960, 3), dtype=np.uint8)
    backend = _backend_or_skip()
    result = backend.analyze(noise)
    assert isinstance(result, EyeGazeResult)
    if result.face_found:
        assert result.open_conf is not None and 0.0 <= result.open_conf <= 1.0
