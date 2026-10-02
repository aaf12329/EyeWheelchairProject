"""MediaPipe 眼动后端：478 点几何判定（EAR 开合比 + 虹膜位置）。

定位：与 vision/yolo_backend.py **并列**的第二引擎，供双引擎对比脚本使用
（blink_preview / gaze_direction_preview）。两者接口形状一致：
    YoloEyeGaze.analyze(frame)        -> EyeGazeResult(open_conf, gaze_score, …)
    MediaPipeEyeGaze.analyze(frame,ts) -> MediaPipeResult(ear, gaze_score, …)

⚠️ 铁律（与 YOLO 线一致）：analyze() 只吃【未镜像帧】——镜像会把"看左/看右"反转；
   本模块已对水平信号做镜像修正（score = 1 - 虹膜在画面中的位置），
   使输出语义与 YOLO 线的 gaze_score 完全一致：**左 0 / 中 0.5 / 右 1**。

信号约定（供状态机直接消费）：
    ear        —— 眼纵横比（连续开合值，闭眼趋近 0）；接 BlinkDetector.update()
    gaze_score —— 虹膜水平位置（左 0 / 中 0.5 / 右 1）；接 GazeDirectionDetector.update()

文件分三段（与 yolo_backend 一致）：
  1) 数据模型与纯逻辑（EAR / 虹膜位置 —— 可单独喂假关键点测试）
  2) 模型加载与推理（FaceLandmarker，VIDEO 模式）
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.paths import FACE_LANDMARKER

# ---- 眼周 6 点（与旧版 blink_preview.py 逐字一致）----
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# ---- 虹膜 5 点 × 2（478 点模型自带）----
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
# ---- 内外眼角（虹膜位置的参考系）----
LEFT_CORNERS = (33, 133)
RIGHT_CORNERS = (362, 263)


# ============================ 1) 数据模型与纯逻辑 ============================


@dataclass(frozen=True)
class MediaPipeResult:
    """一帧的几何判定结果快照。只是"数据袋子"，不含逻辑。

    字段语义：
      face_found  有没有检测到人脸（False 时其余字段为 None）
      ear         双眼平均眼纵横比（连续开合值；闭眼趋近 0）——眨眼状态机直接吃它
      gaze_score  虹膜水平位置（已镜像修正：左 0 / 中 0.5 / 右 1）——方向状态机直接吃它
    """

    face_found: bool
    ear: float | None
    gaze_score: float | None


def _distance(a, b) -> float:
    """两个关键点间的平面距离（忽略 z）。"""
    return float(np.hypot(a.x - b.x, a.y - b.y))


def eye_aspect_ratio(lms) -> float | None:
    """双眼平均眼纵横比：上下眼睑平均距离 ÷ 眼角宽度。闭眼时趋近 0。

    公式与旧版 blink_preview.py 的 eye_aspect_ratio/average_ear 逐字一致
    （单眼 4 段距离：上外-下外、上内-下内、外眼角-内眼角）。
    """
    vals = []
    for idx in (LEFT_EYE, RIGHT_EYE):
        p0, p1, p2, p3, p4, p5 = [lms[i] for i in idx]
        horizontal = _distance(p0, p3)
        if horizontal < 1e-6:
            continue
        vals.append((_distance(p1, p5) + _distance(p2, p4)) / (2.0 * horizontal))
    return sum(vals) / len(vals) if vals else None


def iris_horizontal(lms) -> float | None:
    """虹膜在眼角间的水平归一化位置：0 = 贴画面左眼角，1 = 贴画面右眼角（未修正值）。

    注意：这是"画面坐标"的值。未镜像帧下，使用者看自己的左边 → 虹膜偏画面右侧
    → 该值偏大。镜像修正（1-x）在 gaze_score_from() 里做。
    """
    ratios = []
    for iris, corners in ((LEFT_IRIS, LEFT_CORNERS), (RIGHT_IRIS, RIGHT_CORNERS)):
        a, b = lms[corners[0]].x, lms[corners[1]].x
        span = abs(b - a)
        if span < 1e-5:
            continue
        cx = sum(lms[i].x for i in iris) / len(iris)
        ratios.append((cx - min(a, b)) / span)
    return sum(ratios) / len(ratios) if ratios else None


def gaze_score_from(lms) -> float | None:
    """虹膜水平位置 → 与 YOLO 线同语义的注视信号（左 0 / 中 0.5 / 右 1）。

    镜像修正：未镜像帧上直接测得的值是反的（看左 → 偏大），取 1-x 后
    "看左 → 偏小"，与 YOLO 的 look_left→0.0 映射完全对齐。
    """
    hx = iris_horizontal(lms)
    return None if hx is None else 1.0 - hx


# ============================ 2) 模型加载与推理 ============================


class MediaPipeEyeGaze:
    """MediaPipe 引擎：478 点 → EAR + 虹膜位置。无逐帧状态（模型只读），可多处共用。

    调用关系：双引擎对比脚本在启动时 new 一个（读模型慢，绝不能放进循环）；
    之后每帧调 analyze(未镜像帧, 毫秒时间戳)，只拿 MediaPipeResult。
    """

    def __init__(self, model_path: Path = FACE_LANDMARKER) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:  # pragma: no cover - 环境问题不值得造测试
            raise RuntimeError(
                "缺少 mediapipe 库：请先在虚拟环境里执行 "
                "python -m pip install mediapipe（双引擎对比脚本需要；只跑 YOLO 可加 --no-mp）"
            ) from exc
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"缺少 MediaPipe 人脸模型：{model_path}\n"
                "获取方式：从 main 分支检出（git show main:models/face_landmarker.task），"
                "或从 Yolo_model 项目的 models/ 复制。"
            )
        self._mp = mp
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                num_faces=1,
                running_mode=mp_vision.RunningMode.VIDEO,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._ts_last = -1  # 时间戳必须严格递增（VIDEO 模式的要求）

    def analyze(self, frame: np.ndarray, ts_ms: int) -> MediaPipeResult:
        """吃一帧【未镜像】BGR 画面 + 毫秒时间戳，返回 MediaPipeResult。每帧一次。"""
        if frame is None:
            return MediaPipeResult(False, None, None)
        if ts_ms <= self._ts_last:  # 保险：时钟回绕或调用方传了不增的时间戳
            ts_ms = self._ts_last + 1
        self._ts_last = ts_ms

        rgb = frame[:, :, ::-1]  # BGR → RGB（比 cv2.cvtColor 快一档，避免引 cv2）
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                  data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return MediaPipeResult(False, None, None)
        lms = result.face_landmarks[0]
        return MediaPipeResult(
            face_found=True,
            ear=eye_aspect_ratio(lms),
            gaze_score=gaze_score_from(lms),
        )
