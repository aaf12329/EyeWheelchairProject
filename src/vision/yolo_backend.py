"""YOLO 眼动后端：眼部定位（OpenCV YuNet）+ 双模型推理（睁闭眼 / 5 类注视）。

把 MediaPipe 的"关键点几何判定"整体换成 Yolo_model 项目训练的两个 YOLO26 模型：
  eye_yolo26n.pt   2 类：open_eye / closed_eye（眼部裁剪小图 → 检测框）
  gaze5_yolo26s.pt 5 类：look_up / look_center / look_down / look_left / look_right

⚠️ 两条铁律（沿用 Yolo_model 项目 test_gaze_live.py 的实测结论）：
  1. analyze() 只吃【未镜像帧】——镜像会把"看左/看右"反转；镜像只用于显示层。
     （接口上不强制：你传镜像帧它也跑，但左右语义就是反的。）
  2. 两个模型都吃【眼部裁剪小图】，不吃整幅画面 —— 所以后端先用 YuNet
     （OpenCV 自带 FaceDetectorYN，权重在 models/face_detection_yunet_2023mar.onnx）
     定位人脸与双眼中心，再按训练同款比例裁出小图。

诚实的已知边界（写在最前面，免得后面忘记）：
  - 训练时的眼部框来自 MediaPipe 关键点外接框（6 点 × 扩边 1.6/2.2），现在换成
    "YuNet 眼中心 + 按人脸框比例取框"，中心一致、尺寸近似（两个比例常量可调）；
  - 所有者实测：YOLO 在训练域 95%+，换机器/换摄像头会明显下降（域差距）。
    所以后端把置信度原样交给上层，低置信度的处理（不判定/不候选）由状态机与界面决定。

文件分三段：
  1) 数据模型与纯逻辑（信号换算、双眼合并 —— 可单独跑测试）
  2) 模型加载与推理（YuNet 定位 + ultralytics 推理）
  3) 便捷工具（给显示层用的坐标翻转）

============================ 调用关系总览 ============================

  交互脚本（blink_preview / gaze_direction_preview / gaze_blink_confirm_demo）
  │
  ├─ 启动阶段（只执行一次）
  │   └─ YoloEyeGaze(eye_weights, gaze_weights)     建后端：加载 2 个 YOLO + YuNet 人脸定位
  │
  └─ 每帧循环（★ 每帧都执行）
      ├─★ backend.analyze(frame_raw)               ← 唯一入口：吃【未镜像】BGR 帧
      │    ├─ _locate_eyes(frame)                  YuNet：人脸框 + 双眼中心 → 裁剪框
      │    ├─ _top_pred(eye_model, crop)           睁/闭眼（top 类别 + 置信度）
      │    ├─ _top_pred(gaze_model, crop)          5 类注视（同上）
      │    └─ combine(per_eye)                     双眼合并（一致取之 / 不一致取高置信）
      └─★ 返回 EyeGazeResult：
           open_conf  → 交给 BlinkDetector.update() 当"开合值"（0~1，闭眼时走低）
           gaze_score → 交给方向状态机当"虹膜位置"（左 0 / 中 0.5 / 右 1）
           eye_boxes  → 画框用（未镜像坐标；显示层画框用 flip_box() 翻转 x）
"""
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---- 默认权重路径（与仓库 models/ 一致；也可在构造时传入别的路径）----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EYE_WEIGHTS = PROJECT_ROOT / "models" / "eye_yolo26n.pt"
DEFAULT_GAZE_WEIGHTS = PROJECT_ROOT / "models" / "gaze5_yolo26s.pt"
DEFAULT_YUNET = PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"

# ---- 推理参数（与 Yolo_model/scripts 的 live / eval 同款）----
IMGSZ = 128          # 训练时的输入尺寸，推理必须一致
CONF_THRESHOLD = 0.30  # 检测框置信度下限（eval 用 0.50；live 放宽一些，低置信交给上层处理）
FACE_SCORE_THRESHOLD = 0.60  # YuNet 人脸置信度下限
EYE_CROP_W_RATIO = 0.48  # 眼部裁剪宽 ≈ 人脸框宽的比例（近似训练分布，可调）
EYE_CROP_H_RATIO = 0.20  # 眼部裁剪高 ≈ 人脸框高的比例（同上）
MIN_CROP_PX = 16     # 裁剪太小没有意义（训练数据里不存在这种图）

# 水平注视信号：左 0 / 中 0.5 / 右 1；上下两态在水平轴上归中（与旧虹膜几何等价：上看时虹膜水平居中）
GAZE_SCORE = {
    "look_left": 0.0,
    "look_center": 0.5,
    "look_right": 1.0,
    "look_up": 0.5,
    "look_down": 0.5,
}


# ============================ 1) 数据模型与纯逻辑 ============================


@dataclass(frozen=True)
class EyeGazeResult:
    """一帧的推理结果快照，供状态机与界面消费。只是"数据袋子"，不含逻辑。

    字段语义：
      face_found  FaceDetectorYN 有没有找到人脸（False 时其余字段全为 None/空）
      open_conf   睁眼信号 0~1：双眼平均"是睁眼"的置信度；闭眼时走低。
                  None = 这帧没有可用的眼部检测。直接喂 BlinkDetector.update()。
      gaze_label  双眼合并后的注视类别（look_left / look_center / ...），None = 不可用
      gaze_score  水平注视信号 0~1（左 0 / 中 0.5 / 右 1），直接喂方向状态机
      eye_boxes   YuNet 定位出的眼部裁剪框（未镜像帧坐标，画在镜像显示上要用 flip_box()）
      per_eye     每只眼的 (眼类别, 眼置信度, 注视类别, 注视置信度)，调试用
    """

    face_found: bool
    open_conf: float | None
    gaze_label: str | None
    gaze_score: float | None
    eye_boxes: tuple[tuple[int, int, int, int], ...]
    per_eye: tuple[tuple[str | None, float, str | None, float], ...]


def eye_open_confidence(eye_label: str | None, eye_conf: float) -> float | None:
    """单眼的"睁眼信号"：top 类别是睁眼 → 置信度本身；是闭眼 → 1 - 置信度。

    调用关系：被 YoloEyeGaze.analyze() 每帧每眼调用一次。
    它把"分类结果"翻译成连续信号：睁眼≈高、闭眼≈低 —— 正好接替旧 EAR 的角色，
    让 BlinkDetector（校准/阈值/不应期那套）原封不动继续工作。
    """
    if eye_label is None:
        return None
    return eye_conf if eye_label == "open_eye" else 1.0 - eye_conf


def combine(per_eye: Iterable[tuple[str | None, float]]) -> tuple[str | None, float]:
    """双眼合并：两眼一致 → 取该类别（置信度取较低者，保守）；不一致 → 取置信度高者。

    调用关系：被 YoloEyeGaze.analyze() 调用（眼类别与注视类别各一次）。
    与 Yolo_model/scripts/test_gaze_live.py 的 combine() 同款，保证两边行为一致。
    """
    valid = [(label, conf) for label, conf in per_eye if label is not None]
    if not valid:
        return None, 0.0
    if len(valid) == 2 and valid[0][0] == valid[1][0]:
        return valid[0][0], min(valid[0][1], valid[1][1])
    return max(valid, key=lambda item: item[1])


def gaze_signal(label: str | None) -> float | None:
    """注视类别 → 水平信号（左 0 / 中 0.5 / 右 1）。不可用类别返回 None。"""
    if label is None:
        return None
    return GAZE_SCORE.get(label)


def flip_box(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    """把"未镜像帧坐标"的框换算到"镜像显示帧"坐标（x 轴翻转）。

    调用关系：交互脚本在镜像显示上画框时调用（显示帧 = cv2.flip(原始帧, 1)）。
    """
    x1, y1, x2, y2 = box
    return (width - x2, y1, width - x1, y2)


# ============================ 2) 模型加载与推理 ============================


def _load_yolo(weights: Path):
    """加载一个 YOLO 权重；缺 ultralytics 时给出能落地的中文提示。延迟导入。"""
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - 环境问题不值得造测试
        raise RuntimeError(
            "缺少 ultralytics 库：请先在虚拟环境里执行 "
            "python -m pip install ultralytics（会自动带上 CPU 版 torch）。"
        ) from exc
    if not weights.exists():
        raise FileNotFoundError(f"缺少权重文件：{weights}")
    return YOLO(str(weights))


def _check_yunet(path: Path) -> Path:
    """确认 YuNet 权重存在（FaceDetectorYN 是 OpenCV 自带 API，不需要额外安装）。"""
    if not path.exists():
        raise FileNotFoundError(
            f"缺少 YuNet 模型：{path}（OpenCV 5 已移除 Haar 级联，本仓库统一用 YuNet 定位人脸）"
        )
    return path


class YoloEyeGaze:
    """眼动后端：YuNet 定位眼睛 → 裁剪 → 双 YOLO 推理 → 合并成信号。

    调用关系：交互脚本在启动时 new 一个（读模型慢，绝不能放进循环）；
    之后每帧调 analyze(frame_raw)，只拿 EyeGazeResult，不碰内部状态。
    对象本身无逐帧状态（模型只读），所以多脚本各建各的也互不影响。
    """

    def __init__(
        self,
        eye_weights: Path = DEFAULT_EYE_WEIGHTS,
        gaze_weights: Path = DEFAULT_GAZE_WEIGHTS,
        yunet_path: Path = DEFAULT_YUNET,
        *,
        imgsz: int = IMGSZ,
        conf: float = CONF_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self._imgsz = imgsz
        self._conf = conf
        self._device = device
        self._yunet_path = _check_yunet(yunet_path)
        self._eye_model = _load_yolo(eye_weights)
        self._gaze_model = _load_yolo(gaze_weights)

    # ---- 定位 ----

    def _locate_eyes(self, frame: np.ndarray) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
        """整帧 → [(眼部裁剪框, 所在人脸框), ...]（未镜像坐标）。

        调用关系：只被 analyze() 调用。YuNet 一次给出人脸框与双眼中心，
        裁剪框 = 以眼中心为中心、按人脸框比例取的近似训练分布框。
        """
        h, w = frame.shape[:2]
        detector = cv2.FaceDetectorYN_create(str(self._yunet_path), "", (w, h),
                                             score_threshold=FACE_SCORE_THRESHOLD)
        ok, faces = detector.detect(frame)
        if not ok or faces is None or not len(faces):
            return []
        # 取最大的人脸（驾驶员应该只有一个）
        x, y, fw, fh = max((f for f in faces), key=lambda f: f[2] * f[3])[:4]
        boxes: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
        # faces 行的第 4~7 列是右眼、左眼的中心点（像素坐标）
        for eye_index in (4, 6):
            cx, cy = float(faces[0][eye_index]), float(faces[0][eye_index + 1])
            half_w, half_h = fw * EYE_CROP_W_RATIO / 2, fh * EYE_CROP_H_RATIO / 2
            x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
            x1, y1 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
            if x1 - x0 >= MIN_CROP_PX and y1 - y0 >= MIN_CROP_PX:
                boxes.append(((x0, y0, x1, y1), (int(x), int(y), int(fw), int(fh))))
        return boxes

    # ---- 推理 ----

    def _top_pred(self, model, crop: np.ndarray) -> tuple[str | None, float]:
        """单个裁剪 → 最可信类别 + 置信度（与 Yolo_model 的 top_pred 同款）。"""
        result = model.predict(crop, imgsz=self._imgsz, device=self._device,
                               verbose=False, conf=self._conf)[0]
        if not len(result.boxes):
            return None, 0.0
        top = int(result.boxes.conf.argmax())
        return result.names[int(result.boxes.cls[top])], float(result.boxes.conf[top])

    def analyze(self, frame: np.ndarray) -> EyeGazeResult:
        """吃一帧【未镜像】BGR 画面，返回 EyeGazeResult。每帧调用一次。

        调用关系：被交互脚本的 read_frame() 每帧调用；内部调用 _locate_eyes()、
        _top_pred()、eye_open_confidence()、combine()、gaze_signal()。
        """
        if frame is None:
            return EyeGazeResult(False, None, None, None, (), ())
        located = self._locate_eyes(frame)
        if not located:
            return EyeGazeResult(False, None, None, None, (), ())

        per_eye: list[tuple[str | None, float, str | None, float]] = []
        kept_boxes: list[tuple[int, int, int, int]] = []
        for crop_box, _face in located:
            x0, y0, x1, y1 = crop_box
            crop = frame[y0:y1, x0:x1]
            eye_label, eye_conf = self._top_pred(self._eye_model, crop)
            gaze_label, gaze_conf = self._top_pred(self._gaze_model, crop)
            per_eye.append((eye_label, eye_conf, gaze_label, gaze_conf))
            kept_boxes.append(crop_box)

        eye_label, eye_conf = combine([(e, c) for e, c, _, _ in per_eye])
        gaze_label, _ = combine([(g, c) for _, _, g, c in per_eye])
        open_conf = eye_open_confidence(eye_label, eye_conf)
        return EyeGazeResult(
            face_found=True,
            open_conf=open_conf,
            gaze_label=gaze_label,
            gaze_score=gaze_signal(gaze_label),
            eye_boxes=tuple(kept_boxes),
            per_eye=tuple(per_eye),
        )


# ============================ 3) 便捷工具 ============================


def draw_eye_boxes(frame_display, result: EyeGazeResult, width: int) -> None:
    """在【镜像显示帧】上画眼睛框与注视类别（原始坐标经 flip_box 翻转）。

    调用关系：被交互脚本的绘制段每帧调用（只在 face_found 时）；
    只画图，不参与任何判定 —— 和旧 draw_eye_points() 的定位一样，纯肉眼检查用。
    """
    for box in result.eye_boxes:
        x1, y1, x2, y2 = flip_box(box, width)
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 2)
