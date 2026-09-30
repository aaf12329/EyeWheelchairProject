"""路径注册表：全项目文件位置的唯一登记处（纯地址簿，不做读写）。

任何模块要文件位置都从这里 import，禁止自己拼 `Path(__file__).parents[...]`。
（参考 My_Agent 的 Store.py 原则：人人读它，它不调用任何人。）
"""
from pathlib import Path

# ---- 项目根与资产目录 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# ---- 模型权重（YOLO 线，由 Yolo_model 项目训练交付）----
EYE_WEIGHTS = MODELS_DIR / "eye_yolo26n.pt"          # 2 类：open_eye / closed_eye
GAZE_WEIGHTS = MODELS_DIR / "gaze_yolo26s.pt"        # 3 类主控：look_up/center/down
GAZE5_WEIGHTS = MODELS_DIR / "gaze5_yolo26s.pt"      # 5 类：含 look_left/right
YUNET_WEIGHTS = MODELS_DIR / "face_detection_yunet_2023mar.onnx"  # 人脸定位

# ---- 中文字体探测顺序（msyh 缺失时回退，换机器不会直接崩）----
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)

# ---- 输出目录 ----
SNAPSHOTS_DIR = PROJECT_ROOT / "runs" / "live_snapshots"
