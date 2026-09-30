"""第 2 周（YOLO 版）：眼动模型检测可视化 —— 睁/闭眼 + 5 类注视。

本程序仅把 YOLO 检测结果绘制到屏幕，绝不会向 Arduino 或轮椅发送命令。
它接替旧版 landmarks_preview.py（MediaPipe 478 点 + 手部 21 点）的角色：
交互链路已全面换成 YOLO（见 src/vision/yolo_backend.py），这个脚本就是
"裸看模型输出"的窗口 —— 调阈值、查模型抽没抽风时先开它。

按 Q 退出；按 S 保存当前带检测框的截图。

⚠️ 铁律：YOLO 推理只吃【未镜像帧】（镜像会把看左/看右反转）；
   所以画面显示的是镜像自拍视角，检测框经 flip_box 翻转后画上去。
   详见 vision/yolo_backend.py。

文件分三段：
  1) 结果绘制（把 EyeGazeResult 翻译成框、中文标签）—— 不碰摄像头
  2) 摄像头与后端（打开设备、建 YoloEyeGaze）
  3) 主流程 main()：读帧 → 推理 → 画 → 推窗口 → 按键

============================ 调用关系总览 ============================

  __main__  →  main()
  │
  ├─ 启动阶段（每个只执行一次）
  │   ├─ open_camera()                打开摄像头，返回 cap（后面每帧从它 read）
  │   ├─ make_backend()               建 YOLO 后端（eye_yolo26n + gaze5_yolo26s）
  │   └─ print(...)                   启动提示
  │
  ├─ 每帧循环（★ 每帧都执行；顺序 看 → 画 → 控）
  │   ├─★ read_and_detect(cap, backend)
  │   │    ├─ cap.read()                       取【原始帧】（不镜像！）
  │   │    ├─ backend.analyze(frame)           ← YOLO 推理（未镜像帧，铁律）
  │   │    ├─ cv2.flip(frame, 1)               只为显示做镜像
  │   │    └─ 返回 (显示帧, now, EyeGazeResult)
  │   ├─★ draw_results(显示帧, result, fps)
  │   │    ├─ draw_eye_boxes(...)              双眼黄框（vision.yolo_backend 提供）
  │   │    ├─ draw_labels(...)                 每只眼上方：眼状态 + 注视类别（中文）
  │   │    └─ 左上角黑底状态栏：眼睛 / 注视 / FPS（中文，PIL 渲染）
  │   ├─★ cv2.imshow(WINDOW_NAME, frame)      把这一帧推到窗口
  │   ├─★ window_is_alive()                   问窗口还活着没（点 ✕ 后为假）
  │   └─★ pressed_key(now, key_available_at)  读按键：q 退出 / s 存截图
  │        └─ save_snapshot(frame) → stamp()  存到 data/yolo_snapshots/
  │
  └─ finally（无论怎么退出都执行）
      ├─ cap.release()                  交还摄像头
      └─ cv2.destroyAllWindows()        关窗口
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 本项目各脚本独立运行（没有包结构），把 src/ 加进搜索路径以引入 YOLO 后端
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.yolo_backend import YoloEyeGaze, draw_eye_boxes, flip_box  # noqa: E402  ← 在 sys.path 之后导入

# ---- 运行参数 ----
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
KEY_DEBOUNCE_SECONDS = 0.25   # 两次按键响应的最小间隔：按住不放时不再连发

# ---- 路径与窗口 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "yolo_snapshots"
WINDOW_NAME = "YOLO Eye & Gaze | Q quit | S snapshot"
# 中文字体逐个探测：msyh 缺失时回退，换机器不会直接崩
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)

# 类别 → 中文（与 Yolo_model/scripts/test_gaze_live.py 同款）
EYE_ZH = {"open_eye": "睁眼", "closed_eye": "闭眼"}
GAZE_ZH = {"look_up": "上看", "look_center": "直视", "look_down": "下看",
           "look_left": "看左", "look_right": "看右"}


# ============================ 1) 结果绘制 ============================


@dataclass(frozen=True)
class Snapshot:
    """一次截图需要的东西：显示帧 + 时间戳。只是"数据袋子"。"""

    frame: np.ndarray
    now: float


def stamp(now: float) -> str:
    """把时间戳格式化成文件名友好的字符串（只被 save_snapshot() 调用）。"""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(now)) + f"_{int(now % 1 * 1000):03d}"


_font_cache: dict[tuple[Path, int], "ImageFont.FreeTypeFont"] = {}


def chinese_font(size: int):
    """按 FONT_CANDIDATES 找到第一个可用字体并缓存，避免每帧重复读字体文件。

    调用关系：被 draw_labels() 和 draw_status_bar() 调用；第一次读文件后命中缓存。
    """
    for path in FONT_CANDIDATES:
        if not path.exists():
            continue
        cached = _font_cache.get((path, size))
        if cached is None:
            cached = ImageFont.truetype(str(path), size)
            _font_cache[(path, size)] = cached
        return cached
    raise FileNotFoundError(
        "找不到可用的中文字体，请检查 FONT_CANDIDATES：" + "、".join(str(p) for p in FONT_CANDIDATES)
    )


def _pil_layer(frame, items):
    """一次 PIL 往返画完全部中文。items = [(x, y, 文本, 字号, BGR颜色), ...]。

    调用关系：被 draw_labels() 和 draw_status_bar() 调用。
    它不修改传入的 frame，而是返回一张画好字的新图。
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for x, y, text, size, color in items:
        draw.text((x, y), text, font=chinese_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_labels(frame_display, result, width: int):
    """在每只眼的框上方画"眼状态 + 注视类别"中文标签。

    调用关系：被 draw_results() 调用；内部调用 flip_box()（框是未镜像坐标）与 _pil_layer()。
    """
    items = []
    for box, (eye_label, eye_conf, gaze_label, gaze_conf) in zip(result.eye_boxes, result.per_eye):
        x1, y1, _, _ = flip_box(box, width)
        eye_text = f"{EYE_ZH.get(eye_label, eye_label or '--')} {eye_conf:.2f}"
        gaze_text = f"{GAZE_ZH.get(gaze_label, gaze_label or '--')} {gaze_conf:.2f}"
        items.append((x1, max(4, y1 - 52), eye_text, 18, (0, 255, 255)))
        items.append((x1, max(4, y1 - 28), gaze_text, 18, (120, 255, 120)))
    return _pil_layer(frame_display, items)


def draw_status_bar(frame_display, result, fps: float):
    """左上角黑底状态栏：眼睛信号 / 注视类别 / FPS（中文，PIL 渲染）。"""
    cv2.rectangle(frame_display, (0, 0), (430, 100), (0, 0, 0), -1)
    open_text = "--" if result.open_conf is None else f"{result.open_conf:.2f}"
    gaze_text = GAZE_ZH.get(result.gaze_label, result.gaze_label) if result.gaze_label else "--"
    items = [
        (14, 10, f"睁眼置信度：{open_text}", 20, (0, 255, 0)),
        (14, 38, f"注视类别：{gaze_text}", 24, (255, 255, 0)),
        (14, 72, f"FPS：{fps:.0f}｜推理用未镜像帧，显示为镜像视角", 15, (255, 255, 255)),
    ]
    return _pil_layer(frame_display, items)


def save_snapshot(frame_display: np.ndarray, now: float) -> Path:
    """把当前显示帧存进 data/yolo_snapshots/，返回保存路径。

    调用关系：被 main() 的 S 键分支调用；内部调用 stamp()。
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"yolo_{stamp(now)}.jpg"
    cv2.imwrite(str(path), frame_display)
    return path


# ============================ 2) 摄像头与后端 ============================


def open_camera() -> cv2.VideoCapture:
    """打开摄像头并设置画面宽高；打不开时报一句中文提示。

    调用关系：被 main() 在启动时调用一次，返回的 cap 会一路传给 read_and_detect()。
    """
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("摄像头无法打开，请关闭占用摄像头的软件后重试。")
    return cap


def make_backend() -> YoloEyeGaze:
    """建 YOLO 眼动后端：加载睁闭眼 + 注视两个权重（默认从 models/ 读）。

    调用关系：被 main() 在启动时调用一次（读模型慢，绝不能放进循环）。
    """
    return YoloEyeGaze()


# ============================ 3) 主流程 ============================


def read_and_detect(cap, backend: YoloEyeGaze):
    """读一帧 → 【未镜像】送 YOLO → 翻转出显示帧，返回（显示帧, now, 结果）。

    调用关系：被 main() 每帧调用（循环第一步）；内部调用 cap.read()、
    backend.analyze()（真正推理的那次）和 cv2.flip。
    顺序是铁律：先 analyze 后 flip。
    """
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败。")
    now = time.perf_counter()
    result = backend.analyze(frame)          # ← 推理只吃未镜像帧
    display = cv2.flip(frame, 1)             # ← 镜像只为显示（自拍视角）
    return display, now, result


def draw_results(frame_display, result, fps: float):
    """把一帧的检测全部画好：眼睛框 + 中文标签 + 状态栏。

    调用关系：被 main() 每帧调用；内部调用 draw_eye_boxes()、draw_labels()、draw_status_bar()。
    """
    if result.face_found:
        draw_eye_boxes(frame_display, result, WIDTH)
        frame_display = draw_labels(frame_display, result, WIDTH)
    return draw_status_bar(frame_display, result, fps)


def window_is_alive() -> bool:
    """点窗口右上角 ✕ 不会让 waitKey 返回任何值，只能自己问窗口还活着没。

    调用关系：被 main() 每帧调用；返回 False 时 main() 就 break 退出循环。
    """
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def pressed_key(now: float, key_available_at: float):
    """读一次按键并去抖，返回（按键小写字符 或 None, 新的可响应时间）。

    调用关系：被 main() 每帧调用；内部调用 cv2.waitKey(1)。
    main() 拿返回值决定：'q' 退出、's' 存截图。
    """
    raw_key = cv2.waitKey(1)
    if raw_key == -1:
        return None, key_available_at
    if now < key_available_at:  # 按住不放时系统会连发按键，靠最小间隔挡掉
        return None, key_available_at
    return chr(raw_key & 0xFF).lower(), now + KEY_DEBOUNCE_SECONDS


def main() -> None:
    """程序入口：建好摄像头和 YOLO 后端，每帧走一遍 看→画→控。"""
    cap = open_camera()                    # ① 打开摄像头
    backend = make_backend()               # ② 建 YOLO 后端
    key_available_at = 0.0                 # 按键防抖
    chinese_font(20)                       # ③ 提前加载字体，字体缺失时立刻报错
    frame_count = 0
    started = time.perf_counter()

    print("YOLO 眼动预览已启动：请先点一下视频窗口让它获得焦点，然后 Q 退出、S 存截图。")
    try:
        while True:
            # ---- ① 看：取一帧，未镜像送 YOLO；再翻出显示帧 ----
            frame, now, result = read_and_detect(cap, backend)

            # ---- ② 画：眼睛框 + 中文标签 + 状态栏 ----
            frame_count += 1
            fps = frame_count / max(now - started, 0.001)
            frame = draw_results(frame, result, fps)
            cv2.imshow(WINDOW_NAME, frame)

            # ---- ③ 控：先看窗口还活着没，再读键盘 ----
            if not window_is_alive():
                print("窗口已关闭，退出。")
                break
            key, key_available_at = pressed_key(now, key_available_at)
            if key == "q":
                break
            if key == "s":
                path = save_snapshot(frame, now)
                print(f"截图已保存：{path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
