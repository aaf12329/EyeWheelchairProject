"""第 4 步：单摄像头视线方向选择预览（左/中/右，YOLO 版）。

先看正前方完成校准，再把视线移到屏幕左侧、中间、右侧。
本程序只显示候选方向，不发送确认或任何硬件控制指令。

按 Q 退出；按 C 重新校准；按 I 反转左右方向（若左右显示与实际相反）。

本版把 MediaPipe 虹膜几何判定整体换成了 YOLO26 注视模型
（models/gaze5_yolo26s.pt，5 类：上/中/下/左/右，由 Yolo_model 项目训练交付）：
  旧链路：MediaPipe 478 点 → 虹膜水平位置 score → 阈值判定
  新链路：YuNet 定位眼睛 → 裁剪 → YOLO 判 5 类 → 左0/中0.5/右1 当"score" → 原状态机

**双引擎对比版（2026-10-02）**：同一帧同时跑两套引擎，逐帧数据写入 CSV 供对比参考：
  引擎① YOLO：5 类分类 → 左0/中0.5/右1 信号（如上）
  引擎② MediaPipe：478 点 → 虹膜水平位置（已做镜像修正，语义与①一致：左0/中0.5/右1）
两个 GazeDirectionDetector 状态机**各自独立校准、各自出候选**——同一段视线移动
两边各判各的，CSV 逐帧并列（yolo_score / mp_score / 各自方向与候选），跑完对比。
YOLO 单引擎行为完全保留：加 --no-mp 即回到纯 YOLO；
CSV 落在 data/compare_logs/gaze_compare_<时间戳>.csv（--no-log 可关）。

GazeDirectionDetector 状态机（校准/稳定停留/候选）逻辑没变，只是信号来源换了；
校准保留：它吸收分类器在当前摄像头下的偏置（比如你正视时模型总偏一点）。

⚠️ 铁律：YOLO 推理只吃【未镜像帧】（镜像会把看左/看右反转），镜像只用于显示层。
   详见 vision/yolo_backend.py。

文件分三段：
  1) 判定逻辑（纯逻辑 + 状态机类，只吃数字信号，不碰摄像头也不画图）—— 可以单独跑测试
  2) 摄像头与画面（打开设备、建 YOLO 后端、画中文和卡片）
  3) 主流程 main()：读帧 → 更新状态机 → 画 → 按键

============================ 调用关系总览 ============================
  │   （路径/摄像头/中文字体 来自共用层 src/common/，见 common/paths.py 地址簿）

  __main__  →  main()
  │
  ├─ 启动阶段（每个只执行一次）
  │   ├─ open_camera()  ←common层       打开摄像头，返回 cap（后面每帧从它 read）
  │   ├─ make_backend()                 建 YOLO 后端：加载 eye + gaze5 权重
  │   ├─ make_mp_backend()              建 MediaPipe 引擎（--no-mp 跳过；加载失败自动降级）
  │   ├─ GazeDirectionDetector × 2      两个状态机：一个吃 YOLO 信号、一个吃 MP 虹膜信号
  │   ├─ CsvLogger ←common层            双引擎逐帧数据 → data/compare_logs/（--no-log 跳过）
  │   └─ chinese_font(20) ←common层     预加载中文字体，缺字体时立刻报错
  │
  ├─ 每帧循环（★ 每帧都执行；顺序 看 → 判 → 报 → 控）
  │   ├─★ read_frame(cap, backend, mp_backend, started)
  │   │    ├─ cap.read()                             取【原始帧】（不镜像！）
  │   │    ├─ backend.analyze(frame)                 ← YOLO 推理（未镜像帧，铁律）
  │   │    ├─ mp_backend.analyze(frame, ts_ms)       ← MediaPipe 推理（同一帧）
  │   │    ├─ cv2.flip(frame, 1)                     只为显示做镜像
  │   │    └─ 返回 (显示帧, now, EyeGazeResult, MediaPipeResult 或 None)
  │   ├─★ draw_eye_boxes(显示帧, result, 宽)          画眼睛框 + 注视类别（只为肉眼检查）
  │   ├─★ flow.update(now, result.gaze_score)        ★YOLO 判定（纯逻辑，可单测）
  │   ├─★ flow_mp.update(now, mp.gaze_score)         ★MediaPipe 判定（同款状态机，独立校准）
  │   │    ├─ _update_calibration()           校准中走这里；3 秒后算出正视基准 center
  │   │    ├─ _update_selection()             校准后走这里；偏移量决定 左/中/右，稳定后出候选
  │   │    └─ _status(...) → GazeStatus       把本帧结果打包返回
  │   ├─★ show_status(frame, status, mp_status)  黑底 + YOLO 三行 + MP 一行 + 三张卡片
  │   │    ├─ draw_text(frame, ...)           中文 → chinese_font(20 / 28 / 18)
  │   │    └─ draw_cards(frame, 候选)          左 / 中 / 右 三张卡片（跟 YOLO 侧候选高亮）
  │   ├─★ logger.log(逐帧一行)                        双引擎数据写 CSV（--no-log 跳过）
  │   ├─★ window_is_alive()                   问窗口还活着没（点 ✕ 后为假）
  │   └─★ pressed_key(now, key_available_at)  读按键：q 退出 / c → 两个状态机一起 reset / i → 双双反转
  │
  └─ finally（无论怎么退出都执行）
      ├─ logger.close()                 收尾 CSV，打印路径与行数
      ├─ cap.release()                  交还摄像头
      └─ cv2.destroyAllWindows()        关窗口

这台状态机比 gaze_blink_confirm_demo.py 那台简单，只有两个阶段加两个方向变量：

    校准      ──(满 3 秒 且 样本≥10 个)──▶ 选择方向
    选择方向  ：每帧把"水平注视信号"分成 左 / 中 / 右 三种当前方向；
                同一方向稳定停留 0.70 秒 → 成为"稳定候选"（屏幕上高亮那张卡片）
    任意阶段  ──(信号不可用)──────────────▶ 候选作废、方向重新计时（校准不打断）
    任意阶段  ──(按 C)──────────────────▶ 校准（reset(now) 把一切拨回起点）

注意：这里没有"确认"这一步——它只挑候选方向，眨眼确认由第 5 步的程序负责。
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 本项目各脚本独立运行（没有包结构），把 src/ 加进搜索路径以引入 YOLO/MediaPipe 后端
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.yolo_backend import YoloEyeGaze, draw_eye_boxes  # noqa: E402  ← 在 sys.path 之后导入
from vision.mediapipe_backend import MediaPipeEyeGaze  # noqa: E402  ← 第二引擎（双引擎对比）
from common.camera_utils import open_camera  # noqa: E402  ← 共用层（src/common/）
from common.csv_logger import CsvLogger  # noqa: E402
from common.draw_utils import chinese_font, draw_text  # noqa: E402
from common.paths import COMPARE_LOG_DIR  # noqa: E402

# ---- 运行参数 ----
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0     # 校准时长：这段时间请看屏幕正中间
CALIBRATION_MIN_SAMPLES = 10  # 样本下限：帧率过低时样本太少，基准不可靠
SIDE_THRESHOLD = 0.075        # 相对于正视基准的水平偏移：超过它才算看左/看右
STABLE_SECONDS = 0.70         # 同方向稳定停留多久才成为候选
SMOOTHING_FRAMES = 5          # 注视信号平滑：最近几帧取平均
KEY_DEBOUNCE_SECONDS = 0.25   # 两次按键响应的最小间隔：按住不放时不再连发

MESSAGE_INITIAL = "请看屏幕正中间，正在校准"
MESSAGE_RECALIBRATE = "重新校准：请看屏幕正中间"

# ---- 路径与窗口 ----
WINDOW_NAME = "Gaze Direction (YOLO×MediaPipe) | Q quit | C recalibrate | I invert | visual test only"


# ============================ 1) 判定逻辑（纯逻辑） ============================


@dataclass(frozen=True)
class GazeStatus:
    """一帧的判定结果快照，供界面显示。

    调用关系：由 GazeDirectionDetector._status() 产出 → 被 main() 接住 → 交给 show_status()。
    它只是"数据袋子"，自身不含任何逻辑。
    """

    score: float | None          # 本帧的水平注视信号（平滑后；不可用时是 None）
    center: float | None         # 正视基准（校准完成前是 None）
    raw_direction: str           # 这一帧视线落在哪个方向：左 / 中 / 右
    candidate: str               # 稳定停留够久的方向；还没稳定时是 "无"
    message: str                 # 屏幕上第三行显示的提示语


class GazeDirectionDetector:
    """视线方向的校准 + 选择状态机。逻辑与 MediaPipe 版一致，输入换成信号。

    调用关系：main() 在启动时建一个；之后每帧调 update()；
    按 C 调 reset()，按 I 调 toggle_invert()。update() 内部按阶段分派：
      校准没完成（center 还是 None）→ _update_calibration()
      校准完成                      → _update_selection()
    两种情况都由 _status(...) 打包成 GazeStatus 返回。

    用法：`GazeDirectionDetector(time.perf_counter())`，之后每帧调用
    `update(now, score_raw)` —— score_raw 是 YOLO 后端的水平注视信号
    （左 0 / 中 0.5 / 右 1），不可用时传 None。
    它只依赖浮点数，不 import cv2，所以测试时直接喂数字就行（比旧版喂假关键点更简单）。
    """

    def __init__(
        self,
        now: float,
        *,
        calibration_seconds: float = CALIBRATION_SECONDS,
        calibration_min_samples: int = CALIBRATION_MIN_SAMPLES,
        side_threshold: float = SIDE_THRESHOLD,
        stable_seconds: float = STABLE_SECONDS,
        smoothing_frames: int = SMOOTHING_FRAMES,
    ) -> None:
        self._calibration_seconds = calibration_seconds
        self._calibration_min_samples = calibration_min_samples
        self._side_threshold = side_threshold
        self._stable_seconds = stable_seconds
        # 平滑用的滚动队列：最近几帧注视信号
        self._history: deque[float] = deque(maxlen=smoothing_frames)
        self._invert = False  # 按 I 切换：左右方向是否反转
        self.reset(now)

    def reset(self, now: float, message: str = MESSAGE_INITIAL) -> None:
        """回到校准起点。构造时和按 C 重新校准时都走这里，避免两处手抄不一致。"""
        self._calibration_started = now           # 校准阶段的起始时间
        self._calibration_values: list[float] = []  # 校准期间攒的信号值
        self._center: float | None = None         # 正视基准；它等于 None 就表示"还在校准"
        self._history.clear()
        # ---- 状态机的两个方向变量 ----
        #   _raw_direction : 这一帧视线落在哪个方向（会抖）
        #   _candidate     : 稳定停留够久的那个方向；还没稳定时是 "无"
        self._raw_direction = "中"
        self._candidate = "无"
        self._direction_since: float | None = None  # 当前方向从什么时候开始
        self._message = message

    def toggle_invert(self) -> None:
        """按 I 时调用：把左右方向反过来，并更新屏幕提示。"""
        self._invert = not self._invert
        self._message = "左右方向已反转" if self._invert else "左右方向已恢复"

    def update(self, now: float, score_raw: float | None) -> GazeStatus:
        """吃一帧数据，返回本帧状态。score_raw 传 None 表示这帧信号不可用。"""
        if score_raw is None:
            # 信号不可用：候选作废、方向重新计时；但"校准"不打断（否则一转头就白校准了）
            self._message = "未检测到人脸：请正对摄像头"
            self._candidate = "无"
            self._direction_since = None
            return self._status(score=None)

        self._history.append(score_raw)
        score = sum(self._history) / len(self._history)  # 最近几帧取平均，抹掉抖动

        if self._center is None:
            self._update_calibration(now, score)
        else:
            self._update_selection(now, score)
        return self._status(score=score)

    def _update_calibration(self, now: float, score: float) -> None:
        """校准阶段：攒样本，够 3 秒且样本足够后算出正视基准 _center。

        YOLO 版语义：基准吸收分类器在当前人/摄像头下的偏置 —— 你正视时模型
        未必输出严格的 0.5，校准把它量出来当"中"。
        """
        self._calibration_values.append(score)
        elapsed = now - self._calibration_started
        self._message = f"校准中：请看屏幕正中间 {max(0, self._calibration_seconds - elapsed):.1f} 秒"
        if elapsed < self._calibration_seconds or len(self._calibration_values) < self._calibration_min_samples:
            return
        # 掐掉头尾各 10%（偶发偏移的极值），用中间 80% 的平均值当正视基准
        values = sorted(self._calibration_values)
        trim = max(1, len(values) // 10)
        kept = values[trim:-trim] or values
        self._center = sum(kept) / len(kept)
        self._message = "校准完成：依次看屏幕左、中、右区域"

    def _update_selection(self, now: float, score: float) -> None:
        """校准完成后：把偏移量分成 左/中/右，稳定停留够久就记为候选。"""
        offset = score - self._center
        if self._invert:
            offset = -offset
        if offset < -self._side_threshold:
            new_direction = "左"
        elif offset > self._side_threshold:
            new_direction = "右"
        else:
            new_direction = "中"

        if new_direction != self._raw_direction:
            # 方向变了 → 重新开始计时，候选清空
            self._raw_direction = new_direction
            self._direction_since = now
            self._candidate = "无"
        elif self._direction_since is not None and now - self._direction_since >= self._stable_seconds:
            # 同一方向稳定停留够久 → 成为候选
            self._candidate = self._raw_direction
        self._message = "只是在选择候选方向；本程序不会移动轮椅"

    def _status(self, score: float | None) -> GazeStatus:
        """把当前内部状态打包成 GazeStatus（界面唯一能读到的东西）。"""
        return GazeStatus(
            score=score,
            center=self._center,
            raw_direction=self._raw_direction,
            candidate=self._candidate,
            message=self._message,
        )


# ============================ 2) 摄像头与画面 ============================




def make_backend() -> YoloEyeGaze:
    """建 YOLO 眼动后端：加载睁闭眼 + 注视两个权重（默认从 models/ 读）。

    调用关系：被 main() 在启动时调用一次（读模型慢，绝不能放进循环）；
    返回值一路传给 read_frame()，由它每帧调 analyze()。
    本脚本只用它的 gaze_score / gaze_label。
    """
    return YoloEyeGaze()


def make_mp_backend():
    """建 MediaPipe 引擎（第二引擎）；缺库/缺模型时打印原因并返回 None（自动降级）。

    调用关系：被 main() 在启动时调用一次；返回值传给 read_frame()。
    降级设计：只跑 YOLO 也能完成方向测试，所以这里不抛异常、只提示原因。
    """
    try:
        return MediaPipeEyeGaze()
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"⚠️ MediaPipe 引擎未启用：{exc}")
        print("   → 本次只跑 YOLO 引擎（显式只要 YOLO 可加 --no-mp 去掉本提示）。")
        return None








def draw_cards(frame, candidate: str):
    """画底部三张卡片：左 / 中 / 右；名称与 candidate 相同的那张用实心绿色高亮。

    调用关系：被 show_status() 调用；candidate 由状态机算好（放在 GazeStatus.candidate 里），
    还没稳定时是 "无"，此时三张卡片都不高亮。
    """
    cards = [("左", 35, 390, 280, 515), ("中", 340, 390, 585, 515), ("右", 645, 390, 890, 515)]
    for label, x1, y1, x2, y2 in cards:
        active = label == candidate
        color = (0, 170, 0) if active else (70, 70, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1 if active else 2)
        frame = draw_text(frame, label, (x1 + 98, y1 + 35), 48, (255, 255, 255))
    return frame


# ============================ 3) 主流程 ============================


def read_frame(cap, backend: YoloEyeGaze, mp_backend, started: float):
    """读一帧 → 【未镜像】依次送 YOLO 与 MediaPipe → 翻转出显示帧。

    调用关系：被 main() 每帧调用（循环第一步）；内部调用 cap.read()、
    backend.analyze()、mp_backend.analyze()（mp_backend 为 None 时跳过）和 cv2.flip。
    返回（显示帧, 当前时间, YOLO 结果, MediaPipe 结果 或 None）。
    顺序是铁律：先 analyze 后 flip —— 两个引擎都必须吃未镜像帧，镜像只属于显示层。
    """
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败。")
    now = time.perf_counter()
    result = backend.analyze(frame)                       # ← YOLO 吃未镜像帧
    mp_result = None
    if mp_backend is not None:
        ts_ms = int((now - started) * 1000)               # VIDEO 模式要毫秒时间戳
        mp_result = mp_backend.analyze(frame, ts_ms)      # ← MediaPipe 吃同一帧
    display = cv2.flip(frame, 1)                          # ← 镜像只为显示（自拍视角）
    return display, now, result, mp_result


def show_status(frame, status, mp_status=None) -> None:
    """铺黑色底板 + YOLO 三行 + 可选 MediaPipe 一行 + 底部三张卡片，推给窗口。

    调用关系：被 main() 每帧调用（循环第三步）；内部调用 draw_text() 和 draw_cards()。
    它只读 status，不改任何状态——所以调它不会影响判定。
    卡片跟 YOLO 侧的候选高亮；mp_status 为 None 时（--no-mp）只画 YOLO 部分。
    """
    height = 105 + (28 if mp_status is not None else 0)
    cv2.rectangle(frame, (0, 0), (700, height), (0, 0, 0), -1)
    score_text = "--" if status.score is None else f"{status.score:.3f}"
    base_text = "--" if status.center is None else f"{status.center:.3f}"
    frame = draw_text(frame, f"【YOLO】注视信号：{score_text}｜正视基准：{base_text}", (14, 10), 20, (0, 255, 0))
    frame = draw_text(frame, f"当前方向：{status.raw_direction}｜稳定候选：{status.candidate}", (14, 38), 28, (255, 255, 0))
    frame = draw_text(frame, status.message, (14, 74), 18, (255, 255, 255))
    if mp_status is not None:
        mp_score = "--" if mp_status.score is None else f"{mp_status.score:.3f}"
        frame = draw_text(
            frame,
            f"【MediaPipe】信号：{mp_score}｜方向：{mp_status.raw_direction}｜候选：{mp_status.candidate}",
            (14, 100), 18, (120, 200, 255))
    frame = draw_cards(frame, status.candidate)
    cv2.imshow(WINDOW_NAME, frame)


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

    调用关系：被 main() 每帧调用；内部调用 cv2.waitKey(1)
    （这一次调用同时兼职：取按键 + 让窗口有机会刷新）。
    """
    raw_key = cv2.waitKey(1)
    if raw_key == -1:
        return None, key_available_at
    if now < key_available_at:  # 按住不放时系统会连发按键，靠最小间隔挡掉
        return None, key_available_at
    return chr(raw_key & 0xFF).lower(), now + KEY_DEBOUNCE_SECONDS


def parse_args():
    """命令行参数：默认双引擎 + 写 CSV；--no-mp / --no-log 可分别关掉。"""
    import argparse

    ap = argparse.ArgumentParser(description="视线方向预览（YOLO × MediaPipe 双引擎对比）")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX, help="摄像头编号")
    ap.add_argument("--no-mp", action="store_true", help="只跑 YOLO 引擎（不加载 MediaPipe）")
    ap.add_argument("--no-log", action="store_true", help="不写双引擎对比 CSV")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    """程序入口：建好摄像头和两个引擎后，每帧走一遍 看→判→报→控。"""
    # ==================== 启动阶段（下面每个调用只执行一次）====================
    cap = open_camera(args.camera, WIDTH, HEIGHT)   # ① 打开摄像头
    backend = make_backend()                        # ② 建 YOLO 后端
    mp_backend = None if args.no_mp else make_mp_backend()  # ③ 建 MediaPipe 引擎（可降级）
    started = time.perf_counter()
    flow = GazeDirectionDetector(started)           # ④ 方向状态机（YOLO 信号）
    flow_mp = GazeDirectionDetector(started) if mp_backend else None  # ⑤（MP 虹膜信号）
    logger = None                                   # ⑥ 双引擎逐帧 CSV
    if not args.no_log:
        logger = CsvLogger(COMPARE_LOG_DIR, "gaze_compare", [
            "t_s", "frame", "yolo_score", "yolo_dir", "yolo_candidate", "yolo_center",
            "mp_score", "mp_dir", "mp_candidate", "mp_center",
        ])
    key_available_at = 0.0
    chinese_font(20)                                # 提前加载字体，字体缺失时立刻报错

    print("视线方向预览已启动（YOLO × MediaPipe 双引擎）：请先点一下视频窗口让它获得焦点，"
          "然后 Q 退出、C 重新校准、I 反转左右。")
    if logger:
        print(f"双引擎逐帧数据 → {logger.path}")
    frame_i = 0
    try:
        # ================= 每帧循环：看 → 判 → 报 → 控 =================
        while True:
            # ---- ① 看：取一帧，未镜像送两个引擎，拿回两份结果；再翻出显示帧 ----
            frame, now, result, mp_result = read_frame(cap, backend, mp_backend, started)
            if result.face_found:
                draw_eye_boxes(frame, result, WIDTH)  # 眼睛框（只为肉眼检查，不参与判定）

            # ---- ② 判：两个状态机各自独立校准、各自出候选 ----
            status = flow.update(now, result.gaze_score)              # YOLO：5 类信号
            mp_status = None
            if flow_mp is not None:
                mp_score = mp_result.gaze_score if mp_result else None
                mp_status = flow_mp.update(now, mp_score)             # MP：虹膜位置信号

            # ---- ③ 报：两栏画到窗口；逐帧数据写 CSV ----
            show_status(frame, status, mp_status)
            frame_i += 1
            if logger:
                logger.log([round(now - started, 3), frame_i,
                            result.gaze_score, status.raw_direction, status.candidate,
                            None if status.center is None else round(status.center, 4),
                            None if mp_result is None else mp_result.gaze_score,
                            mp_status.raw_direction if mp_status else None,
                            mp_status.candidate if mp_status else None,
                            None if (mp_status is None or mp_status.center is None)
                            else round(mp_status.center, 4)])

            # ---- ④ 控：先看窗口还活着没，再读键盘 ----
            if not window_is_alive():            # 窗口被点 ✕ 关掉就退出
                print("窗口已关闭，退出。")
                break
            key, key_available_at = pressed_key(now, key_available_at)
            if key == "q":
                break
            if key == "c":
                flow.reset(now, MESSAGE_RECALIBRATE)      # 两个状态机一起重校准
                if flow_mp is not None:
                    flow_mp.reset(now, MESSAGE_RECALIBRATE)
            if key == "i":
                flow.toggle_invert()                      # 左右方向反转 / 恢复（两边一起）
                if flow_mp is not None:
                    flow_mp.toggle_invert()
    finally:
        if logger:
            logger.close()
            print(f"对比数据已保存：{logger.path}（{logger.rows} 行）")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
