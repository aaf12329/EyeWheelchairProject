"""第 3 步：眨眼检测与计数（YOLO 版），仅用于视觉交互验证。

启动后保持睁眼看向摄像头约 3 秒。完成校准后，每完成一次自然眨眼，
屏幕上的眨眼次数会加一。此程序不会输出任何轮椅或 Arduino 指令。
按 Q 退出，按 C 重新校准。

本版把 MediaPipe 关键点几何判定整体换成了 YOLO26 眼动模型
（models/eye_yolo26n.pt，睁/闭眼 2 类，由 Yolo_model 项目训练交付）：
  旧链路：MediaPipe 478 点 → EAR 开合比 → 阈值判定
  新链路：Haar 定位眼睛 → 裁剪 → YOLO 判睁/闭 → 睁眼置信度当"开合值" → 原状态机
BlinkDetector 状态机（校准/计数/不应期）一行没改，只是喂进来的信号换了来源；
置信度是连续值，旧的三帧平滑与两条判定线照常工作。校准保留：它让阈值
适配当前摄像头与光照（模型置信度会随场景漂移）。

⚠️ 铁律：YOLO 推理只吃【未镜像帧】（镜像会把左右反转），镜像只用于显示层——
   所以 read_frame() 先 analyze 原始帧，再翻转到显示。详见 vision/yolo_backend.py。

文件分三段：
  1) 眨眼判定（纯逻辑，只吃数字，不碰摄像头）—— 可以单独跑测试（与旧版完全一致）
  2) 摄像头与画面（打开设备、建 YOLO 后端、画中文字）
  3) 主流程 main()：读帧 → YOLO 推理 → 交给判定 → 画 → 按键

============================ 调用关系总览 ============================

  __main__  →  main()                        ← 唯一入口；main 自己不返回任何东西
  │
  ├─ 启动阶段（每个都只执行一次）
  │   ├─ open_camera()                  打开摄像头，返回 cap（后面每帧从它 read）
  │   ├─ make_backend()                 建 YOLO 后端：加载 eye_yolo26n + gaze5 权重
  │   │    └─ vision.yolo_backend.YoloEyeGaze()   （Haar 定位 + 双模型推理都在里面）
  │   ├─ BlinkDetector(program_started_at)   建眨眼状态机，校准从此刻开始计时
  │   │    └─ self.reset(now)           把所有状态置位（按 C 重新校准时走的也是它）
  │   └─ chinese_font(20)               预加载中文字体，缺字体时立刻报错
  │
  ├─ 每帧循环（下面的 ★ 每帧都执行；五步顺序 看→量→判→报→控）
  │   ├─★ read_frame(cap, backend)
  │   │    ├─ cap.read()                             取【原始帧】（不镜像！）
  │   │    ├─ backend.analyze(frame)                 ← YOLO 推理（未镜像帧，铁律）
  │   │    ├─ cv2.flip(frame, 1)                     只为显示做镜像
  │   │    └─ 返回 (显示帧, now, EyeGazeResult)
  │   ├─★ draw_eye_boxes(显示帧, result, 宽)          画眼睛框（未镜像坐标经 flip_box 翻转）
  │   ├─★ blink.update(now, result.open_conf)        ★核心判定（纯逻辑，与旧版一致）
  │   │    ├─ _update_calibration()   校准未完成时走这里；3 秒后写入 baseline
  │   │    ├─ _update_counting()      校准完成后走这里；眨眼计数在这里 +1
  │   │    └─ _status(ear) → BlinkStatus   把本帧结果打包返回
  │   ├─★ show_status(frame, status)                   把 status 显示到窗口
  │   │    └─ draw_chinese_status(frame, 四行文字) → chinese_font(29 / 20)
  │   ├─★ window_is_alive()                            问窗口还活着没（点 ✕ 后为假）
  │   └─★ pressed_key(now, key_available_at)           读按键：q 退出；c → blink.reset()
  │
  └─ finally（无论怎么退出都执行）
      ├─ cap.release()                   交还摄像头
      └─ cv2.destroyAllWindows()         关窗口

数据在这些函数之间是怎么流动的：

  摄像头原始帧 ──backend.analyze──▶ open_conf（0~1，闭眼走低的连续信号）
  open_conf + 当前时间 ──blink.update──▶ BlinkStatus（计数值 / 状态文字 / 基线 / 提示语）
  BlinkStatus ──show_status──▶ 屏幕左上角的四行中文

（旧版 MediaPipe 的 42 秒退出卡顿随关键点检测一起移除了：torch 模型进程可正常回收。）
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 本项目各脚本独立运行（没有包结构），把 src/ 加进搜索路径以引入 YOLO 后端
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.yolo_backend import YoloEyeGaze, draw_eye_boxes  # noqa: E402  ← 在 sys.path 之后导入


#默认参数设定：
#从第几个摄像头开
#画面高、宽（先写死成固定值）
#校准时长用前三秒去收集信息构建你的眼睛正常是什么样的，睁眼时置信度多高闭眼时多低
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0     # 校准时长：这段时间内请保持睁眼
CALIBRATION_MIN_SAMPLES = 10  # 样本下限：帧率过低时样本太少，基线不可靠
CLOSE_RATIO = 0.78          # 低于睁眼基线的 78% 即进入闭眼候选
REOPEN_RATIO = 0.88         # 高于睁眼基线的 88% 即认为重新睁眼
SMOOTHING_FRAMES = 3        # 轻度平滑：保留自然眨眼的快速变化
MIN_CLOSED_SECONDS = 0.04   # 过滤极短噪声
MAX_CLOSED_SECONDS = 0.80   # 允许较慢/较完整的自然眨眼
MIN_OPEN_SECONDS = 0.10     # 再次计数前只需短暂稳定睁眼
REFRACTORY_SECONDS = 0.30   # 防止一次眨眼多计，但不妨碍连续眨眼
KEY_DEBOUNCE_SECONDS = 0.25 # 两次按键响应的最小间隔：按住不放时不再连发

MESSAGE_INITIAL = "请睁眼正视摄像头，正在校准"
MESSAGE_RECALIBRATE = "重新校准：请睁眼正视摄像头"

WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)

#path写法（里面全是路径）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOW_NAME = "Blink Detection (YOLO) | Q quit | C recalibrate | visual test only"
# 中文字体逐个探测：msyh 缺失时回退，换机器不会直接崩
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)


# ============================ 1) 眨眼判定（纯逻辑） ============================
# ★ 本段与 MediaPipe 版逐字一致（仅注释换口径）：状态机吃的是连续"睁眼信号"，
#   信号来源从 EAR 开合比换成 YOLO 睁眼置信度，判定逻辑一行没动。


@dataclass(frozen=True)
class BlinkStatus:
    """一帧的判定结果快照，供界面显示。

    调用关系：由 BlinkDetector._status() 产出 → 被 main() 接住 → 交给 show_status() 画出来。
    它只是"数据袋子"，自身不含任何逻辑。
    字段说明：ear 字段名沿用旧版（测试与界面都引用它），YOLO 版里它装的是
    "睁眼置信度"（0~1，闭眼走低）——对状态机来说两者是同一种连续信号。
    """

    blink_count: int
    state: str
    message: str
    ear: float | None
    baseline: float | None
    close_threshold: float | None
    is_calibrating: bool


class BlinkDetector:
    """眨眼校准 + 计数状态机。与 MediaPipe 版逐字一致。

    调用关系：main() 在启动时 new 一个；之后每帧调 update()；
    按 C 时调 reset() 重新校准。update() 内部按阶段分派：
      baseline 还是 None  → _update_calibration()
      baseline 已有值      → _update_counting()
      两种情况都由 _status(ear) 打包成 BlinkStatus 返回。

    用法：`BlinkDetector(time.perf_counter())`，之后每帧调用
    `update(now, 睁眼置信度)`；没检测到眼睛时传 None。平滑、校准、阈值比较
    都在内部完成，外面只读返回的 BlinkStatus。
    """

    def __init__(
        self,
        now: float,
        *,
        calibration_seconds: float = CALIBRATION_SECONDS,
        calibration_min_samples: int = CALIBRATION_MIN_SAMPLES,
        close_ratio: float = CLOSE_RATIO,
        reopen_ratio: float = REOPEN_RATIO,
        smoothing_frames: int = SMOOTHING_FRAMES,
        min_closed_seconds: float = MIN_CLOSED_SECONDS,
        max_closed_seconds: float = MAX_CLOSED_SECONDS,
        min_open_seconds: float = MIN_OPEN_SECONDS,
        refractory_seconds: float = REFRACTORY_SECONDS,
    ) -> None:
        self._calibration_seconds = calibration_seconds
        self._calibration_min_samples = calibration_min_samples
        self._close_ratio = close_ratio
        self._reopen_ratio = reopen_ratio
        self._min_closed_seconds = min_closed_seconds
        self._max_closed_seconds = max_closed_seconds
        self._min_open_seconds = min_open_seconds
        self._refractory_seconds = refractory_seconds
        self._history: deque[float] = deque(maxlen=smoothing_frames)  # 最近几帧原始信号，用于平滑
        self.reset(now)

    def reset(self, now: float, message: str = MESSAGE_INITIAL) -> None:
        """初始化/重新校准：把所有状态置位。

        调用关系：被 __init__（建对象时）和 main() 的 C 键分支调用，
        两条路走的是同一份代码，避免两处手抄不一致。
        """
        self._calibration_started_at = now  # 校准阶段的起始时间
        self._calibration_values: list[float] = []  # 校准期间每帧的信号值
        self._baseline: float | None = None  # 睁眼基线，校准完成前是 None
        self._close_threshold: float | None = None  # 闭眼判定线
        self._reopen_threshold: float | None = None  # 重新睁眼判定线
        self._blink_count = 0  # 眨眼计数
        self._is_closed = False  # 是否处于“闭眼候选”状态
        self._closed_since: float | None = None  # 闭眼开始的时间戳
        self._open_since: float | None = None  # 稳定睁眼开始的时间戳
        self._last_blink_at = float("-inf")  # 上次计数时间；负无穷让第一次眨眼不受不应期限制
        self._history.clear()
        self._state = "正在校准"
        self._message = message

    def update(self, now: float, ear_raw: float | None) -> BlinkStatus:
        """吃一帧数据，返回本帧状态。ear_raw 为 None 表示这帧没检测到眼睛。

        调用关系：被 main() 每帧调用一次；内部先做 3 帧平滑，再按阶段分派——
        校准没完成 → _update_calibration()；已完成 → _update_counting()；
        两个分支的最后都经 _status() 打包成 BlinkStatus 返回。
        """
        if ear_raw is None:
            # 眼睛消失期间的时序不可信，未完成的状态全部清空，回来重新开始。
            self._history.clear()
            self._is_closed = False
            self._closed_since = None
            self._open_since = None
            self._state = "未检测到人脸"
            self._message = "未检测到人脸：请正对摄像头"
            return self._status(ear=None)

        self._history.append(ear_raw)
        ear = sum(self._history) / len(self._history)

        if self._baseline is None:
            self._update_calibration(now, ear)
        else:
            self._update_counting(now, ear)
        return self._status(ear=ear)

    def _update_calibration(self, now: float, ear: float) -> None:
        """校准阶段：攒够时间和样本后，算出睁眼基线与两条判定线。

        调用关系：只被 update() 调用，且只在 baseline 还是 None 时进入。
        写完之后 baseline 有了值，下一帧起 update() 就会改走 _update_counting()。
        YOLO 版语义：基线是"当前场景下睁眼置信度"，它随摄像头/光照漂移，所以保留校准。
        """
        self._calibration_values.append(ear)
        elapsed = now - self._calibration_started_at
        remaining = max(0.0, self._calibration_seconds - elapsed)
        self._message = f"校准中：请保持睁眼 {remaining:.1f}s"
        if elapsed < self._calibration_seconds or len(self._calibration_values) < self._calibration_min_samples:
            return

        # 用中间 80% 的平均值，降低偶发眨眼影响。
        values = sorted(self._calibration_values)
        trim = max(1, len(values) // 10)
        kept = values[trim:-trim] or values
        self._baseline = sum(kept) / len(kept)
        self._close_threshold = self._baseline * self._close_ratio
        self._reopen_threshold = self._baseline * self._reopen_ratio
        self._message = "校准完成：现在请自然眨眼"
        self._state = "已准备好"

    def _update_counting(self, now: float, ear: float) -> None:
        """收集数据完成后的计数状态机（与 MediaPipe 版逐字一致）。"""
        # 必须先记录“闭眼前是否已稳定睁眼”，再处理当前闭眼帧。
        # 否则一闭眼就会先清空 open_since，导致永远无法进入计数。
        was_stably_open = (
            self._open_since is not None
            and now - self._open_since >= self._min_open_seconds
            and now - self._last_blink_at >= self._refractory_seconds
        )

        if self._is_closed:
            if ear >= self._reopen_threshold:
                duration = now - self._closed_since if self._closed_since is not None else 0.0
                self._is_closed = False
                self._closed_since = None
                self._open_since = now
                if self._min_closed_seconds <= duration <= self._max_closed_seconds:
                    self._blink_count += 1
                    self._last_blink_at = now
                    self._message = f"检测到一次眨眼（闭眼 {duration:.2f}s）"
                    self._state = "已确认"
                else:
                    self._message = f"忽略非自然眨眼（{duration:.2f}s）"
                    self._state = "已准备好"
            else:
                self._state = "闭眼候选中"
        elif ear >= self._reopen_threshold:
            if self._open_since is None:
                self._open_since = now
            self._state = "已准备好" if was_stably_open else "等待稳定睁眼"
        elif ear < self._close_threshold and was_stably_open:
            self._is_closed = True
            self._closed_since = now
            self._open_since = None
            self._state = "闭眼候选中"
        else:
            # 落在两条判定线之间的灰色地带：保持现状，睁眼计时作废。
            self._open_since = None
            self._state = "等待稳定睁眼"

    def _status(self, ear: float | None) -> BlinkStatus:
        """把当前内部状态打包成 BlinkStatus（界面唯一能读到的东西）。

        调用关系：被 update() 调用；产物交给 main() → show_status() 显示。
        """
        return BlinkStatus(
            blink_count=self._blink_count,
            state=self._state,
            message=self._message,
            ear=ear,
            baseline=self._baseline,
            close_threshold=self._close_threshold,
            is_calibrating=self._baseline is None,
        )


# ============================ 2) 摄像头与画面 ============================


def open_camera() -> cv2.VideoCapture:
    """打开摄像头并设置画面宽高；打不开时报一句中文提示。

    调用关系：被 main() 在启动时调用一次，返回的 cap 会一路传给 read_frame()。
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

    调用关系：被 main() 在启动时调用一次（读模型慢，绝不能放进循环）；
    返回值一路传给 read_frame()，由它每帧调 analyze()。
    本脚本只用它的 open_conf；gaze 部分在第 4/5 步脚本里使用。
    """
    return YoloEyeGaze()


_font_cache: dict[tuple[Path, int], "ImageFont.FreeTypeFont"] = {}


def chinese_font(size: int):
    """按 FONT_CANDIDATES 找到第一个可用字体并缓存，避免每帧重复读字体文件。

    调用关系：被 draw_chinese_status() 调用（每帧两次：29 号大字、20 号小字）；
    第一次真正读文件，之后直接命中缓存。
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


def draw_chinese_status(frame, lines) -> object:
    """用 Windows 中文字体绘制状态栏，避免 OpenCV 英文字体无法显示中文。

    调用关系：被 show_status() 调用；内部调用 chinese_font() 取字体。
    注意它不修改传入的 frame，而是返回一张画好字的新图。
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font_large = chinese_font(29)
    font_small = chinese_font(20)
    draw.text((14, 12), lines[0], font=font_small, fill=(0, 255, 0))
    draw.text((14, 39), lines[1], font=font_large, fill=(255, 255, 0))
    draw.text((14, 76), lines[2], font=font_small, fill=(255, 255, 255))
    draw.text((14, 103), lines[3], font=font_small, fill=(255, 210, 80))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# ============================ 3) 主流程 ============================


def read_frame(cap, backend: YoloEyeGaze):
    """读一帧 → 【未镜像】送 YOLO → 翻转出显示帧，返回（显示帧, 当前时间, 推理结果）。

    调用关系：被 main() 每帧调用（循环的第一步）；
    内部调用 cap.read()、backend.analyze()（真正干活的那次推理）和 cv2.flip。
    顺序是铁律：先 analyze 后 flip —— 模型必须吃未镜像帧，镜像只属于显示层。
    """
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败。")
    now = time.perf_counter()
    result = backend.analyze(frame)          # ← 推理只吃未镜像帧
    display = cv2.flip(frame, 1)             # ← 镜像只为显示（自拍视角）
    return display, now, result


def show_status(frame, status) -> None:
    """铺黑色底板 + 四行中文状态 + 推到窗口显示。

    调用关系：被 main() 每帧调用（循环第 4 步）；内部调用 draw_chinese_status()。
    它只读 status（BlinkStatus），不改任何状态——所以调它不会影响判定。
    """
    cv2.rectangle(frame, (0, 0), (560, 125), (0, 0, 0), -1)
    ear_text = "--" if status.ear is None else f"{status.ear:.3f}"
    base_text = "--" if status.baseline is None else f"{status.baseline:.3f}"
    close_text = "--" if status.close_threshold is None else f"{status.close_threshold:.3f}"
    frame = draw_chinese_status(frame, [
        f"睁眼置信度：{ear_text}｜睁眼基线：{base_text}｜闭眼线：{close_text}",
        f"眨眼次数：{status.blink_count}｜状态：{status.state}",
        status.message,
        "按 Q 退出｜按 C 重新校准｜仅视觉测试，不控制任何硬件",
    ])
    cv2.imshow(WINDOW_NAME, frame)


def window_is_alive() -> bool:
    """点窗口右上角 ✕ 不会让 waitKey 返回任何值，只能自己问窗口还活着没。

    调用关系：被 main() 每帧调用（循环第 5 步的第一件事）；
    返回 False 时 main() 就 break 退出循环。
    """
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def pressed_key(now: float, key_available_at: float):
    """读一次按键并去抖，返回（按键小写字符 或 None, 新的可响应时间）。

    调用关系：被 main() 每帧调用；内部调用 cv2.waitKey(1)
    （这一次调用同时兼职：取按键 + 让窗口有机会刷新）。
    main() 拿返回值决定：'q' 退出、'c' 调 blink.reset()。
    """
    raw_key = cv2.waitKey(1)
    if raw_key == -1:
        return None, key_available_at
    if now < key_available_at:  # 按住不放时系统会连发按键，靠最小间隔挡掉
        return None, key_available_at
    return chr(raw_key & 0xFF).lower(), now + KEY_DEBOUNCE_SECONDS


def main() -> None:

    for i in WELCOME_ART:
        print(i)
    """程序入口：先建好四样东西，然后每帧走一遍 看→量→判→报→控。"""
    # ==================== 启动阶段（下面每个调用只执行一次）====================
    cap = open_camera()  # ① 打开摄像头
    backend = make_backend()  # ② 建 YOLO 后端（加载 eye_yolo26n + gaze5 权重）
    blink = BlinkDetector(time.perf_counter())  # ③ 建眨眼状态机；校准从此刻开始计时
    key_available_at = 0.0  # 按键防抖：下次允许响应的时间点
    chinese_font(20)  # ④ 提前加载字体，字体缺失时立刻报错

    print("眨眼预览已启动（YOLO 版）：请先点一下视频窗口让它获得焦点，然后 Q 退出、C 重新校准。")
    try:
        # ================= 每帧循环：看 → 量 → 判 → 报 → 控 =================
        while True:
            # ---- ① 看：取一帧，未镜像送 YOLO，拿回推理结果；再翻出显示帧 ----
            frame, now, result = read_frame(cap, backend)
            if result.face_found:
                draw_eye_boxes(frame, result, WIDTH)  # 眼睛框（只为肉眼检查，不参与判定）

            # ---- ② 量：睁眼置信度就是"开合值"（没检测到时是 None）----
            ear_raw = result.open_conf

            # ---- ③ 判：校准 / 计数，全部判定逻辑都在 BlinkDetector 里 ----
            status = blink.update(now, ear_raw)

            # ---- ④ 报：把 status 画到窗口上 ----
            show_status(frame, status)

            # ---- ⑤ 控：先看窗口还活着没，再读键盘 ----
            if not window_is_alive():  # 窗口被点 ✕ 关掉就退出
                print("窗口已关闭，退出。")
                break
            key, key_available_at = pressed_key(now, key_available_at)
            if key == "q":
                break
            if key == "c":
                blink.reset(now, MESSAGE_RECALIBRATE)  # 回到校准起点，重新采集一次基线
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
