"""第 3 步：眨眼校准与计数，仅用于视觉交互验证。

启动后保持睁眼看向摄像头约 3 秒。完成校准后，每完成一次自然眨眼，
屏幕上的眨眼次数会加一。此程序不会输出任何轮椅或 Arduino 指令。
按 Q 退出，按 C 重新校准。

文件分三段：
  1) 眨眼判定（纯逻辑，只吃数字，不碰摄像头）—— 可以单独跑测试
  2) 摄像头与画面（打开设备、建检测器、画中文字）
  3) 主流程 main()：读帧 → 量开合值 → 交给判定 → 画 → 按键

============================ 调用关系总览 ============================

  __main__  →  main()                        ← 唯一入口；main 自己不返回任何东西
  │
  ├─ 启动阶段（每个都只执行一次）
  │   ├─ open_camera()                  打开摄像头，返回 cap（后面每帧从它 read）
  │   ├─ face_detector()                建识别器：加载 models/face_landmarker.task
  │   │    └─ MediaPipe: FaceLandmarker.create_from_options()
  │   ├─ BlinkDetector(program_started_at)   建眨眼状态机，校准从此刻开始计时
  │   │    └─ self.reset(now)           把所有状态置位（按 C 重新校准时走的也是它）
  │   └─ chinese_font(20)               预加载中文字体，缺字体时立刻报错
  │
  ├─ 每帧循环（下面的 ★ 每帧都执行；五步顺序 看→量→判→报→控）
  │   ├─★ read_frame(cap, detector, program_started_at)
  │   │    ├─ cap.read() / cv2.flip / cv2.cvtColor     取一帧并做预处理
  │   │    ├─ detector.detect_for_video(图, 毫秒时间戳)  ← MediaPipe 推理
  │   │    └─ 返回 (frame, now, landmarks 或 None)
  │   ├─★ draw_eye_points(frame, landmarks)            画出 12 个眼球点（只为肉眼检查）
  │   ├─★ average_ear(landmarks)                       把 478 个点换算成一个数 ear
  │   │    ├─ eye_aspect_ratio(lm, LEFT_EYE)  → distance() 量 4 段距离
  │   │    └─ eye_aspect_ratio(lm, RIGHT_EYE) → distance()
  │   ├─★ blink.update(now, ear)                       核心判定（纯逻辑，可单测）
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

  摄像头画面 ──read_frame──▶ landmarks（478 个点）
  landmarks ──average_ear──▶ ear（一个浮点数，眼睛张开的程度）
  ear + 当前时间 ──blink.update──▶ BlinkStatus（计数值 / 状态文字 / 基线 / 提示语）
  BlinkStatus ──show_status──▶ 屏幕左上角的四行中文
"""
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


#默认参数设定：
#从第几个摄像头开
#画面高、宽（先写死成固定值）
#校准时长用前三秒去收集信息构建你的眼睛正常是什么样的，睁眼时多大闭眼时多大
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0     # 校准时长：这段时间内请保持睁眼
CALIBRATION_MIN_SAMPLES = 10  # 样本下限：帧率过低时样本太少，基线不可靠
CLOSE_RATIO = 0.78          # 放宽：低于睁眼基线的 78% 即进入闭眼候选
REOPEN_RATIO = 0.88         # 高于睁眼基线的 88% 即认为重新睁眼
SMOOTHING_FRAMES = 3        # 轻度平滑：保留自然眨眼的快速变化
MIN_CLOSED_SECONDS = 0.04   # 过滤极短噪声
MAX_CLOSED_SECONDS = 0.80   # 允许较慢/较完整的自然眨眼
MIN_OPEN_SECONDS = 0.10     # 再次计数前只需短暂稳定睁眼
REFRACTORY_SECONDS = 0.30   # 防止一次眨眼多计，但不妨碍连续眨眼
KEY_DEBOUNCE_SECONDS = 0.25 # 两次按键响应的最小间隔：按住不放时不再连发

MESSAGE_INITIAL = "请睁眼正视摄像头，正在校准"
MESSAGE_RECALIBRATE = "重新校准：请睁眼正视摄像头"


#path写法（里面全是路径）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
WINDOW_NAME = "Blink Detection | Q quit | C recalibrate | visual test only"
# 中文字体逐个探测：msyh 缺失时回退，换机器不会直接崩
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)

# 每只眼睛：外眼角、上外、上内、内眼角、下内、下外。
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
#MediaPipe不会管你具体要什么，他只会输出摄像头信息里面的478个点
#然后这几个点就是每个眼睛的外眼角、上外、上内、内眼角、下内、下外位置


# ============================ 1) 眨眼判定（纯逻辑） ============================


def distance(a, b) -> float:
    """两个关键点之间的平面距离（忽略 z）。只被下面的 eye_aspect_ratio() 调用。"""
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(landmarks, indices) -> float:
    """单只眼睛的开合比：上下眼睑平均距离 ÷ 眼角宽度。闭眼时趋近 0。

    调用关系：被 average_ear() 调用（左眼、右眼各一次）；
    内部调用 distance() 量 4 段距离（两段上下眼睑、一段眼角宽度）。
    """
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in indices]
    horizontal = distance(p0, p3)
    if horizontal < 1e-6:
        return 0.0
    return (distance(p1, p5) + distance(p2, p4)) / (2.0 * horizontal)


def average_ear(landmarks) -> float:
    """双眼平均开合比：取平均可以抵消轻微侧头带来的差异。

    调用关系：被 main() 每帧调用一次；内部调用 eye_aspect_ratio() 两次。
    产出的这个数（ear）就是后面所有判定唯一的输入。
    """
    left = eye_aspect_ratio(landmarks, LEFT_EYE)
    right = eye_aspect_ratio(landmarks, RIGHT_EYE)
    return (left + right) / 2.0


@dataclass(frozen=True)
class BlinkStatus:
    """一帧的判定结果快照，供界面显示。

    调用关系：由 BlinkDetector._status() 产出 → 被 main() 接住 → 交给 show_status() 画出来。
    它只是"数据袋子"，自身不含任何逻辑。
    """

    blink_count: int
    state: str
    message: str
    ear: float | None
    baseline: float | None
    close_threshold: float | None
    is_calibrating: bool


class BlinkDetector:
    """眨眼校准 + 计数状态机。原来散在 main() 里的状态，现在全在这个对象里。

    调用关系：main() 在启动时 new 一个；之后每帧调 update()；
    按 C 时调 reset() 重新校准。update() 内部按阶段分派：
      baseline 还是 None  → _update_calibration()
      baseline 已有值      → _update_counting()
      两种情况都由 _status(ear) 打包成 BlinkStatus 返回。

    用法：`BlinkDetector(time.perf_counter())`，之后每帧调用
    `update(now, 原始开合值)`；没检测到人脸时传 None。平滑、校准、阈值比较
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
        self._history: deque[float] = deque(maxlen=smoothing_frames)  # 最近几帧原始开合值，用于平滑
        self.reset(now)

    def reset(self, now: float, message: str = MESSAGE_INITIAL) -> None:
        """这里在进行初始化（还没有开始收集用户是否睁眼）。

        调用关系：被 __init__（建对象时）和 main() 的 C 键分支调用，
        两条路走的是同一份代码，避免两处手抄不一致。
        """
        self._calibration_started_at = now  # 校准阶段的起始时间
        self._calibration_values: list[float] = []  # 校准期间每帧的开合值
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
        """吃一帧数据，返回本帧状态。ear_raw 为 None 表示这帧没检测到人脸。

        调用关系：被 main() 每帧调用一次；内部先做 3 帧平滑，再按阶段分派——
        校准没完成 → _update_calibration()；已完成 → _update_counting()；
        两个分支的最后都经 _status() 打包成 BlinkStatus 返回。
        """
        if ear_raw is None:
            # 人脸消失期间的时序不可信，未完成的状态全部清空，人脸回来重新开始。
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
        """收集数据完成后的计数状态机。

        调用关系：只被 update() 调用，且只在 baseline 已有值时进入。
        眨眼计数 +1 就发生在这里（下面那段 min ≤ duration ≤ max 的判断里）。
        """
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
    #开摄像头(start)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    #设置画面高度和宽度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("摄像头无法打开，请关闭占用摄像头的软件后重试。")
    #开摄像头(stop)
    return cap


_detector_kept_alive: list = []  # 见 face_detector() 里的说明


#用模型去检测脸部特征返回一个对象（暂时理解成为一个结构体）
def face_detector():
    """建人脸识别器：加载 models/face_landmarker.task，返回识别器对象。

    调用关系：被 main() 在启动时调用一次（读模型很慢，绝不能放进循环）；
    内部调用 MediaPipe 的 FaceLandmarker.create_from_options()。
    返回值一路传给 read_frame()，由它每帧调用 detect_for_video()。
    """
    if not FACE_MODEL.exists():
        raise FileNotFoundError(f"缺少模型文件：{FACE_MODEL}")
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    # 保活：实测 MediaPipe 1.0.1 在 Windows 上回收这个对象要卡约 42 秒
    # （拆推理图时的内部等待）。留一个长期引用不让它被回收，退出时交给系统回收，
    # 整个进程 1 秒内就能结束。
    _detector_kept_alive.append(detector)
    return detector


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


def read_frame(cap, detector, program_started_at: float):
    """读一帧 → 镜像 → 送识别器，返回（画面, 当前时间, 关键点或 None）。

    调用关系：被 main() 每帧调用（循环的第一步）；
    内部调用 cap.read()、cv2.flip / cv2.cvtColor / mp.Image，
    以及 detector.detect_for_video()（真正干活的那一次推理）。
    三个返回值分别给：画图用（frame）、判定用（now）、算开合值用（landmarks）。
    """
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败。")
    frame = cv2.flip(frame, 1)
    now = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, int((now - program_started_at) * 1000))

    #detect_for_video返回对象包含三个东西：
    #  result.face_landmarks                 ← ★ 唯一被用到的：478 个点的坐标
    #  result.face_blendshapes               ← 表情系数，没开启 → 空列表
    #  result.facial_transformation_matrixes ← 人脸姿态矩阵，没开启 → 空列表
    landmarks = result.face_landmarks[0] if result.face_landmarks else None
    return frame, now, landmarks


def draw_eye_points(frame, landmarks) -> None:
    """在画面上画黄点，方便肉眼检查关键点准不准。

    调用关系：被 main() 每帧调用（只在检测到人脸时）；内部什么都不调用，
    只往 frame 上画 12 个圆点——它不参与任何判定。
    """
    h, w = frame.shape[:2]
    for index in LEFT_EYE + RIGHT_EYE:
        lm = landmarks[index]
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1)


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
        f"眼睛开合值：{ear_text}｜睁眼基线：{base_text}｜闭眼线：{close_text}",
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
    """程序入口：先建好四样东西，然后每帧走一遍 看→量→判→报→控。"""
    # ==================== 启动阶段（下面每个调用只执行一次）====================
    cap = open_camera()  # ① 打开摄像头
    detector = face_detector()  # ② 建识别器：调用函数创建MediaPipe人脸关键点检测器对象，用于后续检测人脸
    program_started_at = time.perf_counter()  # 记录程序启动的高精度时间戳，用于给MediaPipe提供视频时间（帧数不稳定毕竟是数字计算）
    blink = BlinkDetector(program_started_at)  # ③ 建眨眼状态机；校准起点 = 程序启动时刻，眨眼全部状态都在里面
    key_available_at = 0.0  # 按键防抖：下次允许响应的时间点
    chinese_font(20)  # ④ 提前加载字体，字体缺失时立刻报错

    print("眨眼预览已启动：请先点一下视频窗口让它获得焦点，然后 Q 退出、C 重新校准。")
    try:
        # ================= 每帧循环：看 → 量 → 判 → 报 → 控 =================
        while True:
            # ---- ① 看：读一帧并交给 MediaPipe，拿回 478 个关键点 ----
            frame, now, landmarks = read_frame(cap, detector, program_started_at)
            if landmarks is not None:
                draw_eye_points(frame, landmarks)  # 顺便把眼睛上的点画出来（只为肉眼检查）

            # ---- ② 量：把关键点换算成"眼睛张多开"这一个数 ----
            #ear是用于判定眼睛的状态标志；没检测到人脸时传 None
            ear_raw = average_ear(landmarks) if landmarks is not None else None

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
        # 不调用 detector.close()：实测它要卡约 42 秒（MediaPipe 1.0.1 拆推理图的等待）。
        # 配合 face_detector() 里的保活引用，进程 1 秒内干净退出，内存由系统回收。
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
