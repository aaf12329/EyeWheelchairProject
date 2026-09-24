"""第 4 步：单摄像头视线方向选择预览（左/中/右）。

先看正前方完成校准，再把视线移到屏幕左侧、中间、右侧。
本程序只显示候选方向，不发送确认或任何硬件控制指令。

按 Q 退出；按 C 重新校准；按 I 反转左右方向（若左右显示与实际相反）。

文件分三段：
  1) 判定逻辑（纯逻辑 + 状态机类，只吃关键点，不碰摄像头也不画图）—— 可以单独跑测试
  2) 摄像头与画面（打开设备、建识别器、画中文和卡片）
  3) 主流程 main()：读帧 → 更新状态机 → 画 → 按键

============================ 调用关系总览 ============================

  __main__  →  main()
  │
  ├─ 启动阶段（每个只执行一次）
  │   ├─ open_camera()                  打开摄像头，返回 cap（后面每帧从它 read）
  │   ├─ create_detector()              建识别器：加载 models/face_landmarker.task
  │   │                                 （和 blink_preview.py 的 face_detector()、
  │   │                                   gaze_blink_confirm_demo.py 的 make_detector()
  │   │                                   是同一段逻辑，只是各文件自己留了一份）
  │   ├─ GazeDirectionDetector(started) 建状态机；校准从此刻开始计时
  │   │    └─ self.reset(now)           把所有状态置位（按 C 重新校准走的也是它）
  │   └─ chinese_font(20)               预加载中文字体，缺字体时立刻报错
  │
  ├─ 每帧循环（★ 每帧都执行；顺序 看 → 判 → 报 → 控）
  │   ├─★ read_frame(cap, detector, started)
  │   │    ├─ cap.read() / cv2.flip / cv2.cvtColor / mp.Image   取一帧并预处理
  │   │    ├─ detector.detect_for_video(图, 毫秒时间戳)          ← MediaPipe 推理
  │   │    └─ 返回 (frame, now, landmarks 或 None)
  │   ├─★ draw_iris_points(frame, landmarks)  把两只眼的虹膜 10 个点画出来（只为肉眼检查）
  │   ├─★ flow.update(now, landmarks)         ★核心判定（纯逻辑，可单测）
  │   │    ├─ gaze_score(landmarks)           虹膜水平位置 → 一个数
  │   │    ├─ _update_calibration()           校准中走这里；3 秒后算出正视基准 center
  │   │    ├─ _update_selection()             校准后走这里；偏移量决定 左/中/右，稳定后出候选
  │   │    └─ _status(...) → GazeStatus       把本帧结果打包返回
  │   ├─★ show_status(frame, status)          黑底 + 三行文字 + 三张卡片 + 推给窗口
  │   │    ├─ draw_text(frame, ...)           三行中文 → chinese_font(20 / 28 / 18)
  │   │    └─ draw_cards(frame, 候选)          左 / 中 / 右 三张卡片 → draw_text()
  │   ├─★ window_is_alive()                   问窗口还活着没（点 ✕ 后为假）
  │   └─★ pressed_key(now, key_available_at)  读按键：q 退出 / c → reset / i → 反转
  │
  └─ finally（无论怎么退出都执行）
      ├─ cap.release()                  交还摄像头
      └─ cv2.destroyAllWindows()        关窗口

这台状态机比 gaze_blink_confirm_demo.py 那台简单，只有两个阶段加两个方向变量：

    校准      ──(满 3 秒 且 样本≥10 个)──▶ 选择方向
    选择方向  ：每帧把"虹膜偏移量"分成 左 / 中 / 右 三种当前方向；
                同一方向稳定停留 0.70 秒 → 成为"稳定候选"（屏幕上高亮那张卡片）
    任意阶段  ──(人脸丢失)──────────────▶ 候选作废、方向重新计时（校准不打断）
    任意阶段  ──(按 C)──────────────────▶ 校准（reset(now) 把一切拨回起点）

注意：这里没有"确认"这一步——它只挑候选方向，眨眼确认由第 5 步的程序负责。
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- 运行参数 ----
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0     # 校准时长：这段时间请看屏幕正中间
CALIBRATION_MIN_SAMPLES = 10  # 样本下限：帧率过低时样本太少，基准不可靠
SIDE_THRESHOLD = 0.075        # 相对于正视基准的水平偏移：超过它才算看左/看右
STABLE_SECONDS = 0.70         # 同方向稳定停留多久才成为候选
SMOOTHING_FRAMES = 5          # 虹膜位置平滑：最近几帧取平均
KEY_DEBOUNCE_SECONDS = 0.25   # 两次按键响应的最小间隔：按住不放时不再连发

MESSAGE_INITIAL = "请看屏幕正中间，正在校准"
MESSAGE_RECALIBRATE = "重新校准：请看屏幕正中间"

# ---- 路径与窗口 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
WINDOW_NAME = "Gaze Direction | Q quit | C recalibrate | I invert | visual test only"
# 中文字体逐个探测：msyh 缺失时回退，换机器不会直接崩
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)

# Face Landmarker 中的虹膜点及眼角点。
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_CORNERS = (33, 133)
RIGHT_CORNERS = (362, 263)


# ============================ 1) 判定逻辑（纯逻辑） ============================


def gaze_score(landmarks) -> float | None:
    """虹膜在双眼水平方向的平均归一化位置：0 = 贴左眼角，1 = 贴右眼角。

    调用关系：被 GazeDirectionDetector.update() 每帧调用一次；它决定了"左 / 中 / 右"。
    内部只用关键点的 x 坐标，不画图、也不做判断（判断在状态机里）。
    眼角间距太小时（画面上几乎重合）会返回 None，表示这个数不可用。
    """
    def average_x(indices):
        return sum(landmarks[i].x for i in indices) / len(indices)

    ratios = []
    for iris, corners in ((LEFT_IRIS, LEFT_CORNERS), (RIGHT_IRIS, RIGHT_CORNERS)):
        a, b = landmarks[corners[0]].x, landmarks[corners[1]].x
        span = abs(b - a)
        if span < 1e-5:
            continue
        ratios.append((average_x(iris) - min(a, b)) / span)
    return sum(ratios) / len(ratios) if ratios else None


@dataclass(frozen=True)
class GazeStatus:
    """一帧的判定结果快照，供界面显示。

    调用关系：由 GazeDirectionDetector._status() 产出 → 被 main() 接住 → 交给 show_status()。
    它只是"数据袋子"，自身不含任何逻辑。
    """

    score: float | None          # 本帧的虹膜位置（平滑后；没人脸或不可用时是 None）
    center: float | None         # 正视基准（校准完成前是 None）
    raw_direction: str           # 这一帧视线落在哪个方向：左 / 中 / 右
    candidate: str               # 稳定停留够久的方向；还没稳定时是 "无"
    message: str                 # 屏幕上第三行显示的提示语


class GazeDirectionDetector:
    """视线方向的校准 + 选择状态机。原来散在 main() 里的状态，现在全在这个对象里。

    调用关系：main() 在启动时建一个；之后每帧调 update()；
    按 C 调 reset()，按 I 调 toggle_invert()。update() 内部按阶段分派：
      校准没完成（center 还是 None）→ _update_calibration()
      校准完成                      → _update_selection()
    两种情况都由 _status(...) 打包成 GazeStatus 返回。

    用法：`GazeDirectionDetector(time.perf_counter())`，之后每帧调用
    `update(now, landmarks)`；没检测到人脸时传 None。
    注意它只依赖关键点对象上的 .x 属性，不 import cv2 或 mediapipe，
    所以测试时可以喂"只有 x、y 的假点"。
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
        # 平滑用的滚动队列：最近几帧虹膜位置
        self._history: deque[float] = deque(maxlen=smoothing_frames)
        self._invert = False  # 按 I 切换：左右方向是否反转
        self.reset(now)

    def reset(self, now: float, message: str = MESSAGE_INITIAL) -> None:
        """回到校准起点。构造时和按 C 重新校准时都走这里，避免两处手抄不一致。"""
        self._calibration_started = now           # 校准阶段的起始时间
        self._calibration_values: list[float] = []  # 校准期间攒的虹膜位置
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

    def update(self, now: float, landmarks) -> GazeStatus:
        """吃一帧数据，返回本帧状态。landmarks 传 None 表示这帧没检测到人脸。"""
        if landmarks is None:
            # 没人脸：候选作废、方向重新计时；但"校准"不打断（否则一转头就白校准了）
            self._message = "未检测到人脸：请正对摄像头"
            self._candidate = "无"
            self._direction_since = None
            return self._status(score=None)

        score = gaze_score(landmarks)
        if score is not None:
            self._history.append(score)
            score = sum(self._history) / len(self._history)  # 最近几帧取平均，抹掉抖动

        if self._center is None and score is not None:
            self._update_calibration(now, score)
        elif self._center is not None and score is not None:
            self._update_selection(now, score)
        return self._status(score=score)

    def _update_calibration(self, now: float, score: float) -> None:
        """校准阶段：攒样本，够 3 秒且样本足够后算出正视基准 _center。"""
        self._calibration_values.append(score)
        elapsed = now - self._calibration_started
        self._message = f"校准中：请看屏幕正中间 {max(0, self._calibration_seconds - elapsed):.1f} 秒"
        if elapsed < self._calibration_seconds or len(self._calibration_values) < self._calibration_min_samples:
            return
        # 掐掉头尾各 10%（偶发眨眼的极值），用中间 80% 的平均值当正视基准
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


_detector_kept_alive: list = []  # 见 create_detector() 里的说明


def create_detector():
    """建人脸识别器：加载 models/face_landmarker.task，返回识别器对象。

    调用关系：被 main() 在启动时调用一次（读模型慢，绝不能放进循环）；
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

    调用关系：被 draw_text() 调用（每帧 3 行文字 + 3 张卡片）；
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


def draw_text(frame, text, xy, size, color):
    """在画面上画一行中文（OpenCV 自带字体画不了中文，所以用 PIL + 系统字体）。

    调用关系：被 show_status() 和 draw_cards() 调用；内部调用 chinese_font()。
    注意它不修改传入的 frame，而是返回一张画好字的新图。
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text(xy, text, font=chinese_font(size), fill=color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


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


def read_frame(cap, detector, started: float):
    """读一帧 → 镜像 → 送识别器，返回（画面, 当前时间, 关键点或 None）。

    调用关系：被 main() 每帧调用（循环第一步）；内部调用 cap.read()、
    cv2.flip / cv2.cvtColor / mp.Image，以及 detector.detect_for_video()（真正推理的那次）。
    """
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败。")
    frame = cv2.flip(frame, 1)
    now = time.perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(image, int((now - started) * 1000))
    # detect_for_video 返回对象里，本项目只用到 face_landmarks（478 个点）
    landmarks = result.face_landmarks[0] if result.face_landmarks else None
    return frame, now, landmarks


def draw_iris_points(frame, landmarks) -> None:
    """把两只眼的虹膜 10 个点画成黄点，方便肉眼检查识别准不准。

    调用关系：被 main() 每帧调用（只在检测到人脸时）；只画图，不参与任何判定。
    """
    h, w = frame.shape[:2]
    for idx in LEFT_IRIS + RIGHT_IRIS:
        lm = landmarks[idx]
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 255), -1)


def show_status(frame, status) -> None:
    """铺黑色底板 + 三行中文 + 底部三张卡片，然后推给窗口。

    调用关系：被 main() 每帧调用（循环第三步）；内部调用 draw_text() 和 draw_cards()。
    它只读 status，不改任何状态——所以调它不会影响判定。
    """
    cv2.rectangle(frame, (0, 0), (650, 105), (0, 0, 0), -1)
    score_text = "--" if status.score is None else f"{status.score:.3f}"
    base_text = "--" if status.center is None else f"{status.center:.3f}"
    frame = draw_text(frame, f"虹膜位置：{score_text}｜正视基准：{base_text}", (14, 10), 20, (0, 255, 0))
    frame = draw_text(frame, f"当前方向：{status.raw_direction}｜稳定候选：{status.candidate}", (14, 38), 28, (255, 255, 0))
    frame = draw_text(frame, status.message, (14, 74), 18, (255, 255, 255))
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


def main() -> None:
    """程序入口：建好摄像头和识别器后，每帧走一遍 看→判→报→控。"""
    # ==================== 启动阶段（下面每个调用只执行一次）====================
    cap = open_camera()                          # ① 打开摄像头
    detector = create_detector()                 # ② 建识别器
    started = time.perf_counter()                # 给 MediaPipe 算毫秒时间戳用
    flow = GazeDirectionDetector(started)        # ③ 建状态机；校准从此刻开始计时
    key_available_at = 0.0                       # 按键防抖：下次允许响应的时间点
    chinese_font(20)                             # ④ 提前加载字体，字体缺失时立刻报错

    print("视线方向预览已启动：请先点一下视频窗口让它获得焦点，然后 Q 退出、C 重新校准、I 反转左右。")
    try:
        # ================= 每帧循环：看 → 判 → 报 → 控 =================
        while True:
            # ---- ① 看：取一帧并识别，拿回 478 个关键点 ----
            frame, now, landmarks = read_frame(cap, detector, started)
            if landmarks is not None:
                draw_iris_points(frame, landmarks)  # 顺便把虹膜上的点画出来（只为肉眼检查）

            # ---- ② 判：校准 / 方向选择，全在 GazeDirectionDetector 里 ----
            status = flow.update(now, landmarks)

            # ---- ③ 报：三行中文 + 三张卡片，推给窗口 ----
            show_status(frame, status)

            # ---- ④ 控：先看窗口还活着没，再读键盘 ----
            if not window_is_alive():            # 窗口被点 ✕ 关掉就退出
                print("窗口已关闭，退出。")
                break
            key, key_available_at = pressed_key(now, key_available_at)
            if key == "q":
                break
            if key == "c":
                flow.reset(now, MESSAGE_RECALIBRATE)  # 回到校准起点
            if key == "i":
                flow.toggle_invert()                   # 左右方向反转 / 恢复
    finally:
        # 不调用 detector.close()：实测它要卡约 42 秒（MediaPipe 1.0.1 拆推理图的等待）。
        # 配合 create_detector() 里的保活引用，进程 1 秒内干净退出，内存由系统回收。
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
