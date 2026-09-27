"""第 5 步：视线选方向 + 眨眼确认（纯屏幕演示）。

流程：看正中校准 3 秒 → 稳定注视 左转/前进/右转 → 自然眨眼确认。
默认（ENABLE_HARDWARE = False）是纯屏幕模拟：确认结果只显示，不发任何指令。
把顶部 ENABLE_HARDWARE 改为 True 并接好 Arduino 后，确认成功的方向才会经
src/hardware/serial_link.py 发给固件（含 1 秒心跳、电池回报、低电强制停车）。
按 Q 退出；C 重新校准；I 反转左右方向。

文件分三段：
  1) 判定逻辑（纯逻辑 + 状态机类，只吃关键点，不碰摄像头也不画图）—— 可以单独跑测试
  2) 摄像头与画面（打开设备、建识别器、画中文、画卡片）
  3) 主流程 main()：读帧 → 更新状态机 → 画 → 按键

============================ 调用关系总览 ============================

  __main__  →  main()
  │
  ├─ 启动阶段（每个只执行一次）
  │   ├─ open_camera()                打开摄像头，返回 cap（后面每帧从它 read）
  │   ├─ make_detector()              建识别器：加载 models/face_landmarker.task
  │   ├─ GazeBlinkDetector(started)   建状态机；校准从此刻开始计时
  │   │    └─ self.reset(now)         把所有状态置位（按 C 重新校准走的也是它）
  │   ├─ WheelchairLink(...)           硬件链路（src/hardware/serial_link.py；默认模拟模式）
  │   │    └─ link.open()             串口打不开时自动退回模拟并记录原因
  │   └─ chinese_font(20)             预加载中文字体，缺字体时立刻报错
  │
  ├─ 每帧循环（★ 每帧都执行；顺序 看 → 判 → 报 → 控）
  │   ├─★ read_frame(cap, detector, started)
  │   │    ├─ cap.read() / cv2.flip / cv2.cvtColor / mp.Image    取一帧并预处理
  │   │    ├─ detector.detect_for_video(图, 毫秒时间戳)           ← MediaPipe 推理
  │   │    └─ 返回 (frame, now, landmarks 或 None)
  │   ├─★ draw_points(frame, landmarks)        把关键点画成黄点（只为肉眼检查）
  │   ├─★ flow.update(now, landmarks)          ★核心判定（纯逻辑，可单测）
  │   │    ├─ gaze_score(landmarks)            虹膜位置 → 看的是左 / 中 / 右
  │   │    ├─ average_ear(landmarks)           眼睛开合值 → 有没有眨眼
  │   │    │    └─ eye_aspect_ratio() → distance()   量 4 段距离
  │   │    ├─ _update_calibration()            校准未完成时走这里
  │   │    ├─ _update_selection()              校准完成后走这里（视线选择 + 眨眼确认）
  │   │    └─ _status(...) → GazeBlinkStatus   把本帧结果打包返回
  │   ├─★ link.update(now)                     硬件链路：心跳重发当前指令 + 收电池回报
  │   │    └─ 确认成功的瞬间：link.set_intent(方向) → 翻译成 L/F/R 发给固件
  │   ├─★ show_status(frame, status, now, fps, link_info)  渲染界面并推窗口
  │   │    └─ render_overlay(...)              面板/指示灯/进度条/卡片/角标（单次 PIL 转换）
  │   ├─★ window_is_alive()                    问窗口还活着没（点 ✕ 后为假）
  │   └─★ pressed_key(now, key_available_at)   读按键：q 退出 / c → reset / i → 反转
  │
  └─ finally（无论怎么退出都执行）
      ├─ link.set_intent("停") + link.close()  先叫停再挂断（模拟模式下是空操作）
      ├─ cap.release()                   交还摄像头
      └─ cv2.destroyAllWindows()         关窗口

四个状态怎么互相转移（这就是"状态机"，全在 GazeBlinkDetector 里）：

    校准      ──(满 3 秒 且 样本≥10 个)──────▶ 选择方向
    选择方向  ──(同一方向稳定停留 0.70 秒)───▶ 等待眨眼（pending = 选中的方向）
    等待眨眼  ──(检测到一次自然眨眼)─────────▶ 确认成功（屏幕显示"已确认"）
    确认成功  ──(1.6 秒后自动)──────────────▶ 选择方向（可以接着选下一项）
    任意状态  ──(人脸丢失)──────────────────▶ 选择方向（未完成的候选作废）
    任意状态  ──(按 C)──────────────────────▶ 校准（reset(now) 把一切拨回起点）

关键一点：**只有在"等待眨眼"这个状态下，一次眨眼才算确认**；在别的状态眨眼没有效果。
"""
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 本项目各脚本独立运行（没有包结构），把 src/ 加进搜索路径以引入硬件接口
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.serial_link import WheelchairLink  # noqa: E402  ← 在 sys.path 之后导入

# ---- 运行参数 ----
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0                            # 校准时长：这段时间请看屏幕正中间
CALIBRATION_MIN_SAMPLES = 10                         # 样本下限：帧率过低时样本太少，标尺不可靠
SIDE_THRESHOLD = 0.075                               # 虹膜偏移超过它才算"看左/看右"
SELECT_STABLE_SECONDS = 0.70                         # 同一方向要稳定停留这么久才成为候选
CLOSE_RATIO, REOPEN_RATIO = 0.78, 0.88               # 闭眼线 / 睁眼线（相对睁眼基线的百分比）
MIN_CLOSED_SECONDS, MAX_CLOSED_SECONDS = 0.04, 0.80  # 一次自然眨眼的闭眼时长范围
MIN_OPEN_SECONDS, REFRACTORY_SECONDS = 0.10, 0.30    # 需要稳定睁眼多久 / 两次计数的最小间隔
GAZE_SMOOTHING_FRAMES = 5                            # 虹膜位置平滑：最近几帧取平均
EAR_SMOOTHING_FRAMES = 3                             # 眼睛开合值平滑：最近几帧取平均
KEY_DEBOUNCE_SECONDS = 0.25                          # 两次按键响应的最小间隔：按住不放时不再连发

# ---- 硬件链路（协议与安全机制见 src/hardware/serial_link.py 和 README）----
ENABLE_HARDWARE = False      # ⚠️ False=纯屏幕模拟（默认）；True 时确认结果会真的发往串口
SERIAL_PORT = "COM3"         # Arduino 的串口号（设备管理器里查）；打不开自动退回模拟模式
FACE_LOST_STOP_SECONDS = 2.0 # 人脸连续丢失这么久 → 主动发"停"（固件的 3 秒超时是兜底）

# ---- 路径与窗口 ----
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "face_landmarker.task"
WINDOW = "Gaze Select + Blink Confirm | Q quit | C recalibrate | I invert | screen demo only"
# 中文字体逐个探测：msyh 缺失时回退，换机器不会直接崩
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)
# ---- 用到的关键点编号（都来自 Face Landmarker 的 478 个点）----
LEFT_IRIS, RIGHT_IRIS = [468, 469, 470, 471, 472], [473, 474, 475, 476, 477]  # 两只眼的虹膜 5 点
LEFT_CORNERS, RIGHT_CORNERS = (33, 133), (362, 263)                           # 两只眼的内外眼角（当尺子）
LEFT_EYE, RIGHT_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]  # 眼睛轮廓 6 点


cat = r'''
__        .-.
             .-"` .`'.    /\\|
     _(\-/)_" ,  .   ,\  /\\\/
    {(#b^d#)} .   ./,  |/\\\/
    `-.(Y).-`  ,  |  , |\.-`
         /~/,_/~~~\,__.-`
        ////~    // ~\\
jgs   ==`==`   ==`   ==`
'''

# ============================ 1) 判定逻辑（纯逻辑） ============================


def distance(a, b):
    """两个关键点之间的平面距离（忽略 z）。只被下面的 eye_aspect_ratio() 调用。"""
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(landmarks, indices):
    """单只眼睛的开合比：上下眼睑平均距离 ÷ 眼角宽度。闭眼时趋近 0。

    调用关系：被 average_ear() 调用（左眼、右眼各一次）；内部调用 distance()。
    """
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in indices]
    horizontal = distance(p0, p3)
    return 0.0 if horizontal < 1e-6 else (distance(p1, p5) + distance(p2, p4)) / (2 * horizontal)


def average_ear(landmarks):
    """双眼平均开合比：取平均可以抵消轻微侧头带来的差异。

    调用关系：被 GazeBlinkDetector.update() 每帧调用一次；
    内部调用 eye_aspect_ratio() 两次。产出的数就是判断"有没有眨眼"的依据。
    """
    return (eye_aspect_ratio(landmarks, LEFT_EYE) + eye_aspect_ratio(landmarks, RIGHT_EYE)) / 2


def gaze_score(landmarks):
    """虹膜在眼睛水平方向上的平均归一化位置：0 = 贴左眼角，1 = 贴右眼角。

    调用关系：被 GazeBlinkDetector.update() 每帧调用一次；它决定了"左转 / 前进 / 右转"。
    内部只用关键点的 x 坐标，不画图、也不做判断（判断在状态机里）。
    """
    ratios = []
    for iris, corners in ((LEFT_IRIS, LEFT_CORNERS), (RIGHT_IRIS, RIGHT_CORNERS)):
        iris_x = sum(landmarks[i].x for i in iris) / len(iris)
        a, b = landmarks[corners[0]].x, landmarks[corners[1]].x
        if abs(a - b) > 1e-5:
            ratios.append((iris_x - min(a, b)) / abs(a - b))
    return sum(ratios) / len(ratios) if ratios else None


@dataclass(frozen=True)
class GazeBlinkStatus:
    """一帧的判定结果快照，供界面显示。

    调用关系：由 GazeBlinkDetector._status() 产出 → 被 main() 接住 → 交给 show_status()。
    它只是"数据袋子"，自身不含任何逻辑。
    """

    state: str            # 四个状态里的哪一个：校准 / 选择方向 / 等待眨眼 / 确认成功
    message: str          # 屏幕上第三行显示的提示语
    score: float | None   # 本帧的虹膜位置（没人脸时是 None）
    ear: float | None     # 本帧的眼睛开合值（没人脸时是 None）
    active: str | None    # 该高亮哪张卡片：左转 / 前进 / 右转 / None
    calibration_progress: float | None = None  # 校准进度 0~1（仅校准状态有值）
    stability_progress: float | None = None    # 方向稳定进度 0~1（仅选择方向状态有值）


class GazeBlinkDetector:
    """视线选择 + 眨眼确认的四状态机。原来散在 main() 里的状态，现在全在这个对象里。

    调用关系：main() 在启动时建一个；之后每帧调 update()；
    按 C 调 reset()，按 I 调 toggle_invert()。update() 内部按状态分派：
      校准没完成 → _update_calibration()
      选择方向 / 等待眨眼 → _update_selection()
      确认成功    → 纯计时，时间到就回选择方向
    每种情况的最后都由 _status() 打包成 GazeBlinkStatus 返回。

    用法：`GazeBlinkDetector(time.perf_counter())`，之后每帧调用
    `update(now, landmarks)`；没检测到人脸时传 None。
    注意它只依赖关键点对象上的 .x / .y 两个属性，不 import cv2 或 mediapipe，
    所以测试时可以喂"只有 x、y 的假点"。
    """

    def __init__(
        self,
        now: float,
        *,
        calibration_seconds: float = CALIBRATION_SECONDS,
        calibration_min_samples: int = CALIBRATION_MIN_SAMPLES,
        side_threshold: float = SIDE_THRESHOLD,
        select_stable_seconds: float = SELECT_STABLE_SECONDS,
        close_ratio: float = CLOSE_RATIO,
        reopen_ratio: float = REOPEN_RATIO,
        min_closed_seconds: float = MIN_CLOSED_SECONDS,
        max_closed_seconds: float = MAX_CLOSED_SECONDS,
        min_open_seconds: float = MIN_OPEN_SECONDS,
        refractory_seconds: float = REFRACTORY_SECONDS,
        gaze_smoothing_frames: int = GAZE_SMOOTHING_FRAMES,
        ear_smoothing_frames: int = EAR_SMOOTHING_FRAMES,
    ) -> None:
        self._calibration_seconds = calibration_seconds
        self._calibration_min_samples = calibration_min_samples
        self._side_threshold = side_threshold
        self._select_stable_seconds = select_stable_seconds
        self._close_ratio = close_ratio
        self._reopen_ratio = reopen_ratio
        self._min_closed_seconds = min_closed_seconds
        self._max_closed_seconds = max_closed_seconds
        self._min_open_seconds = min_open_seconds
        self._refractory_seconds = refractory_seconds
        # 平滑用的滚动队列：最近几帧虹膜位置、最近几帧开合值
        self._gaze_history: deque[float] = deque(maxlen=gaze_smoothing_frames)
        self._ear_history: deque[float] = deque(maxlen=ear_smoothing_frames)
        self._invert = False  # 按 I 切换：左右方向是否反转
        self.reset(now)

    def reset(self, now: float) -> None:
        """回到校准起点。构造时和按 C 重新校准时都走这里，避免两处手抄不一致。"""
        self._calibration_started = now          # 校准阶段的起始时间
        self._gaze_values: list[float] = []      # 校准期间攒的虹膜位置
        self._ear_values: list[float] = []       # 校准期间攒的开合值
        # 三个"标尺"，校准完成后才有值：
        #   _center      = 正视时虹膜该在的位置
        #   _close_line  = 闭眼判定线（低于它算闭眼）
        #   _reopen_line = 睁眼判定线（高于它算睁开）
        self._center: float | None = None
        self._close_line: float | None = None
        self._reopen_line: float | None = None
        self._gaze_history.clear()
        self._ear_history.clear()
        # ---- 状态机的全部状态（本程序的"记忆"）----
        #   _state         : 现在是四个状态里的哪一个
        #   _raw_choice    : 这一帧视线落在哪个方向（会抖）
        #   _stable_choice : 稳定够久、已经确认过的方向（用来高亮卡片）
        #   _pending       : 已经选中、正等着眨眼确认的方向
        self._state, self._raw_choice, self._stable_choice, self._pending = "校准", "前进", None, None
        # 四个时间戳（方向稳定的起点 / 睁眼起点 / 闭眼起点 / 确认成功时刻）+ 是否闭着眼
        self._direction_since = self._open_since = self._closed_since = self._confirm_started = None
        self._is_closed = False
        self._last_blink = float("-inf")  # 上次计数时刻；负无穷让第一次眨眼不受不应期限制
        self._message = "请睁眼看屏幕正中间，正在校准"

    def toggle_invert(self) -> None:
        """按 I 时调用：把左右方向反过来，并更新屏幕提示。"""
        self._invert = not self._invert
        self._message = "左右方向已反转" if self._invert else "左右方向已恢复"

    def update(self, now: float, landmarks) -> GazeBlinkStatus:
        """吃一帧数据，返回本帧状态。landmarks 传 None 表示这帧没检测到人脸。"""
        if landmarks is None:
            # 没人脸：不作任何判断，候选作废；但"校准"状态不打断（否则一转头就白校准了）
            self._message = "未检测到人脸：已暂停，重新正对摄像头"
            if self._state != "校准":
                self._state, self._stable_choice, self._pending = "选择方向", None, None
            return self._status(score=None, ear=None,
                                calibration_progress=None, stability_progress=None)

        score = gaze_score(landmarks)                 # 眼睛在看哪边
        current_ear = average_ear(landmarks)          # 眼睛张多开
        # 两个数各做一次"最近几帧平均"，把抖动抹掉
        if score is not None:
            self._gaze_history.append(score)
            score = sum(self._gaze_history) / len(self._gaze_history)
        self._ear_history.append(current_ear)
        current_ear = sum(self._ear_history) / len(self._ear_history)

        # 进度条数据按"本帧进入时的状态"算：这样校准完成的那一帧也能看到 100% 满条
        calibration_progress = stability_progress = None
        if self._state == "校准":
            calibration_progress = min((now - self._calibration_started) / self._calibration_seconds, 1.0)
        elif self._state == "选择方向" and self._direction_since is not None:
            stability_progress = min((now - self._direction_since) / self._select_stable_seconds, 1.0)

        # ---- 按状态分派 ----
        # 【状态一：校准】攒够 3 秒且样本足够 → 算出三个"标尺"，转入选择方向
        if self._state == "校准" and score is not None:
            self._update_calibration(now, score, current_ear)
        # 【状态二、三：选择方向 / 等待眨眼】视线偏移决定候选方向，顺便跑眨眼状态机
        elif self._state in ("选择方向", "等待眨眼") and self._center is not None:
            self._update_selection(now, score, current_ear)
        # 【状态四：确认成功】停 1.6 秒让人看清结果，然后自动回到"选择方向"
        elif self._state == "确认成功" and self._confirm_started and now - self._confirm_started >= 1.6:
            self._state, self._stable_choice, self._pending, self._direction_since = "选择方向", None, None, now
            self._message = "请继续选择下一项方向"
        return self._status(score=score, ear=current_ear,
                            calibration_progress=calibration_progress,
                            stability_progress=stability_progress)

    def _update_calibration(self, now: float, score: float, current_ear: float) -> None:
        """校准阶段：攒样本，够 3 秒且样本足够后算出三个标尺。"""
        self._gaze_values.append(score)
        self._ear_values.append(current_ear)
        elapsed = now - self._calibration_started
        self._message = f"校准中：请看正中间 {max(0, self._calibration_seconds - elapsed):.1f} 秒"
        if elapsed < self._calibration_seconds or len(self._gaze_values) < self._calibration_min_samples:
            return
        # _center = 正视时的虹膜位置；base = 睁眼时的开合值
        self._center = sum(self._gaze_values) / len(self._gaze_values)
        base = sum(self._ear_values) / len(self._ear_values)
        self._close_line, self._reopen_line = base * self._close_ratio, base * self._reopen_ratio
        self._state, self._message = "选择方向", "校准完成：看左转、前进或右转，稳定后等待眨眼"

    def _update_selection(self, now: float, score: float, current_ear: float) -> None:
        """选择方向 / 等待眨眼：先看视线选了哪个方向，再跑眨眼状态机。"""
        # offset = 相对正视基准的偏移量；按 I 反转时取负号
        offset = score - self._center if score is not None else 0.0
        if self._invert:
            offset = -offset
        # 偏移量越过阈值就判成左/右，否则算"前进"（中）
        new_choice = "左转" if offset < -self._side_threshold else "右转" if offset > self._side_threshold else "前进"
        if self._state == "选择方向":
            if new_choice != self._raw_choice:
                # 方向变了 → 重新开始计时，候选清空
                self._raw_choice, self._direction_since, self._stable_choice = new_choice, now, None
            elif self._direction_since is not None and now - self._direction_since >= self._select_stable_seconds:
                # 同一方向稳定停留够久 → 定为候选，进入"等待眨眼"
                self._stable_choice, self._pending, self._state = self._raw_choice, self._raw_choice, "等待眨眼"
                self._message = f"已选择“{self._pending}”，请自然眨眼确认"

        # 眨眼状态机：仅在“等待眨眼”时才把一次眨眼当确认。
        # stable_open 三个条件：有睁眼起点、睁眼已满 0.1 秒、离上次计数已满 0.3 秒
        stable_open = (
            self._open_since is not None
            and now - self._open_since >= self._min_open_seconds
            and now - self._last_blink >= self._refractory_seconds
        )
        blink_confirmed = False
        if self._is_closed:
            # 本来闭着 → 现在睁开了：量一下这次闭了多久
            if current_ear >= self._reopen_line:
                duration = now - self._closed_since if self._closed_since else 0.0
                self._is_closed, self._closed_since, self._open_since = False, None, now
                # 闭眼时长落在自然眨眼窗口内 → 记一次"可用的眨眼"
                if self._min_closed_seconds <= duration <= self._max_closed_seconds:
                    self._last_blink, blink_confirmed = now, True
            # 否则持续闭眼，等待睁开。
        elif current_ear >= self._reopen_line:
            # 睁着眼：记下"开始睁眼"的时刻（只在还没有的时候记一次）
            if self._open_since is None:
                self._open_since = now
        elif current_ear < self._close_line and stable_open:
            # 掉到闭眼线以下，且刚才确实稳定睁着 → 进入闭眼候选
            self._is_closed, self._closed_since, self._open_since = True, now, None
        else:
            # 落在两条判定线之间的灰色地带 → 睁眼计时作废
            self._open_since = None

        # 【状态三 → 四】只有"等待眨眼"期间的眨眼才被当作确认
        if self._state == "等待眨眼" and blink_confirmed:
            self._state, self._confirm_started = "确认成功", now
            # 是否真的发往硬件看右上角链路状态（模拟/串口），状态机保持纯逻辑
            self._message = f"已确认“{self._pending}”"

    def _status(self, score: float | None, ear: float | None,
                calibration_progress: float | None, stability_progress: float | None) -> GazeBlinkStatus:
        """把当前内部状态打包成 GazeBlinkStatus（界面唯一能读到的东西）。

        两个进度由 update() 按"本帧进入时的状态"算好传进来（详见那里的注释）。
        """
        # 高亮哪张卡片：等待眨眼/确认成功时高亮 pending，其余时候高亮已稳定的候选
        active = self._pending if self._state in ("等待眨眼", "确认成功") else self._stable_choice
        return GazeBlinkStatus(
            state=self._state,
            message=self._message,
            score=score,
            ear=ear,
            active=active,
            calibration_progress=calibration_progress,
            stability_progress=stability_progress,
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
        raise RuntimeError("摄像头无法打开。")
    return cap


_detector_kept_alive: list = []  # 见 make_detector() 里的说明


def make_detector():
    """建人脸识别器：加载 models/face_landmarker.task，返回识别器对象。

    调用关系：被 main() 在启动时调用一次（读模型慢，绝不能放进循环）；
    内部调用 MediaPipe 的 FaceLandmarker.create_from_options()。
    返回值一路传给 read_frame()，由它每帧调用 detect_for_video()。
    """
    if not MODEL.exists():
        raise FileNotFoundError(f"缺少模型：{MODEL}")
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
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

    调用关系：被 chinese_overlay() 和 cards() 调用；第一次真正读文件，之后命中缓存。
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


# ---- 界面配色（BGR）与几何常量 ----
PANEL_BG, PANEL_BORDER = (30, 30, 36), (90, 120, 160)   # 深色面板 + 蓝灰描边
STATE_COLORS = {                        # 每个状态一个指示灯颜色
    "校准": (60, 200, 255),             # 黄
    "选择方向": (120, 200, 120),        # 绿
    "等待眨眼": (40, 170, 255),         # 橙
    "确认成功": (90, 255, 130),         # 亮绿
}
BAR_BG = (62, 62, 70)                   # 进度条底槽
CARD_BG, CARD_BORDER = (45, 45, 52), (95, 95, 105)
CARD_CANDIDATE = (40, 180, 255)         # 琥珀色：已稳定候选
CARD_CONFIRM_FILL = (70, 170, 95)       # 绿色：等待眨眼 / 已确认
TEXT_VALUES, TEXT_MAIN, TEXT_HINT = (120, 255, 120), (255, 255, 255), (175, 175, 175)
CARD_RECTS = [("左转", 35, 390, 280, 515), ("前进", 340, 390, 585, 515), ("右转", 645, 390, 890, 515)]
PANEL_W, PANEL_H = 640, 164
BAR_X1, BAR_X2 = 70, 330


def _pil_text(frame, items):
    """一次 PIL 往返画完全部中文（旧版每帧最多 4 次 BGR↔RGB 转换，这里只做 1 次）。

    items = [(x, y, 文本, 字号, BGR颜色), ...]
    调用关系：被 render_overlay() 调用；内部调用 chinese_font()（带缓存）。
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for x, y, text, size, color in items:
        draw.text((x, y), text, font=chinese_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _card_mode(status) -> str:
    """三张卡片该用哪种画法：none=普通 / candidate=琥珀候选 / confirmed=绿色已确认。"""
    if status.state in ("等待眨眼", "确认成功") and status.active:
        return "confirmed"
    if status.state == "选择方向" and status.active:
        return "candidate"
    return "none"


def _draw_panel(frame, status):
    """状态面板的底板、状态指示灯和两条进度条（文字统一在 _pil_text 里画）。"""
    color = STATE_COLORS.get(status.state, TEXT_MAIN)
    cv2.rectangle(frame, (0, 0), (PANEL_W, PANEL_H), PANEL_BG, -1)
    cv2.rectangle(frame, (0, 0), (PANEL_W, PANEL_H), PANEL_BORDER, 1)
    cv2.circle(frame, (26, 32), 8, color, -1)                     # 状态指示灯
    for y, progress in ((80, status.calibration_progress), (100, status.stability_progress)):
        cv2.rectangle(frame, (BAR_X1, y), (BAR_X2, y + 8), BAR_BG, -1)       # 进度条底槽
        if progress is not None:
            fill_w = int((BAR_X2 - BAR_X1) * max(0.0, min(1.0, progress)))
            if fill_w > 0:
                cv2.rectangle(frame, (BAR_X1, y), (BAR_X1 + fill_w, y + 8), color, -1)
    return frame


def _draw_cards(frame, status, now, mode):
    """底部三张卡片：普通 / 已稳定候选（琥珀边+顶部色条）/ 等待确认与已确认（绿底+呼吸灯+对勾）。"""
    pulse = 0.5 + 0.5 * math.sin(now * 6.0)                       # 呼吸灯相位
    for label, x1, y1, x2, y2 in CARD_RECTS:
        if label != status.active:
            cv2.rectangle(frame, (x1, y1), (x2, y2), CARD_BG, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), CARD_BORDER, 2)
        elif mode == "confirmed":
            cv2.rectangle(frame, (x1, y1), (x2, y2), CARD_CONFIRM_FILL, -1)
            glow = tuple(int(c * (0.65 + 0.35 * pulse)) for c in (150, 255, 190))
            cv2.rectangle(frame, (x1, y1), (x2, y2), glow, 4)     # 边框随呼吸明暗
            if status.state == "确认成功":                        # 右上角对勾
                pts = np.array([(x2 - 52, y1 + 34), (x2 - 40, y1 + 46), (x2 - 16, y1 + 12)])
                cv2.polylines(frame, [pts], False, (255, 255, 255), 4)
        else:  # candidate
            cv2.rectangle(frame, (x1, y1), (x2, y2), CARD_BG, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), CARD_CANDIDATE, 4)
            cv2.rectangle(frame, (x1, y1), (x2, y1 + 8), CARD_CANDIDATE, -1)  # 顶部色条
    return frame


def _draw_chips(frame, link_info):
    """右上角两枚角标的底板：帧率、硬件链路状态（模式/电量）。文字在 _pil_text 里画。"""
    cv2.rectangle(frame, (WIDTH - 120, 8), (WIDTH - 8, 40), PANEL_BG, -1)
    cv2.rectangle(frame, (WIDTH - 340, 8), (WIDTH - 128, 40), PANEL_BG, -1)
    return frame


def render_overlay(frame, status, now, fps, link_info):
    """把本帧全部界面元素画好并返回新帧（纯绘制：不 imshow、不碰任何状态，可单测）。

    调用关系：被 show_status() 调用（main 每帧一次）；
    内部调用 _draw_panel / _draw_cards / _draw_chips / _pil_text。
    """
    mode = _card_mode(status)
    frame = _draw_panel(frame, status)
    frame = _draw_cards(frame, status, now, mode)
    frame = _draw_chips(frame, link_info)

    score_t = "--" if status.score is None else f"{status.score:.3f}"
    ear_t = "--" if status.ear is None else f"{status.ear:.3f}"
    items = [
        (44, 18, f"当前状态：{status.state}", 24, STATE_COLORS.get(status.state, TEXT_MAIN)),
        (16, 52, f"虹膜位置 {score_t}    眼睛开合 {ear_t}", 18, TEXT_VALUES),
        (16, 78, "校准", 15, TEXT_HINT),
        (16, 98, "稳定", 15, TEXT_HINT),
        (16, 118, status.message, 18, TEXT_MAIN),
        (16, 142, "Q 退出｜C 重新校准｜I 反转左右", 16, TEXT_HINT),
        (WIDTH - 108, 14, f"{fps:.0f} FPS", 18, TEXT_VALUES),
    ]
    hw_text, hw_color = "模拟模式", TEXT_HINT
    if link_info is not None:
        if link_info.connected:
            hw_text = f"{link_info.mode} 已发 {link_info.send_count}"
            if link_info.battery is not None:
                hw_text += f" 电 {link_info.battery[0]}% {link_info.battery[1]:.1f}V"
            hw_color = (60, 60, 255) if link_info.is_low_battery else (120, 255, 120)
        elif link_info.last_error:
            hw_text = "模拟（串口未连上）"
    items.append((WIDTH - 332, 14, hw_text, 16, hw_color))
    for label, x1, _, y1, _ in CARD_RECTS:
        tcolor = TEXT_MAIN
        if label == status.active:
            tcolor = (255, 255, 255) if mode == "confirmed" else CARD_CANDIDATE
        items.append((x1 + 90, y1 + 40, label, 40, tcolor))
    return _pil_text(frame, items)


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
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - started) * 1000))
    # detect_for_video 返回对象里，本项目只用到 face_landmarks（478 个点）
    landmarks = result.face_landmarks[0] if result.face_landmarks else None
    return frame, now, landmarks


def draw_points(frame, landmarks) -> None:
    """把用到的关键点画成黄点，方便肉眼检查识别准不准。

    调用关系：被 main() 每帧调用（只在检测到人脸时）；只画图，不参与任何判定。
    """
    h, w = frame.shape[:2]
    for idx in LEFT_IRIS + RIGHT_IRIS + LEFT_EYE + RIGHT_EYE:
        p = landmarks[idx]
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 255), -1)


def show_status(frame, status, now, fps, link_info) -> None:
    """渲染整帧界面（render_overlay）并推给窗口。被 main() 每帧调用。"""
    cv2.imshow(WINDOW, render_overlay(frame, status, now, fps, link_info))


def window_is_alive() -> bool:
    """点窗口右上角 ✕ 不会让 waitKey 返回任何值，只能自己问窗口还活着没。

    调用关系：被 main() 每帧调用；返回 False 时 main() 就 break 退出循环。
    """
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
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
    print(cat)
    """程序入口：建好摄像头、识别器、硬件链路后，每帧走一遍 看→判→连→报→控。"""
    # ==================== 启动阶段（下面每个调用只执行一次）====================
    cap = open_camera()                          # ① 打开摄像头
    detector = make_detector()                   # ② 建识别器
    started = time.perf_counter()                # 给 MediaPipe 算毫秒时间戳用
    flow = GazeBlinkDetector(started)            # ③ 建状态机；校准从此刻开始计时
    link = WheelchairLink(SERIAL_PORT if ENABLE_HARDWARE else None)  # ④ 硬件链路（默认模拟）
    link.open()                                  #    串口打不开时自动退回模拟模式
    key_available_at = 0.0                       # 按键防抖：下次允许响应的时间点
    chinese_font(20)                             # ⑤ 提前加载字体，字体缺失时立刻报错
    frame_count = 0                              # 只用来算 FPS
    prev_state = None                            # 用来捕捉"确认成功"这一瞬间
    face_lost_since = None                       # 人脸持续丢失的起点（超时就发"停"）

    print("视线选择演示已启动：请先点一下视频窗口让它获得焦点，然后 Q 退出、C 重新校准、I 反转左右。")
    if link.connected:
        print(f"硬件链路已连接：{link.mode}。确认成功的方向会真的发往固件（低电时被强制为停止）。")
    else:
        reason = f"（{link.last_error}）" if link.last_error else ""
        print(f"模拟模式{reason}：确认结果只显示在屏幕上，不会发出任何指令。")
    try:
        # ================= 每帧循环：看 → 判 → 连 → 报 → 控 =================
        while True:
            # ---- ① 看：取一帧并识别，拿回 478 个关键点 ----
            frame, now, landmarks = read_frame(cap, detector, started)
            if landmarks is not None:
                draw_points(frame, landmarks)    # 顺便把关键点画出来（只为肉眼检查）

            # ---- ② 判：视线 + 眨眼，四状态机全在 GazeBlinkDetector 里 ----
            status = flow.update(now, landmarks)

            # ---- ③ 连：硬件链路（心跳/收电池）+ 状态变化时下发意图 ----
            link_info = link.update(now)
            if prev_state == "等待眨眼" and status.state == "确认成功" and status.active:
                link.set_intent(status.active)   # "左转"→L / "前进"→F / "右转"→R；低电强制为"停"
            prev_state = status.state
            if landmarks is None:
                face_lost_since = face_lost_since if face_lost_since is not None else now
                if now - face_lost_since >= FACE_LOST_STOP_SECONDS:
                    link.set_intent("停")        # 人不见了 → 主动停车（固件超时是兜底）
            else:
                face_lost_since = None

            # ---- ④ 报：状态面板 + 进度条 + 卡片 + 角标，推给窗口 ----
            frame_count += 1
            fps = frame_count / max(now - started, 0.001)
            show_status(frame, status, now, fps, link_info)

            # ---- ⑤ 控：先看窗口还活着没，再读键盘 ----
            if not window_is_alive():            # 窗口被点 ✕ 关掉就退出
                print("窗口已关闭，退出。")
                break
            key, key_available_at = pressed_key(now, key_available_at)
            if key == "q":
                break
            if key == "c":
                flow.reset(now)                  # 回到校准起点，重新采集一次标尺
                link.set_intent("停")            # 校准期间不该挂着旧指令
            if key == "i":
                flow.toggle_invert()             # 左右方向反转 / 恢复
    finally:
        link.set_intent("停")                    # 无论怎么退出先叫停（模拟模式下是空操作）
        link.close()
        # 不调用 detector.close()：实测它要卡约 42 秒（MediaPipe 1.0.1 拆推理图的等待）。
        # 配合 make_detector() 里的保活引用，进程 1 秒内干净退出，内存由系统回收。
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
