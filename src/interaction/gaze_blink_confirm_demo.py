"""第 5 步：视线选方向 + 眨眼确认（纯屏幕演示，YOLO 版）。

流程：看正中校准 3 秒 → 稳定注视 左转/前进/右转 → 自然眨眼确认。
默认（ENABLE_HARDWARE = False）是纯屏幕模拟：确认结果只显示，不发任何指令。
把顶部 ENABLE_HARDWARE 改为 True 并接好 Arduino 后，确认成功的方向才会经
src/hardware/serial_link.py 发给固件（含 1 秒心跳、电池回报、低电强制停车）。
按 Q 退出；C 重新校准；I 反转左右方向。

本版把 MediaPipe 关键点几何判定整体换成了 YOLO26 眼动模型
（Yolo_model 项目交付：eye_yolo26n.pt 睁/闭眼 2 类 + gaze5_yolo26s.pt 注视 5 类）：
  旧链路：MediaPipe 478 点 → EAR 开合比 + 虹膜位置 → 双状态机
  新链路：YuNet 定位眼睛 → 裁剪 → YOLO 双模型 → 睁眼置信度 + 水平注视信号 → 原状态机
GazeBlinkDetector 四状态机（校准/选择/等待眨眼/确认）逻辑没变，只是信号来源换了；
校准保留：它吸收模型置信度在当前摄像头/光照下的偏置。
动作语义沿用 Yolo_model 实测约定：模型输出 look_left/look_right 在【未镜像帧】上
分别对应"左转/右转"；显示是镜像自拍视角，按 I 可反转。

⚠️ 铁律：YOLO 推理只吃【未镜像帧】，镜像只用于显示层。详见 vision/yolo_backend.py。

文件分三段：
  1) 判定逻辑（纯逻辑 + 状态机类，只吃数字信号，不碰摄像头也不画图）—— 可以单独跑测试
  2) 摄像头与画面（打开设备、建 YOLO 后端、画中文、画卡片）
  3) 主流程 main()：读帧 → 更新状态机 → 画 → 按键

============================ 调用关系总览 ============================
  │   （路径/摄像头/中文字体 来自共用层 src/common/，见 common/paths.py 地址簿）

  __main__  →  main()
  │
  ├─ 启动阶段（每个只执行一次）
  │   ├─ open_camera()  ←common层     打开摄像头，返回 cap（后面每帧从它 read）
  │   ├─ make_backend()               建 YOLO 后端：加载 eye_yolo26n + gaze5 权重
  │   ├─ GazeBlinkDetector(started)   建状态机；校准从此刻开始计时
  │   │    └─ self.reset(now)         把所有状态置位（按 C 重新校准走的也是它）
  │   ├─ WheelchairLink(...)           硬件链路（src/hardware/serial_link.py；默认模拟模式）
  │   │    └─ link.open()             串口打不开时自动退回模拟并记录原因
  │   └─ chinese_font(20) ←common层   预加载中文字体，缺字体时立刻报错
  │
  ├─ 每帧循环（★ 每帧都执行；顺序 看 → 判 → 连 → 报 → 控）
  │   ├─★ read_frame(cap, backend)
  │   │    ├─ cap.read()                             取【原始帧】（不镜像！）
  │   │    ├─ backend.analyze(frame)                 ← YOLO 推理（未镜像帧，铁律）
  │   │    ├─ cv2.flip(frame, 1)                     只为显示做镜像
  │   │    └─ 返回 (显示帧, now, EyeGazeResult)
  │   ├─★ draw_eye_boxes(显示帧, result, 宽)          画眼睛框（只为肉眼检查）
  │   ├─★ flow.update(now, result.gaze_score,        ★核心判定（纯逻辑，可单测）
  │   │                    result.open_conf)
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
    任意状态  ──(信号不可用)────────────────▶ 选择方向（未完成的候选作废）
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
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 本项目各脚本独立运行（没有包结构），把 src/ 加进搜索路径以引入硬件接口与 YOLO 后端
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hardware.serial_link import WheelchairLink  # noqa: E402  ← 在 sys.path 之后导入
from vision.yolo_backend import YoloEyeGaze, draw_eye_boxes  # noqa: E402  ← 在 sys.path 之后导入
from common.camera_utils import open_camera  # noqa: E402  ← 共用层（src/common/）
from common.draw_utils import chinese_font  # noqa: E402

# ---- 运行参数 ----
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0                            # 校准时长：这段时间请看屏幕正中间
CALIBRATION_MIN_SAMPLES = 10                         # 样本下限：帧率过低时样本太少，标尺不可靠
SIDE_THRESHOLD = 0.075                               # 注视信号偏移超过它才算"看左/看右"
SELECT_STABLE_SECONDS = 0.70                         # 同一方向要稳定停留这么久才成为候选
CLOSE_RATIO, REOPEN_RATIO = 0.78, 0.88               # 闭眼线 / 睁眼线（相对睁眼基线的百分比）
MIN_CLOSED_SECONDS, MAX_CLOSED_SECONDS = 0.04, 0.80  # 一次自然眨眼的闭眼时长范围
MIN_OPEN_SECONDS, REFRACTORY_SECONDS = 0.10, 0.30    # 需要稳定睁眼多久 / 两次计数的最小间隔
GAZE_SMOOTHING_FRAMES = 5                            # 注视信号平滑：最近几帧取平均
EAR_SMOOTHING_FRAMES = 3                             # 睁眼置信度平滑：最近几帧取平均
KEY_DEBOUNCE_SECONDS = 0.25                          # 两次按键响应的最小间隔：按住不放时不再连发

# ---- 硬件链路（协议与安全机制见 src/hardware/serial_link.py 和 README）----
ENABLE_HARDWARE = False      # ⚠️ False=纯屏幕模拟（默认）；True 时确认结果会真的发往串口
SERIAL_PORT = "COM3"         # Arduino 的串口号（设备管理器里查）；打不开自动退回模拟模式
FACE_LOST_STOP_SECONDS = 2.0 # 眼睛连续丢失这么久 → 主动发"停"（固件的 3 秒超时是兜底）

# ---- 路径与窗口 ----
ROOT = Path(__file__).resolve().parents[2]
WINDOW = "Gaze Select + Blink Confirm (YOLO) | Q quit | C recalibrate | I invert | screen demo only"


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
# ★ 与 MediaPipe 版相比，本段删掉了所有关键点几何（EAR / 虹膜位置），状态机改为
#   直接吃 YOLO 后端的两个连续信号（注视信号 / 睁眼置信度），状态转移逻辑逐字保留。


@dataclass(frozen=True)
class GazeBlinkStatus:
    """一帧的判定结果快照，供界面显示。

    调用关系：由 GazeBlinkDetector._status() 产出 → 被 main() 接住 → 交给 show_status()。
    它只是"数据袋子"，自身不含任何逻辑。
    字段说明：score 装注视信号（左 0 / 中 0.5 / 右 1）；ear 字段名沿用旧版，
    装睁眼置信度（0~1，闭眼走低）——对状态机两者是同一种连续信号。
    """

    state: str            # 四个状态里的哪一个：校准 / 选择方向 / 等待眨眼 / 确认成功
    message: str          # 屏幕上第三行显示的提示语
    score: float | None   # 本帧的注视信号（不可用时是 None）
    ear: float | None     # 本帧的睁眼置信度（不可用时是 None）
    active: str | None    # 该高亮哪张卡片：左转 / 前进 / 右转 / None
    calibration_progress: float | None = None  # 校准进度 0~1（仅校准状态有值）
    stability_progress: float | None = None    # 方向稳定进度 0~1（仅选择方向状态有值）


class GazeBlinkDetector:
    """视线选择 + 眨眼确认的四状态机。状态转移逻辑与 MediaPipe 版逐字一致。

    调用关系：main() 在启动时建一个；之后每帧调 update()；
    按 C 调 reset()，按 I 调 toggle_invert()。update() 内部按状态分派：
      校准没完成 → _update_calibration()
      选择方向 / 等待眨眼 → _update_selection()
      确认成功    → 纯计时，时间到就回选择方向
    每种情况的最后都由 _status() 打包成 GazeBlinkStatus 返回。

    用法：`GazeBlinkDetector(time.perf_counter())`，之后每帧调用
    `update(now, score_raw, ear_raw)` —— 两个信号来自 YOLO 后端
    （score_raw：左 0 / 中 0.5 / 右 1；ear_raw：睁眼置信度），不可用时传 None。
    它只依赖浮点数，不 import cv2，所以测试时直接喂合成数字就行。
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
        # 平滑用的滚动队列：最近几帧注视信号、最近几帧睁眼置信度
        self._gaze_history: deque[float] = deque(maxlen=gaze_smoothing_frames)
        self._ear_history: deque[float] = deque(maxlen=ear_smoothing_frames)
        self._invert = False  # 按 I 切换：左右方向是否反转
        self.reset(now)

    def reset(self, now: float) -> None:
        """回到校准起点。构造时和按 C 重新校准时都走这里，避免两处手抄不一致。"""
        self._calibration_started = now          # 校准阶段的起始时间
        self._gaze_values: list[float] = []      # 校准期间攒的注视信号
        self._ear_values: list[float] = []       # 校准期间攒的睁眼置信度
        # 三个"标尺"，校准完成后才有值：
        #   _center      = 正视时的注视信号
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

    def update(self, now: float, score_raw: float | None, ear_raw: float | None) -> GazeBlinkStatus:
        """吃一帧数据，返回本帧状态。两个信号任一不可用就按"没检测到"处理。"""
        if score_raw is None or ear_raw is None:
            # 信号不可用：不作任何判断，候选作废；但"校准"状态不打断（否则一转头就白校准了）
            self._message = "未检测到人脸：已暂停，重新正对摄像头"
            if self._state != "校准":
                self._state, self._stable_choice, self._pending = "选择方向", None, None
            return self._status(score=None, ear=None,
                                calibration_progress=None, stability_progress=None)

        # 两个数各做一次"最近几帧平均"，把抖动抹掉
        self._gaze_history.append(score_raw)
        score = sum(self._gaze_history) / len(self._gaze_history)
        self._ear_history.append(ear_raw)
        current_ear = sum(self._ear_history) / len(self._ear_history)

        # 进度条数据按"本帧进入时的状态"算：这样校准完成的那一帧也能看到 100% 满条
        calibration_progress = stability_progress = None
        if self._state == "校准":
            calibration_progress = min((now - self._calibration_started) / self._calibration_seconds, 1.0)
        elif self._state == "选择方向" and self._direction_since is not None:
            stability_progress = min((now - self._direction_since) / self._select_stable_seconds, 1.0)

        # ---- 按状态分派 ----
        # 【状态一：校准】攒够 3 秒且样本足够 → 算出三个"标尺"，转入选择方向
        if self._state == "校准":
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
        """校准阶段：攒样本，够 3 秒且样本足够后算出三个标尺。

        YOLO 版语义：_center 吸收注视分类器的个体/场景偏置；base 是当前场景下
        睁眼置信度的基线 —— 两者都随摄像头与光照漂移，所以校准保留。
        """
        self._gaze_values.append(score)
        self._ear_values.append(current_ear)
        elapsed = now - self._calibration_started
        self._message = f"校准中：请看正中间 {max(0, self._calibration_seconds - elapsed):.1f} 秒"
        if elapsed < self._calibration_seconds or len(self._gaze_values) < self._calibration_min_samples:
            return
        # _center = 正视时的注视信号；base = 睁眼置信度基线
        self._center = sum(self._gaze_values) / len(self._gaze_values)
        base = sum(self._ear_values) / len(self._ear_values)
        self._close_line, self._reopen_line = base * self._close_ratio, base * self._reopen_ratio
        self._state, self._message = "选择方向", "校准完成：看左转、前进或右转，稳定后等待眨眼"

    def _update_selection(self, now: float, score: float, current_ear: float) -> None:
        """选择方向 / 等待眨眼：先看视线选了哪个方向，再跑眨眼状态机。"""
        # offset = 相对正视基准的偏移量；按 I 反转时取负号
        offset = score - self._center
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




def make_backend() -> YoloEyeGaze:
    """建 YOLO 眼动后端：加载睁闭眼 + 注视两个权重（默认从 models/ 读）。

    调用关系：被 main() 在启动时调用一次（读模型慢，绝不能放进循环）；
    返回值一路传给 read_frame()，由它每帧调 analyze()。
    """
    return YoloEyeGaze()






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
        (16, 52, f"注视信号 {score_t}    睁眼置信 {ear_t}", 18, TEXT_VALUES),
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


def read_frame(cap, backend: YoloEyeGaze):
    """读一帧 → 【未镜像】送 YOLO → 翻转出显示帧，返回（显示帧, 当前时间, 推理结果）。

    调用关系：被 main() 每帧调用（循环第一步）；内部调用 cap.read()、
    backend.analyze()（真正推理的那次）和 cv2.flip。
    顺序是铁律：先 analyze 后 flip —— 模型必须吃未镜像帧，镜像只属于显示层。
    """
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("摄像头读取失败。")
    now = time.perf_counter()
    result = backend.analyze(frame)          # ← 推理只吃未镜像帧
    display = cv2.flip(frame, 1)             # ← 镜像只为显示（自拍视角）
    return display, now, result


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
    """程序入口：建好摄像头、YOLO 后端、硬件链路后，每帧走一遍 看→判→连→报→控。"""
    # ==================== 启动阶段（下面每个调用只执行一次）====================
    cap = open_camera(CAMERA_INDEX, WIDTH, HEIGHT)                          # ① 打开摄像头
    backend = make_backend()                     # ② 建 YOLO 后端
    started = time.perf_counter()                # 只用来算 FPS
    flow = GazeBlinkDetector(started)            # ③ 建状态机；校准从此刻开始计时
    link = WheelchairLink(SERIAL_PORT if ENABLE_HARDWARE else None)  # ④ 硬件链路（默认模拟）
    link.open()                                  #    串口打不开时自动退回模拟模式
    key_available_at = 0.0                       # 按键防抖：下次允许响应的时间点
    chinese_font(20)                             # ⑤ 提前加载字体，字体缺失时立刻报错
    frame_count = 0                              # 只用来算 FPS
    prev_state = None                            # 用来捕捉"确认成功"这一瞬间
    face_lost_since = None                       # 眼睛持续丢失的起点（超时就发"停"）

    print("视线选择演示已启动（YOLO 版）：请先点一下视频窗口让它获得焦点，然后 Q 退出、C 重新校准、I 反转左右。")
    if link.connected:
        print(f"硬件链路已连接：{link.mode}。确认成功的方向会真的发往固件（低电时被强制为停止）。")
    else:
        reason = f"（{link.last_error}）" if link.last_error else ""
        print(f"模拟模式{reason}：确认结果只显示在屏幕上，不会发出任何指令。")
    try:
        # ================= 每帧循环：看 → 判 → 连 → 报 → 控 =================
        while True:
            # ---- ① 看：取一帧，未镜像送 YOLO，拿回推理结果；再翻出显示帧 ----
            frame, now, result = read_frame(cap, backend)
            if result.face_found:
                draw_eye_boxes(frame, result, WIDTH)  # 眼睛框（只为肉眼检查，不参与判定）

            # ---- ② 判：视线 + 眨眼，四状态机全在 GazeBlinkDetector 里 ----
            status = flow.update(now, result.gaze_score, result.open_conf)

            # ---- ③ 连：硬件链路（心跳/收电池）+ 状态变化时下发意图 ----
            link_info = link.update(now)
            if prev_state == "等待眨眼" and status.state == "确认成功" and status.active:
                link.set_intent(status.active)   # "左转"→L / "前进"→F / "右转"→R；低电强制为"停"
            prev_state = status.state
            if not result.face_found:
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
