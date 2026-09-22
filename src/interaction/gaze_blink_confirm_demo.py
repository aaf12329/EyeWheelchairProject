"""第 5 步：视线选方向 + 眨眼确认（纯屏幕演示）。

流程：看正中校准 3 秒 → 稳定注视 左转/前进/右转 → 自然眨眼确认。
确认结果仅显示在屏幕上，不会向 Arduino、驱动板或轮椅发送任何指令。
按 Q 退出；C 重新校准；I 反转左右方向。

============================ 调用关系总览 ============================

  __main__  →  main()
  │
  ├─ 启动阶段（每个只执行一次）
  │   ├─ cv2.VideoCapture(...)     打开摄像头（这段直接写在 main 里，没有抽成函数）
  │   ├─ make_detector()           建识别器：加载 models/face_landmarker.task
  │   └─ def reset(now)            ← 注意：这是"写在 main 里面的小函数"（嵌套函数）。
  │                                  它自己不执行，只有按 C 键时才被调用，
  │                                  作用是把所有状态拨回"校准起点"
  │
  ├─ 每帧循环（★ 每帧都执行；顺序 看 → 量 → 判 → 报 → 控）
  │   ├─★ cap.read() / cv2.flip / cv2.cvtColor / mp.Image    取一帧并预处理
  │   ├─★ detector.detect_for_video(图, 毫秒时间戳)           ← MediaPipe 推理
  │   ├─★ gaze_score(lm)               虹膜在眼睛里的水平位置 → 看的是左 / 中 / 右
  │   ├─★ ear(lm, LEFT_EYE / RIGHT_EYE) → distance()         眼睛开合值 → 有没有眨眼
  │   ├─★ 状态机 if / elif（本文件的核心，状态图见下）
  │   ├─★ chinese_overlay(frame, 四行文字)   左上角四行中文
  │   ├─★ cards(frame, active)               底部三张卡片：左转 / 前进 / 右转
  │   └─★ cv2.waitKey(1)                     读按键：Q 退出、C → reset(now)、I 反转左右
  │
  └─ finally（无论怎么退出都执行）
      ├─ detector.close()
      ├─ cap.release()
      └─ cv2.destroyAllWindows()

四个状态怎么互相转移（这就是"状态机"）：

    校准      ──(满 3 秒 且 样本≥10 个)──────▶ 选择方向
    选择方向  ──(同一方向稳定停留 0.70 秒)───▶ 等待眨眼（pending = 选中的方向）
    等待眨眼  ──(检测到一次自然眨眼)─────────▶ 确认成功（屏幕显示"已确认"）
    确认成功  ──(1.6 秒后自动)──────────────▶ 选择方向（可以接着选下一项）
    任意状态  ──(人脸丢失)──────────────────▶ 选择方向（未完成的候选作废）
    任意状态  ──(按 C)──────────────────────▶ 校准（reset(now) 把一切拨回起点）

关键一点：**只有在"等待眨眼"这个状态下，一次眨眼才算确认**；在别的状态眨眼没有效果。
"""
from collections import deque
from pathlib import Path
import math
import time

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- 运行参数 ----
CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
CALIBRATION_SECONDS = 3.0                            # 校准时长：这段时间请看屏幕正中间
SIDE_THRESHOLD = 0.075                               # 虹膜偏移超过它才算"看左/看右"
SELECT_STABLE_SECONDS = 0.70                         # 同一方向要稳定停留这么久才成为候选
CLOSE_RATIO, REOPEN_RATIO = 0.78, 0.88               # 闭眼线 / 睁眼线（相对睁眼基线的百分比）
MIN_CLOSED_SECONDS, MAX_CLOSED_SECONDS = 0.04, 0.80  # 一次自然眨眼的闭眼时长范围
MIN_OPEN_SECONDS, REFRACTORY_SECONDS = 0.10, 0.30    # 需要稳定睁眼多久 / 两次计数的最小间隔

# ---- 路径与窗口 ----
ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "face_landmarker.task"
FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
WINDOW = "视线选择 + 眨眼确认｜Q 退出｜C 校准｜I 反转左右｜仅模拟演示"
# ---- 用到的关键点编号（都来自 Face Landmarker 的 478 个点）----
LEFT_IRIS, RIGHT_IRIS = [468, 469, 470, 471, 472], [473, 474, 475, 476, 477]  # 两只眼的虹膜 5 点
LEFT_CORNERS, RIGHT_CORNERS = (33, 133), (362, 263)                           # 两只眼的内外眼角（当尺子）
LEFT_EYE, RIGHT_EYE = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]  # 眼睛轮廓 6 点


def make_detector():
    """建人脸识别器：加载 models/face_landmarker.task，返回识别器对象。

    调用关系：被 main() 在启动时调用一次（读模型慢，绝不能放进循环）；
    内部调用 MediaPipe 的 FaceLandmarker.create_from_options()。
    返回值留给每帧的 detector.detect_for_video() 使用。
    """
    if not MODEL.exists():
        raise FileNotFoundError(f"缺少模型：{MODEL}")
    vision = mp.tasks.vision
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def distance(a, b):
    """两个关键点之间的平面距离（忽略 z）。只被下面的 ear() 调用。"""
    return math.hypot(a.x - b.x, a.y - b.y)


def ear(landmarks, eye):
    """单只眼睛的开合比：上下眼睑平均距离 ÷ 眼角宽度。闭眼时趋近 0。

    调用关系：被 main() 每帧调用两次（左眼、右眼），两个值取平均；内部调用 distance()。
    注意：它和 blink_preview.py 里的 eye_aspect_ratio() 是同一个公式的两份实现。
    """
    p0, p1, p2, p3, p4, p5 = [landmarks[i] for i in eye]
    horizontal = distance(p0, p3)
    return 0.0 if horizontal < 1e-6 else (distance(p1, p5) + distance(p2, p4)) / (2 * horizontal)


def gaze_score(landmarks):
    """虹膜在眼睛水平方向上的平均归一化位置：0 = 贴左眼角，1 = 贴右眼角。

    调用关系：被 main() 每帧调用一次；它决定了"左转 / 前进 / 右转"往哪边选。
    内部只用关键点的 x 坐标，不画图、也不做判断（判断在 main 的状态机里）。
    """
    ratios = []
    for iris, corners in ((LEFT_IRIS, LEFT_CORNERS), (RIGHT_IRIS, RIGHT_CORNERS)):
        iris_x = sum(landmarks[i].x for i in iris) / len(iris)
        a, b = landmarks[corners[0]].x, landmarks[corners[1]].x
        if abs(a - b) > 1e-5:
            ratios.append((iris_x - min(a, b)) / abs(a - b))
    return sum(ratios) / len(ratios) if ratios else None


def chinese_overlay(frame, lines):
    """把四行中文画到画面上（OpenCV 自带字体画不了中文，所以用 PIL + 系统字体）。

    调用关系：被 main() 每帧调用一次；返回一张"画好字的新图"，
    不修改传进来的 frame——所以调用处必须写成 frame = chinese_overlay(...)。
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    small, large = ImageFont.truetype(str(FONT), 20), ImageFont.truetype(str(FONT), 29)
    positions = [(14, 10, small, (0, 255, 0)), (14, 37, large, (255, 255, 0)),
                 (14, 76, small, (255, 255, 255)), (14, 103, small, (255, 210, 80))]
    for text, (x, y, font, color) in zip(lines, positions):
        draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cards(frame, active):
    """画底部三张卡片：左转 / 前进 / 右转；active 那一张用实心绿色高亮。

    调用关系：被 main() 每帧调用一次。active 是 main 根据状态机算出来的：
    正在等待眨眼或刚确认成功时高亮 pending，否则高亮已经稳定的候选方向。
    """
    options = [("左转", 35, 390, 280, 515), ("前进", 340, 390, 585, 515), ("右转", 645, 390, 890, 515)]
    for label, x1, y1, x2, y2 in options:
        color = (0, 170, 0) if label == active else (70, 70, 70)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1 if label == active else 2)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(image).text((x1 + 64, y1 + 38), label, font=ImageFont.truetype(str(FONT), 36), fill=(255, 255, 255))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return frame


def main():
    """程序入口：建好摄像头和识别器后，每帧走一遍 看→量→判→报→控。

    调用关系：由文件最后一行 __main__ 调用；内部调用 make_detector()、gaze_score()、
    ear()、distance()、chinese_overlay()、cards()，以及写在它里面的嵌套函数 reset()。
    """
    # ==================== 启动阶段（下面每个调用只执行一次）====================
    # 打开摄像头（和 blink_preview.py 里的 open_camera() 是同一段逻辑，这里没有抽成函数）
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened(): raise RuntimeError("摄像头无法打开。")
    # 建识别器
    detector = make_detector()
    # 两个时间基准：started 用来给 MediaPipe 算毫秒时间戳；calibration_started 用来算校准倒计时
    started = calibration_started = time.perf_counter()
    # 校准期间攒的样本：gaze_values 是虹膜位置，ear_values 是眼睛开合值
    gaze_values, ear_values = [], []
    # 三个"标尺"，校准完成后才有值：
    #   center       = 正视时虹膜该在的位置
    #   close_line   = 闭眼判定线（低于它算闭眼）
    #   reopen_line  = 睁眼判定线（高于它算睁开）
    center = close_line = reopen_line = None
    # 平滑用的滚动队列：最近 5 帧虹膜位置、最近 3 帧开合值
    gaze_history, ear_history = deque(maxlen=5), deque(maxlen=3)
    # ---- 状态机的全部状态（本程序的"记忆"）----
    #   state        : 现在是四个状态里的哪一个
    #   raw_choice   : 这一帧视线落在哪个方向（会抖）
    #   stable_choice: 稳定够久、已经确认过的方向（用来高亮卡片）
    #   pending      : 已经选中、正等着眨眼确认的方向
    state, raw_choice, stable_choice, pending = "校准", "前进", None, None
    # 四个时间戳（方向稳定的起点 / 睁眼起点 / 闭眼起点 / 确认成功时刻）+ 是否闭着眼
    direction_since = open_since = closed_since = confirm_started = None
    is_closed, invert, key_held = False, False, False
    last_blink = float("-inf")   # 上次计数时刻；负无穷让第一次眨眼不受不应期限制
    message = "请睁眼看屏幕正中间，正在校准"

    # ---- 写在 main 里面的小函数（嵌套函数）：按 C 键时调用它，回到校准起点 ----
    # nonlocal 是干什么的：下面这些名字都是 main 的局部变量，嵌套函数默认只能读不能改；
    # nonlocal 就是在声明"我要改的是外面 main 里的那个变量，不是新建一个同名的"。
    def reset(now):
        nonlocal calibration_started, gaze_values, ear_values, center, close_line, reopen_line
        nonlocal state, raw_choice, stable_choice, pending, direction_since, open_since, closed_since
        nonlocal is_closed, last_blink, confirm_started, message
        calibration_started = now; gaze_values, ear_values = [], []
        center = close_line = reopen_line = None
        gaze_history.clear(); ear_history.clear()
        state, raw_choice, stable_choice, pending = "校准", "前进", None, None
        direction_since = open_since = closed_since = confirm_started = None
        is_closed, last_blink = False, float("-inf")
        message = "请睁眼看屏幕正中间，正在校准"

    try:
        # ================= 每帧循环：看 → 量 → 判 → 报 → 控 =================
        while True:
            # ---- ① 看：取一帧、镜像、送 MediaPipe，拿回 478 个关键点 ----
            ok, frame = cap.read()
            if not ok: raise RuntimeError("摄像头读取失败。")
            frame = cv2.flip(frame, 1); now = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now-started)*1000))
            # ---- ② 量：本帧的两个数，先按"没数据"（None）处理，检测到人脸才填 ----
            score = current_ear = None    # score = 虹膜位置；current_ear = 眼睛开合值
            blink_confirmed = False       # 本帧是否完成了一次"可用的眨眼"
            if result.face_landmarks:
                lm = result.face_landmarks[0]
                score = gaze_score(lm)                                        # 眼睛在看哪边
                current_ear = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2    # 眼睛张多开
                # 两个数各做一次"最近几帧平均"，把抖动抹掉
                if score is not None: gaze_history.append(score); score = sum(gaze_history)/len(gaze_history)
                ear_history.append(current_ear); current_ear = sum(ear_history)/len(ear_history)
                # 把用到的关键点画成黄点，方便肉眼检查识别准不准
                h, w = frame.shape[:2]
                for idx in LEFT_IRIS + RIGHT_IRIS + LEFT_EYE + RIGHT_EYE:
                    p = lm[idx]; cv2.circle(frame, (int(p.x*w), int(p.y*h)), 2, (0,255,255), -1)

                # ---- ③ 判：状态机 ----
                # 【状态一：校准】攒够 3 秒且样本足够 → 算出三个"标尺"，转入选择方向
                if state == "校准" and score is not None:
                    gaze_values.append(score); ear_values.append(current_ear)
                    elapsed = now-calibration_started
                    message = f"校准中：请看正中间 {max(0, CALIBRATION_SECONDS-elapsed):.1f} 秒"
                    if elapsed >= CALIBRATION_SECONDS and len(gaze_values) >= 10:
                        # center = 正视时的虹膜位置；base = 睁眼时的开合值
                        center = sum(gaze_values)/len(gaze_values); base = sum(ear_values)/len(ear_values)
                        close_line, reopen_line = base*CLOSE_RATIO, base*REOPEN_RATIO
                        state, message = "选择方向", "校准完成：看左转、前进或右转，稳定后等待眨眼"

                # 【状态二、三：选择方向 / 等待眨眼】视线偏移决定候选方向，顺便跑眨眼状态机
                elif state in ("选择方向", "等待眨眼") and center is not None:
                    # offset = 相对正视基准的偏移量；按 I 反转时取负号
                    offset = score-center if score is not None else 0.0
                    if invert: offset = -offset
                    # 偏移量越过阈值就判成左/右，否则算"前进"（中）
                    new_choice = "左转" if offset < -SIDE_THRESHOLD else "右转" if offset > SIDE_THRESHOLD else "前进"
                    if state == "选择方向":
                        if new_choice != raw_choice:
                            # 方向变了 → 重新开始计时，候选清空
                            raw_choice, direction_since, stable_choice = new_choice, now, None
                        elif direction_since is not None and now-direction_since >= SELECT_STABLE_SECONDS:
                            # 同一方向稳定停留够久 → 定为候选，进入"等待眨眼"
                            stable_choice, pending, state = raw_choice, raw_choice, "等待眨眼"
                            message = f"已选择“{pending}”，请自然眨眼确认"

                    # 眨眼状态机：仅在“等待眨眼”时才把一次眨眼当确认。
                    # stable_open 三个条件：有睁眼起点、睁眼已满 0.1 秒、离上次计数已满 0.3 秒
                    stable_open = open_since is not None and now-open_since >= MIN_OPEN_SECONDS and now-last_blink >= REFRACTORY_SECONDS
                    if is_closed:
                        # 本来闭着 → 现在睁开了：量一下这次闭了多久
                        if current_ear >= reopen_line:
                            duration = now-closed_since if closed_since else 0.0
                            is_closed, closed_since, open_since = False, None, now
                            # 闭眼时长落在自然眨眼窗口内 → 记一次"可用的眨眼"
                            if MIN_CLOSED_SECONDS <= duration <= MAX_CLOSED_SECONDS:
                                last_blink, blink_confirmed = now, True
                        # 否则持续闭眼，等待睁开。
                    elif current_ear >= reopen_line:
                        # 睁着眼：记下"开始睁眼"的时刻（只在还没有的时候记一次）
                        if open_since is None: open_since = now
                    elif current_ear < close_line and stable_open:
                        # 掉到闭眼线以下，且刚才确实稳定睁着 → 进入闭眼候选
                        is_closed, closed_since, open_since = True, now, None
                    else:
                        # 落在两条判定线之间的灰色地带 → 睁眼计时作废
                        open_since = None

                    # 【状态三 → 四】只有"等待眨眼"期间的眨眼才被当作确认
                    if state == "等待眨眼" and blink_confirmed:
                        state, confirm_started = "确认成功", now
                        message = f"已确认“{pending}”——这是模拟指令，不会移动轮椅"
                # 【状态四：确认成功】停 1.6 秒让人看清结果，然后自动回到"选择方向"
                elif state == "确认成功" and confirm_started and now-confirm_started >= 1.6:
                    state, stable_choice, pending, direction_since = "选择方向", None, None, now
                    message = "请继续选择下一项方向"
            else:
                # 没人脸：不作任何判断，候选作废；但"校准"状态不打断（否则一转头就白校准了）
                message = "未检测到人脸：已暂停，重新正对摄像头"
                if state != "校准": state, stable_choice, pending = "选择方向", None, None

            # ---- ④ 报：铺黑底 + 四行中文 + 三张卡片，推给窗口 ----
            cv2.rectangle(frame, (0,0), (780,130), (0,0,0), -1)
            score_t = "--" if score is None else f"{score:.3f}"; ear_t = "--" if current_ear is None else f"{current_ear:.3f}"
            # 高亮哪张卡片：等待眨眼/确认成功时高亮 pending，其余时候高亮已稳定的候选
            active = pending if state in ("等待眨眼", "确认成功") else stable_choice
            frame = chinese_overlay(frame, [f"虹膜位置：{score_t}｜眼睛开合值：{ear_t}", f"当前状态：{state}", message, "Q 退出｜C 重新校准｜I 反转左右｜仅屏幕模拟，不控制硬件"])
            frame = cards(frame, active)
            cv2.imshow(WINDOW, frame)

            # ---- ⑤ 控：键盘。注意 key_held 这个防抖是"哪一帧没有按键就解锁"，
            #      长按某个键时系统会连发按键事件，它挡不住（已知缺陷，尚未修）----
            key = cv2.waitKey(1)
            if key == -1: key_held = False; continue
            if key_held: continue
            key_held = True; key &= 0xFF
            if key in (ord('q'),ord('Q')): break
            if key in (ord('c'),ord('C')): reset(now)   # 调 main 里那个嵌套函数，回到校准起点
            if key in (ord('i'),ord('I')):
                invert = not invert; message = "左右方向已反转" if invert else "左右方向已恢复"
    finally:
        # 收摊三件套。注意：detector.close() 在 mediapipe 1.0.1 + Windows 上要卡约 42 秒
        #（原因和绕开办法见 blink_preview.py 里的说明），本文件还没处理，所以按 Q 后进程会僵一会儿。
        detector.close(); cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
