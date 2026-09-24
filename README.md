README有中文版本和英文版本，内容一样中文版本在上面英文版本请往下翻
The README has a Chinese version and an English version with the same content. The Chinese version is above; please scroll down for the English version.


## 中文版（Chinese）

# EyeWheelchairProject（眼控轮椅原型）

Eye-controlled wheelchair prototype — gaze + blink interaction, vision-only stage (no hardware control yet).

用普通 USB 摄像头识别"眼睛在看哪个方向"和"有没有眨眼"，先把**眼控交互**这条路走通。

> ⚠️ **当前阶段一切只输出到屏幕**：所有程序都不会向 Arduino、电机驱动板或轮椅发送任何指令，
> 也不包含任何硬件控制代码。等识别链路的正确性能用测试和实测证明之后，才会考虑接串口。

---

## 交互设计

最终要做的是"注视选方向 + 眨眼当确认键"，替代摇杆：

1. **校准（3 秒）**：睁眼看屏幕正中间，程序量出你自己的"睁眼基线"，并派生出两条判定线（闭眼线 / 睁眼线）。
   每个人的眼型、眼镜、坐姿距离都不同，所以阈值必须由本人现场标定，不能写死。
2. **选方向**：视线移到屏幕左 / 中 / 右（或左转 / 前进 / 右转），同一方向稳定停留约 0.7 秒才成为候选。
3. **眨眼确认**：一次自然眨眼（闭眼约 0.04~0.8 秒）才算数；闭得太短当噪声忽略，闭得太久（眯眼、闭目休息）不算。
   两次计数之间留 0.3 秒不应期，避免一次眨眼被数成两三次。

为什么用两个维度（视线 + 眨眼）：视线负责"选"，眨眼负责"确认"，避免视线一扫过就把指令发出去。

## 目录结构

```
EyeWheelchairProject/
├─ src/
│  ├─ camera/
│  │  └─ camera_preview.py            第 1 周：摄像头基线（预览 / 录像 / 截图）
│  ├─ vision/
│  │  └─ landmarks_preview.py         第 2 周：人脸 478 点 + 手部 21 点可视化
│  └─ interaction/
│     ├─ blink_preview.py             第 3 步：眨眼校准与计数（已重构，含测试）
│     ├─ gaze_direction_preview.py    第 4 步：视线方向选择（左 / 中 / 右）
│     └─ gaze_blink_confirm_demo.py   第 5 步：视线选方向 + 眨眼确认（完整演示）
├─ tests/                             pytest：眨眼判定行为测试 + 无摄像头冒烟测试
├─ models/                            MediaPipe 模型文件（face / hand landmarker，已入库）
├─ data/                              录制素材（raw_videos / snapshots）
├─ requirements.md                    依赖清单
└─ README.md
```

五个脚本**互不 import**，各自独立可运行（按周推进的开发顺序保留下来，方便逐步验证）。

## 代码组织

每个脚本内部统一分三段，用注释横幅隔开：

1. **判定逻辑**：只吃关键点或数字，不碰摄像头、不画图 —— 可以脱离硬件单独测试
2. **摄像头与画面**：打开设备、构建识别器、绘制文字与图形
3. **主流程 `main()`**：读帧 → 更新状态 → 画 → 按键

另外：

- 每个文件**开头的注释是一张调用关系图**（谁调用谁、哪些每帧调用、哪些只执行一次），
  每个函数的文档串里写明"被谁调用、内部调用谁"，可以按图读代码。
- 状态机（第 3 步的眨眼计数、第 4 步的方向选择、第 5 步的"选方向 + 眨眼确认"）
  各自封装成一个类，状态存在对象里，对外只暴露 `update(now, ...)` 与 `reset(now)`。
- 这样组织的目的：**阈值与判定逻辑能脱离摄像头被自动测试** —— 改动前后各跑一次 `pytest`，
  就知道有没有破坏原有行为，不必开摄像头靠肉眼验证。
- 第 2、3、4、5 步四个脚本已按这套办法整理完毕；`camera_preview.py` 是另一版重构，
  尚未按这套整理，详见"已知问题"第 2 条。

## 环境准备

需要 Python 3.10 以上（开发环境是 3.13 / 3.14）。

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install opencv-python mediapipe numpy pillow pyserial pytest
```

在 Git Bash 里把 `venv\Scripts\python.exe` 写成 `venv/Scripts/python.exe`，下文同理。

## 运行

```bash
venv\Scripts\python.exe src\interaction\blink_preview.py
```

| 脚本 | 按键 |
|---|---|
| `src/camera/camera_preview.py` | `Q` 退出 · `R` 开始/停止录像 · `S` 截图 |
| `src/vision/landmarks_preview.py` | `Q` 退出 · `S` 保存带关键点的截图 |
| `src/interaction/blink_preview.py` | `Q` 退出 · `C` 重新校准 |
| `src/interaction/gaze_direction_preview.py` | `Q` 退出 · `C` 重新校准 · `I` 反转左右 |
| `src/interaction/gaze_blink_confirm_demo.py` | `Q` 退出 · `C` 重新校准 · `I` 反转左右 |

两个使用前提：

- **先点一下视频窗口再按键**——OpenCV 的按键来自窗口焦点，焦点在终端上时按 Q 是没有反应的。
- 摄像头打不开时：关掉占用摄像头的软件（会议、相机 App），或把脚本顶部的 `CAMERA_INDEX` 从 `0` 改成 `1`。

## 测试

```bash
venv\Scripts\python.exe -m pytest -q
```

约 1 秒跑完，覆盖眨眼的校准、正常计数、过短/过长忽略、不应期、人脸丢失重置等行为，另有一个加载真实模型的冒烟测试。
测试全部使用合成数据，**不需要摄像头**；`data/raw_videos` 里没有可用录像时会自动跳过视频那条。

改代码的习惯：**改动前后各跑一次 `pytest`**。它会把"这次改动有没有破坏原有行为"直接告诉你，不用开摄像头靠肉眼验证。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第 1 周 | 摄像头基线（预览 / 录像 / 截图） | ⚠️ 重构版待实机完整验证 |
| 第 2 周 | MediaPipe 人脸与手部关键点可视化 | ✅ 已重构（初步测试通过） |
| 第 3 步 | 眨眼校准与计数（重构 + 单元测试） | ✅ 已实测通过 |
| 第 4 步 | 视线方向选择（左 / 中 / 右） | ✅ 已重构（初步测试通过） |
| 第 5 步 | 视线选方向 + 眨眼确认（屏幕演示） | ✅ 已重构（初步测试通过） |
| 代码整理 | 四个脚本统一为三段式结构 + 调用关系注释，判定逻辑可单测 | ✅ |
| 下一步 | 串口输出（Arduino / 电机驱动） | ⬜ 未开始，需先确定指令协议与急停方案 |

## 已知问题与注意事项

1. **退出行为（五个脚本已统一处理）**：MediaPipe 1.0.1 在 Windows 上销毁识别器要卡约 42 秒（实测）。
   涉及 MediaPipe 的四个脚本现在都不再调用 `close()`，改为在检测器构建函数里保留一个长期引用（保活），
   让进程退出时由系统一次性回收 —— 按 Q 或点窗口 ✕ 之后**1 秒内**结束。
   `camera_preview.py` 不使用 MediaPipe，本来就没有这个问题。
2. **`camera_preview.py` 尚未实机完整验证**：它由另一版重构完成（不是本次统一套路），
   使用前请先跑一遍预览 / 录像 / 截图，确认与旧版行为一致。
3. **依赖未锁版本**：`requirements.md` 里没有写版本号，换机器或升级可能踩到 API 变化。
   当前开发环境实测为 mediapipe 1.0.1 + opencv 5.0.0，交付前建议锁定版本。
4. **左右镜像约定**：交互类脚本对画面做了水平镜像（`cv2.flip`），让屏幕里的"左/右"与使用者的体感一致；
   `camera_preview.py` 不做镜像，所以它录下来的视频是"对面看你"的视角，与截图视角相反。
   如果实机上发现左右判断相反，按 `I` 反转。
5. **窗口标题**：Windows 版 OpenCV 对中文窗口标题支持不好（会显示成乱码），
   所以五个脚本的窗口标题一律用英文；窗口内的中文状态文字由 PIL + 系统字体渲染，不受影响。
6. **中文字体依赖**：三个需要显示中文的交互脚本按 `msyh.ttc → simhei.ttf → arial.ttf` 顺序探测系统字体，
   并缓存已加载的字体（不再每帧重复读文件）；三者都不存在才会报错。
   `camera_preview.py` 与 `landmarks_preview.py` 只用英文，不依赖中文字体。
7. `data/raw_videos/` 里 2026-09-16 的录像文件都是 0 字节空壳（当时录屏未写入成功），已从版本库中删除。

### 有意保留的行为边界

下面三条是系统当前的真实行为边界，属于"已知、暂时不动"——改动它们需要单独评估
（本次重构的验收标准是"行为不变"，所以一处都没有调）：

- 一直盯着屏幕正中间**不会**产生候选：候选的计时只在**方向发生变化**时开始，而初始方向本来就是"中"。
- 进入"等待眨眼"后把视线移开**不会取消**已选中的方向，必须眨一次眼（或人脸丢失）才能退出该状态。
- 在 30fps 下，只闭 1 帧也会被 3 帧平滑拉长成约 0.10 秒的闭眼，因此可能被计为一次眨眼
  （`MIN_CLOSED_SECONDS = 0.04` 实际不触发，真正抗噪的是平滑本身）。

## 后续计划

1. **实机完整验证 `camera_preview.py`**：重构版还没完整跑过，用之前先测预览 / 录像 / 截图三项。
2. **补常驻测试**：目前 `tests/` 只覆盖第 3 步（眨眼）与一个加载真实模型的冒烟测试；
   第 4、5 步的行为验证目前是一次性脚本，还没落成常驻用例。
3. **抽公共部分**：打开摄像头、构建检测器、中文绘制在五个脚本里各有一份拷贝，改一处要同步五处，
   是下一个该消除的重复。
4. 接串口控制前先定三件事：**指令协议**（左转/前进/右转/停的编码）、**急停**（任何异常或超时立即停）、
   **输出层与控制层分离**（识别只产出意图，发送动作单独一层，便于测试和回放）。


## English Version
# EyeWheelchairProject (Eye-Controlled Wheelchair Prototype)

Eye-controlled wheelchair prototype — gaze + blink interaction, vision-only stage (no hardware control yet).

Using an ordinary USB camera to recognize "which direction the eyes are looking" and "whether there is a blink," the goal is to first get the **eye-control interaction** path working.

> ⚠️ **At the current stage, everything is output only to the screen**: none of the programs send any commands to Arduino, motor driver boards, or the wheelchair, and no hardware control code is included. Only after the correctness of the recognition chain can be proven through tests and real-world trials will serial communication be considered.

---

## Interaction Design

The final goal is "gaze to select direction + blink as confirmation key," replacing the joystick:

1. **Calibration (3 seconds)**: Keep your eyes open and look at the center of the screen; the program measures your own "open-eye baseline" and derives two decision lines (closed-eye line / open-eye line). Everyone's eye shape, glasses, and sitting distance are different, so the thresholds must be calibrated on-site by the user and cannot be hard-coded.
2. **Select direction**: Move your gaze to the left / center / right of the screen (or left turn / forward / right turn). The same direction must remain stable for about 0.7 seconds before becoming a candidate.
3. **Blink confirmation**: Only a natural blink (eyes closed for about 0.04–0.8 seconds) counts; too short is ignored as noise, too long (squinting, eyes closed resting) does not count. A 0.3-second refractory period is left between two counts to avoid a single blink being counted two or three times.

Why use two dimensions (gaze + blink): gaze is responsible for "selecting," blink is responsible for "confirming," preventing an instruction from being sent just because the gaze swept across.

## Directory Structure

```
EyeWheelchairProject/
├─ src/
│  ├─ camera/
│  │  └─ camera_preview.py            Week 1: Camera baseline (preview / record / snapshot)
│  ├─ vision/
│  │  └─ landmarks_preview.py         Week 2: Face 478 points + hand 21 points visualization
│  └─ interaction/
│     ├─ blink_preview.py             Step 3: Blink calibration and counting (refactored, with tests)
│     ├─ gaze_direction_preview.py    Step 4: Gaze direction selection (left / center / right)
│     └─ gaze_blink_confirm_demo.py   Step 5: Gaze selects direction + blink confirmation (full demo)
├─ tests/                             pytest: blink decision behavior tests + no-camera smoke tests
├─ models/                            MediaPipe model files (face / hand landmarker, checked in)
├─ data/                              Recorded materials (raw_videos / snapshots)
├─ requirements.md                    Dependency list
└─ README.md
```

The five scripts **do not import each other** and can each run independently (the weekly development order is preserved for gradual verification).

## Code Organization

Each script is internally split into three sections separated by comment banners:

1. **Decision logic**: takes only landmarks or numbers — no camera, no drawing. It can be tested without any hardware.
2. **Camera and rendering**: opening the device, constructing the detectors, drawing text and shapes.
3. **Main flow `main()`**: read a frame → update state → draw → handle keys.

Also:

- The comment at the **top of each file is a call graph** (who calls whom, which calls happen every frame, which happen only once), and every function's docstring states "who calls me / whom I call" — you can read the code along that map.
- Each state machine (blink counting in step 3, direction selection in step 4, "select + blink to confirm" in step 5) is wrapped in a class: the state lives on the object, and only `update(now, ...)` and `reset(now)` are exposed.
- The purpose of this layout: **thresholds and decision logic can be tested without a camera** — run `pytest` before and after a change and you know whether existing behaviour broke, instead of opening the camera and verifying by eye.
- Steps 2, 3, 4 and 5 have been reorganised this way; `camera_preview.py` is a separate refactor that has not been reorganised — see Known Issues item 2.

## Environment Setup

Requires Python 3.10 or above (development environment is 3.13 / 3.14).

```bash
python -m venv venv
venv\Scripts\python.exe -m pip install opencv-python mediapipe numpy pillow pyserial pytest
```

In Git Bash, write `venv\Scripts\python.exe` as `venv/Scripts/python.exe`; the same applies below.

## Running

```bash
venv\Scripts\python.exe src\interaction\blink_preview.py
```

| Script | Keys |
|---|---|
| `src/camera/camera_preview.py` | `Q` quit · `R` start/stop recording · `S` snapshot |
| `src/vision/landmarks_preview.py` | `Q` quit · `S` save snapshot with landmarks |
| `src/interaction/blink_preview.py` | `Q` quit · `C` recalibrate |
| `src/interaction/gaze_direction_preview.py` | `Q` quit · `C` recalibrate · `I` invert left/right |
| `src/interaction/gaze_blink_confirm_demo.py` | `Q` quit · `C` recalibrate · `I` invert left/right |

Two prerequisites:

- **Click the video window first, then press keys** — OpenCV key events come from window focus; when focus is on the terminal, pressing Q has no effect.
- If the camera cannot open: close software occupying the camera (meetings, camera apps), or change `CAMERA_INDEX` at the top of the script from `0` to `1`.

## Testing

```bash
venv\Scripts\python.exe -m pytest -q
```

Runs in about 1 second, covering blink calibration, normal counting, too-short/too-long ignoring, refractory period, face-loss reset, and other behaviors, plus a smoke test that loads the real model. All tests use synthetic data and **do not require a camera**; the video test is automatically skipped when there is no usable recording in `data/raw_videos`.

Habit when changing code: **run `pytest` once before and after each change**. It will directly tell you "whether this change broke existing behavior," without needing to open the camera and verify by eye.

## Progress

| Stage | Content | Status |
|---|---|---|
| Week 1 | Camera baseline (preview / record / snapshot) | ⚠️ Refactored version awaits full on-device verification |
| Week 2 | MediaPipe face and hand landmark visualization | ✅ Refactored (preliminary tests pass) |
| Step 3 | Blink calibration and counting (refactor + unit tests) | ✅ Verified on the real device |
| Step 4 | Gaze direction selection (left / center / right) | ✅ Refactored (preliminary tests pass) |
| Step 5 | Gaze selects direction + blink confirmation (screen demo) | ✅ Refactored (preliminary tests pass) |
| Code tidy-up | Four scripts unified into the three-section layout with call-graph comments; decision logic unit-testable | ✅ |
| Next | Serial output (Arduino / motor driver) | ⬜ Not started; command protocol and emergency stop must be defined first |

## Known Issues and Notes

1. **Exit behaviour (handled in all five scripts)**: MediaPipe 1.0.1 takes about 42 seconds to destroy a recognizer on Windows (measured). The four MediaPipe-based scripts no longer call `close()`; instead the detector factory keeps a long-lived reference (keep-alive), so everything is reclaimed by the system at exit — pressing Q or clicking the window ✕ now ends the process **within 1 second**. `camera_preview.py` does not use MediaPipe and never had this problem.
2. **`camera_preview.py` is not fully verified on the real machine**: it was refactored in a separate pass (not the unified pattern). Run preview / record / snapshot once before relying on it, and confirm the behaviour matches the old version.
3. **Dependencies are not version-locked**: `requirements.md` lists no version numbers, so another machine or an upgrade may hit API changes. The development environment measures mediapipe 1.0.1 + opencv 5.0.0; lock the versions before delivery.
4. **Left-right mirror convention**: interaction scripts mirror the image horizontally (`cv2.flip`) so that "left/right" on the screen matches the user's felt left/right; `camera_preview.py` does not mirror, so recorded video is from the "person facing you" perspective — the opposite of the snapshots. If left/right turns out reversed on the real device, press `I` to invert.
5. **Window title**: Windows OpenCV renders Chinese window titles badly (garbled text), so all five scripts use English titles; Chinese text inside the window is drawn by PIL + system fonts and is unaffected.
6. **Chinese font dependency**: the three interaction scripts that display Chinese probe system fonts in the order `msyh.ttc → simhei.ttf → arial.ttf` and cache the loaded font (no more re-reading the file every frame); an error is raised only if none of the three exists. `camera_preview.py` and `landmarks_preview.py` use English only and do not depend on fonts.
7. The 2026-09-16 recording files in `data/raw_videos/` are all 0-byte empty shells (the recording never wrote any data) and have been deleted from the repository.

### Behaviour boundaries we intentionally keep

These three are the current, real boundaries of the system and are "known, deliberately untouched" — changing any of them needs a separate decision (the acceptance bar for this refactor was *behaviour unchanged*, so nothing was tuned):

- Staring at the exact centre of the screen **never** produces a candidate: the stability timer starts only when the direction **changes**, and the initial direction is already "centre".
- After entering "waiting for blink", looking away does **not** cancel the selected direction — only a blink (or losing the face) leaves that state.
- At 30 fps a single closed frame is stretched by the 3-frame smoothing into a ~0.10 s closure and can therefore count as a blink (`MIN_CLOSED_SECONDS = 0.04` effectively never fires; the smoothing itself is what filters noise).

## Future Plans

1. **Fully verify `camera_preview.py` on the real machine**: the refactored version has never been run end to end — test preview / record / snapshot first.
2. **Add permanent tests**: `tests/` currently covers step 3 (blink) plus one smoke test that loads the real model; the step 4 and 5 behaviour checks are still one-off scripts.
3. **Extract the common parts**: opening the camera, building detectors and drawing Chinese text are copied in all five scripts, so one change means five edits — the next duplication to remove.
4. Before connecting serial control, define three things first: **command protocol** (encoding for left turn / forward / right turn / stop), **emergency stop** (stop immediately on any exception or timeout), and **separation of output layer and control layer** (recognition only produces intent; sending actions is a separate layer for easier testing and playback).