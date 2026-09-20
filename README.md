README有中文版本和英文版本，内容一样中文版本在上面英文版本请往下翻
The README has a Chinese version and an English version with the same content. The Chinese version is above; please scroll down for the English version.


README(CH)中文版本README

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
| 第 1 周 | 摄像头基线（预览 / 录像 / 截图） | ✅ |
| 第 2 周 | MediaPipe 人脸与手部关键点可视化 | ✅ |
| 第 3 步 | 眨眼校准与计数（重构 + 单元测试） | ✅ |
| 第 4 步 | 视线方向选择（左 / 中 / 右） | ✅ |
| 第 5 步 | 视线选方向 + 眨眼确认（屏幕演示） | ✅ |
| 下一步 | 串口输出（Arduino / 电机驱动） | ⬜ 未开始，需先确定指令协议与急停方案 |

## 已知问题与注意事项

1. **退出卡顿（重要）**：MediaPipe 1.0.1 在 Windows 上销毁识别器要卡约 42 秒。
   `blink_preview.py` 已处理（保活引用 + 不调用 `close()`，退出不到 1 秒），
   **其余 4 个脚本尚未处理**，按 Q 之后进程会僵约 42 秒才真正结束。
2. **依赖未锁版本**：`requirements.md` 里没有写版本号，换机器或升级可能踩到 API 变化。
   当前开发环境实测为 mediapipe 1.0.1 + opencv 5.0.0，交付前建议锁定版本。
3. **左右镜像约定**：交互类脚本对画面做了水平镜像（`cv2.flip`），让屏幕里的"左/右"与使用者的体感一致；
   `camera_preview.py` 不做镜像，所以它录下来的视频是"对面看你"的视角，与截图视角相反。
   如果实机上发现左右判断相反，按 `I` 反转；两个脚本的命令行参数或阈值符号是排查的第二处。
4. **窗口标题**：Windows 版 OpenCV 对中文窗口标题支持不好（会显示成乱码），
   `blink_preview.py` 已改成英文标题，窗口内的中文由 PIL + 系统字体渲染，不受影响。
5. **中文字体依赖**：脚本按 `msyh.ttc → simhei.ttf → arial.ttf` 顺序探测系统字体，都没有才会报错。
6. `data/raw_videos/` 里 2026-09-16 的录像文件都是 0 字节空壳（当时录屏未写入成功），已从版本库中删除。

## 后续计划

1. 把 `gaze_direction_preview.py` 与 `gaze_blink_confirm_demo.py` 按 `blink_preview.py` 同样的办法收拾一遍
   （同文件内拆出可测试的判定类、补测试、修退出卡顿与按键去抖）。
2. 抽公共部分（打开摄像头、中文绘制、检测器构建），减少各脚本之间的复制粘贴。
3. 接串口控制前先定三件事：**指令协议**（左转/前进/右转/停的编码）、**急停**（任何异常或超时立即停）、
   **输出层与控制层分离**（识别只产出意图，发送动作单独一层，便于测试和回放）。


README(EN):
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
| Week 1 | Camera baseline (preview / record / snapshot) | ✅ |
| Week 2 | MediaPipe face and hand landmark visualization | ✅ |
| Step 3 | Blink calibration and counting (refactor + unit tests) | ✅ |
| Step 4 | Gaze direction selection (left / center / right) | ✅ |
| Step 5 | Gaze selects direction + blink confirmation (screen demo) | ✅ |
| Next | Serial output (Arduino / motor driver) | ⬜ Not started; command protocol and emergency stop must be defined first |

## Known Issues and Notes

1. **Exit lag (important)**: MediaPipe 1.0.1 takes about 42 seconds to destroy the recognizer on Windows. `blink_preview.py` has already handled this (keep-alive reference + do not call `close()`, exits in under 1 second); **the other 4 scripts are not yet handled**, and the process will hang for about 42 seconds after pressing Q before actually ending.
2. **Dependencies not version-locked**: `requirements.md` does not specify version numbers; moving to another machine or upgrading may hit API changes. The current development environment is tested with mediapipe 1.0.1 + opencv 5.0.0; it is recommended to lock versions before delivery.
3. **Left-right mirror convention**: Interaction scripts horizontally mirror the image (`cv2.flip`) so that "left/right" on the screen matches the user's physical sense; `camera_preview.py` does not mirror, so the video it records is from the "opposite person looking at you" perspective, opposite to the snapshot perspective. If left/right judgment is found to be reversed on the real device, press `I` to invert; the command-line arguments or threshold signs of the two scripts are the second place to check.
4. **Window title**: Windows OpenCV does not support Chinese window titles well (they show as garbled text). `blink_preview.py` has been changed to an English title; Chinese inside the window is rendered by PIL + system fonts and is not affected.
5. **Chinese font dependency**: Scripts probe system fonts in the order `msyh.ttc → simhei.ttf → arial.ttf`; only if none are found will an error be reported.
6. The 2026-09-16 recording files in `data/raw_videos/` are all 0-byte empty shells (screen recording did not write successfully at the time) and have been deleted from the version library.

## Future Plans

1. Clean up `gaze_direction_preview.py` and `gaze_blink_confirm_demo.py` in the same way as `blink_preview.py` (extract testable decision classes within the same file, add tests, fix exit lag and key debouncing).
2. Extract common parts (opening camera, Chinese drawing, detector construction) to reduce copy-paste between scripts.
3. Before connecting serial control, define three things first: **command protocol** (encoding for left turn / forward / right turn / stop), **emergency stop** (stop immediately on any exception or timeout), and **separation of output layer and control layer** (recognition only produces intent; sending actions is a separate layer for easier testing and playback).