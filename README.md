README有中文版本和英文版本，内容一样中文版本在上面英文版本请往下翻
The README has a Chinese version and an English version with the same content. The Chinese version is above; please scroll down for the English version.


## 中文版（Chinese）

# EyeWheelchairProject（眼控轮椅原型）

Eye-controlled wheelchair prototype — gaze + blink interaction. Python side runs as a pure screen simulation by default; the Arduino firmware and serial layer are written but not yet verified on the real chair.

用普通 USB 摄像头识别"眼睛在看哪个方向"和"有没有眨眼"，先把**眼控交互**这条路走通。

> ⚠️ **安全边界**：Python 端默认是**纯屏幕模拟**（`gaze_blink_confirm_demo.py` 顶部
> `ENABLE_HARDWARE = False`），不向任何硬件发指令。Arduino 固件（`firmware/`）与串口输出层
> （`src/hardware/`）已经写好，但**未台架验证、默认关闭**；台架清单全过且加装物理急停按钮之前，
> 不进行任何载人测试。

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
│     └─ gaze_blink_confirm_demo.py   第 5 步：视线选方向 + 眨眼确认（界面美化版，接串口层）
│  └─ hardware/
│     └─ serial_link.py               串口输出层：意图→指令、心跳、电池、低电锁（默认模拟）
├─ tests/                             pytest：眨眼判定 / 串口链路 / 界面渲染 + 模型冒烟测试
├─ firmware/
│  └─ wheelchair_controller.ino       Arduino 修订版固件（含台架验证清单，待实机验证）
├─ Related_materials/                 硬件照片、参考固件（ardino_demo.ino）等资料
├─ models/                            MediaPipe 模型文件（face / hand landmarker，已入库）
├─ data/                              录制素材（raw_videos / snapshots）
├─ requirements.md                    依赖清单
├─ GIT_GUIDE.md                       组员 Git 上手指南（分支规则 / 日常循环 / 提交格式）
└─ README.md
```

五个演示脚本各自独立可运行（按周推进的开发顺序保留下来，方便逐步验证）。唯一的跨文件依赖是
第 5 步 → `src/hardware/serial_link.py`（串口输出层），且默认以模拟模式运行。

## 协作与分支（2026-09-30 起）

多人协作启动，分支规则如下：

| 分支 | 用途 | 规则 |
|---|---|---|
| `main` | 主干 | 由所有者维护与合并，**组员不要直接 push** |
| `huang` / `chen` | 组员开发分支 | 一人一分支，**只推自己的分支** |

- **组员上手**：先读 [GIT_GUIDE.md](GIT_GUIDE.md)（一次性配置、克隆、每日循环、
  提交信息格式、入库红线、问题急救、速查表），10 分钟可开工。
- **提交信息**：沿用本仓库惯例——`CH:` / `EN:` 双语、中文在前、一行说清做了什么。
- **干完活**：push 自己分支 → 所有者审核后合并进 main（`git switch main && git merge huang`）。
- **红线**：虚拟环境（`myvenv/` 等）、密钥、大体积原始数据永不入库（见 `.gitignore`）。

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

约 1 秒跑完，共 29 条：眨眼判定的校准 / 计数 / 过短过长忽略 / 不应期 / 人脸丢失重置，
串口链路的意图翻译 / 心跳 / 电池解析 / 低电锁，界面渲染冒烟，外加一个加载真实模型的冒烟测试。
全部使用合成数据，**不需要摄像头和硬件**；`data/raw_videos` 里没有可用录像时会自动跳过视频那条。

改代码的习惯：**改动前后各跑一次 `pytest`**。它会把"这次改动有没有破坏原有行为"直接告诉你，不用开摄像头靠肉眼验证。

## 硬件与固件

视觉端（Python）之外，下位机方案已经成型：

| 部件 | 说明 |
|---|---|
| 主控 | Arduino Uno 兼容板（ATmega328P-AU，16MHz，5V 逻辑） |
| 电机驱动 | BTS7960 大电流双 H 桥模块（带散热片） |
| 电机 | 12-24V 直流电机 ×2（左右轮差速转向） |
| 电压传感器 | 0-25V 分压模块 → A0（监测电池） |
| 供电 | 锂电池组（满电约 25V）；电机电源直接取自电池，**不经过** Uno 的 5V 稳压 |

```
PC（Python + pyserial，9600 波特）──串口指令──▶ ATmega328P ──INA/INB/PWM──▶ BTS7960 ×2 ──▶ 左右电机
电池 ──▶ 0-25V 电压传感器（1:5 分压）──▶ A0
```

### 串口指令协议（由固件定义，Python 端 `serial_link.py` 已按此实现）

| 指令 | 动作 |
|---|---|
| `0` / `S` | 停止 |
| `1` / `F` | 前进 |
| `2` / `B` | 后退 |
| `L` / `R` | 原地左转 / 右转 |
| `3` / `4` | 左半速 / 右半速前进（弧线行驶） |
| `GET_DATA` | 回报 `BATTERY:百分比,电压×10`（如 `BATTERY:87,218` = 87%、21.8V） |

固件内置四层安全机制：**3 秒收不到指令自动急停**（因此 Python 端必须周期性重发当前指令当心跳）、
换向/转向**先减速停稳再执行**、低电（≤19.0V）**只允许停止指令**（回升到 19.5V 才解锁）、
**看门狗**（程序卡死 2 秒自动复位停机）。

固件文件两个：

- `Related_materials/ardino_demo.ino` —— **参考件**，原样保留；
- `firmware/wheelchair_controller.ino` —— **修订版**：看门狗、串口 char 缓冲、转向先停、
  PWM 频率修正（约 18kHz）、电池平均+迟滞、低电限流等，每处改动带【修订】标记，
  文件底部有 **7 条台架验证清单**（接线核对、噪音/温度、看门狗、长跑、转向先停、低电模拟、电压对表）。

Python 端的串口输出层也已落地：`src/hardware/serial_link.py`
（`WheelchairLink` 类：意图翻译成协议字母、当前指令每 1 秒心跳重发、每 2 秒 `GET_DATA`
轮询电池、低电双重保险强制停车；`port=None` 时为**模拟模式**，无硬件也能跑全套测试）。
`gaze_blink_confirm_demo.py` 已接入：默认 `ENABLE_HARDWARE = False` 纯屏幕模拟；
改为 `True` 并填好串口号后，"确认成功"的方向才会真的发往固件。

> ⚠️ 烧录前需在 Arduino IDE 安装 **TimerOne** 库；台架清单全部通过**且加装物理急停按钮**（直接切断
> 电机电源）之后，才允许载人测试。

## 进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 第 1 周 | 摄像头基线（预览 / 录像 / 截图） | ⚠️ 重构版待实机完整验证 |
| 第 2 周 | MediaPipe 人脸与手部关键点可视化 | ✅ 已重构（初步测试通过） |
| 第 3 步 | 眨眼校准与计数（重构 + 单元测试） | ✅ 已实测通过 |
| 第 4 步 | 视线方向选择（左 / 中 / 右） | ✅ 已重构（初步测试通过） |
| 第 5 步 | 视线选方向 + 眨眼确认（屏幕演示） | ✅ 已重构（初步测试通过） |
| 代码整理 | 四个脚本统一为三段式结构 + 调用关系注释，判定逻辑可单测 | ✅ |
| 固件 | Arduino 控制器安全修订版（看门狗/char 缓冲/转向先停/PWM 频率），含台架清单 | 🔶 已写好，待台架验证 |
| 串口输出层 | `src/hardware/serial_link.py`（心跳/电池轮询/低电锁/模拟模式），含 13 条测试 | 🔶 已写好，待接实机 |
| 下一步 | 台架验证固件 → 插上 Arduino 实机联调串口层 | ⬜ 未开始 |

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
2. **补常驻测试**：`tests/` 已覆盖第 3 步（眨眼）、串口层与界面渲染；第 4 步（视线方向）的行为
   验证仍是一次性脚本，可照 `test_serial_link.py` 的样子落成常驻用例。
3. **抽公共部分**：打开摄像头、构建检测器、中文绘制在五个脚本里各有一份拷贝，改一处要同步五处。
   → **已在 `chen` 分支完成（2026-09-29）**：新增 `src/common/` 共用层（`paths.py` 路径地址簿 +
   `camera_utils.open_camera()` + `draw_utils.chinese_font/draw_text()`），5 个脚本去重、净减 143 行；
   **合并 chen 后 main 一并获得**（main 自身尚未引入）。
4. **台架验证 `firmware/wheelchair_controller.ino`**：按文件底部 7 条清单逐项过
   （接线核对 → PWM 噪音/温度 → 看门狗 → 长跑 → 转向先停 → 低电模拟 → 电压对表），
   全过后加装物理急停按钮。
5. **实机联调串口层**：插上 Arduino，把 `gaze_blink_confirm_demo.py` 顶部 `ENABLE_HARDWARE`
   改为 `True` 并填好串口号，验证心跳保活、`GET_DATA` 电池回报、低电强制停车三条链路。


## English Version
# EyeWheelchairProject (Eye-Controlled Wheelchair Prototype)

Eye-controlled wheelchair prototype — gaze + blink interaction. The Python side runs as a pure screen simulation by default; the Arduino firmware and serial layer are written but not yet verified on the real chair.

Using an ordinary USB camera to recognize "which direction the eyes are looking" and "whether there is a blink," the goal is to first get the **eye-control interaction** path working.

> ⚠️ **Safety boundary**: the Python side defaults to **pure screen simulation** (`ENABLE_HARDWARE = False` at the top of `gaze_blink_confirm_demo.py`) and sends no commands to any hardware. The Arduino firmware (`firmware/`) and the serial output layer (`src/hardware/`) are written but **not bench-verified and off by default**; no human ride testing before the bench checklist fully passes **and** a physical e-stop button is installed.

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
│     └─ gaze_blink_confirm_demo.py   Step 5: Gaze selects direction + blink confirmation (restyled UI, wired to the serial layer)
│  └─ hardware/
│     └─ serial_link.py               Serial output layer: intent→command, heartbeat, battery, low-battery lock (simulated by default)
├─ tests/                             pytest: blink logic / serial link / UI rendering + model smoke test
├─ firmware/
│  └─ wheelchair_controller.ino       Revised Arduino firmware (with bench checklist, pending verification)
├─ Related_materials/                 Hardware photos, reference firmware (ardino_demo.ino), etc.
├─ models/                            MediaPipe model files (face / hand landmarker, checked in)
├─ data/                              Recorded materials (raw_videos / snapshots)
├─ requirements.md                    Dependency list
├─ GIT_GUIDE.md                       Teammate Git onboarding guide (branch rules / daily loop / commit format)
└─ README.md
```

The five demo scripts run independently (the weekly development order is preserved for gradual verification). The only cross-file dependency is step 5 → `src/hardware/serial_link.py` (serial output layer), which itself runs in simulated mode by default.

## Collaboration and Branches (since 2026-09-30)

Multi-contributor collaboration has started, with the following branch rules:

| Branch | Purpose | Rule |
|---|---|---|
| `main` | Trunk | Maintained and merged by the owner — **members do not push directly** |
| `huang` / `chen` | Per-member dev branches | One branch per member; **push only your own branch** |

- **New members**: read [GIT_GUIDE.md](GIT_GUIDE.md) first (one-time setup, clone, daily loop, commit message format, red lines, first aid, cheat sheet) — 10 minutes and you can start.
- **Commit messages** keep this repo's convention: `CH:` / `EN:` bilingual, Chinese first, one line saying what changed.
- **When done**: push your branch → the owner reviews and merges into main (`git switch main && git merge huang`).
- **Red lines**: virtualenvs (`myvenv/` etc.), secrets and large raw data files never get committed (see `.gitignore`).

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

Runs in about 1 second, 29 tests in total: blink calibration / counting / too-short & too-long ignoring / refractory period / face-loss reset; serial-link intent translation / heartbeat / battery parsing / low-battery lock; UI rendering smoke tests; plus one smoke test that loads the real model. All use synthetic data and **need no camera or hardware**; the video test is automatically skipped when there is no usable recording in `data/raw_videos`.

Habit when changing code: **run `pytest` once before and after each change**. It will directly tell you "whether this change broke existing behavior," without needing to open the camera and verify by eye.

## Hardware and Firmware

Beyond the vision side (Python), the low-level controller design is now in place:

| Part | Notes |
|---|---|
| MCU | Arduino Uno compatible board (ATmega328P-AU, 16 MHz, 5 V logic) |
| Motor driver | BTS7960 high-current dual H-bridge module (with heatsinks) |
| Motors | 12–24 V DC motors ×2 (differential steering) |
| Voltage sensor | 0–25 V divider module → A0 (battery monitoring) |
| Power | Li-ion pack (~25 V full); motor power comes straight from the battery, **not** through the Uno's 5 V regulator |

```
PC (Python + pyserial, 9600 baud) ──commands──▶ ATmega328P ──INA/INB/PWM──▶ BTS7960 ×2 ──▶ left/right motors
Battery ──▶ 0–25 V voltage sensor (1:5 divider) ──▶ A0
```

### Serial command protocol (defined by the firmware; implemented on the Python side in `serial_link.py`)

| Command | Action |
|---|---|
| `0` / `S` | Stop |
| `1` / `F` | Forward |
| `2` / `B` | Backward |
| `L` / `R` | Pivot turn left / right |
| `3` / `4` | Half-speed left / right (arc driving) |
| `GET_DATA` | Replies `BATTERY:percent,voltage×10` (e.g. `BATTERY:87,218` = 87%, 21.8 V) |

The firmware has four built-in safety layers: **auto-stop after 3 s without a command** (so the Python side must periodically re-send the current command as a heartbeat), turns and direction changes **ramp to a full stop before executing**, low battery (≤19.0 V) **only accepts stop commands** (unlocks at 19.5 V), and a **watchdog** (a hung program resets within 2 s and stops the motors).

Two firmware files:

- `Related_materials/ardino_demo.ino` — **reference copy**, kept untouched;
- `firmware/wheelchair_controller.ino` — **revised version**: watchdog, serial char buffer, stop-before-turn, PWM frequency fix (~18 kHz), battery averaging + hysteresis, throttled low-battery warnings, etc. Every change carries a 【修订】 marker, and the file ends with a **7-item bench verification checklist** (wiring check, noise/temperature, watchdog, long run, stop-before-turn, low-battery simulation, voltage cross-check).

The Python serial output layer is also in place: `src/hardware/serial_link.py`
(the `WheelchairLink` class: intent → protocol letters, 1 s heartbeat re-send of the current
command, `GET_DATA` battery polling every 2 s, a second low-battery lock that forces stop;
with `port=None` it runs in a **simulated mode**, so the full test suite needs no hardware).
`gaze_blink_confirm_demo.py` is wired to it: by default `ENABLE_HARDWARE = False` (pure screen
simulation); set it to `True` with the right port and only then does a confirmed direction
actually go out to the firmware.

> ⚠️ Install the **TimerOne** library in the Arduino IDE before compiling. The bench checklist must fully pass **and a physical e-stop button** (cutting motor power directly) must be added before any human ride testing.

## Progress

| Stage | Content | Status |
|---|---|---|
| Week 1 | Camera baseline (preview / record / snapshot) | ⚠️ Refactored version awaits full on-device verification |
| Week 2 | MediaPipe face and hand landmark visualization | ✅ Refactored (preliminary tests pass) |
| Step 3 | Blink calibration and counting (refactor + unit tests) | ✅ Verified on the real device |
| Step 4 | Gaze direction selection (left / center / right) | ✅ Refactored (preliminary tests pass) |
| Step 5 | Gaze selects direction + blink confirmation (screen demo) | ✅ Refactored (preliminary tests pass) |
| Code tidy-up | Four scripts unified into the three-section layout with call-graph comments; decision logic unit-testable | ✅ |
| Firmware | Safety-revised Arduino controller (watchdog / char buffer / stop-before-turn / PWM frequency), with bench checklist | 🔶 Written, awaiting bench verification |
| Serial output layer | `src/hardware/serial_link.py` (heartbeat / battery polling / low-battery lock / simulated mode), 13 tests | 🔶 Written, awaiting on-device integration |
| Next | Bench-verify the firmware → plug in the Arduino and integrate the serial layer | ⬜ Not started |

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
2. **Add permanent tests**: `tests/` now covers step 3 (blink), the serial layer and UI rendering; the step 4 (gaze direction) behaviour checks are still one-off scripts and could be turned into permanent cases like `test_serial_link.py`.
3. **Extract the common parts**: opening the camera, building detectors and drawing Chinese text are copied in all five scripts, so one change means five edits.
   → **Done on the `chen` branch (2026-09-29)**: new `src/common/` shared layer (`paths.py` registry + `camera_utils.open_camera()` + `draw_utils.chinese_font/draw_text()`); five scripts deduplicated, -143 lines. **Merging `chen` brings it into `main`** (main itself does not have it yet).
4. **Bench-verify `firmware/wheelchair_controller.ino`**: work through the 7-item checklist at the end of the file (wiring check → PWM noise/temperature → watchdog → long run → stop-before-turn → low-battery simulation → voltage cross-check); then add a physical e-stop button.
5. **On-device integration of the serial layer**: plug in the Arduino, set `ENABLE_HARDWARE = True` with the right port in `gaze_blink_confirm_demo.py`, and verify the three chains — heartbeat keep-alive, `GET_DATA` battery replies, and forced stop on low battery.