/*
 * 轮椅主控固件（基于 Related_materials/ardino_demo.ino 的修订版）
 *
 * 与原版的关系：原版是参考件，保持原样不动；本文件是它的修订版，
 * 每处行为差异都在对应位置用【修订】标注，并在文件末尾附上台架验证清单。
 *
 * 修订总览：
 *  【P0-1】PWM 频率：删掉与 TimerOne 打架的 TCCR1B 手动分频，只留 Timer1.initialize(55)
 *          （约 18 kHz，在 BTS7960 约 25 kHz 的规格内）
 *  【P0-3】看门狗：2 秒不喂狗自动复位，复位后 setup() 里 stopAll() 停机 —— 程序跑飞也会停
 *  【P0-4】前进中打转向也先停稳再转（原来只有后退中才有这层保护）
 *  【P0-5】串口读取 String → 定长 char 缓冲（消除 2KB SRAM 堆碎片崩溃 + 1 秒阻塞；
 *          rev2 曾把行尾 '\n' 当分隔符跳过导致命令永不执行，已修正：行尾 = 提交命令）
 *  【P1-6】低电量告警限流：状态变化时打一条，之后每 5 秒提醒一次（原来每圈刷屏）
 *  【P1-7】电池读数：8 次采样平均 + 迟滞（19.0V 落锁 / 19.5V 解锁；rev2 曾把迟滞
 *          方向写反成 19.5 落锁/18.5 解锁导致横跳，已修正）
 *  【P1-8】注释纠错：ACCEL_TIME 实为 5 秒；电池为 6 串锂电（约 25V 满电），不是 2 节
 *  【P2-9】提取 handleTurnRequest()：L/R 的"后退先停"逻辑去重
 *  【P2-10】提取 readBatteryVoltage()：两处重复的 ADC 换算合并
 *  【P2-11】删除 Motor 结构体里从没被读过的 isAccelerating 字段
 *  【保持不变】指令协议、速度/时间参数（FULL_SPEED 等，标定过的安全值）、
 *             两相架构（setMotorTarget 写意图 / updateMotor 每帧执行）
 *
 * 待核对（需要看实际接线，代码两侧各留了方案）：
 *  【P0-2】BTS7960 是 RPWM/LPWM 四线制，原代码是 INA/INB/PWM 三线制。
 *          若实接为 BTS7960 标准接法，倒车时占空比可能没生效 —— 见文件末尾清单第 1 条。
 */
#include <TimerOne.h>
#include <avr/wdt.h>
#include <string.h>   // strcmp（rev2 漏了这句，靠 Arduino.h 间接带入才侥幸能编）

// ========== 引脚定义 ==========
const int M1_INA = 4, M1_INB = 6, M1_PWM = 9;
const int M2_INA = 5, M2_INB = 7, M2_PWM = 10;
const int BATTERY_PIN = A0;

// ========== 系统参数 ==========
const int FULL_SPEED = 70;   // PWM 0-255 满速（标定值，勿随意改）
const int HALF_SPEED = 35;   // PWM 0-255 半速
const int TURN_SPEED = 35;   // PWM 0-255 转向速度
const unsigned long ACCEL_TIME = 5000;       // 加速/减速全程 5 秒（原注释写 3 秒是错的）
const float STOP_THRESHOLD = 1.0;            // |speed| 低于它视为已停稳（0-255 刻度）
const unsigned long CMD_TIMEOUT = 3000;      // 3 秒收不到指令 → 急停（失联保护）
const unsigned long DIR_CHANGE_TIMEOUT = 7000;  // 换向等待停稳的超时
const unsigned long TURN_CHANGE_TIMEOUT = 5000; // 转向等待停稳的超时
const unsigned long LOW_BATT_REPEAT_MS = 5000;  // 低电告警的重复间隔

// 电池参数（0-25V 电压传感器模块，内部 1:5 分压）
const float VOLTAGE_DIVIDER = 5;
const float LOCKOUT_VOLTAGE = 19.0;   // 电压跌破它 → 落锁：只接受停止指令（沿用原版阈值）
const float UNLOCK_VOLTAGE = 19.5;    // 【修订 2026-09-27】电压回升超过它才解锁。
                                      // 迟滞方向必须是"解锁阈值高于落锁阈值"：
                                      // 放电时电压只会越来越低，跌到 19.0 落锁后，
                                      // 要涨回 19.5 以上才算真恢复。rev2 写反成
                                      // 19.5 落锁 / 18.5 解锁，在 18.5~19.5V 之间
                                      // 每圈循环都会落锁↔解锁来回横跳。
const float BATTERY_MAX = 25.0;       // 满电电压（6 串锂电 ≈ 25.2V）
const float BATTERY_MIN = 19.0;       // 建议充电电压

// 串口指令缓冲：定长 char 数组，杜绝 String 在 2KB SRAM 上反复分配
const int CMD_BUFFER_SIZE = 16;

// ========== 电机控制结构 ==========
struct Motor {
  int pwmPin;
  int inA;
  int inB;
  float currentSpeed;
  float targetSpeed;
  bool dirA;
  bool dirB;
  unsigned long accelStartTime;
};

Motor m1 = {M1_PWM, M1_INA, M1_INB, 0, 0, LOW, LOW, 0};
Motor m2 = {M2_PWM, M2_INA, M2_INB, 0, 0, LOW, LOW, 0};

// ========== 系统状态变量 ==========
unsigned long lastCmdTime = 0;
char lastCommand = 'S';          // 【P0-5】原来是 String，改 char：状态只有单字符
bool isBatteryLow = false;

// 低电告警限流用
bool lowBatteryAnnounced = false;
unsigned long lastLowBatteryMsgAt = 0;

// 方向切换状态（等停稳后执行）
bool pendingDirectionChange = false;
bool targetDirA, targetDirB;
unsigned long directionChangeStart = 0;

// 转向请求状态（等停稳后执行）
bool pendingTurnChange = false;
char pendingTurnType = ' '; // 'L' 或 'R'
unsigned long turnChangeStart = 0;

// ========== 函数声明 ==========
void updateMotors();
void updateMotor(Motor* m);
float easeMotor(float t);
void applyMotorState(Motor* m);
void setMotorTarget(Motor* m, int target, bool dirA, bool dirB);
void stopAll();
bool needDirectionChange(bool newDirA, bool newDirB);
void handleDirectionChange(bool newDirA, bool newDirB);
void handleTurnRequest(char type);          // 【P2-9】L/R 共用的"后退先停"逻辑
void checkDirectionChange();
void checkTurnChange();
void checkCommandTimeout();
float readBatteryVoltage();                 // 【P2-10】ADC → 电压，唯一实现
void checkBatteryStatus();
void sendSensorData();
void processCommand(const char* cmd);

// ========== 主程序 ==========
void setup() {
  wdt_enable(WDTO_2S);              // 【P0-3】看门狗：2 秒不喂 → 自动复位
                                    // 复位后从这里重新跑，stopAll() 把电机停住
                                    // （看门狗复位不会清串口缓冲，无碍）
  Serial.begin(9600);

  // 初始化电机引脚
  pinMode(M1_INA, OUTPUT);
  pinMode(M1_INB, OUTPUT);
  pinMode(M1_PWM, OUTPUT);
  pinMode(M2_INA, OUTPUT);
  pinMode(M2_INB, OUTPUT);
  pinMode(M2_PWM, OUTPUT);

  // 【P0-1】PWM 频率只由 Timer1.initialize(55) 决定（周期 55µs ≈ 18kHz）。
  // 原版这里还有一句 TCCR1B 手动改分频，两处设置叠加后实际频率远超 BTS7960
  // 规格上限，已删除。若台架实测嫌 18kHz 有轻啸，可把 55 调小到 40（≈24kHz，
  // 仍在规格内）；不要恢复手动分频那句。
  Timer1.initialize(55);
  Timer1.pwm(M1_PWM, 0);
  Timer1.pwm(M2_PWM, 0);

  stopAll();
  lastCmdTime = millis();

  Serial.println("Arduino wheelchair controller ready (rev2: wdt+charbuf)");
}

void loop() {
  wdt_reset();                      // 【P0-3】每圈喂狗。任何一处卡死超过 2 秒
                                    // （比如死循环、堆损坏）→ 芯片自动复位 → 停机

  // 检查电池状态
  checkBatteryStatus();

  // 处理串口命令：【P0-5】按字节累积到定长缓冲，不再用 readStringUntil
  //（原来那句最长会阻塞 1 秒，期间加减速全部停摆）
  //【修订 2026-09-27】修复致命 bug：rev2 把 '\n' 当"可忽略的分隔符"跳过，
  // 而命令的提交恰恰靠行尾触发——导致以换行结尾的正常命令永远攒不完整、
  // 永远不执行（直到攒满 16 字符被截断成 Unknown）。现在行尾 = 提交命令。
  static char buffer[CMD_BUFFER_SIZE];
  static int len = 0;
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {                // 行尾 = 一条命令结束，立即处理
        buffer[len] = '\0';
        len = 0;
        processCommand(buffer);
        lastCmdTime = millis();
      }
      continue;                     // 连续换行/空行在这里被自然过滤
    }
    if (c >= 32 && c < 127 && len < CMD_BUFFER_SIZE - 1) {
      buffer[len++] = c;            // 可打印 ASCII 才收；超长命令截断（防御异常输入）
    }
  }
  // 兜底：发送方不带换行时，攒满 16 字符也在本圈处理（截断成 Unknown，不悬挂）
  if (len >= CMD_BUFFER_SIZE - 1) {
    buffer[len] = '\0';
    len = 0;
    processCommand(buffer);
    lastCmdTime = millis();
  }

  // 检查指令超时
  checkCommandTimeout();

  // 更新电机状态
  updateMotors();

  // 检查方向切换状态
  checkDirectionChange();

  // 检查转向请求
  checkTurnChange();
}

// ========== 电机控制核心函数 ==========
void updateMotors() {
  updateMotor(&m1);
  updateMotor(&m2);
}

void updateMotor(Motor* m) {
  if (m->currentSpeed == m->targetSpeed) return;

  float progress = (float)(millis() - m->accelStartTime) / ACCEL_TIME;
  progress = constrain(progress, 0, 1);

  float newSpeed = m->targetSpeed > m->currentSpeed ?
                 easeMotor(progress) * (m->targetSpeed - m->currentSpeed) + m->currentSpeed :
                 (1 - easeMotor(progress)) * (m->currentSpeed - m->targetSpeed) + m->targetSpeed;

  // 当接近停止时直接设为0
  if (abs(newSpeed) < STOP_THRESHOLD) {
    newSpeed = 0;
  }
  else if (progress >= 1) {
    newSpeed = m->targetSpeed;
  }

  m->currentSpeed = newSpeed;
  applyMotorState(m);
}

float easeMotor(float t) {
  // 二次方缓动，更平滑的加速曲线
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

void applyMotorState(Motor* m) {
  digitalWrite(m->inA, m->dirA);
  digitalWrite(m->inB, m->dirB);

  int speedValue = constrain(abs(m->currentSpeed), 0, 255);
  int duty = map(speedValue, 0, 255, 0, 1023);
  Timer1.setPwmDuty(m->pwmPin, duty);
}

// ========== 运动控制函数 ==========
void stopMotors() {
  setMotorTarget(&m1, 0, m1.dirA, m1.dirB);
  setMotorTarget(&m2, 0, m2.dirA, m2.dirB);
  lastCommand = 'S';
  Serial.println("Stopped");
}

void moveForward() {
  if (needDirectionChange(LOW, HIGH)) {
    handleDirectionChange(LOW, HIGH);
  } else {
    setMotorTarget(&m1, FULL_SPEED, LOW, HIGH);
    setMotorTarget(&m2, FULL_SPEED, LOW, HIGH);
    lastCommand = 'F';
    Serial.println("Moving forward");
  }
}

void moveBackward() {
  if (needDirectionChange(HIGH, LOW)) {
    handleDirectionChange(HIGH, LOW);
  } else {
    setMotorTarget(&m1, FULL_SPEED, HIGH, LOW);
    setMotorTarget(&m2, FULL_SPEED, HIGH, LOW);
    lastCommand = 'B';
    Serial.println("Moving backward");
  }
}

void turnLeft() {
  // 原地左转：左轮后退，右轮前进
  setMotorTarget(&m1, TURN_SPEED, HIGH, LOW);
  setMotorTarget(&m2, TURN_SPEED, LOW, HIGH);
  lastCommand = 'L';
  Serial.println("Turning left");
}

void turnRight() {
  // 原地右转：左轮前进，右轮后退
  setMotorTarget(&m1, TURN_SPEED, LOW, HIGH);
  setMotorTarget(&m2, TURN_SPEED, HIGH, LOW);
  lastCommand = 'R';
  Serial.println("Turning right");
}

void mixSpeedLeftHalf() {
  setMotorTarget(&m1, HALF_SPEED, LOW, HIGH);
  setMotorTarget(&m2, FULL_SPEED, LOW, HIGH);
  lastCommand = '3';
  Serial.println("Mixed speed: Left half, Right full");
}

void mixSpeedRightHalf() {
  setMotorTarget(&m1, FULL_SPEED, LOW, HIGH);
  setMotorTarget(&m2, HALF_SPEED, LOW, HIGH);
  lastCommand = '4';
  Serial.println("Mixed speed: Left full, Right half");
}

// ========== 方向切换处理 ==========
bool needDirectionChange(bool newDirA, bool newDirB) {
  // 当前停止状态时不要求方向切换
  if (m1.targetSpeed == 0 && m2.targetSpeed == 0) {
    return false;
  }
  return (m1.dirA != newDirA || m1.dirB != newDirB) ||
         (m2.dirA != newDirA || m2.dirB != newDirB);
}

void handleDirectionChange(bool newDirA, bool newDirB) {
  Serial.println("Direction change initiated");

  // 设置减速停止
  setMotorTarget(&m1, 0, m1.dirA, m1.dirB);
  setMotorTarget(&m2, 0, m2.dirA, m2.dirB);

  // 记录待处理的方向切换请求
  pendingDirectionChange = true;
  targetDirA = newDirA;
  targetDirB = newDirB;
  directionChangeStart = millis();
}

void checkDirectionChange() {
  if (!pendingDirectionChange) return;

  // 超时检查
  if (millis() - directionChangeStart > DIR_CHANGE_TIMEOUT) {
    pendingDirectionChange = false;
    Serial.println("Direction change timeout");
    return;
  }

  // 使用阈值判断是否停止
  if (abs(m1.currentSpeed) < STOP_THRESHOLD &&
      abs(m2.currentSpeed) < STOP_THRESHOLD) {
    // 电机已停止，执行方向切换
    setMotorTarget(&m1, FULL_SPEED, targetDirA, targetDirB);
    setMotorTarget(&m2, FULL_SPEED, targetDirA, targetDirB);
    pendingDirectionChange = false;
    Serial.println("Direction changed");
  }
}

// ========== 转向切换处理 ==========
// 【P2-9】原来 processCommand 里 L/R 两个分支各抄了一遍"后退中先停再挂起"，
// 提到这里共用；【P0-4】原来只有"后退中"才走这条路，现在任何运动状态中
// 收到转向请求且方向需要翻转，都先停稳再转。
void handleTurnRequest(char type) {
  // 当前在运动且转向会造成至少一个电机方向翻转 → 必须先停稳
  // （转向指令本身就要一正一反两个方向，运动中直接翻转 = 瞬间换向，大电流冲击）
  bool moving = (m1.targetSpeed != 0 || m2.targetSpeed != 0);
  if (moving) {
    // 设置减速停止
    setMotorTarget(&m1, 0, m1.dirA, m1.dirB);
    setMotorTarget(&m2, 0, m2.dirA, m2.dirB);

    // 记录待处理的转向请求
    pendingTurnChange = true;
    pendingTurnType = type;
    turnChangeStart = millis();
    Serial.println(type == 'L' ? "Pending left turn after stop" : "Pending right turn after stop");
  } else {
    // 静止状态直接转向
    if (type == 'L') {
      turnLeft();
    } else {
      turnRight();
    }
  }
}

void checkTurnChange() {
  if (!pendingTurnChange) return;

  // 超时检查
  if (millis() - turnChangeStart > TURN_CHANGE_TIMEOUT) {
    pendingTurnChange = false;
    Serial.println("Turn change timeout");
    return;
  }

  // 使用阈值判断是否停止
  if (abs(m1.currentSpeed) < STOP_THRESHOLD &&
      abs(m2.currentSpeed) < STOP_THRESHOLD) {
    // 电机已停止，执行转向
    if (pendingTurnType == 'L') {
      turnLeft();
    } else if (pendingTurnType == 'R') {
      turnRight();
    }
    pendingTurnChange = false;
    Serial.println("Turn direction changed");
  }
}

// ========== 系统功能函数 ==========
void setMotorTarget(Motor* m, int target, bool dirA, bool dirB) {
  m->dirA = dirA;
  m->dirB = dirB;
  m->targetSpeed = target;
  m->accelStartTime = millis();
}

void stopAll() {
  setMotorTarget(&m1, 0, LOW, LOW);
  setMotorTarget(&m2, 0, LOW, LOW);
  applyMotorState(&m1);
  applyMotorState(&m2);
}

void checkCommandTimeout() {
  // 如果有待处理的切换，不触发超时停止
  if (pendingDirectionChange || pendingTurnChange) {
    lastCmdTime = millis(); // 重置超时计时器
    return;
  }

  if (lastCommand == 'S' || lastCommand == '0') return;

  unsigned long currentTime = millis();

  if (currentTime - lastCmdTime > CMD_TIMEOUT) {
    Serial.println("Command timeout - stopping motors");
    stopMotors();
  }
}

// 【P2-10】原来这段换算在 checkBatteryStatus 和 sendSensorData 里各有一份。
// 【P1-7】8 次采样取平均，压掉 ADC 单次读数的抖动。
float readBatteryVoltage() {
  long sum = 0;
  for (int i = 0; i < 8; i++) {
    sum += analogRead(BATTERY_PIN);
  }
  float batteryValue = sum / 8.0;
  return (batteryValue * (5.0 / 1023.0)) * VOLTAGE_DIVIDER;
}

void checkBatteryStatus() {
  float batteryVoltage = readBatteryVoltage();

  // 【P1-7】【修订 2026-09-27】迟滞方向修正：跌破 19.0V 落锁，回升过 19.5V 才解锁
  // （原 rev2 把方向写反，18.5~19.5V 区间每圈横跳；原始参考件是单阈值 19.0V，
  //   临界电压附近同样会横跳——两种毛病都堵上了）
  if (!isBatteryLow && batteryVoltage <= LOCKOUT_VOLTAGE) {
    isBatteryLow = true;
  } else if (isBatteryLow && batteryVoltage >= UNLOCK_VOLTAGE) {
    isBatteryLow = false;
    lowBatteryAnnounced = false;   // 解锁后允许下一次落锁再提示
  }

  // 【P1-6】原来这里每圈 println（loop 每秒跑几千圈，串口被灌满，
  // Python 端没法正常收 BATTERY 回报）。现在：落锁瞬间打一条，
  // 之后每 5 秒提醒一次，平时一条都不打。
  if (isBatteryLow) {
    if (!lowBatteryAnnounced) {
      lowBatteryAnnounced = true;
      lastLowBatteryMsgAt = millis();
      Serial.println("Low battery voltage, please charge");
    } else if (millis() - lastLowBatteryMsgAt >= LOW_BATT_REPEAT_MS) {
      lastLowBatteryMsgAt = millis();
      Serial.println("Low battery voltage, please charge");
    }
  }
}

void sendSensorData() {
  float batteryVoltage = readBatteryVoltage();
  int batteryPercentage = map((int)(batteryVoltage * 10), (int)(BATTERY_MIN * 10), (int)(BATTERY_MAX * 10), 0, 100);
  batteryPercentage = constrain(batteryPercentage, 0, 100);

  Serial.print("BATTERY:");
  Serial.print(batteryPercentage);
  Serial.print(",");
  Serial.println((int)(batteryVoltage * 10));  // 保留一位小数（×10 整数传输）
}

// ========== 命令处理函数 ==========
// 【P0-5】参数从 String 改为 const char*（配合 char 缓冲，堆上零分配）
void processCommand(const char* cmd) {
  // 低电量时只允许停止命令
  bool isStopCmd = (strcmp(cmd, "S") == 0 || strcmp(cmd, "0") == 0);
  if (isBatteryLow && !isStopCmd) {
    Serial.println("Low battery, only stop command allowed");
    return;
  }

  // 新命令到来时清除待处理的切换请求
  pendingDirectionChange = false;
  pendingTurnChange = false;

  // 处理停止命令
  if (isStopCmd) {
    stopMotors();
  }
  // 处理前进命令
  else if (strcmp(cmd, "1") == 0 || strcmp(cmd, "F") == 0) {
    moveForward();
  }
  // 处理后退命令
  else if (strcmp(cmd, "2") == 0 || strcmp(cmd, "B") == 0) {
    moveBackward();
  }
  // 处理混合速度
  else if (strcmp(cmd, "3") == 0) {
    mixSpeedLeftHalf();
  }
  else if (strcmp(cmd, "4") == 0) {
    mixSpeedRightHalf();
  }
  // 处理转向命令：【P0-4】任何运动状态中都先停稳再转（原来只有后退中才保护）；
  // 【P2-9】L/R 共用 handleTurnRequest()
  else if (strcmp(cmd, "L") == 0) {
    handleTurnRequest('L');
  }
  else if (strcmp(cmd, "R") == 0) {
    handleTurnRequest('R');
  }
  // 处理数据请求
  else if (strcmp(cmd, "GET_DATA") == 0) {
    sendSensorData();
  }
  else {
    Serial.print("Unknown command: ");
    Serial.println(cmd);
  }
}

/* ============================ 台架验证清单 ============================
 * 每一项都对应上面一处【修订】。烧录本固件后按顺序做，任何一条不过
 * 就停下来分析，不要带病上车。
 *
 * 1.【P0-2 接口核对，最先做】
 *    a. 断电，对照实物接线确认：M1_PWM/M2_PWM 到底接的是 BTS7960 的哪根线
 *       （RPWM？还是 R_EN/L_EN 短接后接 RPWM？）。
 *    b. 台架上悬空车轮，发 "2"（后退）：倒车应该和前进一样是 70/255 的慢速。
 *       如果倒车明显比前进快 → 占空比没吃到 LPWM，接线或代码要按 BTS7960
 *       标准接法重做（两个电机需要 4 路 PWM，Uno 够用但需重新分配定时器）。
 * 2.【P0-1 PWM 频率】
 *    a. 静止听噪音：上电后电机应无明显啸叫（18kHz 已接近人耳上限）。
 *    b. 低速跑 5 分钟，摸 BTS7960 散热片：温热正常，烫手说明开关损耗过大。
 *    c. 有条件用示波器/逻辑分析仪看 9/10 脚：应为约 18kHz 方波，占空比随指令变化。
 * 3.【P0-3 看门狗】
 *    a. 正常跑 10 分钟不应复位（串口反复打印 "ready" 就是异常）。
 *    b. 人为验证：临时在 loop() 里加一句 while(1); 编译烧录 → 2 秒内电机
 *       应停、串口应重新打印 ready → 验证完删掉这行。
 * 4.【P0-5 串口缓冲】
 *    a. 快速连发指令：Python 端每 100ms 发一条，跑 10 分钟不崩、不丢指令。
 *    b. 发一条超长垃圾（>16 字符）+ 发空行：应回 "Unknown command" 或被忽略，
 *       之后正常指令仍可用。
 * 5.【P0-4 转向先停】
 *    前进中发 "L"：串口应先打 "Pending left turn after stop"，车减速停稳后才
 *    开始原地转（约 5 秒内），而不是立刻打轮。
 * 6.【P1-6/7 低电逻辑】
 *    a. 用可调电源模拟电池：从 21V 缓降到 19.5V → 恰好打一条低电告警，
 *       之后每 5 秒一条（不是刷屏）。
 *    b. 在 19.2~18.8V 之间晃动：不应反复落锁/解锁。
 *    c. 落锁后发 "F" 应被拒（"Low battery, only stop command allowed"），
 *       发 "S" 应正常停。
 * 7.【GET_DATA】
 *    发 "GET_DATA"，回包形如 "BATTERY:87,218"（87%、21.8V），与万用表实测
 *    电池电压误差 < 0.3V（分压电阻有偏差时可校 VOLTAGE_DIVIDER）。
 * ===================================================================== */
