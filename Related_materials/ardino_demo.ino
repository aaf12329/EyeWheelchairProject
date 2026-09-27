#include <TimerOne.h>

// ========== 引脚定义 ==========
const int M1_INA = 4, M1_INB = 6, M1_PWM = 9;
const int M2_INA = 5, M2_INB = 7, M2_PWM = 10;
const int BATTERY_PIN = A0;

// ========== 系统参数 ==========
const int FULL_SPEED = 70;  // 恢复255范围 (PWM 0-255)
const int HALF_SPEED = 35;  // 恢复255范围
const int TURN_SPEED = 35;  // 恢复255范围
const int ACCEL_TIME = 5000;       // 3秒加速/减速
const float STOP_THRESHOLD = 1.0;  // 停止阈值
const unsigned long CMD_TIMEOUT = 3000; // 3秒指令超时
const unsigned long DIR_CHANGE_TIMEOUT = 7000; // 7秒方向切换超时
const unsigned long TURN_CHANGE_TIMEOUT = 5000; // 5秒转向切换超时

// 电池参数
const float VOLTAGE_DIVIDER = 5;
const float LOW_BATTERY_THRESHOLD = 19;
const float BATTERY_MAX = 25;  // 最大电压 (2节锂电串联)
const float BATTERY_MIN = 19;  // 最低电压

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
  bool isAccelerating;
};

Motor m1 = {M1_PWM, M1_INA, M1_INB, 0, 0, LOW, LOW, 0, false};
Motor m2 = {M2_PWM, M2_INA, M2_INB, 0, 0, LOW, LOW, 0, false};

// ========== 系统状态变量 ==========
unsigned long lastCmdTime = 0;
String lastCommand = "S";  // 使用字符串记录最后命令
bool isBatteryLow = false;

// 方向切换状态
bool pendingDirectionChange = false;
bool targetDirA, targetDirB;
unsigned long directionChangeStart = 0;

// 转向请求状态
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
void checkDirectionChange();
void checkTurnChange();
void checkCommandTimeout();
void checkBatteryStatus();
void sendSensorData();
void processCommand(String cmd);

// ========== 主程序 ==========
void setup() {
  Serial.begin(9600);
  
  // 初始化电机引脚
  pinMode(M1_INA, OUTPUT);
  pinMode(M1_INB, OUTPUT);
  pinMode(M1_PWM, OUTPUT);
  pinMode(M2_INA, OUTPUT);
  pinMode(M2_INB, OUTPUT);
  pinMode(M2_PWM, OUTPUT);

  // 提升 Timer1（引脚 9、10）PWM 频率到 31 kHz，消除异响
  TCCR1B = (TCCR1B & 0b11111000) | 0x01;

  // 初始化 PWM 定时器
  Timer1.initialize(55);
  Timer1.pwm(M1_PWM, 0);
  Timer1.pwm(M2_PWM, 0);
  
  stopAll();
  lastCmdTime = millis();

  Serial.println("Arduino wheelchair controller ready");
}

void loop() {
  // 检查电池状态
  checkBatteryStatus();
  
  // 处理串口命令
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    processCommand(command);
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
    m->isAccelerating = false;
  }
  else if (progress >= 1) {
    newSpeed = m->targetSpeed;
    m->isAccelerating = false;
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
  
  // 正确使用 constrain 函数
  int speedValue = constrain(abs(m->currentSpeed), 0, 255);
  int duty = map(speedValue, 0, 255, 0, 1023);
  Timer1.setPwmDuty(m->pwmPin, duty);
}

// ========== 运动控制函数 ==========
void stopMotors() {
  setMotorTarget(&m1, 0, m1.dirA, m1.dirB);
  setMotorTarget(&m2, 0, m2.dirA, m2.dirB);
  lastCommand = "S";
  Serial.println("Stopped");
}

void moveForward() {
  if (needDirectionChange(LOW, HIGH)) {
    handleDirectionChange(LOW, HIGH);
  } else {
    setMotorTarget(&m1, FULL_SPEED, LOW, HIGH);
    setMotorTarget(&m2, FULL_SPEED, LOW, HIGH);
    lastCommand = "F";
    Serial.println("Moving forward");
  }
}

void moveBackward() {
  if (needDirectionChange(HIGH, LOW)) {
    handleDirectionChange(HIGH, LOW);
  } else {
    setMotorTarget(&m1, FULL_SPEED, HIGH, LOW);
    setMotorTarget(&m2, FULL_SPEED, HIGH, LOW);
    lastCommand = "B";
    Serial.println("Moving backward");
  }
}

void turnLeft() {
  // 原地左转：左轮前进，右轮后退
  setMotorTarget(&m1, TURN_SPEED, HIGH, LOW);   // 左轮正转
  setMotorTarget(&m2, TURN_SPEED, LOW, HIGH);   // 右轮反转
  lastCommand = "L";
  Serial.println("Turning left");
}

void turnRight() {
  // 原地右转：左轮后退，右轮前进
  setMotorTarget(&m1, TURN_SPEED, LOW, HIGH);   // 左轮反转
  setMotorTarget(&m2, TURN_SPEED, HIGH, LOW);   // 右轮正转
  lastCommand = "R";
  Serial.println("Turning right");
}

void mixSpeedLeftHalf() {
  setMotorTarget(&m1, HALF_SPEED, LOW, HIGH);
  setMotorTarget(&m2, FULL_SPEED, LOW, HIGH);
  lastCommand = "3";
  Serial.println("Mixed speed: Left half, Right full");
}

void mixSpeedRightHalf() {
  setMotorTarget(&m1, FULL_SPEED, LOW, HIGH);
  setMotorTarget(&m2, HALF_SPEED, LOW, HIGH);
  lastCommand = "4";
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
  m->isAccelerating = true;
}

void stopAll() {
  setMotorTarget(&m1, 0, LOW, LOW);
  setMotorTarget(&m2, 0, LOW, LOW);
  applyMotorState(&m1);
  applyMotorState(&m2);
}

void checkCommandTimeout() {
  // 修复：如果有待处理的切换，不触发超时停止
  if (pendingDirectionChange || pendingTurnChange) {
    lastCmdTime = millis(); // 重置超时计时器
    return;
  }

  if (lastCommand == "S" || lastCommand == "0") return;
  
  unsigned long currentTime = millis();
  
  if (currentTime - lastCmdTime > CMD_TIMEOUT) {
    Serial.println("Command timeout - stopping motors");
    stopMotors();
  }
}

void checkBatteryStatus() {
  int batteryValue = analogRead(BATTERY_PIN);
  float batteryVoltage = (batteryValue * (5.0 / 1023.0)) * VOLTAGE_DIVIDER;
  isBatteryLow = (batteryVoltage <= LOW_BATTERY_THRESHOLD);

  if (isBatteryLow) {
    Serial.println("Low battery voltage, please charge");
  }
}

void sendSensorData() {
  int batteryValue = analogRead(BATTERY_PIN);
  float batteryVoltage = (batteryValue * (5.0 / 1023.0)) * VOLTAGE_DIVIDER;
  int batteryPercentage = map(batteryVoltage * 10, BATTERY_MIN * 10, BATTERY_MAX * 10, 0, 100);
  batteryPercentage = constrain(batteryPercentage, 0, 100);

  Serial.print("BATTERY:");
  Serial.print(batteryPercentage);
  Serial.print(",");
  Serial.println((int)(batteryVoltage * 10));  // 保留一位小数
}

// ========== 命令处理函数 ==========
void processCommand(String cmd) {
  // 低电量时只允许停止命令
  if (isBatteryLow && cmd != "S" && cmd != "0") {
    Serial.println("Low battery, only stop command allowed");
    return;
  }

  // 新命令到来时清除待处理的切换请求
  pendingDirectionChange = false;
  pendingTurnChange = false;

  // 处理停止命令
  if (cmd == "0" || cmd == "S") {
    stopMotors();
  }
  // 处理前进命令
  else if (cmd == "1" || cmd == "F") {
    moveForward();
  }
  // 处理后退命令
  else if (cmd == "2" || cmd == "B") {
    moveBackward();
  }
  // 处理混合速度 (代码1功能)
  else if (cmd == "3") {
    mixSpeedLeftHalf();
  }
  else if (cmd == "4") {
    mixSpeedRightHalf();
  }
  // 处理转向命令 (代码2功能)
  else if (cmd == "L") {
    // 如果在后退状态，先减速再转向
    if (lastCommand == "B" || lastCommand == "2") {
      // 设置减速停止
      setMotorTarget(&m1, 0, m1.dirA, m1.dirB);
      setMotorTarget(&m2, 0, m2.dirA, m2.dirB);
      
      // 记录待处理的转向请求
      pendingTurnChange = true;
      pendingTurnType = 'L';
      turnChangeStart = millis();
      Serial.println("Pending left turn after stop");
    } else {
      // 非后退状态直接转向
      turnLeft();
    }
  }
  else if (cmd == "R") {
    // 如果在后退状态，先减速再转向
    if (lastCommand == "B" || lastCommand == "2") {
      // 设置减速停止
      setMotorTarget(&m1, 0, m1.dirA, m1.dirB);
      setMotorTarget(&m2, 0, m2.dirA, m2.dirB);
      
      // 记录待处理的转向请求
      pendingTurnChange = true;
      pendingTurnType = 'R';
      turnChangeStart = millis();
      Serial.println("Pending right turn after stop");
    } else {
      // 非后退状态直接转向
      turnRight();
    }
  }
  // 处理数据请求
  else if (cmd == "GET_DATA") {
    sendSensorData();
  }
  else {
    Serial.println("Unknown command: " + cmd);
  }
}
