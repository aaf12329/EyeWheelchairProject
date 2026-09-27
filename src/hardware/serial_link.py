"""轮椅硬件接口（串口输出层）：把界面的"意图"翻译成固件指令并保持心跳。

对应固件：firmware/wheelchair_controller.ino（协议表见 README"硬件与固件"一节）。

职责与固件安全机制的配合：
  - set_intent("左转"/"前进"/"右转"/"停") → 翻译成协议字母 L/F/R/S 立即下发
  - update(now) 每帧调用：当前指令每 1 秒重发一次（固件 3 秒收不到指令会自动急停，
    这里留 3 倍余量）、每 2 秒发一次 GET_DATA 收电池回报、随时接收固件消息
  - 低电双重保险：固件本身会拒绝非停止指令；本模块收到低电信号后也会把意图强制为"停"
  - port=None 时进入模拟模式：除真实收发外行为完全一致，方便无硬件开发与测试

用法（见 gaze_blink_confirm_demo.py 的 main()）：
    link = WheelchairLink("COM3")      # 或 None → 模拟模式
    link.open()                        # 串口打不开会自动退回模拟模式
    每帧: info = link.update(now)      # 拿 LinkStatus 画到界面上
    确认成功时: link.set_intent("左转")
    退出前: link.set_intent("停"); link.close()
"""
from dataclasses import dataclass
import time

HEARTBEAT_INTERVAL = 1.0        # 当前指令重发间隔（秒）
BATTERY_POLL_INTERVAL = 2.0     # GET_DATA 轮询间隔（秒）
LOW_BATTERY_ENTER_V = 19.0      # 电压跌破它 → 视为低电（与固件同一阈值）
LOW_BATTERY_EXIT_V = 19.5       # 回升超过它 → 解除低电（迟滞，方向同固件）

# 界面意图 → 固件协议字母
COMMAND_MAP = {"左转": "L", "右转": "R", "前进": "F", "停": "S"}
VALID_COMMANDS = {"S", "F", "B", "L", "R", "3", "4"}

FIRMWARE_LOW_MSG = "Low battery voltage, please charge"
FIRMWARE_RECOVERED_MSG = "Battery voltage recovered"


@dataclass(frozen=True)
class LinkStatus:
    """一帧的链路快照，供界面显示。由 WheelchairLink.update() 产出。"""

    mode: str                            # "模拟" 或 串口名（如 "COM3"）
    connected: bool                      # 串口是否真的打开
    intent: str                          # 当前意图（协议字母 S/F/L/R/3/4；"" = 无）
    send_count: int                      # 已发送的指令条数（含心跳）
    battery: tuple[int, float] | None    # (百分比, 电压V)；没收到过回报时是 None
    is_low_battery: bool                 # 低电标志（电压或固件消息触发）
    firmware_message: str                # 固件最近发来的一行消息（如 "Direction changed"）
    last_error: str                      # 最近一次错误说明（空串 = 无错）


class WheelchairLink:
    """串口硬件链路。所有收发都不阻塞主循环（timeout=0 + 随到随收）。

    port=None 时为模拟模式：照常计数与计时，只是不碰真串口。
    测试时可以注入 port_factory 来替代 pyserial（见 tests/test_serial_link.py）。
    """

    def __init__(
        self,
        port: str | None = None,
        *,
        baudrate: int = 9600,
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        battery_poll_interval: float = BATTERY_POLL_INTERVAL,
        port_factory=None,
    ) -> None:
        self._port_path = port
        self._baudrate = baudrate
        self._heartbeat_interval = heartbeat_interval
        self._battery_poll_interval = battery_poll_interval
        self._port_factory = port_factory  # None → 用 pyserial
        self._port = None
        self._mode = "模拟"
        self._connected = False
        self._intent = ""
        self._send_count = 0
        self._last_sent_at = float("-inf")   # 上次下发当前指令的时刻（心跳用）
        self._last_poll_at = float("-inf")   # 上次 GET_DATA 的时刻
        self._battery = None
        self._is_low = False
        self._firmware_message = ""
        self._last_error = ""
        self._rx = b""                       # 串口接收拼行缓冲

    # ---- 只读属性（main 里判断用）----
    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def last_error(self) -> str:
        return self._last_error

    def open(self) -> None:
        """打开串口；port=None 或打开失败都落回模拟模式（不抛异常，错误记在 last_error）。"""
        if self._port_path is None:
            return
        try:
            import serial
            factory = self._port_factory or serial.Serial
            self._port = factory(self._port_path, baudrate=self._baudrate, timeout=0, write_timeout=1)
            self._port.reset_input_buffer()
            self._connected = True
            self._mode = self._port_path
        except Exception as exc:  # noqa: BLE001 —— 任何打不开的原因都退回模拟
            self._port = None
            self._connected = False
            self._mode = "模拟"
            self._last_error = f"无法打开串口 {self._port_path}：{exc}"

    def set_intent(self, intent, now: float | None = None) -> None:
        """设定运动意图（"左转"/"前进"/"右转"/"停" 或协议字母），变化时立即下发。

        低电时除停止外的意图都会被强制为"停"（固件端也会再拦一次，双保险）。
        """
        letter = COMMAND_MAP.get(intent, str(intent).strip().upper())
        if letter not in VALID_COMMANDS:
            self._last_error = f"未知意图：{intent!r}"
            return
        if self._is_low and letter != "S":
            self._last_error = "低电量：意图已强制为停止"
            letter = "S"
        if letter == self._intent:
            return                           # 意图没变：不重复下发（心跳会保持）
        self._intent = letter
        self._send(letter, now if now is not None else time.perf_counter())

    def update(self, now: float) -> LinkStatus:
        """每帧调用：心跳重发、GET_DATA 轮询、接收固件消息。返回本帧 LinkStatus。"""
        if self._connected:
            if self._intent and now - self._last_sent_at >= self._heartbeat_interval:
                self._send(self._intent, now)                    # 心跳
            if now - self._last_poll_at >= self._battery_poll_interval:
                self._last_poll_at = now
                self._send("GET_DATA", now, resets_heartbeat=False)
        self._drain()
        return self._status()

    def close(self) -> None:
        """先发"停"再挂断；模拟模式下只是清引用。"""
        if self._connected and self._intent != "S":
            self._send("S", time.perf_counter())
        if self._port is not None:
            try:
                self._port.close()
            except Exception:  # noqa: BLE001 —— 关闭失败也无需上抛
                pass
            self._port = None
        self._connected = False

    # ---- 内部实现 ----
    def _send(self, letter: str, now: float, *, resets_heartbeat: bool = True) -> None:
        if self._connected and self._port is not None:
            try:
                self._port.write((letter + "\n").encode("ascii"))
            except Exception as exc:  # noqa: BLE001 —— 设备拔掉等情况
                self._connected = False
                self._last_error = f"串口写入失败：{exc}"
                return
        self._send_count += 1                # 模拟模式也计数（界面显示"已发 N 条"）
        if resets_heartbeat:
            self._last_sent_at = now

    def _drain(self) -> None:
        """把串口里已到达的字节全部收下来，按行交给 _handle_line。非阻塞。"""
        if not self._connected or self._port is None:
            return
        try:
            while self._port.in_waiting:
                self._rx += self._port.read(self._port.in_waiting)
        except Exception as exc:  # noqa: BLE001
            self._connected = False
            self._last_error = f"串口读取失败：{exc}"
            return
        if len(self._rx) > 256:              # 防御：垃圾数据再长也只留尾部
            self._rx = self._rx[-256:]
        while b"\n" in self._rx:
            raw, self._rx = self._rx.split(b"\n", 1)
            self._handle_line(raw.decode("utf-8", errors="ignore").strip())

    def _handle_line(self, line: str) -> None:
        """处理固件发来的一行：BATTERY 回报 / 低电消息 / 其他消息。"""
        if not line:
            return
        if line.startswith("BATTERY:"):
            try:
                pct_text, v_text = line[len("BATTERY:"):].split(",")
                voltage = int(v_text) / 10.0
                self._battery = (int(pct_text), voltage)
                # 与固件一致的迟滞：跌破 19.0 判低，回升过 19.5 才解除
                if not self._is_low and voltage <= LOW_BATTERY_ENTER_V:
                    self._is_low = True
                elif self._is_low and voltage >= LOW_BATTERY_EXIT_V:
                    self._is_low = False
                if self._is_low and self._intent != "S":
                    self.set_intent("停")    # 双保险：Python 端也停
            except ValueError:
                self._last_error = f"电池回报格式异常：{line}"
        else:
            self._firmware_message = line
            if line == FIRMWARE_LOW_MSG:
                self._is_low = True
            elif line == FIRMWARE_RECOVERED_MSG:
                self._is_low = False

    def _status(self) -> LinkStatus:
        return LinkStatus(
            mode=self._mode,
            connected=self._connected,
            intent=self._intent,
            send_count=self._send_count,
            battery=self._battery,
            is_low_battery=self._is_low,
            firmware_message=self._firmware_message,
            last_error=self._last_error,
        )
