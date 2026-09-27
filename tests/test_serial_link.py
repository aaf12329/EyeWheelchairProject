"""WheelchairLink 的行为测试：全部用注入的假串口，不需要真硬件。

锁定的行为（与 firmware/wheelchair_controller.ino 的协议和安全机制对应）：
  1. 模拟模式（port=None）：不碰串口，其余行为一致
  2. 意图翻译：左转/前进/右转/停 → L/F/R/S，变化时立即下发
  3. 心跳：当前指令每 1 秒重发一次（配合固件 3 秒急停）
  4. 电池回报解析 + 低电锁（跌破 19.0 落锁、回升过 19.5 解锁，与固件一致）
  5. 低电时非停止意图被强制为"停"
  6. close() 先发"停"再挂断
"""
import pytest

from hardware.serial_link import COMMAND_MAP, LinkStatus, WheelchairLink


class FakeSerial:
    """假串口：write 记录到 tx，read 从预置的 rx 里取。"""

    def __init__(self, rx: bytes = b""):
        self._rx = bytearray(rx)
        self.tx = []          # 每次write的原始bytes
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def read(self, n=-1):
        if n in (-1, None):
            n = len(self._rx)
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def write(self, data) -> int:
        self.tx.append(bytes(data))
        return len(data)

    def close(self) -> None:
        self.closed = True

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    @property
    def written(self) -> bytes:
        return b"".join(self.tx)


def make_link(rx: bytes = b"") -> tuple[WheelchairLink, FakeSerial]:
    fake = FakeSerial()
    link = WheelchairLink("COM9", port_factory=lambda port, **kw: fake)
    link.open()
    if rx:
        # open() 会清空输入缓冲（清掉上次残留是正确行为），所以测试数据要在 open 之后灌
        fake._rx.extend(rx)
    return link, fake


def sent_letters(fake: FakeSerial) -> list[str]:
    """把 write 进去的 b"F\\n" 还原成 ["F", ...]，方便断言。"""
    return [chunk[:-1].decode("ascii") for chunk in fake.tx if chunk.endswith(b"\n")]


def movement_letters(fake: FakeSerial) -> list[str]:
    """只看运动指令，滤掉 GET_DATA 轮询（第一次 update 就会发一次拿电池）。"""
    return [x for x in sent_letters(fake) if x != "GET_DATA"]


# ---------------------------- 模拟模式 ----------------------------


def test_simulated_mode_counts_without_serial():
    link = WheelchairLink(None)
    link.open()
    status = link.set_intent("前进", now=0.0)
    info = link.update(0.0)
    assert info.mode == "模拟"
    assert info.connected is False
    assert info.intent == "F"
    assert info.send_count == 1


# ---------------------------- 意图翻译与心跳 ----------------------------


def test_intent_maps_to_protocol_letters():
    assert COMMAND_MAP == {"左转": "L", "右转": "R", "前进": "F", "停": "S"}
    link, fake = make_link()
    link.set_intent("前进", now=0.0)
    assert sent_letters(fake) == ["F"]
    link.set_intent("左转", now=0.1)
    assert sent_letters(fake) == ["F", "L"]


def test_duplicate_intent_is_not_resent_immediately():
    link, fake = make_link()
    link.set_intent("前进", now=0.0)
    link.set_intent("前进", now=0.2)          # 相同意图：跳过
    assert sent_letters(fake) == ["F"]


def test_heartbeat_resends_current_command_every_second():
    link, fake = make_link()
    link.set_intent("前进", now=0.0)
    link.update(0.0)
    link.update(0.5)                           # 未到 1 秒：不发
    assert movement_letters(fake) == ["F"]
    link.update(1.05)                          # 到 1 秒：心跳重发
    assert movement_letters(fake) == ["F", "F"]
    link.update(1.6)                           # 间隔不足：仍只有两次
    assert movement_letters(fake) == ["F", "F"]


def test_battery_poll_sends_get_data():
    link, fake = make_link()
    link.update(0.0)                           # 无意图也要轮询电池
    link.update(1.0)                           # 轮询间隔 2 秒：不发
    link.update(2.05)                          # 发 GET_DATA
    assert "GET_DATA" in sent_letters(fake)


# ---------------------------- 电池回报与低电锁 ----------------------------


def test_battery_reply_is_parsed():
    link, _ = make_link(rx=b"BATTERY:87,218\n")
    info = link.update(0.0)
    assert info.battery == (87, 21.8)
    assert info.is_low_battery is False


def test_low_battery_locks_intent_to_stop():
    link, fake = make_link(rx=b"BATTERY:10,188\n")
    info = link.update(0.0)
    assert info.is_low_battery is True
    link.set_intent("前进", now=0.1)           # 低电时的非停止意图
    info = link.update(0.2)
    assert info.intent == "S"
    assert movement_letters(fake) == ["S"]     # 前进没发出去，发的是停（滤掉 GET_DATA 轮询）


def test_low_battery_hysteresis_direction():
    link, _ = make_link()
    link._handle_line("BATTERY:0,189")         # 18.9V → 落锁
    assert link.update(0.0).is_low_battery is True
    link._handle_line("BATTERY:0,192")         # 19.2V：在迟滞区间内 → 仍锁
    assert link.update(0.0).is_low_battery is True
    link._handle_line("BATTERY:0,196")         # 19.6V：过解锁线 → 解锁
    assert link.update(0.0).is_low_battery is False


def test_firmware_low_message_also_locks():
    link, _ = make_link()
    link._handle_line("Low battery voltage, please charge")
    assert link.update(0.0).is_low_battery is True
    link._handle_line("Battery voltage recovered")
    assert link.update(0.0).is_low_battery is False


def test_other_firmware_lines_are_kept_as_message():
    link, _ = make_link(rx=b"Direction changed\n")
    info = link.update(0.0)
    assert info.firmware_message == "Direction changed"


# ---------------------------- 生命周期 ----------------------------


def test_close_sends_stop_then_closes():
    link, fake = make_link()
    link.set_intent("前进", now=0.0)
    assert sent_letters(fake) == ["F"]
    link.close()
    assert sent_letters(fake)[-1] == "S"
    assert fake.closed is True
    assert link.update(0.1).connected is False


def test_open_failure_falls_back_to_simulated():
    def broken_factory(port, **kw):
        raise OSError("port not found")
    link = WheelchairLink("COM99", port_factory=broken_factory)
    link.open()
    info = link.update(0.0)
    assert info.connected is False
    assert "COM99" in info.last_error
    assert info.mode == "模拟"
