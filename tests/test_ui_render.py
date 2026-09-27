"""render_overlay 的渲染冒烟测试：四种状态 × 三种卡片画法都不崩、都真的画了东西。"""
import numpy as np
import pytest

from gaze_blink_confirm_demo import GazeBlinkStatus, render_overlay
from hardware.serial_link import LinkStatus

FRAME = 1 / 30


def make_status(state, active, cal=None, stab=None):
    return GazeBlinkStatus(
        state=state, message="测试消息", score=0.5, ear=0.5,
        active=active, calibration_progress=cal, stability_progress=stab,
    )


def fake_link(connected, battery=None, low=False):
    return LinkStatus(
        mode="COM9" if connected else "模拟", connected=connected, intent="F",
        send_count=3, battery=battery, is_low_battery=low,
        firmware_message="", last_error="",
    )


CASES = [
    make_status("校准", None, cal=0.4),
    make_status("选择方向", "左转", stab=0.6),
    make_status("等待眨眼", "左转"),
    make_status("确认成功", "右转"),
]


@pytest.mark.parametrize("status", CASES, ids=[s.state for s in CASES])
def test_all_states_render(status):
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    out = render_overlay(frame, status, now=0.37, fps=29.6, link_info=fake_link(False))
    assert out.shape == frame.shape
    assert out.any()                      # 至少画了点东西（面板/文字像素）


def test_panel_and_cards_draw_pixels():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    out = render_overlay(frame, make_status("选择方向", "左转", stab=0.5), now=0.0, fps=30.0, link_info=None)
    # 面板底色 (30,30,36) 应大量出现
    panel = out[:164, :640]
    assert int(np.count_nonzero(np.all(panel == np.array((30, 30, 36), dtype=np.uint8), axis=2))) > 5000
    # 左转卡片是琥珀边 (40,180,255)，应出现在卡片区域
    card = out[390:515, 35:280]
    assert int(np.count_nonzero(np.all(card == np.array((40, 180, 255), dtype=np.uint8), axis=2))) > 100


def test_progress_bar_fills_with_progress():
    empty = render_overlay(np.zeros((540, 960, 3), dtype=np.uint8),
                           make_status("校准", None, cal=0.0), now=0.0, fps=30.0, link_info=None)
    full = render_overlay(np.zeros((540, 960, 3), dtype=np.uint8),
                          make_status("校准", None, cal=1.0), now=0.0, fps=30.0, link_info=None)
    row_empty = empty[84, 70:330]         # 进度条所在的一行
    row_full = full[84, 70:330]
    assert int(np.count_nonzero(np.all(row_empty == np.array((62, 62, 70), dtype=np.uint8), axis=1))) > 200
    assert int(np.count_nonzero(np.all(row_full == np.array((62, 62, 70), dtype=np.uint8), axis=1))) == 0
