"""pytest 配置：让 tests/ 能直接 import 项目源码。

- src/interaction：blink_preview / gaze_blink_confirm_demo（脚本式模块）
- src：hardware.serial_link（包式模块）
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
INTERACTION_DIR = SRC_DIR / "interaction"
for path in (str(SRC_DIR), str(INTERACTION_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
