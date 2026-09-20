"""pytest 配置：让 tests/ 能直接 import src/interaction 下的脚本模块。"""
import sys
from pathlib import Path

INTERACTION_DIR = Path(__file__).resolve().parents[1] / "src" / "interaction"
if str(INTERACTION_DIR) not in sys.path:
    sys.path.insert(0, str(INTERACTION_DIR))
