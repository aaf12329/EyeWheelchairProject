"""通用基础包：路径地址簿 / 摄像头 / 绘图。

依赖方向（参考 My_Agent 的 Store 原则）：
    入口脚本(interaction/*) ──► vision/yolo_backend ──► common/paths（只读地址簿）
    入口脚本 ──► common/camera_utils、common/draw_utils
common/ 内各模块之间零 import；paths.py 纯地址簿，不调用任何人。
"""
