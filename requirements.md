# 运行依赖（ultralytics 会自动带上 CPU 版 torch）
python -m pip install opencv-python ultralytics numpy pillow pyserial pytest
# 双引擎对比脚本（blink_preview / gaze_direction_preview）额外需要 mediapipe；
# 只用 YOLO 引擎（--no-mp）则不需要
python -m pip install mediapipe
