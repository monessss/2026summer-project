# Jetson 部署说明

## 1. 记录平台

- Jetson 型号：TODO
- JetPack 版本：TODO
- CUDA/cuDNN/TensorRT 版本：TODO
- ROS2 发行版：TODO
- 功耗模式：TODO
- 摄像头型号和分辨率：TODO

## 2. 环境

1. 安装与 JetPack 匹配的 NVIDIA PyTorch/torchvision wheel。
2. 确认 `python3 -c "import torch; print(torch.cuda.is_available())"` 输出 `True`。
3. 安装 ROS2，并确认 `rclpy` 和 `std_msgs` 可导入。
4. 安装剩余依赖：

```bash
cd task1
python3 -m pip install -r requirements-jetson.txt
source /opt/ros/<distro>/setup.bash
```

Jetson 上的 OpenCV 建议使用 JetPack 自带版本，不要盲目用 pip 覆盖 CUDA/GStreamer 支持。

## 3. 检查摄像头

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

## 4. 运行

```bash
python3 src/jetson_ros2_node.py \
  --model models/best.pt \
  --camera 0 \
  --width 640 \
  --height 480 \
  --imgsz 640 \
  --conf 0.50 \
  --device 0 \
  --half \
  --topic /yolo/detections \
  --save-video results/videos/jetson_demo.mp4 \
  --save-jsonl results/jetson_detections.jsonl \
  --metrics-csv results/fps_jetson.csv \
  --warmup-frames 30 \
  --max-measured-frames 500
```

无显示器时加 `--headless`。

## 5. ROS2 验证

```bash
ros2 node list
ros2 topic list
ros2 topic type /yolo/detections
ros2 topic echo /yolo/detections
ros2 topic hz /yolo/detections
```

## 6. FPS 验收

1. Jetson 开启实验要求的固定功耗模式。
2. 先预热 30–50 帧。
3. 连续运行至少 500 帧。上述命令会在预热后自动停止。
4. 视频中保留 FPS 叠加显示，同时保存 JSONL、`fps_jetson.csv` 和 `fps_jetson.json`。
5. JSON 摘要自动给出平均、中位、最低和最高 FPS，以平均 FPS 判定是否达到 5 FPS。
6. 训练电脑的 RTX 速度不得代替 Jetson 实测结果。

## 7. 可选 TensorRT

如果 PyTorch FP16 不能稳定达到 5 FPS，可在 Jetson 上导出：

```bash
yolo export model=models/best.pt format=engine imgsz=640 half=True device=0
```

导出后用 `.engine` 文件替换 `--model`，并在报告中说明精度和速度差异。
