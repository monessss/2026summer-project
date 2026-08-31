# Jetson 部署说明

当前状态：使用者已确认 ROS2 实机发布成功；本次未提交 Jetson FPS、话题频率、JSONL 或结果视频，因此以下步骤同时作为复现说明和后续数值验收流程。

## 1. 部署契约

本程序运行在支持 CUDA 的 NVIDIA Jetson 上，使用设备中与 JetPack 匹配的 NVIDIA PyTorch、OpenCV、ROS2 Python 环境和 `models/best.pt`。代码不绑定某一个 Jetson 型号或 ROS2 发行版；实际软件版本、CUDA 设备、摄像头参数和模型 SHA-256 会写入 `results/fps_jetson.json`，验收报告只引用该文件中的实测记录。

固定运行参数如下：

| 参数 | 数值 |
|---|---:|
| 摄像头索引 | 0 |
| 请求分辨率 | 640×480 |
| 训练/推理输入尺寸 | 640 |
| 置信度阈值 | 0.50 |
| IoU 参数 | 0.70 |
| CUDA 设备 | 0 |
| FP16 | 开启 |
| 预热帧数 | 30 |
| 计时帧数 | 500 |
| ROS2 话题 | `/yolo/detections` |

## 2. 环境检查

Jetson 运行环境必须先提供与设备当前 JetPack 完全匹配、CUDA 可用的 PyTorch/torchvision，以及 ROS2 的 `rclpy` 和 `std_msgs`。PyTorch/torchvision 的取得方式以 NVIDIA 官方 [PyTorch for Jetson compatibility table](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html) 为准，不安装桌面版 PyTorch。随后创建能够读取 ROS2 系统包的虚拟环境并安装本项目依赖。应用层 OpenCV 固定为 `opencv-python==4.11.0.86`；PyPI 为 Linux aarch64 提供该版本 wheel。

```bash
cd task1
python3 -m venv --system-site-packages .venv-jetson
source .venv-jetson/bin/activate
python3 -m pip install -r requirements-jetson.txt
python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python3 -c "import cv2, rclpy, std_msgs, ultralytics; print(cv2.__version__, ultralytics.__version__)"
test -n "$ROS_DISTRO" && echo "$ROS_DISTRO"
```

三条命令全部成功后再运行检测程序。

## 3. 摄像头检查

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

摄像头必须能够提供 640×480 画面，程序会把驱动实际返回的宽度、高度和帧率写入 FPS 摘要。

## 4. 正式运行

```bash
python3 src/jetson_ros2_node.py \
  --model models/best.pt \
  --camera 0 \
  --width 640 \
  --height 480 \
  --camera-fps 30 \
  --imgsz 640 \
  --conf 0.50 \
  --iou 0.70 \
  --device 0 \
  --half \
  --topic /yolo/detections \
  --save-video results/videos/jetson_demo.mp4 \
  --save-jsonl results/jetson_detections.jsonl \
  --metrics-csv results/fps_jetson.csv \
  --warmup-frames 30 \
  --max-measured-frames 500
```

无显示器运行时增加 `--headless`。JSONL 每次运行都会覆盖旧文件，避免不同模型或不同实验轮次混在同一结果中。

## 5. ROS2 验证

另开一个已加载相同 ROS2 环境的终端：

```bash
ros2 node list
ros2 topic type /yolo/detections
ros2 topic echo /yolo/detections
ros2 topic hz /yolo/detections
```

节点列表必须包含 `/task1_yolo_detector`，话题类型必须为 `std_msgs/msg/String`。

## 6. FPS 判定

程序跳过前 30 帧，在完整处理循环上连续测量 500 帧。计时范围包含摄像头读取、模型推理、绘图、ROS2 发布和结果记录。程序生成：

- `results/fps_jetson.csv`：逐帧耗时和 FPS。
- `results/fps_jetson.json`：平均值、中位数、最小值、最大值、运行参数、环境版本、模型 SHA-256 和通过状态。
- `results/jetson_detections.jsonl`：每帧发布内容。
- `results/videos/jetson_demo.mp4`：带检测框、类别、置信度和 FPS 的视频。

通过条件为：固定参数与本页“部署契约”一致，摄像头实际返回 640×480，模型与输出路径符合下方一种制品契约，`measured_frames == 500`，并且 `mean_fps >= 5.0`。这些条件由 FPS JSON 的 `protocol_matches` 和 `passed` 字段统一判定；未通过时程序退出码为 1，通过时为 0。训练电脑速度不计入 Jetson 验收。MP4 固定以请求帧率 30 FPS 编码，画面叠加的 FPS 与 CSV/JSON 中的统计值来自实际处理循环。

| 制品契约 | 模型 | 视频 | ROS2 JSONL | FPS CSV/JSON |
|---|---|---|---|---|
| `pytorch_fp16` | `models/best.pt` | `results/videos/jetson_demo.mp4` | `results/jetson_detections.jsonl` | `results/fps_jetson.csv`、`results/fps_jetson.json` |
| `tensorrt_fp16` | `models/best.engine` | `results/videos/jetson_demo_engine.mp4` | `results/jetson_detections_engine.jsonl` | `results/fps_jetson_engine.csv`、`results/fps_jetson_engine.json` |

## 7. TensorRT 路径

本仓库的基准验收先使用 `best.pt` 和 FP16。只有该基准结果低于 5 FPS 时才执行 TensorRT 导出：

```bash
yolo export model=models/best.pt format=engine imgsz=640 half=True device=0
```

导出文件固定为 `models/best.engine`。TensorRT 使用相同参数单独运行，并写入独立结果文件：

```bash
python3 src/jetson_ros2_node.py \
  --model models/best.engine \
  --camera 0 --width 640 --height 480 --camera-fps 30 \
  --imgsz 640 --conf 0.50 --iou 0.70 --device 0 --half \
  --topic /yolo/detections \
  --save-video results/videos/jetson_demo_engine.mp4 \
  --save-jsonl results/jetson_detections_engine.jsonl \
  --metrics-csv results/fps_jetson_engine.csv \
  --warmup-frames 30 --max-measured-frames 500
```

TensorRT 结果不得覆盖 `pytorch_fp16` 基准；最终报告引用 `passed=true` 的制品契约及其模型 SHA-256。
