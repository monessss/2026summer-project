
# 实验一：目标检测与识别

本项目使用 Ultralytics YOLO26n 训练轻量级桌面物体检测模型，并在 Jetson 上实时显示类别、检测框、置信度和 FPS，同时通过 ROS2 发布结构化检测结果。选择 Nano 版本是为了兼顾三类物体的检测精度和 Jetson 上不低于 5 FPS 的实时性要求。

## 当前状态

| 项目 | 状态 |
|---|---|
| 三类数据集及严格审计 | 已完成并通过 |
| YOLO26n 训练、评估和模型发布程序 | 已完成；100 轮正式训练及独立测试集评估通过 |
| 20 件实物验收程序 | 已完成，固定为 7 个键盘、7 个农夫山泉、6 个手机 |
| Jetson 检测、视频记录、FPS 统计和 ROS2 发布程序 | 已完成；ROS2 实机发布已由使用者确认成功 |
| `models/best.pt` 与训练指标 | 已生成；SHA-256 为 `7ab7a1cf4c08e2eb803d5666a6caa6931256cfabfe04dc8e9c98385322fbcdcf` |
| 20 件实物正确率 | 未测量；仓库不声明 80% 已达成 |
| Jetson 平均 FPS | 未测量；仓库不声明 5 FPS 已达成 |
| 结果视频 | Windows DJI 40 秒演示已在本机生成，按要求不上传；Jetson 视频未提交 |

正式训练于 2026-08-31 在 NVIDIA GeForce RTX 4060 Laptop GPU 上完成。最佳权重来自第 93 个 epoch，验证集 `mAP50=0.884`、`mAP50-95=0.695`。独立测试集包含 88 张图片、335 个实例，结果为 `Precision=0.791`、`Recall=0.842`、`mAP50=0.864`、`mAP50-95=0.697`。这些电脑端指标不替代 20 件实物正确率或 Jetson FPS 验收。

Windows DJI 演示视频已在本机录制：40.0 秒、1280×720、15 FPS，画面包含检测框、类别、置信度和完整循环 FPS，并出现键盘与手机同时检测。按本次提交要求，视频文件不上传 GitHub。ROS2 实机发布已由使用者确认成功；仓库没有提交 `topic echo`、话题频率或 Jetson FPS 记录，因此不声明这些数值。

## 目录

```text
task1/
├── configs/                    # 数据集和训练参数
├── data/                       # YOLO 数据集
├── models/                     # 最佳权重和模型说明
├── src/
│   ├── check_dataset.py        # 数据集审计
│   ├── train.py                # 模型训练
│   ├── evaluate.py             # 测试指标与错误案例
│   ├── acceptance_test.py      # 20 件实物验收
│   └── jetson_ros2_node.py     # Jetson 推理与 ROS2 发布
├── results/                    # 指标、图片、CSV 和视频
└── docs/                       # 运行、部署、ROS2 和报告文档
```

## 数据集

当前数据来自 [`boff868/2026Summer-integrated-robot-grouptask/dataset_self`](https://github.com/boff868/2026Summer-integrated-robot-grouptask/tree/main/dataset_self)，固定在源仓库提交 `d06626c71780a5c0a8283d76d09e7b90d0238680`。本项目修正 1 个舍入越界框，并将 5 个跨集合的连续拍摄序列归并到单一集合。最终划分为 train 433 张、val 75 张、test 88 张。详细来源、统计和逐项修正记录见 [`data/README.md`](data/README.md)。

数据已按标准 YOLO Detection 格式放入：

```text
data/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

图片和标签必须同名，例如 `images/train/phone_001.jpg` 对应 `labels/train/phone_001.txt`。标签行为：

```text
class_id x_center y_center width height
```

数据描述和类别编号见 [`configs/data.yaml`](configs/data.yaml) 和 [`data/README.md`](data/README.md)。训练固定从 COCO 预训练权重 `yolo26n.pt` 迁移学习，不从零开始训练。

## 训练电脑运行

训练环境固定使用 Python 3.10；应用依赖固定在 `requirements-train.txt`。PyTorch 使用与训练电脑 NVIDIA 驱动匹配的官方构建，实际 PyTorch、CUDA、GPU 和 Ultralytics 版本由训练程序写入 `models/model_info.json`。

Linux/macOS：

```bash
cd task1
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

Windows PowerShell：

```powershell
cd C:\yolo\2026summer-project\task1
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
```

Windows 正式训练和 `evaluate.py` 必须从纯 ASCII 路径运行；固定路径为 `C:\yolo\2026summer-project\task1`。这是为了避开第三方图像加载器对中文路径的兼容问题。数据审计脚本本身支持中文路径。

先检查数据，再训练：

```bash
python src/check_dataset.py --data configs/data.yaml
python src/train.py --config configs/train.yaml
```

训练成功后，程序会将验证集表现最好的权重复制到 `models/best.pt`，并在 `models/model_info.json` 中记录 SHA-256、类别和训练配置。

## 完整测试集评估

```bash
python src/evaluate.py \
  --model models/best.pt \
  --data configs/data.yaml \
  --output results
```

输出包括：

- `results/metrics.json`：Precision、Recall、mAP50、mAP50-95 和速度。
- `results/figures/`：混淆矩阵、PR/F1 曲线等。
- `results/error_examples/`：典型误检、漏检和类别错误图片。
- `results/error_cases.csv`：错误案例的可追溯记录。

## Windows 摄像头预览

在训练电脑上连接摄像头后运行：

```powershell
python src/windows_demo.py --model models/best.pt --camera 0 --device 0
```

窗口会实时显示类别、检测框、置信度和完整循环 FPS；按 `q` 或 `Esc` 退出。该入口不依赖 ROS2，也不会生成或覆盖正式的 20 件验收记录。若相机 0 无法打开，可依次尝试 `--camera 1`、`--camera 2`。

录制固定 40 秒演示视频：

```powershell
python src/windows_demo.py --camera 1 --width 1280 --height 720 --duration 40 --save-video results/videos/windows_dji_demo.mp4
```

## 20 件实物验收

```bash
python src/acceptance_test.py --model models/best.pt --camera 0
```

每次画面只放置一个待测物体，按真实类别编号记录样本：`0` 为键盘，`1` 为农夫山泉，`2` 为手机。程序固定记录 20 件，配额为 7、7、6；以最高置信度检测的类别作为最终预测。完成后自动生成：

- `results/acceptance_test/test_20_objects.csv`
- `results/acceptance_test/summary.json`
- `results/acceptance_test/images/`

只有三个类别配额全部完成且至少 16 件预测正确时，`passed` 才为 `true`。

## Jetson + ROS2

```bash
python src/jetson_ros2_node.py \
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
  --max-measured-frames 500
```

另开终端验证 ROS2：

```bash
ros2 topic echo /yolo/detections
ros2 topic hz /yolo/detections
```

程序预热 30 帧后测量 500 帧并自动结束；按 `q` 可提前终止。详细部署和验收方法见 [`docs/Jetson部署说明.md`](docs/Jetson部署说明.md)。

## 验收证据对照

| 验收项 | 对应证据 |
|---|---|
| 同时识别不少于 2 类 | Jetson 结果视频和 ROS2 JSONL |
| 20 件物体正确率≥80% | `results/acceptance_test/summary.json` 和 CSV |
| Jetson 实时速度≥5 FPS | 结果视频、ROS2 日志和实验报告的实测数据 |
| 显示类别、框和置信度 | `jetson_ros2_node.py` 及结果视频 |
| ROS2 发布 | `ros2 topic echo` 截图/JSONL 和消息说明 |
| 典型错误案例 | `results/error_examples/` 和 `error_cases.csv` |

## 提交材料

- 数据集：`data/`
- 模型：`models/best.pt` 和 `models/model_info.json`
- 程序：`src/`
- 结果视频：`results/videos/`
- 运行说明：[`docs/运行说明.md`](docs/运行说明.md)
- 实验报告：[`docs/实验报告.md`](docs/实验报告.md)
