
# 实验一：目标检测与识别

本项目使用 Ultralytics YOLO26n 训练轻量级桌面物体检测模型，并在 Jetson 上实时显示类别、检测框、置信度和 FPS，同时通过 ROS2 发布结构化检测结果。选择 Nano 版本是为了兼顾三类物体的检测精度和 Jetson 上不低于 5 FPS 的实时性要求。

## 验收进度

- [x] 建立不少于 2 类物体的数据和训练接口（`keyboard`、`nongfu_spring`、`phone`）
- [x] 导入 596 张实际采集并标注的数据（train/val/test = 428/73/95）
- [x] 数据集完整性、标签合法性和跨集重复检查程序
- [x] YOLO 迁移学习程序和可复现训练配置
- [x] 完整测试集指标、曲线和典型错误样例保存程序
- [x] 20 件实物交互式验收程序
- [x] Jetson 实时检测、视频保存和 ROS2 发布程序
- [ ] 添加训练后的 `models/best.pt`
- [ ] 完成 20 件实物测试，正确率达到 80% 以上
- [ ] 在 Jetson 实测完整检测速度，平均 FPS 达到 5 以上
- [ ] 添加结果视频和完成后的实验报告

> 未打勾项目需要实际数据集、训练权重或 Jetson 实测结果，不在仓库中伪造数值。

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

当前数据来自 [`boff868/2026Summer-integrated-robot-grouptask/dataset_self`](https://github.com/boff868/2026Summer-integrated-robot-grouptask/tree/main/dataset_self)，固定在源仓库提交 `d06626c71780a5c0a8283d76d09e7b90d0238680`。源数据的 `valid` 目录在导入时规范为本项目的 `val`；除 1 个舍入导致的微小越界框经过可追溯修正外，其余图片和标签内容不变。详细来源、数量、修正和授权信息见 [`data/README.md`](data/README.md)。

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

数据描述和类别编号见 [`configs/data.yaml`](configs/data.yaml) 和 [`data/README.md`](data/README.md)。训练默认从 COCO 预训练权重 `yolo26n.pt` 迁移学习，而不是从零开始训练。

## 训练电脑运行

```bash
cd task1
python -m venv .venv
```

Linux/macOS：

```bash
source .venv/bin/activate
pip install -r requirements-train.txt
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements-train.txt
```

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

## 20 件实物验收

```bash
python src/acceptance_test.py --model models/best.pt --camera 0 --samples 20
```

窗口中按真实物体的类别编号记录当前样本，例如按 `0` 记录键盘、按 `2` 记录手机。完成 20 件后自动生成：

- `results/acceptance_test/test_20_objects.csv`
- `results/acceptance_test/summary.json`
- `results/acceptance_test/images/`

只有完成 20 件并且正确率不低于 80% 时，`passed` 才会为 `true`。

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

按 `q` 结束检测。详细的 Jetson 环境、自启动和验收方法见 [`docs/Jetson部署说明.md`](docs/Jetson部署说明.md)。

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
