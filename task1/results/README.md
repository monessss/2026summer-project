# 结果与验收证据

本目录只保存程序真实生成的结果，不手工编造指标。

当前状态：正式训练、独立测试集评估、20 件实物验收和 Jetson 实测均未执行，因此本目录没有指标、验收 CSV、错误案例或视频结果。目录中的 `.gitkeep` 仅用于保留目录结构。

| 文件/目录 | 来源 | 用途 |
|---|---|---|
| `dataset_report.json` | `check_dataset.py` | 数据数量、类别分布和标签错误 |
| `training_results.csv` | `train.py` | 每个 epoch 的损失和验证指标 |
| `metrics.json` | `evaluate.py` | Precision、Recall、mAP 和速度 |
| `figures/training/` | `train.py` | 训练曲线和训练阶段图表 |
| `figures/test/` | `evaluate.py` | 独立测试集混淆矩阵、PR/F1 曲线 |
| `error_examples/` | `evaluate.py` | 典型误检、漏检和类别错误 |
| `error_cases.csv` | `evaluate.py` | 错误案例清单 |
| `acceptance_test/` | `acceptance_test.py` | 20 件实物正确率验收 |
| `jetson_detections.jsonl` | `jetson_ros2_node.py` | ROS2 发布内容留档 |
| `fps_jetson.csv`、`fps_jetson.json` | `jetson_ros2_node.py` | PyTorch/FP16 基准逐帧速度和通过状态 |
| `fps_jetson_engine.csv`、`fps_jetson_engine.json` | `jetson_ros2_node.py` | TensorRT 后备测试逐帧速度和通过状态 |
| `videos/` | `jetson_ros2_node.py` | Jetson 结果视频 |

正式验收只采用上述程序生成且能与模型 SHA-256 对应的文件。
