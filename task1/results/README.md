# 结果与验收证据

本目录只保存程序真实生成的结果，不手工编造指标。

| 文件/目录 | 来源 | 用途 |
|---|---|---|
| `dataset_report.json` | `check_dataset.py` | 数据数量、类别分布和标签错误 |
| `training_results.csv` | `train.py` | 每个 epoch 的损失和验证指标 |
| `metrics.json` | `evaluate.py` | Precision、Recall、mAP 和速度 |
| `figures/` | `evaluate.py` | 混淆矩阵、PR/F1 曲线 |
| `error_examples/` | `evaluate.py` | 典型误检、漏检和类别错误 |
| `error_cases.csv` | `evaluate.py` | 错误案例清单 |
| `acceptance_test/` | `acceptance_test.py` | 20 件实物正确率验收 |
| `jetson_detections.jsonl` | `jetson_ros2_node.py` | ROS2 发布内容留档 |
| `videos/` | `jetson_ros2_node.py` | Jetson 结果视频 |

最终提交前，请确保所有数值和文件都已由真实运行生成。
