# 数据集说明

## 放置方法

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

图片和标签必须同名。每行标签使用归一化 YOLO Detection 格式：

```text
class_id x_center y_center width height
```

## 类别

| ID | 类别 | 中文名 |
|---:|---|---|
| 0 | `keyboard` | 键盘 |
| 1 | `nongfu_spring` | 农夫山泉 |
| 2 | `phone` | 手机 |

## 数据集卡片（放入数据后填写）

- 采集人：TODO
- 采集日期：TODO
- 摄像头：TODO
- 原始分辨率：TODO
- 标注工具：TODO
- 数据授权/隐私说明：TODO

| 划分 | 图片数 | 实例数 | keyboard | nongfu_spring | phone |
|---|---:|---:|---:|---:|---:|
| train | TODO | TODO | TODO | TODO | TODO |
| val | TODO | TODO | TODO | TODO | TODO |
| test | TODO | TODO | TODO | TODO | TODO |

## 划分原则

1. 训练、验证和测试互不重复。
2. 同一段视频的连续相似帧放在同一划分中，避免泄漏。
3. 测试集应覆盖遮挡、侧视、远距离、明暗变化和多物体场景。
4. 用 `python src/check_dataset.py` 检查标签与跨集重复。
