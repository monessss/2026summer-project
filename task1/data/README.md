# 数据集说明

## 来源与版本

- 源仓库：[`boff868/2026Summer-integrated-robot-grouptask/dataset_self`](https://github.com/boff868/2026Summer-integrated-robot-grouptask/tree/main/dataset_self)
- 固定源提交：`d06626c71780a5c0a8283d76d09e7b90d0238680`
- 数据来源：项目组使用 DJI 设备实拍，经 Roboflow 项目 `2026summer_nongfu_checked` 导出
- 导入规则：源目录 `train`、`valid`、`test` 分别映射为本项目的 `train`、`val`、`test`
- 内容处理：调整目录位置，并修正 1 个因六位小数舍入而越过图像边界 `0.0000015` 的框；详见下方“数据修正记录”
- 授权：源仓库采用 MIT License，副本见 [`SOURCE_LICENSE.txt`](SOURCE_LICENSE.txt)

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

## 数据集卡片

- 采集人：源仓库项目组（个人名单以源仓库记录为准）
- 采集日期：源文件名显示为 2026 年 8 月，精确信息以源仓库为准
- 采集设备：DJI 设备（具体型号未在源数据说明中提供）
- 图片处理：最长边压缩到 800 px
- 标注工具：Roboflow
- 标注形式：归一化 YOLO Detection 框；原多边形已转为最小外接框
- 数据授权/隐私说明：遵循源仓库 MIT License；提交实验前仍应由采集者确认画面不含敏感或未授权内容

| 划分 | 图片数 | 实例数 | keyboard | nongfu_spring | phone |
|---|---:|---:|---:|---:|---:|
| train | 428 | 1314 | 763 | 248 | 303 |
| val | 73 | 300 | 163 | 49 | 88 |
| test | 95 | 366 | 228 | 29 | 109 |
| **合计** | **596** | **1980** | **1154** | **326** | **500** |

## 数据修正记录

为保证严格的数据审计可以通过，导入后只进行了一处最小数值修正：

- 文件：`labels/train/old__keyboard__DJI_20260826133343_0209_D_002_jpg.rf.itcaVq7eW3cwLnJphaI0.txt`
- 行号：第 6 行
- 原标注：`1 0.366140 0.110330 0.074780 0.220663`
- 修正后：`1 0.366140 0.110330 0.074780 0.220660`
- 原因：原框的上边界为 `-0.0000015`，属于坐标保留六位小数导致的微小越界；将高度裁到合法最大值，中心和宽度均不变

## 划分原则

1. 训练、验证和测试互不重复。
2. 同一段视频的连续相似帧放在同一划分中，避免泄漏。
3. 测试集应覆盖遮挡、侧视、远距离、明暗变化和多物体场景。
4. 用 `python src/check_dataset.py` 检查标签与跨集重复。
