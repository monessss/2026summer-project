# 数据集说明

## 来源与版本

- 源仓库：[`boff868/2026Summer-integrated-robot-grouptask/dataset_self`](https://github.com/boff868/2026Summer-integrated-robot-grouptask/tree/main/dataset_self)
- 固定源提交：`d06626c71780a5c0a8283d76d09e7b90d0238680`
- 源仓库说明：数据为 DJI 实拍画面，由 Roboflow 项目 `2026summer_nongfu_checked` 导出
- 导入规则：源目录 `train`、`valid`、`test` 首先映射为本项目的 `train`、`val`、`test`，随后按完整 DJI 拍摄序列消除跨集泄漏
- 内容处理：修正 1 个因六位小数舍入而越过图像边界 `0.0000015` 的框，并归并 5 个被拆散的连续拍摄序列；详见下方“数据修正记录”
- 最终数据指纹（类别映射、图片原始字节、规范化标签、文件名和划分）：`366ccefb5118171059c4ad824fec05d21e87c98a87925b275d622cc1b60d0681`
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

- 文件名时间戳日期：2026-08-26、2026-08-28
- 图像来源设备：DJI
- 图片处理：最长边压缩到 800 px
- 标注工具：Roboflow
- 标注形式：归一化 YOLO Detection 框；原多边形已转为最小外接框
- 数据许可：源仓库根目录采用 MIT License，本目录保存其许可证副本

| 划分 | 图片数 | 实例数 | keyboard | nongfu_spring | phone |
|---|---:|---:|---:|---:|---:|
| train | 433 | 1330 | 772 | 248 | 310 |
| val | 75 | 315 | 174 | 53 | 88 |
| test | 88 | 335 | 208 | 25 | 102 |
| **合计** | **596** | **1980** | **1154** | **326** | **500** |

## 数据修正记录

为保证严格的数据审计可以通过，导入后只进行了一处最小数值修正：

- 文件：`labels/train/old__keyboard__DJI_20260826133343_0209_D_002_jpg.rf.itcaVq7eW3cwLnJphaI0.txt`
- 行号：第 6 行
- 原标注：`1 0.366140 0.110330 0.074780 0.220663`
- 修正后：`1 0.366140 0.110330 0.074780 0.220660`
- 原因：原框的上边界为 `-0.0000015`，属于坐标保留六位小数导致的微小越界；将高度裁到合法最大值，中心和宽度均不变

源数据有 5 个 DJI 连续拍摄序列被分到两个集合。按照“整段序列只属于一个集合”的规则，将每组全部归入原本图片数占多数的集合：

| 拍摄序列 | 原集合 | 归并后集合 | 移动图片数 |
|---|---|---|---:|
| `old__keyboard__DJI_20260826133326_0206` | train、val | train | 3 |
| `old__keyboard__DJI_20260826133332_0207` | train、val | train | 1 |
| `old__phone__DJI_20260826132430_0188` | train、val | train | 1 |
| `old__phone__DJI_20260826132438_0189` | val、test | val | 3 |
| `old__phone__DJI_20260826132444_0190` | val、test | val | 4 |

## 划分原则

1. 74 个 DJI 拍摄序列均只属于一个划分。
2. 三个划分不存在 SHA-256 完全相同的图片。
3. 596 张图片全部可解码，596 个标签文件全部配对。
4. 1980 行标注的类别编号、字段数量、归一化坐标和边界均合法。
5. `python src/check_dataset.py` 会重复检查以上约束，任一约束失败时禁止训练。
6. 审计报告中的 `dataset_fingerprint_sha256` 必须等于本页记录的数据指纹。
