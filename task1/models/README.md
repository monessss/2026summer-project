# 模型说明

训练完成后，`src/train.py` 会将验证集表现最好的权重复制为 `models/best.pt`，并生成 `models/model_info.json`，包含生成时间、原始 checkpoint、SHA-256、类别名称和训练参数。

当前状态：**待放入数据集并完成真实训练，仓库中没有伪造权重。**

如果权重超过 GitHub 普通文件限制，请使用 Git LFS 或 GitHub Release，并在本文件中补充下载链接和 SHA-256。
