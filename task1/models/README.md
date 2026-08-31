# 模型说明

训练完成后，`src/train.py` 会将验证集表现最好的权重复制为 `models/best.pt`，并生成 `models/model_info.json`，包含生成时间、原始 checkpoint、SHA-256、类别名称和训练参数。

当前状态：100 轮正式训练已于 2026-08-31 完成。发布模型为 `best.pt`，大小 5,371,141 字节，SHA-256 为 `7ab7a1cf4c08e2eb803d5666a6caa6931256cfabfe04dc8e9c98385322fbcdcf`。训练环境、类别、配置和数据指纹见 `model_info.json`。

`models/best.pt` 与 `models/model_info.json` 必须作为同一组提交材料保存，并以 `model_info.json` 中的 SHA-256 校验值确认权重文件完整性。独立测试指标保存在 `../results/metrics.json`；模型指标不替代 Jetson 实机速度和 20 件实物正确率验收。
