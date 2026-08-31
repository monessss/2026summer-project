# 模型说明

训练完成后，`src/train.py` 会将验证集表现最好的权重复制为 `models/best.pt`，并生成 `models/model_info.json`，包含生成时间、原始 checkpoint、SHA-256、类别名称和训练参数。

当前状态：数据集已经导入并通过审计，正式训练尚未执行，因此仓库中不存在 `best.pt` 和 `model_info.json`，也不声明任何模型精度。

模型生成后，`models/best.pt` 与 `models/model_info.json` 必须作为同一组提交材料保存，并以 `model_info.json` 中的 SHA-256 校验值确认权重文件完整性。
