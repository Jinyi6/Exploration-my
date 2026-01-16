2025-01-14
- 新增 locomo 的 LLM 评测 reward function（按 data_source 路由）。
- 文件：`verl/verl/utils/reward_score/locomo.py`。
- 使用方式：数据预处理将 `data_source` 设为 `locomo`，并在 `reward_model.ground_truth` 写入标准答案，训练会通过 `verl/verl/utils/reward_score/__init__.py` 自动调用。
- 仅 locomo 用途：评测 prompt 需要问题文本，默认从 `extra_info.question` 读取（或 `prompt`/`problem`），若没有则传空字符串。
