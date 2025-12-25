# dp\_critic response\_mask 错位问题分析

## 背景
- 在 actor/trainer 侧，我们统一使用 `response_mask = attention_mask[:, -response_length:]` 来刻画每个 response token（对应 actor 的“action”）的有效性。  
- `megatron_critic` 也遵循这套定义，所以 GAE、returns 以及后续的风险加权操作都假设 mask 和 response token 一一对应。  
- `dp_critic`（`verl/verl/workers/critic/dp_critic.py`）内部为了适配 model 输出，在 `_forward_micro_batch` 里把 value 序列切成 `values[:, -response_length-1:-1]`，并在 `compute_values` / `update_critic` 中使用 `attention_mask[:, -response_length-1:-1]` 当作 `response_mask`。

## 产生原因
1. dp\_critic 的 response\_mask 切片方式与其它组件不一致（actor、megatron\_critic 均使用 `[:, -response_length:]`），导致 mask **落在 response 区间的前一位**（通常是 prompt 的最后一个 token 或 padding）。  
2. `_forward_micro_batch` 同时把 `values/value_quantiles` 也切到了前一位，使得「错位 value + 错位 mask」看上去自洽，但实际输出的 value 表示的是 state 的有效性而非 action 的有效性。  
3. 当 actor 在 risk-aware 模式下直接用自己的 `response_mask` 去加工 dp\_critic 给出的 `values/value_quantiles` 时，mask 和数据不再匹配：  
   - Response 的最后一个真实 token 被视作 mask=1，但 value 来源却是前一个 token。  
   - Padding（mask=0）那一位在 dp\_critic 中被乘上 mask=1，从而把噪声写入 `values/value_quantiles/value_logits`。  
4. `compute_gae_advantage_return` 默认不再根据 mask 截断 TD 递推，只在 whiten 时用 mask，因此错位值会直接污染优势函数与 returns。  

## 表现
- 纯风险中性 PPO/QRDQN 仍能训练，是因为 “错位 value + 错位 mask” 在 critic 内部互相抵消，并且 reward mask 通常在最后一位；但 value/advantage 的最后一个 token 实际缺失。  
- 一旦启用 `risk_apply_to=baseline1/reweight/target1` 或 `risk_level != neutral` 等需要精确对齐 response token 的场景，actor 侧的 mask（正确区间）和 critic 侧输出（错位区间）无法匹配：  
  - tail reweight / risk baseline 读取的 quantiles 会把 padding 当作有效样本，导致权重或 rho 估计失真；  
  - risk target 直接使用错位的 rho 作为 `risk_returns`，在 response 末端出现明显跳变。  

## 修改建议
1. **统一 mask 定义**：将 `dp_critic` 中所有 `response_mask = attention_mask[:, -response_length - 1 : -1]` 改为 `[:, -response_length:]`，与 actor、megatron\_critic 保持一致。涉及位置至少包括：
   - `compute_values`（~line 251）
   - `update_critic`（~line 362）
   - 所有对 distributional 输出乘 mask 的位置（`values = values * response_mask`、`values * response_mask.unsqueeze(-1)` 等）
2. **同步截取区间**：`_forward_micro_batch` 里的 `values = values[:, -response_length - 1 : -1]`、`hidden_states = ...` 也需要改为 `[:, -response_length:]`，保证返回的张量与 mask 对同一批 response tokens。  
3. **验证**：修改后跑一次最小化的 PPO/QRDQN 训练（risk-neutral & risk-aware 各一次）并检查以下诊断：
   - `values`/`returns` 在 response 最后一位不再为零；  
   - risk baseline/reweight/target 的指标（如 `adv/reweight_tail_frac`、`risk_returns`）在 padding case 下不再出现异常值。  

这样可以确保 actor 端的 GAE、风险 reshaping、tail reweight 等逻辑都能对准真实的 response token，同时消除 padding 污染。  
