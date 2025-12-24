# 本仓库中 PPO 与 Distributional PPO 的实现原理总结（LLM 训练视角）

本文总结当前代码库中，针对大模型（LLM）的 PPO 训练实现原理，重点包括：

- 各个网络的角色与输入/输出
- PPO 中 reward、优势函数（尤其是 GAE）的计算流程
- Distributional PPO（值分布 Critic）的实现细节
- 在什么配置下会启用 Distributional PPO

特别说明：

- 本文**暂不总结** risk-sensitive RL（`risk_apply_to` / `risk_level` 等）的实现细节，只聚焦在风险中性（risk-neutral）的 PPO 和 Distributional Critic。
- GAE 及其在 Distributional Critic 下的形态是重点部分。

---

## 一、整体框架概览：RLHF/PPO 在本仓库中的结构

主要相关文件和模块：

- 训练主入口：`verl/verl/trainer/ppo/ray_trainer.py`
- PPO 核心公式：`verl/verl/trainer/ppo/core_algos.py`
- Actor（策略网络）：`verl/verl/workers/actor/dp_actor.py`
- Critic（价值/分布网络）：`verl/verl/workers/critic/dp_critic.py`
- Reward 计算：`verl/verl/trainer/ppo/reward.py`，以及 `verl/verl/workers/fsdp_workers.py` 中 reward 展开逻辑
- PPO 默认配置：`verl/verl/trainer/config/ppo_trainer.yaml`

### 1.1 角色划分（Role）

在 `ray_trainer.py` 中定义的 `Role`：

- `Actor` / `ActorRollout`：策略网络，用于生成回答（rollout）和计算 log_prob。
- `Critic`：价值网络，用于估计每个 token 的价值（标量或分布）。
- `RefPolicy`：参考策略，用于 KL 惩罚。
- `RewardModel`：奖励模型或自定义 reward 函数，用来给生成结果打分。

训练时，通过 Ray WorkerGroup 管理这几个角色的并行 worker：

- `self.actor_rollout_wg`：负责生成回答 + 计算 policy log_prob。
- `self.critic_wg`：负责计算 values（供 GAE 等优势估计使用）以及更新 Critic 参数。
- `self.ref_policy_wg`（如果启用）：用于计算 `ref_log_prob` 做 KL 惩罚。
- `self.rm_wg`（如果用 RM）：用于计算 reward。

### 1.2 数据结构：DataProto 与关键字段

`DataProto` 是本仓库里在 worker 间传递数据的统一容器，包含：

- `data.batch[...]`：Tensor 字段（PyTorch 张量），如：
  - `input_ids`：完整输入（prompt + response），形状 `(B, S)`。
  - `attention_mask`：与 `input_ids` 对应的 mask，形状 `(B, S)`。
  - `position_ids`：位置编码索引，形状 `(B, S)`。
  - `responses`：仅 response 部分的 token id，形状 `(B, T)`。
  - `token_level_scores`：token 级 reward（通常只在最后一个 response token 上非零），形状 `(B, T)`。
  - `token_level_rewards`：在 `token_level_scores` 基础上加 KL 惩罚后的 reward，形状 `(B, T)`。
  - `old_log_probs`：旧策略（上一轮 actor）对 response 的 log_prob，形状 `(B, T)`。
  - `ref_log_prob`：参考策略的 log_prob（如果启用 KL in reward / KL loss），形状 `(B, T)`。
  - `values`：Critic 估计的 value 序列，形状 `(B, T)`。
  - `advantages`：优势函数，形状 `(B, T)`。
  - `returns`：用于训练 Critic 的 target return 序列，形状 `(B, T)`。
  - `response_mask`：只针对 response 部分的 mask，形状 `(B, T)`。
- `data.non_tensor_batch[...]`：Python 对象，如：
  - `uid`：同一 prompt 下多条 response 的分组 id。
  - `raw_prompt` 等元信息。

典型流程（非常简化）：

1. Rollout：Actor 根据 `input_ids` 生成 `responses`，记录 `old_log_probs`。
2. Reward：RewardModel 或自定义 reward 函数对 `(prompt, response)` 打分，得到每个样本的标量 reward，并在 `fsdp_workers.py` 内展开为 `token_level_scores`（只在最后一个 response token 非零）。
3. KL 惩罚：如果启用 `algorithm.use_kl_in_reward`，用当前策略和参考策略的 log_prob 计算 per-token KL，得到 `token_level_rewards = token_level_scores - β * KL`。
4. Critic 估值：Critic 对序列做前向，得到 `values`（标量或由分布坍缩得到的期望值）。
5. Advantage：在 `compute_advantage` 中，结合 `token_level_rewards` 和 `values`，通过 GAE 或其它策略得到 `advantages` 和 `returns`。
6. Actor 更新：Actor 使用 `advantages`（以及 `old_log_probs`）计算 PPO policy loss，反向更新策略。
7. Critic 更新：Critic 使用 `returns` 作为监督目标，计算 value loss 或 distributional loss，更新 Critic。

---

## 二、Actor（策略网络）的结构与 PPO Loss

### 2.1 Actor 网络结构与前向输出

Actor 实现在 `verl/verl/workers/actor/dp_actor.py` 中的 `DataParallelPPOActor`：

- 底层是一个 LLM（如 Qwen3），通过 `self.actor_module` 封装。
- 使用 FSDP / Ulysses 进行分布式并行和去 padding（`use_remove_padding`）。

关键接口：

1. `compute_log_prob(self, data: DataProto)`：
   - 输入：
     - `input_ids`, `attention_mask`, `position_ids`, `responses`。
   - 流程：
     - 按 `micro_batch_size` 分批，将 batch 拆成多个 micro_batch。
     - 对每个 micro_batch 调用 `_forward_micro_batch`。
   - `_forward_micro_batch`：
     - 将 `input_ids` 输入 actor 模型，得到 logits：
       - `logits` 形状 `(B, S, vocab_size)`。
     - 利用 `responses` 进行对齐，计算 response 部分的 token 级 `log_probs`：
       - 形状 `(B, T)`，T 为 response 长度。
     - 若 `calculate_entropy=True`，则同时返回 per-token entropy `(B, T)`。
   - 返回：
     - `log_probs`：当前 actor 对 response 的 log_prob `(B, T)`。
     - `entropys`（可选）：per-token entropy `(B, T)`。

2. `update_policy(self, data: DataProto)`：
   - 输入：包含
     - `responses`, `input_ids`, `attention_mask`, `position_ids`
     - `old_log_probs`, `advantages`
     - 若启用 KL loss：`ref_log_prob`
     - 若使用 group entropy：`group_entropys`
   - 核心步骤：
     1. 根据 `ppo_mini_batch_size` 和 `ppo_micro_batch_size_per_gpu` 将大 batch 拆成多个 mini_batch，再拆成 micro_batch。
     2. 对每个 micro_batch：
        - 计算 `response_mask`：通常是 `attention_mask` 的末尾 `T` 个位置。
        - 调用 `_forward_micro_batch` 得到当前策略的 `log_prob` 和 entropy。
        - 使用 `core_algos.compute_policy_loss` 计算 PPO policy loss：
          - 输入：
            - `old_log_prob`（行为 π_old）
            - `log_prob`（当前 π）
            - `advantages`
            - `response_mask`
            - `clip_ratio(=ε)`、`clip_ratio_low`、`clip_ratio_high`、`clip_ratio_c`（dual-clip）
          - 输出：
            - `pg_loss`：最终聚合的 policy gradient loss（带 clipping）。
            - `pg_clipfrac`：被 clip 的比例。
            - `ppo_kl`：近似 KL。
            - `pg_clipfrac_lower`：下界 clip 的比例。
        - 结合 entropy 构造最终 loss：
          - 若 `entropy_coeff != 0`，则
            - `entropy_loss = agg_loss(entropy, response_mask, loss_agg_mode)`
            - `policy_loss = pg_loss - entropy_coeff * entropy_loss`
          - 否则 `policy_loss = pg_loss`。
        - 若启用 `use_kl_loss`，再加上额外的 KL loss：
          - `kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=kl_loss_type)`
          - `kl_loss = agg_loss(kld, response_mask, loss_agg_mode)`
          - `policy_loss += kl_loss * kl_loss_coef`
        - 根据 dynamic bsz / gradient_accumulation 做尺度修正后 `loss.backward()`。
     3. 完成一个 mini_batch 内多个 micro_batch 的 backward 后，调用 `_optimizer_step`：
        - 包含梯度裁剪（`grad_clip`）和优化器 step。

### 2.2 PPO Policy Loss 的数学形式

按照 `core_algos.compute_policy_loss`，本实现采用的是带 dual-clip 的 PPO 变体（dual-clip 部分可以关掉）：

- 定义
  - 比例：`r_t = exp(log_prob - old_log_prob)`。
  - 原始 PPO clip loss：
    \[
    L^{\text{CLIP}}_t(\theta) = \min\left( r_t(\theta) A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right)
    \]
- dual-clip 在优势为负时追加一个下界剪切：
  - 另设 `c > 1`，当 `A_t < 0` 时：
    \[
    L^{\text{dual}}_t(\theta) = \min\Big( \max(r_t A_t,\; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t),\; c A_t \Big)
    \]
- 实现：
  - `pg_losses1 = - advantages * ratio`
  - `pg_losses2 = - advantages * clamp(ratio, 1-cliprange_low, 1+cliprange_high)`
  - `clip_pg_losses1 = max(pg_losses1, pg_losses2)`
  - `pg_losses3 = - advantages * clip_ratio_c`
  - `clip_pg_losses2 = min(pg_losses3, clip_pg_losses1)`
  - 最终：
    - 若 `advantages < 0`：`pg_losses = where(advantages<0, clip_pg_losses2, clip_pg_losses1)`
    - 然后用 `agg_loss(pg_losses, response_mask, loss_agg_mode)` 聚合为标量 loss。

---

## 三、Critic 的实现：标量 PPO Critic 与 Distributional Critic

Critic 的基础抽象在 `verl/verl/workers/critic/base.py`，具体实现为 `DataParallelPPOCritic`（`dp_critic.py`）。

### 3.1 Critic 的前向：`compute_values`

`compute_values(self, data: DataProto)` 用于在 rollouts 收集完 reward 后，给每个 token 估计一个 value baseline。核心流程：

1. 将 `data` 按 `micro_batch_size` 或 dynamic bsz 切分成多个 micro_batch。
2. 对每个 micro_batch 调用 `_forward_micro_batch`：
   - 输入：
     - `input_ids`, `attention_mask`, `position_ids`, `responses`（同 actor）。
   - 根据 `self.is_distributional`（即 `critic.distributional`）决定不同分支：
     - 若 `distributional == False`：
       - 直接在 Critic 模型中走“普通 value head”，输出 `values`，形状 `(B, S, 1)` 或 `(B, S)`。
       - 再裁切为 response 区间，得到 `(B, T)`。
     - 若 `distributional == True`：
       - 打开 `output_hidden_states=True`，从底层 LLM 取出最后一层 `hidden_states`，形状 `(B, S, H)`。
       - 再根据 `quantile_mode` 调用不同的 distributional head：
         - `quantile_mode == "iqn"`：`iqn_head(hidden, taus)` → `(B, S, K)` quantiles + τ。
         - `quantile_mode == "fixed"`：`qr_head(hidden)` → `(B, S, K)` quantiles。
         - `quantile_mode == "c51"`：`c51_head(hidden)` → `(B, S, N_atoms)` logits。
3. 将多个 micro_batch 拼回完整 batch，得到：
   - 非分布式：`values`，形状 `(B, S)`。
   - 分布式：`values` 为 quantiles 或 logits，形状 `(B, S, K)` 或 `(B, S, N_atoms)`。

在 `compute_values` 的后半部分，会将序列裁切到 response 部分，并与 `response_mask` 相乘，得到最终的：

- `values_mean`，形状都是 `(B, T)`：
  - 非 distributional：
    - 直接 `values * response_mask`。
  - Distributional + risk-neutral（本总结只关心这部分）：
    - `quantile_mode == "c51"`：
      - `probs = softmax(logits, dim=-1)`。
      - 定义 atoms：`atoms = linspace(c51_v_min, c51_v_max, num_atoms)`。
      - 期望：`expect = (probs * atoms).sum(-1)`，再乘 `response_mask` 得到 `(B, T)`。
    - `quantile_mode` 为 `"iqn"` 或 `"fixed"`：
      - 先 `values = values * response_mask.unsqueeze(-1)`（只保留 response 区域的 quantiles）。
      - 然后对 quantile 维度取平均：`values_mean = values.mean(dim=-1)`，形状 `(B, T)`。

**结论**：无论是否 Distributional，`compute_values` 对外暴露的都是一个形状 `(B, T)` 的标量 value 序列 `values_mean`，供 GAE 等优势函数使用；区别在于：

- 非 Distributional：`values_mean` 直接来自一个 value head 的标量输出。
- Distributional：`values_mean` 是对预测的回报分布求期望后的结果。

### 3.2 Critic 的训练：`update_critic`

`update_critic(self, data: DataProto)` 负责在 actor 更新之后，用当前 batch 的 `returns` 和 `values` 来更新 Critic。

- 输入 `data.batch` 中主要用到的键：
  - `input_ids`, `responses`, `attention_mask`, `position_ids`
  - `values`（旧 value，来自之前 `compute_values`）
  - `returns`（目标 return，由 `compute_advantage` 得到）
  - 若是 distributional + IQN，还可能有：
    - `target_quantiles`, `target_taus`（可选）

数据流程：

1. 拆 mini_batch / micro_batch，同 actor。
2. 对每个 micro_batch 调用 `_forward_micro_batch` 得到当前的 vpreds：
   - 非 distributional：
     - `vpreds` 形状 `(B, T)`，标量 value。
   - Distributional：
     - `quantile_mode == "c51"`：`vpreds` 是 `(B, T, N_atoms)` 的 logits。
     - 其它（IQN / Fixed）：`vpreds` 是 `(B, T, K)` 的 quantiles；同时还可以得到 `taus`。
3. 根据 `self.is_distributional` 选择不同的 loss：

#### 3.2.1 标量 Critic：`compute_value_loss`

当 `critic.distributional == false` 时，使用 `core_algos.compute_value_loss`，公式与标准 PPO 一致：

- 输入：`vpreds`, `returns`, `values`, `response_mask`。
- 先做 value clipping：
  - `vpredclipped = clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)`。
- 计算两种平方误差：
  - `vf_losses1 = (vpreds - returns)^2`
  - `vf_losses2 = (vpredclipped - returns)^2`
- 取 “更坏的” 那个（和 PPO policy loss 类似的理念）：
  - `vf_loss = 0.5 * masked_mean(max(vf_losses1, vf_losses2), response_mask)`。
- 得到标量 loss `vf_loss` 和被 clip 比例 `vf_clipfrac`。

#### 3.2.2 Distributional Critic：Quantile / C51 Loss

当 `critic.distributional == true` 时：

1. `quantile_mode == "c51"`：Categorical (C51) Loss
   - vpreds 被视为 logits。
   - 通过固定的 atoms：`atoms = linspace(c51_v_min, c51_v_max, num_atoms)`。
   - 使用 `compute_categorical_value_loss`：
     1. 将标量 `returns` clamp 到 `[v_min, v_max]`。
     2. 将每个 return 按线性插值投影到相邻的两个 atom 上（构造 target_probs）。
     3. 对 logits 做 softmax，得到预测概率分布 `probs`。
     4. loss = masked_mean( CrossEntropy(target_probs, softmax(logits)), response_mask )。
   - 这是典型的 C51 训练方式。

2. `quantile_mode` 为 `"iqn"` / `"fixed"`：Quantile Huber Loss
   - vpreds 被视为 `(B, T, K)` 个 quantile 值。
   - 使用 `compute_quantile_value_loss`：
     1. 若无 `taus`，生成 τ（IQN：随机均匀；Fixed：固定网格）。
     2. 构造 diff：`diff = returns - vpreds`（在适配维度后），形状 `(B, T, K, 1)`。
     3. 计算 Huber 损失：
        - 若 |diff| ≤ κ：`0.5 * diff^2`。
        - 否则：`κ(|diff| - 0.5κ)`。
     4. 乘以 quantile 损失权重：`|τ - I(diff < 0)|`。
     5. 对 quantile 维度和 K'（target）维度求平均，再做 `masked_mean`，得到标量 loss。
   - 这是 QR-DQN / IQN 类型的标准 quantile regression 损失。

---

## 四、GAE 的实现与细节（重点）

GAE 的核心在 `verl/verl/trainer/ppo/core_algos.py` 中的 `compute_gae_advantage_return`，被 `ray_trainer.py` 中的 `compute_advantage` 调用。

### 4.1 GAE 的数学形式

标准 GAE 定义（单步 TD δ）：

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

\[
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\]

在有限长度响应、且只在 response 区域有效时，代码实现做了：

- 设 response 长度为 T，索引 t=0,...,T-1，对每个样本独立计算。
- 终止处（t=T-1）没有 `V(s_T)`，用 `nextvalues = 0.0` 近似。

### 4.2 代码实现流程

`compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam)`：

- 输入：
  - `token_level_rewards`：形状 `(B, T)`，对应每个 response token 的 reward。
    - 通常是：最后一个 token 上是 sequence reward + 可能的后处理；其他 token 为 0。
  - `values`：形状 `(B, T)`，来自 Critic 的 value baseline。
  - `response_mask`：形状 `(B, T)`，response 区域 mask（非 EOS 之后为 1，EOS 之后为 0）。
  - `gamma`、`lam`：标量超参数。
- 内部逻辑（伪代码）：
  ```python
  lastgaelam = 0
  advantages_reversed = []
  gen_len = token_level_rewards.shape[-1]

  for t in reversed(range(gen_len)):  # 从 T-1 到 0
      nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
      delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
      lastgaelam = delta + gamma * lam * lastgaelam
      advantages_reversed.append(lastgaelam)

  advantages = stack(advantages_reversed[::-1], dim=1)  # (B, T)
  returns = advantages + values  # (B, T)
  advantages = masked_whiten(advantages, response_mask)
  ```
- 解释：
  - 循环逆序，从最后一个有效 token 反向累积。
  - `lastgaelam` 存的其实就是
    \[
    A_t = \delta_t + \gamma \lambda \delta_{t+1} + \gamma^2 \lambda^2 \delta_{t+2} + \cdots
    \]
  - 最终 `returns = advantages + values`，即
    \[
    \text{return}_t = A_t + V(s_t)
    \]
    这通常视作对折扣回报的估计。
  - 最后调用 `masked_whiten` 对 `advantages` 按 `response_mask` 做标准化：
    - 减均值、除以标准差，只在有效 token 区内统计；无效 token 保持 0。

### 4.3 GAE 在训练流程中的位置

在 `ray_trainer.py` 中的 `compute_advantage`：

1. 如果 `data.batch` 中还没有 `response_mask`，先通过 `compute_response_mask` 从 `attention_mask` 中切出 response 部分。
2. 读取 `data.batch["token_level_rewards"]` 和 `data.batch["values"]`。
3. 根据 `adv_estimator` 分支：
   - `GAE`：调用 `core_algos.compute_gae_advantage_return(...)`。
   - 其它（GRPO / REINFORCE++ / RLOO / QAE 等）走对应分支。
4. 将 `advantages` 和 `returns` 写回：
   - `data.batch["advantages"] = advantages`
   - `data.batch["returns"] = returns`
5. 非 risk 模式下，直接返回 `data`，后续：
   - `advantages` 进入 Actor 的 `update_policy`。
   - `returns` 进入 Critic 的 `update_critic`。

---

## 五、GAE 在 Distributional Critic 下的不同点

**核心结论**：  
在当前实现中（忽略 risk-sensitive 逻辑），**GAE 的形式完全不变**，只是在「values 从哪里来」这件事上发生了变化：

- 标量 Critic：
  - `values` 直接是神经网络的标量输出 `V(s_t)`。
- Distributional Critic：
  - Critic 学的是回报分布 `Z(s_t)`，在 `compute_values` 中先将分布坍缩为期望值 `E[Z(s_t)]`：
    - Quantile 模式：对 quantile 取均值；
    - C51 模式：对 categorical 分布与 atoms 求加权和。
  - GAE 中使用的 `values` 就是这个期望。

因此，从 GAE 的公式角度来看：

- δ_t 的形式没变：
  \[
  \delta_t = r_t + \gamma V_{\text{dist}}(s_{t+1}) - V_{\text{dist}}(s_t)
  \]
- 只是 `V_{\text{dist}}(s)` 是一个 *由分布估计出来的期望值*，而不是直接的标量 head 输出。

这带来几个实际差异（直观理解）：

1. **Baseline 质量提升的潜力**  
   - 分布学习可以让 Critic 在高方差场景下学到更稳定的表示。
   - 即便最后只用期望，内部结构更丰富，可能使得 `V(s)` 的偏差更小，进而降低 GAE 的方差。

2. **GAE 的数值稳定性**  
   - 因为 `values` 的来源从「单一标量 head」变成了「分布 + 坍缩」，在某些情况下能缓解 over-estimation / under-estimation 问题，使 δ_t 更平滑。

3. **实现层面完全兼容**  
   - 对 `compute_gae_advantage_return` 来说，只要传入的 `values` 是 `(B, T)` 的标量序列即可；它对 Distributional 与否完全无感。
   - Distributional 与否只影响：
     - `compute_values` 如何生成 `values`。
     - `update_critic` 采用哪种 loss。

**小结**：  
> 在当前代码中，“Distributional PPO” 的核心是：**Critic 用分布化方式训练 + 推理时用分布的期望作为 baseline**，而 GAE 本身的形式保持不变。

---

## 六、KL 惩罚与 `token_level_rewards` 的构造

虽然不是本问重点，但理解 reward 形态有助于掌握 GAE 输入。

### 6.1 从 sequence reward 到 `token_level_scores`

在 FSDP worker 侧（`verl/verl/workers/fsdp_workers.py`）：

- RewardModel 或自定义 reward 函数通常输出的是每条样本的一个标量得分 `scores`，形状 `(B,)`。
- 通过 `_expand_to_token_level` 展开为 token 级：
  - 构造一个全 0 的 token 矩阵 `token_level_scores`，形状 `(B, S)`。
  - 利用 `eos_mask_idx`（最后一个有效 token 的位置），在该位置写入 sequence score。
  - 然后切出 response 区域：
    - `token_level_scores = token_level_scores[:, -response_length:]`，形状 `(B, T)`。

这就是 `data.batch["token_level_scores"]` 的由来。

### 6.2 加上 KL 惩罚：`token_level_rewards`

在 `ray_trainer.py` 中的 `apply_kl_penalty`：

- 读取：
  - `token_level_scores`（形状 `(B, T)`）。
  - `old_log_probs`, `ref_log_prob`。
- 计算 KL：
  - `kld = core_algos.kl_penalty(old_log_probs, ref_log_prob, kl_penalty=kl_penalty)`（逐 token）。
  - 再乘上 `response_mask` 避免 padding 干扰。
- 得到：
  - `token_level_rewards = token_level_scores - beta * kld`。
  - 这里 `beta` 通过 `AdaptiveKLController` 自适应调整。
- 写回：
  - `data.batch["token_level_rewards"] = token_level_rewards`。
- 后续 GAE 就是直接用这个 `token_level_rewards` 作为 `r_t`。

因此，从 GAE 的视角看：

- `r_t` 只有在最后几个 response token（尤其是 EOS）可能非零。
- KL 惩罚是以 token 级方式施加的，reward 里已经包含了 KL 信息。

---

## 七、哪些配置会启用 Distributional PPO

### 7.1 基础配置项

在 `verl/verl/trainer/config/ppo_trainer.yaml` 中，Critic 默认配置为：

```yaml
critic:
  distributional: false
  num_quantiles: 32
  quantile_mode: iqn  # iqn | fixed | c51
  quantile_huber_kappa: 1.0
  c51_v_min: -10.0
  c51_v_max: 10.0
  risk_apply_to: "none"   # 本文暂不展开
  risk_level: "neutral"   # 本文暂不展开
  ...
```

因此，要在训练中真正启用 Distributional Critic，需要在命令行或脚本中覆盖这些配置。

### 7.2 典型脚本示例

在 `scripts/train/qwem2.5-1.5b/distributional_RL_ppo.sh` 中，有如下关键片段：

```bash
CRITIC_DISTRIBUTIONAL=true
CRITIC_NUM_QUANTILES=32
CRITIC_QUANTILE_KAPPA=1.0
CRITIC_QUANTILE_MODE=iqn
...
python3 -m verl.trainer.main_ppo \
    ...
    ++critic.distributional=${CRITIC_DISTRIBUTIONAL} \
    ++critic.num_quantiles=${CRITIC_NUM_QUANTILES} \
    ++critic.quantile_huber_kappa=${CRITIC_QUANTILE_KAPPA} \
    ++critic.quantile_mode=${CRITIC_QUANTILE_MODE} \
    ...
```

这类脚本对应的行为：

- `critic.distributional=true` → 启用 Distributional Critic。
- `critic.quantile_mode=iqn` → 使用 IQN 形式的 quantile head。
- `critic.num_quantiles=32` → 每个 token 输出 32 个分位值。
- `critic.quantile_huber_kappa=1.0` → Quantile Huber loss 中的 κ。

类似地，还有：

- `scripts/train/qwem2.5-1.5b/QRDQN.sh`：
  - 一般会设置 `quantile_mode=fixed`，对应 QR-DQN 风格的固定 quantile。
- `scripts/train/qwem2.5-1.5b/C51.sh`：
  - 设置 `quantile_mode=c51`，开启 C51 分类分布 head。
- `scripts/train/qwem2.5-1.5b/risk_neutral.sh` 等：
  - 也是以 `++critic.distributional` 控制是否分布化，但会结合不同的 risk 配置（本文不展开 risk 部分，仅指出：**是否 Distributional 由 `critic.distributional` 控制，与 risk 配置正交**）。

### 7.3 总结：何时可以称为 “Distributional PPO”

在当前代码库中，从实现上看：

- 只要满足：
  1. `critic.distributional == true`；
  2. `critic.quantile_mode` 设为 `"iqn"` / `"fixed"` / `"c51"` 之一；
  3. Actor 仍然使用 PPO 的 policy loss（这是默认）；
- 那么训练过程就是：
  - **PPO actor + Distributional Critic baseline**，即可视为一种 “Distributional PPO”。

在风险中性设置下（`risk_apply_to="none"`, `risk_level="neutral"`）：

- Critic 学习回报分布；
- GAE 使用分布期望作为 baseline；
- Actor 的目标仍然是最大化期望回报（包含 KL 惩罚），只是 baseline 更强。

---

## 八、小结

1. **Actor 部分**：
   - 是一个标准的 LLM 策略头，通过 `compute_log_prob` 和 `update_policy` 实现 PPO policy update。
   - 使用 dual-clip PPO loss + 可选的 entropy 正则和 KL loss。

2. **Critic 部分**：
   - 在非分布模式下，跟 HuggingFace TRL 的标量 Critic 非常类似，使用 value clipping 的 MSE loss。
   - 在分布模式下：
     - 使用 IQN / Fixed quantiles / C51 来建模回报分布；
     - 训练时用 Quantile Huber loss 或 Categorical CrossEntropy；
     - 推理时对分布求期望，得到 scalar `values` 供 GAE 使用。

3. **GAE**：
   - GAE 的实现和公式完全是标准形式，只依赖：
     - token-level rewards（含 KL 惩罚）；
     - Critic 给出的 `values` baseline。
   - 在 Distributional Critic 下，`values` 是从回报分布坍缩得到的期望值，因此在不改变 GAE 公式的前提下，引入了更丰富的价值估计。

4. **Distributional PPO 的判定**：
   - 当 `critic.distributional=true` 且 `critic.quantile_mode` 为 `"iqn"/"fixed"/"c51"` 时，就处在 Distributional Critic 模式。
   - Actor 仍然用 PPO 更新，因此整体组合可称为 “Distributional PPO”。

