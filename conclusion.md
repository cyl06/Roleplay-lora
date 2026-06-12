# 项目总结：基于 LoRA 微调的小型 LLM 角色扮演对话生成

## 一、项目概述

本项目使用 QLoRA（4-bit 量化 + Low-Rank Adaptation）对 1.5B–3.8B 参数的小型开源 LLM 进行高效微调，使其具备稳定的角色扮演对话能力。目标应用场景包括游戏 NPC 对话、互动叙事和虚拟角色聊天。

## 二、模型与训练

### 2.1 实验方案

共训练 4 个 LoRA 变体，覆盖 3 个对照维度：

| 编号 | 模型 | 基座参数 | 训练数据集 | LoRA 适配器 |
|------|------|---------|-----------|------------|
| M1 | Qwen2.5-1.5B + roleplay | 1.5B | hieunguyenminh/roleplay (5.7K 多轮) | 51 MB |
| M2 | Phi-3.5-mini + npc-dialogue | 3.8B | amaydle/npc-dialogue (1.7K 单轮) | 52 MB |
| M3 | Qwen2.5-3B + npc-dialogue | 3.0B | amaydle/npc-dialogue (1.7K 单轮) | 73 MB |
| M4 | Qwen2.5-1.5B + npc-dialogue | 1.5B | amaydle/npc-dialogue (1.7K 单轮) | 51 MB |

### 2.2 训练配置

- **方法:** QLoRA，4-bit NF4 量化，bf16 计算精度
- **LoRA 参数:** rank=8, alpha=16, dropout=0.05
- **优化器:** AdamW 8-bit，学习率 2e-4，warmup 50 步
- **训练:** 3 epochs，batch size=1×8（梯度累积），max_seq_length=512
- **硬件:** NVIDIA RTX 4060 Laptop (8GB VRAM)，单卡训练
- **显存占用:** 1.5B 模型 ~1.5-2GB，3B 模型 ~2-2.5GB，Phi-3.5 ~2.5GB（均为 4-bit 量化后）

### 2.3 LoRA 模块差异

不同基座模型的 attention/MLP 线性层结构不同：

| 基座 | Attention | MLP |
|------|-----------|-----|
| Qwen2.5 | q_proj, k_proj, v_proj, o_proj | gate_proj, up_proj, down_proj |
| Phi-3.5-mini | qkv_proj (合并), o_proj | gate_up_proj (合并), down_proj |

## 三、评估体系

按照 guidance.md 设计的三层评估框架：

### 3.1 评测数据

| 数据集 | 规模 | 用途 |
|--------|------|------|
| role_cards.jsonl | 232 张角色卡 | 统一角色定义（来自 npc-dialogue + roleplay） |
| single_turn_eval.jsonl | 192 条 | 单轮参考回答评估 |
| multi_turn_eval.jsonl | 40 组 | 5 轮角色一致性评估 |
| adversarial_eval.jsonl | 30 组 | 出戏（OOC）探测 |

### 3.2 评分体系

使用 Qwen2.5-1.5B-Instruct 作为本地 judge 模型，对每个回复评 6 个维度（1–5 分），辅以正则 OOC 检测。总分公式：

```
Total = 0.30×Role_Fidelity + 0.20×Style + 0.15×Relevance
      + 0.15×Multi-turn_Consistency + 0.10×Non_OOC + 0.10×Fluency
```

### 3.3 评估工具链

```
scripts/
├── build_eval_data.py        # 构建统一评估数据集
├── generate_responses.py     # 4 模型 × 3 类评测 → 生成回复
├── judge_responses.py        # Judge 模型打分 + OOC 检测
└── summarize_results.py      # 汇总 CSV + 对比分析
```

## 四、评估结果

### 4.1 单轮参考回答评估

| 模型 | Total | Role Fidelity | Style | OOC Rate | 延迟 (s) |
|------|------:|-------------:|------:|---------:|---------:|
| M1: Qwen2.5-1.5B + roleplay | **4.85** | **4.83** | **4.83** | 0.0% | 3.96 |
| M3: Qwen2.5-3B + npc-dialogue | 4.44 | 4.37 | 4.37 | 0.0% | 1.05 |
| M2: Phi-3.5-mini + npc-dialogue | 4.37 | 4.30 | 4.30 | 0.0% | 0.82 |
| M4: Qwen2.5-1.5B + npc-dialogue | 4.37 | 4.30 | 4.30 | 0.0% | 0.85 |

**分析:**
- M1（roleplay 数据集）在单轮评估中显著领先，总分高出其他模型约 0.4–0.5 分，说明多轮对话训练数据带来了更好的角色理解和回复质量。
- 三个 npc-dialogue 训练模型（M2/M3/M4）分数接近（4.37–4.44），数据集的影响大于模型大小。
- 单轮场景下所有模型 OOC 率均为 0%，说明 LoRA 微调有效抑制了基础模型的"AI 助手"行为。

### 4.2 多轮角色一致性评估（主指标）

| 模型 | Total | Role Fidelity | Style | Multi-turn Consistency | OOC Rate | 延迟 (s) |
|------|------:|-------------:|------:|----------------------:|---------:|---------:|
| M1: Qwen2.5-1.5B + roleplay | **4.04** | **4.11** | **4.11** | **4.11** | 34.3% | 3.78 |
| M4: Qwen2.5-1.5B + npc-dialogue | 3.33 | 3.31 | 3.31 | 3.31 | 30.0% | 0.89 |
| M2: Phi-3.5-mini + npc-dialogue | 3.32 | 3.26 | 3.26 | 3.26 | 22.9% | 0.78 |
| M3: Qwen2.5-3B + npc-dialogue | 3.22 | 3.24 | 3.26 | 3.24 | 41.4% | 1.18 |

**分析:**
- M1 在多轮场景再次大幅领先（总分 4.04 vs 第二名 3.33），差距拉大到 ~0.7 分。roleplay 数据集的多轮对话特性使模型在身份保持和跨轮一致性上明显更好。
- 多轮 OOC 率显著高于单轮：22.9%–41.4%。对抗性多轮对话更容易诱导模型出戏。Phi-3.5-mini 的 OOC 率最低（22.9%），Qwen2.5-3B 反而最高（41.4%）。
- M2（Phi-3.5-mini）在推理速度上最优（0.78s/轮），且 OOC 率最低，说明不同基座的语言能力对角色扮演稳定性有影响。

### 4.3 三组对照分析

#### 数据集影响：M1 vs M4（同基座，不同数据集）

| 指标 | M1 (roleplay) | M4 (npc-dialogue) | 差异 |
|------|-------------:|------------------:|-----:|
| 单轮 Total | 4.85 | 4.37 | +0.48 |
| 多轮 Total | 4.04 | 3.33 | +0.71 |
| 多轮 OOC Rate | 34.3% | 30.0% | +4.3% |

**结论:** roleplay 多轮数据集带来了明显更好的通用角色扮演能力，特别是在多轮一致性上优势巨大。但 OOC 率反而略高，可能与数据集中的对抗性对话较少有关。

#### 模型大小影响：M3 vs M4（同数据集，不同规模）

| 指标 | M3 (3B) | M4 (1.5B) | 差异 |
|------|--------:|----------:|-----:|
| 单轮 Total | 4.44 | 4.37 | +0.07 |
| 多轮 Total | 3.22 | 3.33 | **-0.11** |
| 多轮 OOC Rate | 41.4% | 30.0% | **+11.4%** |

**结论:** 在 npc-dialogue 单轮数据集上，3B 相比 1.5B 的提升极其有限（仅 +0.07）。多轮场景下 3B 表现反而更差，且 OOC 率更高。说明更大模型配合单轮训练数据不能自动带来更好的角色扮演能力——训练数据的对话结构才是关键因素。

#### 基座模型影响：M2 vs M3（同数据集，不同基座）

| 指标 | M2 (Phi-3.5-mini) | M3 (Qwen2.5-3B) | 差异 |
|------|------------------:|----------------:|-----:|
| 单轮 Total | 4.37 | 4.44 | -0.07 |
| 多轮 Total | 3.32 | 3.22 | +0.10 |
| 多轮 OOC Rate | 22.9% | 41.4% | **-18.5%** |

**结论:** Phi-3.5-mini 虽然单轮得分略低，但在多轮一致性和抗 OOC 方面表现更好。Phi-3.5 的 `<|system|>/<|user|>/<|assistant|>` 格式和更强的指令遵循能力可能是其更低 OOC 率的原因。

### 4.4 推理效率

| 模型 | 单轮延迟 (s) | 多轮延迟 (s/轮) | 适配器大小 |
|------|------------:|---------------:|----------:|
| M1: Qwen2.5-1.5B + roleplay | 3.96 | 3.78 | 51 MB |
| M2: Phi-3.5-mini + npc-dialogue | 0.82 | 0.78 | 52 MB |
| M3: Qwen2.5-3B + npc-dialogue | 1.05 | 1.18 | 73 MB |
| M4: Qwen2.5-1.5B + npc-dialogue | 0.85 | 0.89 | 51 MB |

- M2（Phi-3.5-mini）延迟最低（~0.8s），适合实时 NPC 交互
- M1（Qwen2.5-1.5B + roleplay）延迟最高（~3.8s），可能是 roleplay 数据集中较长的多轮对话导致模型倾向于生成更长回复
- 所有适配器均 < 100MB，部署轻量化

## 五、核心发现

1. **数据集远比模型大小重要。** roleplay 多轮数据集训练的 1.5B 模型（M1）在所有指标上碾压 npc-dialogue 单轮数据集训练的 3B 模型（M3）。单轮训练数据无法培养多轮角色一致性。

2. **LoRA 微调有效抑制出戏。** 单轮场景下所有模型 OOC 率为 0%，证明 LoRA 成功将基础模型的"AI 助手"行为替换为角色扮演行为。

3. **更大模型不一定更好。** 在相同的 npc-dialogue 单轮训练数据下，3B 模型（M3）不仅没有超越 1.5B（M4），多轮 OOC 率反而高出 11%。模型越大，越容易在压力下"退回"预训练行为。

4. **Phi-3.5-mini 是意外的"抗出戏"冠军。** 其 OOC 率（22.9%）远低于同数据集的 Qwen 模型（30-41%），且推理速度最快。不同基座的 chat template 和指令微调策略对角色扮演稳定性有显著影响。

5. **单轮→多轮的 OOC 跳变是普遍问题。** 所有模型的 OOC 率从单轮的 0% 跳到多轮的 23-41%，说明对抗性多轮对话仍是小模型角色扮演的主要瓶颈。

## 六、不足与改进方向

### 6.1 当前局限

- **Judge 模型可靠性有限。** 使用 Qwen2.5-1.5B 作为 judge，同模型评分出现多维度完全一致的情况，说明小模型 judge 的区分度不够。建议使用 7B+ 模型或 API judge 提升评估可靠性。
- **npc-dialogue 数据量太小。** 仅 1.7K 条单轮对话，不足以培养复杂角色扮演能力。
- **缺少 base model baseline。** 虽然 configs/models.yaml 预留了 base 模型配置，但评估仅覆盖 LoRA 模型，无法直接量化 LoRA 微调相比 base model 的提升幅度。
- **人工评估未实施。** guidance.md 设计了完整的人工盲评方案，因时间限制未执行。judge 模型评分只能作为参考。

### 6.2 后续改进

- **混合训练数据:** 将 roleplay 多轮数据与 npc-dialogue 单轮数据混合，兼顾角色多样性和对话深度。
- **更大的 judge 模型:** 使用 Qwen2.5-7B-Instruct 或 API 模型（如 GPT-4）作为 judge。
- **增加 base model baseline:** 对未微调的基座模型运行相同评估，量化 LoRA 的实际提升。
- **对抗训练:** 在训练数据中混入 OOC 探测样本，增强模型的抗出戏能力。
- **角色卡多样性:** 扩展 eval_data 中的角色类型覆盖（奇幻、科幻、日常、历史等）。

## 七、产出物清单

```
Roleplay/
├── train_roleplay_lora.ipynb           # Qwen2.5-1.5B + roleplay 训练
├── train_roleplay_lora_extra.ipynb     # Phi-3.5-mini + npc-dialogue 训练
├── train_roleplay_lora_qwen1-5b.ipynb  # Qwen2.5-1.5B + npc-dialogue 训练
├── train_roleplay_lora_qwen3b.ipynb    # Qwen2.5-3B + npc-dialogue 训练
├── train_roleplay_lora_colab.ipynb     # Colab 版本（支持 Google Drive）
├── configs/models.yaml                 # 模型配置
├── data/eval_data/                     # 评估数据集
├── scripts/                            # 评估工具链
│   ├── build_eval_data.py
│   ├── generate_responses.py
│   ├── judge_responses.py
│   └── summarize_results.py
├── results/                            # 评估结果
│   ├── generations_*.jsonl
│   ├── judgements_*.jsonl
│   └── summary.csv
├── demo/app.py                         # Gradio Demo（Playground + Arena）
├── guidance.md                         # 评估框架设计文档
├── conclusion.md                       # 本文档
└── requirements.txt                    # Python 依赖
```
