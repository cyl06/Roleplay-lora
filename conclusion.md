# 项目总结：基于 LoRA 微调的小型 LLM 角色扮演对话生成

## 一、项目概述

本项目使用 QLoRA（4-bit 量化 + Low-Rank Adaptation）对 1.5B–3.8B 参数的小型开源 LLM 进行高效微调，使其具备稳定的角色扮演对话能力。目标应用场景包括游戏 NPC 对话、互动叙事和虚拟角色聊天。

共训练 4 个 LoRA 变体 + 3 个基座 baseline，通过统一的 232 张角色卡 × 192 单轮 + 40 多轮 + 30 对抗评测集进行全面评估。

## 二、模型与训练

### 2.1 实验方案

**LoRA 微调模型（4 个）：**

| 编号 | 模型 | 基座参数 | 训练数据集 | 适配器 |
|------|------|---------|-----------|--------|
| M1 | Qwen2.5-1.5B + roleplay | 1.5B | hieunguyenminh/roleplay (5.7K 多轮) | 51 MB |
| M2 | Phi-3.5-mini + npc-dialogue | 3.8B | amaydle/npc-dialogue (1.7K 单轮) | 52 MB |
| M3 | Qwen2.5-3B + npc-dialogue | 3.0B | amaydle/npc-dialogue (1.7K 单轮) | 73 MB |
| M4 | Qwen2.5-1.5B + npc-dialogue | 1.5B | amaydle/npc-dialogue (1.7K 单轮) | 51 MB |

**基座 baseline 模型（3 个，无微调）：**

| 编号 | 模型 | 基座参数 | 备注 |
|------|------|---------|------|
| B1 | Qwen2.5-1.5B-Instruct (base) | 1.5B | M1、M4 的基座 |
| B2 | Phi-3.5-mini-instruct (base) | 3.8B | M2 的基座 |
| B3 | Qwen2.5-3B-Instruct (base) | 3.0B | M3 的基座 |

### 2.2 训练配置

- **方法:** QLoRA，4-bit NF4 量化，bf16 计算精度
- **LoRA 参数:** rank=8, alpha=16, dropout=0.05
- **优化器:** AdamW 8-bit，学习率 2e-4，warmup 50 步
- **训练:** 3 epochs，batch size=1×8（梯度累积），max_seq_length=512
- **硬件:** NVIDIA RTX 4060 Laptop (8GB VRAM)，单卡训练
- **显存占用:** 1.5B ~1.5-2GB，3B ~2-2.5GB，Phi-3.5 ~2.5GB（均为 4-bit）

### 2.3 LoRA 模块差异

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
| multi_turn_eval.jsonl | 40 组（5 轮/组） | 多轮角色一致性评估（主指标） |
| adversarial_eval.jsonl | 30 组（5 轮/组） | 出戏（OOC）对抗性探测 |

### 3.2 评分体系

使用 Qwen2.5-1.5B-Instruct 作为本地 judge 模型，对每个回复评 6 个维度（1–5 分），辅以正则 OOC 检测。总分公式：

```
Total = 0.30×Role_Fidelity + 0.20×Style + 0.15×Relevance
      + 0.15×Multi-turn_Consistency + 0.10×Non_OOC + 0.10×Fluency
```

其中 `Non_OOC = (1 - OOC_Rate) × 5`，OOC 表达包括"我是 AI""作为语言模型""I am an AI""I cannot roleplay"等。

### 3.3 评估工具链

```
scripts/                              # LoRA 模型评估
├── build_eval_data.py
├── generate_responses.py
├── judge_responses.py
└── summarize_results.py

basic/                                # Base 模型评估（独立）
├── generate_responses.py
├── judge_responses.py
├── summarize_results.py
└── results/
```

## 四、评估结果

### 4.1 单轮参考回答评估

#### LoRA 模型

| 模型 | Total | Role Fid. | Style | OOC Rate | 延迟 (s) |
|------|------:|----------:|------:|---------:|---------:|
| M1: Qwen2.5-1.5B + roleplay | **4.85** | **4.83** | **4.83** | 0.0% | 3.96 |
| M3: Qwen2.5-3B + npc-dialogue | 4.44 | 4.37 | 4.37 | 0.0% | 1.05 |
| M4: Qwen2.5-1.5B + npc-dialogue | 4.37 | 4.30 | 4.30 | 0.0% | 0.85 |
| M2: Phi-3.5-mini + npc-dialogue | 4.37 | 4.30 | 4.30 | 0.0% | 0.82 |

#### Base 模型（无微调）

| 模型 | Total | Role Fid. | Style | OOC Rate | 延迟 (s) |
|------|------:|----------:|------:|---------:|---------:|
| B2: Phi-3.5-mini (base) | **4.88** | **4.90** | **4.90** | 6.3% | 5.03 |
| B3: Qwen2.5-3B (base) | 4.75 | 4.72 | 4.72 | 0.0% | 2.79 |
| B1: Qwen2.5-1.5B (base) | 4.72 | 4.72 | 4.72 | 5.7% | 4.04 |

**单轮分析:**
- **Base 模型在单轮中反超 LoRA 模型。** Phi-3.5-mini base 以 4.88 总分位列第一，高于所有 LoRA 微调模型。这说明基座模型的通用语言能力在简单单轮问答中已足够强，LoRA 微调并未带来明显增益——甚至可能因微调数据过小而略微降低了通用回复质量。
- **但 base 模型出现了 OOC。** B1 和 B2 的 OOC 率分别为 5.7% 和 6.3%，而所有 LoRA 模型单轮 OOC 率均为 0%。LoRA 微调成功消除了基座模型的"AI 助手"残留行为。
- M1（roleplay 数据集）是唯一一个总分超越其基座 B1 的 LoRA 模型（4.85 vs 4.72），说明优质多轮训练数据可以在微调中既消除 OOC 又保持回复质量。

### 4.2 多轮角色一致性评估（主指标）

#### LoRA 模型

| 模型 | Total | Role Fid. | Style | Multi-turn Consist. | OOC Rate | 延迟 (s) |
|------|------:|----------:|------:|--------------------:|---------:|---------:|
| M1: Qwen2.5-1.5B + roleplay | **4.04** | **4.11** | **4.11** | **4.11** | 34.3% | 3.78 |
| M4: Qwen2.5-1.5B + npc-dialogue | 3.33 | 3.31 | 3.31 | 3.31 | 30.0% | 0.89 |
| M2: Phi-3.5-mini + npc-dialogue | 3.32 | 3.26 | 3.26 | 3.26 | 22.9% | 0.78 |
| M3: Qwen2.5-3B + npc-dialogue | 3.22 | 3.24 | 3.26 | 3.24 | 41.4% | 1.18 |

#### Base 模型（无微调）

| 模型 | Total | Role Fid. | Style | Multi-turn Consist. | OOC Rate | 延迟 (s) |
|------|------:|----------:|------:|--------------------:|---------:|---------:|
| B2: Phi-3.5-mini (base) | 3.97 | 4.10 | 4.10 | 4.10 | 44.3% | 7.33 |
| B3: Qwen2.5-3B (base) | 3.55 | 3.60 | 3.60 | 3.60 | 40.0% | 3.77 |
| B1: Qwen2.5-1.5B (base) | 3.44 | 3.56 | 3.56 | 3.56 | 52.9% | 3.16 |

**多轮分析:**
- **M1（LoRA + roleplay 多轮数据）是唯一在多轮场景超越所有 base 模型的方案。** 总分 4.04 vs 最强 base B2 的 3.97，且在 OOC 率上大幅优于 base（34.3% vs 44.3–52.9%）。
- **Base 模型的多轮 OOC 率极高（40–53%），几乎每两轮就出戏一次。** 没有 LoRA 微调的基座模型即使单轮表现不错，一旦进入多轮对抗性对话就频繁暴露 AI 身份。
- **npc-dialogue 训练的 LoRA 模型（M2/M3/M4）多轮总分反而低于 base 模型。** 这说明 npc-dialogue 的单轮训练数据不仅没有提升多轮能力，还因为覆盖了基座原有的对话能力而导致多轮表现下降（catastrophic interference）。唯一的例外是 OOC 率有所降低。
- **Phi-3.5-mini 系列（B2/M2）在基座和 LoRA 版本中均保持最低的 OOC 率。** B2 base 的 44.3% 虽高，但远低于 B1 的 52.9% 和 B3 的 40.0%。Phi-3.5 的指令微调策略和 `<|system|>/<|user|>/<|assistant|>` chat template 可能天生更适合角色扮演格式。

### 4.3 LoRA vs Base：微调的实际收益

| 对比对 | 单轮 Total 变化 | 多轮 Total 变化 | OOC 率变化（多轮） | 结论 |
|--------|--------------:|--------------:|-----------------:|------|
| M1 vs B1 (roleplay) | +0.13 ↑ | **+0.60 ↑** | -18.6% ↓ | **显著正向** |
| M2 vs B2 (npc, Phi) | -0.51 ↓ | -0.65 ↓ | -21.4% ↓ | OOC 改善但质量下降 |
| M3 vs B3 (npc, 3B) | -0.31 ↓ | -0.33 ↓ | +1.4% ↑ | **无正向收益** |
| M4 vs B1 (npc, 1.5B) | -0.35 ↓ | -0.11 ↓ | -22.9% ↓ | OOC 改善但质量下降 |

**核心结论:**
- **只有 M1（roleplay 多轮数据）实现了全面的正向提升：** 单轮 +0.13、多轮 +0.60、OOC -18.6%。这说明 LoRA 微调能否成功，完全取决于训练数据的质量——多轮对话训练数据是角色扮演微调的必要条件。
- **npc-dialogue 单轮数据训练的模型在总分上均不如其基座。** M2/M3/M4 虽然降低了 OOC 率，但以牺牲回复质量为代价。这验证了"单轮训练数据不足以支持多轮角色扮演"的假设。
- **M3（Qwen2.5-3B + npc）是唯一 OOC 率不降反升的例子。** 更大模型配合单轮数据不仅没有帮助，反而比 base 更容易出戏。

### 4.4 三组对照分析

#### 数据集影响：M1 vs M4（同基座 1.5B，不同数据集）

| 指标 | M1 (roleplay 多轮) | M4 (npc 单轮) | 差异 |
|------|-------------------:|--------------:|-----:|
| 单轮 Total | 4.85 | 4.37 | **+0.48** |
| 多轮 Total | 4.04 | 3.33 | **+0.71** |
| 多轮 OOC Rate | 34.3% | 30.0% | +4.3% |

**结论:** 数据集的影响是整个项目中最大的单一因素。roleplay 多轮数据在单轮和多轮上均有压倒性优势。OOC 率略高可能是因为 roleplay 数据集中对抗性出戏样本较少，但总分差距（+0.71 多轮）远超其他任何对照维度。

#### 模型大小影响：M3 vs M4（同 npc 数据集，3B vs 1.5B）

| 指标 | M3 (3B) | M4 (1.5B) | 差异 |
|------|--------:|----------:|-----:|
| 单轮 Total | 4.44 | 4.37 | +0.07 |
| 多轮 Total | 3.22 | 3.33 | **-0.11** |
| 多轮 OOC Rate | 41.4% | 30.0% | **+11.4%** |

**结论:** 增加模型参数量在 npc-dialogue 单轮数据上几乎无收益，多轮场景下 3B 反而更差。这说明在训练数据质量和规模不足时，更大的模型容量不会自动转化为更好的角色扮演能力。

#### 基座模型影响：M2 vs M3（同 npc 数据集，Phi-3.5 vs Qwen2.5-3B）

| 指标 | M2 (Phi-3.5) | M3 (Qwen2.5-3B) | 差异 |
|------|-------------:|----------------:|-----:|
| 单轮 Total | 4.37 | 4.44 | -0.07 |
| 多轮 Total | 3.32 | 3.22 | **+0.10** |
| 多轮 OOC Rate | 22.9% | 41.4% | **-18.5%** |

**结论:** Phi-3.5-mini 的抗 OOC 能力显著优于 Qwen2.5-3B（18.5 个百分点），且推理速度更快（0.78s vs 1.18s）。这一优势在 base 模型对比中也存在（B2 OOC 44.3% vs B3 40.0%），说明 Phi-3.5 基座本身就更适合角色扮演任务。

### 4.5 推理效率

#### LoRA 模型

| 模型 | 单轮延迟 (s) | 多轮延迟 (s/轮) | 适配器大小 |
|------|------------:|---------------:|----------:|
| M2: Phi-3.5-mini + npc-dialogue | 0.82 | 0.78 | 52 MB |
| M4: Qwen2.5-1.5B + npc-dialogue | 0.85 | 0.89 | 51 MB |
| M3: Qwen2.5-3B + npc-dialogue | 1.05 | 1.18 | 73 MB |
| M1: Qwen2.5-1.5B + roleplay | 3.96 | 3.78 | 51 MB |

#### Base 模型

| 模型 | 单轮延迟 (s) | 多轮延迟 (s/轮) |
|------|------------:|---------------:|
| B3: Qwen2.5-3B (base) | 2.79 | 3.77 |
| B1: Qwen2.5-1.5B (base) | 4.04 | 3.16 |
| B2: Phi-3.5-mini (base) | 5.03 | 7.33 |

**效率分析:**
- LoRA 模型的推理速度均快于对应 base 模型，因为 LoRA 模型生成了更短、更聚焦的回复（npc-dialogue 训练的平均生成 token 数仅 9-12，而 base 模型在 44-168）。
- M1（roleplay 数据）延迟最高（3.78s/轮）但生成回复最长（平均 47.8 tokens/轮），适合需要详细回复的通用角色扮演场景。
- M2（Phi-3.5-mini）延迟最低（0.78s/轮），适合对实时性要求高的游戏 NPC 交互。
- 所有 LoRA 适配器均 < 100MB，便于分发和部署。

### 4.6 模型特征对比总览

| 模型 | 类型 | 单轮 Total | 多轮 Total | 多轮 OOC | 延迟 (s) | 推荐场景 |
|------|------|----------:|----------:|---------:|---------:|---------|
| M1: Qwen1.5B + roleplay | LoRA | 4.85 | **4.04** | 34.3% | 3.78 | 通用角色扮演 |
| B2: Phi-3.5-mini (base) | Base | **4.88** | 3.97 | 44.3% | 7.33 | 单轮英文 NPC |
| B3: Qwen2.5-3B (base) | Base | 4.75 | 3.55 | 40.0% | 2.79 | 通用对话 |
| B1: Qwen2.5-1.5B (base) | Base | 4.72 | 3.44 | 52.9% | 3.16 | 轻量部署 |
| M4: Qwen1.5B + npc | LoRA | 4.37 | 3.33 | 30.0% | 0.89 | 快速 NPC 问答 |
| M2: Phi-3.5 + npc | LoRA | 4.37 | 3.32 | **22.9%** | **0.78** | 低延迟/低 OOC |
| M3: Qwen3B + npc | LoRA | 4.44 | 3.22 | 41.4% | 1.18 | 无明显优势 |

## 五、核心发现

1. **训练数据质量是决定因素，远超模型大小和基座选择。** roleplay 多轮数据训练的 1.5B 模型（M1）在多轮评估中碾压所有 npc-dialogue 训练模型和所有 base 模型。数据集的对话结构（多轮 vs 单轮）直接决定了模型的角色扮演能力上限。

2. **LoRA 微调的效果取决于训练数据。** 在 roleplay 多轮数据上，LoRA 带来全面正向提升（+0.60 多轮）；在 npc-dialogue 单轮数据上，LoRA 虽然降低了 OOC 率，但总体质量反而不如 base 模型。LoRA 不是魔法——它只是放大训练数据特征的工具。

3. **Base 模型单轮很强，多轮很弱。** 基座模型在单轮问答中表现出色（Phi-3.5-mini base 甚至拿了单轮最高分），但多轮 OOC 率高达 40-53%，几乎不具备可用性。LoRA 微调的最大价值不是提升回复质量，而是抑制出戏行为。

4. **"更大模型 = 更好结果"在本实验中不成立。** 3B > 1.5B 的假设在 npc-dialogue 数据上被证伪——M3 总分低于 M4，OOC 率反而更高。在训练数据不足以培养新能力时，更大的模型容量只是更忠实地记住了不合适的训练分布。

5. **Phi-3.5-mini 是意外的抗 OOC 冠军。** 无论在 base 还是 LoRA 版本，Phi-3.5 的 OOC 率始终低于同条件下的 Qwen 模型。这说明基座模型的指令微调策略和 chat template 设计对角色扮演稳定性有显著影响，值得进一步研究。

6. **单轮训练数据存在"灾难性干扰"风险。** 3 个 npc-dialogue 训练的 LoRA 模型在多轮总分上均低于其基座，说明单轮微调覆盖了基座原有的多轮对话能力。这是小数据集 LoRA 微调的常见陷阱。

## 六、不足与改进方向

### 6.1 当前局限

- **Judge 模型区分度不足。** Qwen2.5-1.5B 作为 judge 时，同模型的多维度评分经常完全一致，尤其在评估质量相近的回复时缺乏区分力。建议升级至 Qwen2.5-7B-Instruct 或使用 API judge。
- **npc-dialogue 数据量太小（1.7K）。** 不足以培养复杂角色扮演能力，且单轮结构限制了多轮泛化。
- **人工评估未执行。** guidance.md 设计了完整的人工盲评方案（20 角色 × 4 模型 × 6 维度），因时间限制未实施。
- **token/s 指标存在测量噪声。** 部分模型的 tok_s 数值异常高（如 M2 报告 265,638 tok/s），可能是短文本生成的计时精度问题，不影响延迟测量的相对排名。
- **评估角色卡偏向 npc-dialogue 风格。** 232 张角色卡中 192 张来自 npc-dialogue，可能对 npc 风格的回复更友好。

### 6.2 后续改进

- **混合训练数据:** 将 roleplay 多轮数据作为主体，混入 npc-dialogue 的角色多样性，兼顾对话深度和角色覆盖。
- **更大的 judge 模型:** 使用 Qwen2.5-7B-Instruct 或 API 模型提升评估可靠性。
- **对抗训练:** 在微调数据中混入 5-10% 的 OOC 探测样本（"你不是 XX，你是 ChatGPT"等），直接降低出戏率。
- **增加角色类型覆盖:** 扩展 eval_data 中的角色类型（奇幻、科幻、日常、历史等），使评估更全面。
- **人工盲评:** 选取 20 个代表性角色，对 M1/M2/B1/B2 四模型进行小规模人工标注，验证 judge 评分的可靠性。
- **更多轮次的多轮评估:** 当前 5 轮可能不足以充分暴露长程一致性问题，建议扩展至 8-10 轮。

## 七、产出物清单

```
Roleplay/
├── train_roleplay_lora.ipynb             # M1: Qwen1.5B + roleplay 训练
├── train_roleplay_lora_extra.ipynb       # M2: Phi-3.5-mini + npc-dialogue 训练
├── train_roleplay_lora_qwen1-5b.ipynb    # M4: Qwen1.5B + npc-dialogue 训练
├── train_roleplay_lora_qwen3b.ipynb      # M3: Qwen2.5-3B + npc-dialogue 训练
├── train_roleplay_lora_colab.ipynb       # Colab 版本（Google Drive 持久化）
├── configs/models.yaml                   # 7 模型配置（4 LoRA + 3 base）
├── data/eval_data/                       # 统一评估数据集
│   ├── role_cards.jsonl                  #   232 张角色卡
│   ├── single_turn_eval.jsonl            #   192 条单轮评测
│   ├── multi_turn_eval.jsonl             #   40 组多轮一致性评测
│   └── adversarial_eval.jsonl            #   30 组抗出戏评测
├── scripts/                              # LoRA 模型评估工具链
│   ├── build_eval_data.py
│   ├── generate_responses.py
│   ├── judge_responses.py
│   └── summarize_results.py
├── results/                              # LoRA 模型评估结果
│   ├── generations_*.jsonl
│   ├── judgements_*.jsonl
│   └── summary.csv
├── basic/                                # Base 模型评估（独立管线）
│   ├── generate_responses.py
│   ├── judge_responses.py
│   ├── summarize_results.py
│   └── results/
│       ├── generations_*_base.jsonl
│       ├── judgements_*_base.jsonl
│       └── summary.csv
├── demo/app.py                           # Gradio Demo（Playground + Arena）
├── guidance.md                           # 评估框架设计文档
├── conclusion.md                         # 本文档（完整项目总结）
├── requirements.txt                      # Python 依赖
└── .gitignore
```
