# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoRA fine-tuning of small open-source LLMs (1.5B–4B parameters) for consistent character roleplay dialogue generation. Target use cases: game NPC dialogue, interactive storytelling, virtual companion chat.

## Tech Stack (planned)

- **Model training:** LoRA via PEFT (HuggingFace `peft`), likely with `transformers` + `datasets`
- **Target base models:** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Phi-3.5-mini-instruct (3.8B)
- **Training frameworks to consider:** Axolotl (mentioned in README for dataset compatibility), or direct `transformers` Trainer with PEFT
- **Datasets:** See README.md for the full list — key ones are PIPPA (~16K dialogues), PIPPA-shareGPT (Axolotl-compatible format), hieunguyenminh/roleplay (5K+ Gemini-generated), and agentlans/multi-character-dialogue (10K+ multi-character scenes)

## Key Design Decisions (from README)

1. **LoRA, not full fine-tuning** — freeze base model weights, train only low-rank decomposition matrices. Low VRAM footprint.
2. **Small models (1.5B–4B)** — targets consumer GPU training and low-latency inference.
3. **System-prompt-driven** — character identity (name, persona, backstory, speech style) is injected via system prompt; the model must maintain that identity across multi-turn conversations.
4. **Three task variants:** general character-card roleplay, game NPC dialogue (world-aware, quest-aware), and multi-character scene dialogue.

## Development Roadmap

1. Dataset preparation — select/download datasets, convert to a unified training format (likely ChatML or ShareGPT conversation format)
2. Environment setup — PyTorch, transformers, peft, bitsandbytes (if using QLoRA)
3. Training pipeline — LoRA fine-tuning script with configurable base model, dataset, and LoRA hyperparameters (rank `r`, alpha, target modules)
4. Evaluation — qualitative (human eval of character consistency across multi-turn chats) and quantitative (perplexity on held-out roleplay dialog)
5. Inference demo — Gradio or terminal-based chat interface for interactive testing
