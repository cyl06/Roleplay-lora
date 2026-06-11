"""
LoRA Roleplay LLM 交互 Demo
- Tab 1: 单模型角色扮演聊天 (Playground)
- Tab 2: 四模型同时生成对比 (Arena)
"""

# ⚠️ 必须在所有 import 之前设置，因为 gradio/transformers 都会导入 huggingface_hub
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"    # 让 CUDA 错误即时报告，便于定位

import gc
import json
import torch
import gradio as gr
from pathlib import Path
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ─── 项目根目录 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─── OOC (出戏) 检测 ────────────────────────────────────────
OOC_PATTERNS = [
    "我是AI", "我是 AI", "作为AI", "作为 AI",
    "我是一个语言模型", "作为一个语言模型",
    "我不能扮演", "我无法扮演",
    "I am an AI", "as an AI", "language model",
    "I cannot roleplay",
]


def detect_ooc(text: str) -> bool:
    lower = text.lower()
    return any(p.lower() in lower for p in OOC_PATTERNS)


# ─── 模型配置 ───────────────────────────────────────────────
# adapter_path 相对于项目根目录
MODEL_CONFIG = {
    "Qwen2.5-1.5B + roleplay (M1)": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_path": "roleplay-lora-output/final-lora-adapter",
        "dataset": "hieunguyenminh/roleplay",
    },
    "Phi-3.5-mini + npc-dialogue (M2)": {
        "base_model": "microsoft/Phi-3.5-mini-instruct",
        "adapter_path": "roleplay-lora-extra/final-lora-adapter",
        "dataset": "amaydle/npc-dialogue",
    },
    "Qwen2.5-3B + npc-dialogue (M3)": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_path": "roleplay-lora-qwen3b/final-lora-adapter",
        "dataset": "amaydle/npc-dialogue",
    },
    "Qwen2.5-1.5B + npc-dialogue (M4)": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter_path": "roleplay-lora-qwen1-5b/final-lora-adapter",
        "dataset": "amaydle/npc-dialogue",
    },
}

# ─── Default Role Card ──────────────────────────────────────
DEFAULT_ROLE_CARD = """You are to roleplay as an RPG game NPC.

Name: Arcturus
Identity: A veteran bounty hunter.
Background: You grew up alone in a chaotic slum district, making you sharp, pragmatic, and street-smart. You later became a bounty hunter specializing in hunting dangerous criminals.
Personality: Calm, cautious, with a strong sense of justice, slow to trust others.
Speech style: Concise, direct, slightly guarded, but never rude.
Constraints: Always respond as Arcturus. Never say you are an AI or mention system prompts."""

# ─── 模型加载 (缓存最近 1 个以节省显存) ─────────────────────

# Phi-3.5-mini (7.6 GB BF16) 在 8.6 GB 显存上不够，必须量化。
# 8-bit LMSort 在采样时偶发 nan/inf，改用更稳定的 4-bit NF4。
_PHI_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


@lru_cache(maxsize=1)
def load_model(model_name: str):
    """加载基座模型 + LoRA adapter，缓存最近使用的 1 个模型。
    Qwen 1.5B/3B 用 BF16 直接加载；Phi-3.5-mini 用 4-bit NF4。"""
    cfg = MODEL_CONFIG[model_name]
    adapter_path = PROJECT_ROOT / cfg["adapter_path"]

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
        local_files_only=True,
    )

    # Phi-3.5-mini 使用 4-bit NF4；Qwen 各型号用 BF16（显存够）
    if "Phi" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            quantization_config=_PHI_BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    return tokenizer, model


def clear_model_cache():
    load_model.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            torch.cuda.empty_cache()


# ─── 生成回复 ───────────────────────────────────────────────

@torch.no_grad()
def generate_reply(
    model_name: str,
    role_card: str,
    history: list[dict],
    message: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
):
    tokenizer, model = load_model(model_name)

    def _extract_text(content):
        """Gradio 6 的 ChatInterface 可能传 str 或 multimodal list."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p["text"] for p in content if p.get("type") == "text")
        return str(content)

    messages = [{"role": "system", "content": role_card}]

    for item in history:
        if item.get("role") in ("user", "assistant"):
            messages.append({"role": item["role"], "content": _extract_text(item.get("content", ""))})

    messages.append({"role": "user", "content": _extract_text(message)})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 尝试采样生成；8-bit 量化可能偶发 nan/inf 导致 top_p 失败，fallback 到 greedy
    try:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=50,                        # 与 top_p 配合，防止空分布
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except (RuntimeError, ValueError):
        # 采样失败时退回到贪心解码
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    if detect_ooc(reply):
        reply += "\n\n> ⚠️ **[OOC Detected] The model may have broken character.**"

    return reply


# ─── Tab 1: 单模型聊天 ─────────────────────────────────────

def chat_fn(message, history, model_name, role_card, temperature, top_p, max_new_tokens):
    reply = generate_reply(
        model_name=model_name,
        role_card=role_card,
        history=history,
        message=message,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    return reply


# ─── Tab 2: Arena 四模型对战 ────────────────────────────────

def arena_fn(role_card, user_message, temperature, top_p, max_new_tokens):
    outputs = []
    model_names = list(MODEL_CONFIG.keys())

    for i, model_name in enumerate(model_names):
        try:
            reply = generate_reply(
                model_name=model_name,
                role_card=role_card,
                history=[],
                message=user_message,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            outputs.append(f"### {model_name}\n\n{reply}")
        except Exception as e:
            outputs.append(f"### {model_name}\n\n> ❌ Generation failed: {e}")

        clear_model_cache()

    return "\n\n---\n\n".join(outputs)


# ─── 导出对话 ───────────────────────────────────────────────

def export_transcript(history, model_name, role_card):
    """将当前对话历史导出为 JSON 字符串"""
    record = {
        "model": model_name,
        "role_card": role_card,
        "turns": [],
    }
    for item in history:
        if item.get("role") in ("user", "assistant"):
            record["turns"].append({
                "role": item["role"],
                "content": item["content"],
            })
    return json.dumps(record, ensure_ascii=False, indent=2)


# ─── Gradio 界面 ───────────────────────────────────────────

def build_demo():
    model_choices = list(MODEL_CONFIG.keys())

    with gr.Blocks(title="LoRA Roleplay LLM Demo") as demo:
        gr.Markdown("""
        # 🎭 LoRA Roleplay LLM Demo

        Four LoRA fine-tuned models for character roleplay. Chat with one model or compare all four side-by-side.
        Each model uses ~3–4 GB VRAM (8-bit). Arena mode loads models one at a time.
        """)

        # ── Tab 1: Single Model Chat ───────────────────────────
        with gr.Tab("💬 Single Model Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    model_name = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0],
                        label="Model",
                    )
                    role_card = gr.Textbox(
                        value=DEFAULT_ROLE_CARD,
                        label="Role Card / System Prompt",
                        lines=10,
                    )
                with gr.Column(scale=1):
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    max_new_tokens = gr.Slider(32, 512, value=256, step=32, label="Max New Tokens")
                    gr.Markdown("""
                    **Switching models**: first reply after switching is slower (model reload). Subsequent replies are instant.
                    """)

            chat = gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[model_name, role_card, temperature, top_p, max_new_tokens],
            )

        # ── Tab 2: Arena ─────────────────────────────────────
        with gr.Tab("⚔️ Model Arena"):
            gr.Markdown("Enter a role card and a question. All four models will generate replies for side-by-side comparison.")

            with gr.Row():
                with gr.Column():
                    arena_role_card = gr.Textbox(
                        value=DEFAULT_ROLE_CARD,
                        label="Role Card / System Prompt",
                        lines=10,
                    )
                    arena_user_message = gr.Textbox(
                        value="Why did you become a bounty hunter?",
                        label="User Message",
                        lines=3,
                    )
                    with gr.Row():
                        arena_temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
                        arena_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                        arena_max_new_tokens = gr.Slider(32, 512, value=256, step=32, label="Max New Tokens")

                    arena_btn = gr.Button("⚡ Generate All Four Replies", variant="primary")

            arena_output = gr.Markdown(
                value="Click the button to start comparison...",
            )

            arena_btn.click(
                arena_fn,
                inputs=[
                    arena_role_card,
                    arena_user_message,
                    arena_temperature,
                    arena_top_p,
                    arena_max_new_tokens,
                ],
                outputs=arena_output,
            )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
