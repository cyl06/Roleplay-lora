"""Run all models over eval datasets and dump predictions to results/generations.jsonl."""
import json
import os
import time
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def load_jsonl(path):
    if not Path(path).exists():
        print(f"  SKIP: {path} not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- Model loading ----
def load_model(cfg):
    """Load base model + optional LoRA adapter.

    LoRA models are loaded with 4-bit (QLoRA) to match training and fit 8GB VRAM.
    Base models are loaded in bf16.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    has_adapter = cfg.get("adapter_path") and Path(cfg["adapter_path"]).exists()

    print(f"    Loading tokenizer & model: {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )

    if has_adapter:
        # Load with 4-bit QLoRA — same as training, fits 8GB VRAM
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"    Loading LoRA adapter: {cfg['adapter_path']}")
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_model"],
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print("    No adapter — using base model")

    model.eval()
    return tokenizer, model


# ---- Generation ----
@torch.no_grad()
def generate_reply(tokenizer, model, messages, max_new_tokens=256,
                   temperature=0.7, top_p=0.9):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency = time.time() - t0

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    tok_s = len(new_tokens) / max(latency, 1e-6)

    return text, {"latency": latency, "num_new_tokens": int(len(new_tokens)), "tok_s": tok_s}


# ---- Single-turn ----
def run_single_turn(model_key, tokenizer, model, role_cards, samples):
    role_map = {r["role_id"]: r for r in role_cards}
    records = []

    for s in tqdm(samples, desc=f"  single-turn {model_key}"):
        role = role_map.get(s["role_id"])
        if role is None:
            continue
        messages = [
            {"role": "system", "content": role["system_prompt"]},
            {"role": "user", "content": s["user"]},
        ]
        pred, perf = generate_reply(tokenizer, model, messages)
        records.append({
            "eval_type": "single_turn",
            "model_key": model_key,
            "sample_id": s["sample_id"],
            "role_id": s["role_id"],
            "user": s["user"],
            "reference": s.get("reference"),
            "emotion": s.get("emotion"),
            "prediction": pred,
            "performance": perf,
        })

    return records


# ---- Multi-turn ----
def run_multi_turn(model_key, tokenizer, model, role_cards, samples):
    role_map = {r["role_id"]: r for r in role_cards}
    records = []

    for s in tqdm(samples, desc=f"  multi-turn {model_key}"):
        role = role_map.get(s["role_id"])
        if role is None:
            continue
        messages = [{"role": "system", "content": role["system_prompt"]}]
        transcript = []

        for user_msg in s["turns"]:
            messages.append({"role": "user", "content": user_msg})
            pred, perf = generate_reply(tokenizer, model, messages)
            messages.append({"role": "assistant", "content": pred})
            transcript.append({"user": user_msg, "assistant": pred, "performance": perf})

        records.append({
            "eval_type": "multi_turn",
            "model_key": model_key,
            "dialogue_id": s["dialogue_id"],
            "role_id": s["role_id"],
            "transcript": transcript,
        })

    return records


# ---- Main ----
def main():
    with open("configs/models.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    role_cards = load_jsonl("data/eval_data/role_cards.jsonl")
    single_turn_samples = load_jsonl("data/eval_data/single_turn_eval.jsonl")
    multi_turn_samples = load_jsonl("data/eval_data/multi_turn_eval.jsonl")
    adversarial_samples = load_jsonl("data/eval_data/adversarial_eval.jsonl")

    # Only run LoRA models
    eval_models = ["phi35_npc"]
    # eval_models = ["qwen15_roleplay", "qwen15_npc", "qwen3_npc", "phi35_npc"]

    for model_key in eval_models:
        model_cfg = cfg["models"][model_key]
        records = []
        per_model_path = f"results/generations_{model_key}.jsonl"

        # Skip if already done
        if Path(per_model_path).exists():
            existing = load_jsonl(per_model_path)
            if existing:
                print(f"\n{'='*50}\nSKIP {model_cfg['display_name']}: {per_model_path} already exists ({len(existing)} records)\n{'='*50}")
                continue

        print(f"\n{'='*50}\nRunning: {model_cfg['display_name']}\n{'='*50}")

        tokenizer, model = None, None
        try:
            tokenizer, model = load_model(model_cfg)

            if single_turn_samples:
                records.extend(
                    run_single_turn(model_key, tokenizer, model, role_cards, single_turn_samples)
                )

            if multi_turn_samples:
                records.extend(
                    run_multi_turn(model_key, tokenizer, model, role_cards, multi_turn_samples)
                )

            if adversarial_samples:
                records.extend(
                    run_multi_turn(model_key, tokenizer, model, role_cards, adversarial_samples)
                )
        finally:
            # Save per-model results even on error
            if records:
                dump_jsonl(records, per_model_path)
                print(f"  Saved {len(records)} records → {per_model_path}")
            # Aggressively free memory
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Merge all per-model files
    all_records = []
    for model_key in eval_models:
        path = f"results/generations_{model_key}.jsonl"
        if Path(path).exists():
            all_records.extend(load_jsonl(path))

    if all_records:
        dump_jsonl(all_records, "results/generations.jsonl")
        print(f"\nTotal: {len(all_records)} records → results/generations.jsonl")


if __name__ == "__main__":
    main()
