"""Run BASE models (no LoRA) over eval datasets, save to basic/results/."""
import json
import os
import time
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJ_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJ_ROOT / "basic" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJ_ROOT / path
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(records, path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJ_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- Model loading ----
def load_model(cfg):
    """Load base model (no adapter) — 4-bit for large models, bf16 for small."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"    Loading tokenizer & model: {cfg['base_model']} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )

    # Use 4-bit for all base models to match eval conditions
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
    print("    Base model (no LoRA)")

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
    cfg_path = PROJ_ROOT / "configs" / "models.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    role_cards = load_jsonl("data/eval_data/role_cards.jsonl")
    single_turn_samples = load_jsonl("data/eval_data/single_turn_eval.jsonl")
    multi_turn_samples = load_jsonl("data/eval_data/multi_turn_eval.jsonl")
    adversarial_samples = load_jsonl("data/eval_data/adversarial_eval.jsonl")

    # Only base models — no LoRA
    eval_models = ["qwen15_base", "qwen3_base", "phi35_base"]

    for model_key in eval_models:
        model_cfg = cfg["models"][model_key]
        records = []
        per_model_path = RESULT_DIR / f"generations_{model_key}.jsonl"

        if per_model_path.exists():
            existing = load_jsonl(str(per_model_path))
            if existing:
                print(f"\n{'='*50}\nSKIP {model_cfg['display_name']}: {per_model_path.name} already exists ({len(existing)} records)\n{'='*50}")
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
            if records:
                dump_jsonl(records, str(per_model_path))
                print(f"  Saved {len(records)} records → {per_model_path}")
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Merge all per-model files
    all_records = []
    for model_key in eval_models:
        path = RESULT_DIR / f"generations_{model_key}.jsonl"
        if path.exists():
            all_records.extend(load_jsonl(str(path)))

    if all_records:
        dump_jsonl(all_records, str(RESULT_DIR / "generations.jsonl"))
        print(f"\nTotal: {len(all_records)} records → basic/results/generations.jsonl")


if __name__ == "__main__":
    main()
