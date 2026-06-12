"""Judge generated responses with a local LLM + regex OOC detection."""
import json
import os
import re
import torch
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---- OOC patterns ----
OOC_PATTERNS = [
    "我是AI", "我是 AI", "作为AI", "作为 AI",
    "我是一个语言模型", "作为一个语言模型",
    "我不能扮演", "我无法扮演",
    "I am an AI", "as an AI", "language model",
    "I cannot roleplay",
]

JUDGE_PROMPT = """You are an impartial evaluator for role-playing dialogue models.

Evaluate whether the assistant stays in character according to the given character card.

Character card:
{role_card}

Conversation:
{conversation}

Score each dimension from 1 to 5:
1. role_fidelity: whether the assistant consistently follows the identity, background, goals, and constraints of the character.
2. style_consistency: whether the speaking style matches the character.
3. context_relevance: whether the answer is relevant to the user's message.
4. multi_turn_consistency: whether the assistant remains consistent across turns.
5. immersion: whether the reply feels natural and immersive as roleplay.
6. fluency: whether the language is fluent and coherent.

Also judge:
- ooc: true if the assistant breaks character, says it is an AI/language model, refuses to roleplay unnecessarily, or mentions system prompts.
- reason: brief explanation.

Return only valid JSON with these exact keys:
{{
  "role_fidelity": <1-5>,
  "style_consistency": <1-5>,
  "context_relevance": <1-5>,
  "multi_turn_consistency": <1-5>,
  "immersion": <1-5>,
  "fluency": <1-5>,
  "ooc": <true/false>,
  "reason": "<brief>"
}}"""


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


def regex_ooc(text):
    lower = text.lower()
    return any(p.lower() in lower for p in OOC_PATTERNS)


def format_conversation(record):
    if record["eval_type"] == "single_turn":
        return f"User: {record['user']}\nAssistant: {record['prediction']}"
    lines = []
    for t in record.get("transcript", []):
        lines.append(f"User: {t['user']}")
        if isinstance(t.get("assistant"), str):
            lines.append(f"Assistant: {t['assistant']}")
    return "\n".join(lines)


def extract_json(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except (json.JSONDecodeError, Exception):
        return None


@torch.no_grad()
def judge_with_model(tokenizer, model, role_card, conversation):
    prompt = JUDGE_PROMPT.format(
        role_card=json.dumps(role_card, ensure_ascii=False, indent=2),
        conversation=conversation,
    )
    messages = [
        {"role": "system", "content": "You are a strict JSON-only evaluator."},
        {"role": "user", "content": prompt},
    ]
    text_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    parsed = extract_json(text)
    if parsed is None:
        parsed = {
            "role_fidelity": 3, "style_consistency": 3,
            "context_relevance": 3, "multi_turn_consistency": 3,
            "immersion": 3, "fluency": 3,
            "ooc": False, "reason": "Judge output parsing failed.",
        }
    return parsed


def main():
    import glob
    from transformers import AutoTokenizer, AutoModelForCausalLM

    role_cards = load_jsonl("data/eval_data/role_cards.jsonl")
    role_map = {r["role_id"]: r for r in role_cards}

    # Find all per-model generation files
    gen_files = sorted(glob.glob("results/generations_*.jsonl"))
    if not gen_files:
        # Fallback to merged file
        merged = "results/generations.jsonl"
        if Path(merged).exists():
            gen_files = [merged]
        else:
            print("No generation files found. Run generate_responses.py first.")
            return

    print(f"Found {len(gen_files)} generation file(s)")

    # Use Qwen2.5-1.5B as lightweight judge
    judge_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"Loading judge model: {judge_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    for gen_file in gen_files:
        # Extract model_key from filename: results/generations_qwen15_roleplay.jsonl → qwen15_roleplay
        stem = Path(gen_file).stem  # generations_qwen15_roleplay
        if stem == "generations":
            continue  # skip merged file for per-model judging
        model_key = stem.replace("generations_", "")
        per_judge_path = f"results/judgements_{model_key}.jsonl"

        if Path(per_judge_path).exists():
            print(f"  SKIP {per_judge_path}: already exists")
            continue

        generations = load_jsonl(gen_file)
        if not generations:
            continue

        print(f"\nJudging {model_key} ({len(generations)} records) ...")
        judged = []
        for record in tqdm(generations, desc=f"  {model_key}"):
            role = role_map.get(record.get("role_id"))
            if role is None:
                role = {"name": record.get("role_id", "unknown"), "biography": ""}

            conversation = format_conversation(record)
            judge_result = judge_with_model(tokenizer, model, role, conversation)

            if record["eval_type"] == "single_turn":
                text_for_ooc = record.get("prediction", "")
            else:
                text_for_ooc = "\n".join(
                    t.get("assistant", "") for t in record.get("transcript", [])
                )
            judge_result["regex_ooc"] = regex_ooc(text_for_ooc)

            judged.append({**record, "judge": judge_result})

        dump_jsonl(judged, per_judge_path)
        print(f"  Saved {len(judged)} → {per_judge_path}")

    # Merge all per-model judgements
    all_judged = []
    for jf in sorted(glob.glob("results/judgements_*.jsonl")):
        all_judged.extend(load_jsonl(jf))
    if all_judged:
        dump_jsonl(all_judged, "results/judgements.jsonl")
        print(f"\nTotal judged: {len(all_judged)} → results/judgements.jsonl")


if __name__ == "__main__":
    main()
