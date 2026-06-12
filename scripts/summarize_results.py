"""Aggregate judged records into summary CSV and print score tables."""
import json
import yaml
import pandas as pd
from pathlib import Path


def load_jsonl(path):
    if not Path(path).exists():
        print(f"  SKIP: {path} not found")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_display_name(model_key):
    try:
        with open("configs/models.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg["models"].get(model_key, {}).get("display_name", model_key)
    except Exception:
        return model_key


def get_perf(record):
    if record["eval_type"] == "single_turn":
        return record.get("performance", {"latency": 0, "tok_s": 0, "num_new_tokens": 0})
    perfs = [t.get("performance", {}) for t in record.get("transcript", [])]
    if not perfs:
        return {"latency": 0, "tok_s": 0, "num_new_tokens": 0}
    return {
        "latency": sum(p.get("latency", 0) for p in perfs) / len(perfs),
        "tok_s": sum(p.get("tok_s", 0) for p in perfs) / len(perfs),
        "num_new_tokens": sum(p.get("num_new_tokens", 0) for p in perfs) / len(perfs),
    }


def safe_float(x, default=3.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main():
    import glob

    # Load from per-model judgement files if they exist, otherwise from merged
    judge_files = sorted(glob.glob("results/judgements_*.jsonl"))
    if not judge_files:
        judge_files = ["results/judgements.jsonl"]

    records = []
    for jf in judge_files:
        if Path(jf).exists():
            records.extend(load_jsonl(jf))

    if not records:
        print("No judged records. Run judge_responses.py first.")
        return

    rows = []
    for r in records:
        j = r.get("judge", {})
        perf = get_perf(r)

        ooc = bool(j.get("ooc", False)) or bool(j.get("regex_ooc", False))
        non_ooc = 0 if ooc else 1

        rf = safe_float(j.get("role_fidelity", 3), 3)
        sc = safe_float(j.get("style_consistency", 3), 3)
        cr = safe_float(j.get("context_relevance", 3), 3)
        mtc = safe_float(j.get("multi_turn_consistency", 3), 3)
        imm = safe_float(j.get("immersion", 3), 3)
        flu = safe_float(j.get("fluency", 3), 3)

        total = (
            0.30 * rf + 0.20 * sc + 0.15 * cr +
            0.15 * mtc + 0.10 * non_ooc * 5 + 0.10 * flu
        )

        rows.append({
            "model_key": r["model_key"],
            "eval_type": r["eval_type"],
            "role_fidelity": rf,
            "style_consistency": sc,
            "context_relevance": cr,
            "multi_turn_consistency": mtc,
            "immersion": imm,
            "fluency": flu,
            "ooc": int(ooc),
            "total": total,
            "latency": perf["latency"],
            "tok_s": perf["tok_s"],
            "num_new_tokens": perf["num_new_tokens"],
        })

    df = pd.DataFrame(rows)

    # ---- Per-model summary ----
    summary = df.groupby(["model_key", "eval_type"]).agg({
        "role_fidelity": "mean",
        "style_consistency": "mean",
        "context_relevance": "mean",
        "multi_turn_consistency": "mean",
        "immersion": "mean",
        "fluency": "mean",
        "ooc": "mean",
        "total": "mean",
        "latency": "mean",
        "tok_s": "mean",
        "num_new_tokens": "mean",
        "role_fidelity": "count",  # will be overwritten with count column
    }).reset_index()

    # Rename the count column
    summary = summary.rename(columns={"role_fidelity": "count", "ooc": "ooc_rate"})

    # Fix: the last agg overwrites role_fidelity with count
    # Let's recompute properly
    summary = df.groupby(["model_key", "eval_type"]).agg(
        role_fidelity=("role_fidelity", "mean"),
        style_consistency=("style_consistency", "mean"),
        context_relevance=("context_relevance", "mean"),
        multi_turn_consistency=("multi_turn_consistency", "mean"),
        immersion=("immersion", "mean"),
        fluency=("fluency", "mean"),
        ooc_rate=("ooc", "mean"),
        total=("total", "mean"),
        latency=("latency", "mean"),
        tok_s=("tok_s", "mean"),
        num_new_tokens=("num_new_tokens", "mean"),
        count=("total", "count"),
    ).reset_index()

    summary["display_name"] = summary["model_key"].apply(get_display_name)

    # ---- Print ----
    print("\n" + "=" * 80)
    print("Evaluation Summary (all eval types)")
    print("=" * 80)
    cols = ["display_name", "eval_type", "total", "role_fidelity", "style_consistency",
            "context_relevance", "multi_turn_consistency", "immersion", "fluency",
            "ooc_rate", "tok_s", "count"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- Multi-turn only ----
    mt = summary[summary["eval_type"] == "multi_turn"]
    if len(mt) > 0:
        print("\n" + "=" * 80)
        print("Multi-turn Evaluation (main metric)")
        print("=" * 80)
        mt_cols = ["display_name", "total", "role_fidelity", "style_consistency",
                   "multi_turn_consistency", "immersion", "ooc_rate", "tok_s"]
        print(mt[mt_cols].sort_values("total", ascending=False)
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- Save ----
    Path("results").mkdir(exist_ok=True)
    summary.to_csv("results/summary.csv", index=False)
    print(f"\nSaved → results/summary.csv")

    # ---- Comparison tables ----
    print("\n" + "=" * 80)
    print("Comparison: Dataset Effect (same base, different datasets)")
    print("=" * 80)
    compare_datasets = summary[
        summary["model_key"].isin(["qwen15_roleplay", "qwen15_npc"])
    ]
    print(compare_datasets[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 80)
    print("Comparison: Model Size Effect (same dataset, different sizes)")
    print("=" * 80)
    compare_size = summary[
        summary["model_key"].isin(["qwen3_npc", "qwen15_npc"])
    ]
    print(compare_size[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 80)
    print("Comparison: Base Model Effect (same dataset, different base)")
    print("=" * 80)
    compare_base = summary[
        summary["model_key"].isin(["qwen3_npc", "phi35_npc"])
    ]
    print(compare_base[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
