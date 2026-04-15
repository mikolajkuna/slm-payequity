"""
Ablation evaluation only — runs evaluate_peft.py for each trained adapter.
Assumes training already done by run_ablation.py.

Usage:
    $env:PYTHONUTF8="1"
    venv313\Scripts\python.exe run_ablation_eval.py
"""

import os
os.environ["PYTHONUTF8"] = "1"

import json
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import importlib.util

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"

runs = [
    ("n20", "./output_peft/llama-ablation-n20/lora_adapter"),
    ("n40", "./output_peft/llama-ablation-n40/lora_adapter"),
    ("n60", "./output_peft/llama-ablation-n60/lora_adapter"),
]

KNOWN_N144 = {"TC1": 0.990, "TC2": 0.655, "TC3": 0.870, "Overall": 0.839}

# ── LOAD evaluate_peft scorers ────────────────────────────────────────────────

spec = importlib.util.spec_from_file_location("ep", "evaluate_peft.py")
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)

# ── SHARED BNB CONFIG ─────────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── EVALUATE EACH ADAPTER ─────────────────────────────────────────────────────

summary = {}

for label, adapter_path in runs:
    n = label
    eval_dir = f"./eval_results/ablation/llama-{n}"

    # Skip if already evaluated
    result_file = os.path.join(eval_dir, "results.json")
    if os.path.isfile(result_file):
        print(f"\n[{n}] Already evaluated — loading from {result_file}")
        with open(result_file, encoding="utf-8") as f:
            results = json.load(f)
        tc_scores = {"TC1": [], "TC2": [], "TC3": []}
        for r in results:
            tc = r.get("task_category", "TC1")
            if tc in tc_scores:
                tc_scores[tc].append(r["score"])
        summary[n] = {
            tc: round(sum(v) / len(v), 3) for tc, v in tc_scores.items() if v
        }
        summary[n]["Overall"] = round(
            sum(r["score"] for r in results) / len(results), 3)
        continue

    os.makedirs(eval_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"EVALUATING {n}: {adapter_path}")
    print(f"{'='*60}")

    if not os.path.isdir(adapter_path):
        print(f"  ERROR: adapter not found at {adapter_path} — skipping")
        continue

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print(f"  Adapter loaded from: {adapter_path}")

    # Patch ep config and evaluate
    ep.BASE_MODEL   = BASE_MODEL
    ep.ADAPTER_PATH = adapter_path
    ep.OUTPUT_DIR   = eval_dir
    ep.COMPARE_ZEROSHOT = False

    results = ep.evaluate(model, tokenizer, label=f"ft-{n}")

    # Save results
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {result_file}")

    # Collect scores
    tc_scores = {"TC1": [], "TC2": [], "TC3": []}
    for r in results:
        tc = r.get("task_category", "TC1")
        if tc in tc_scores:
            tc_scores[tc].append(r["score"])
    summary[n] = {
        tc: round(sum(v) / len(v), 3) for tc, v in tc_scores.items() if v
    }
    summary[n]["Overall"] = round(
        sum(r["score"] for r in results) / len(results), 3)

    # Free VRAM before next run
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(8)

# ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────────

print("\n\n" + "=" * 60)
print("ABLATION SUMMARY — Llama 3.1 8B (data efficiency)")
print("=" * 60)
print(f"{'n':>6}  {'TC1':>6}  {'TC2':>6}  {'TC3':>6}  {'Overall':>8}")
print("-" * 44)

for key in ["n20", "n40", "n60"]:
    s = summary.get(key, {})
    tc1 = f"{s['TC1']:.3f}" if s.get("TC1") is not None else "  —  "
    tc2 = f"{s['TC2']:.3f}" if s.get("TC2") is not None else "  —  "
    tc3 = f"{s['TC3']:.3f}" if s.get("TC3") is not None else "  —  "
    ov  = f"{s['Overall']:.3f}" if s.get("Overall") is not None else "  —  "
    print(f"{key:>6}  {tc1:>6}  {tc2:>6}  {tc3:>6}  {ov:>8}")

k = KNOWN_N144
print(f"{'n144':>6}  {k['TC1']:>6.3f}  {k['TC2']:>6.3f}  {k['TC3']:>6.3f}  {k['Overall']:>8.3f}"
      "  <- full training set")

# Save summary JSON
summary["n144"] = KNOWN_N144
with open("ablation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: ablation_summary.json")
print("Done.")
