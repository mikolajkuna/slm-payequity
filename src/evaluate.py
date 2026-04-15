"""
Evaluation script for fine-tuned QLoRA adapter
Tests TC1 / TC2 / TC3 tasks with and without LoRA adapter (zero-shot vs fine-tuned)

Usage:
    python evaluate_peft.py

Output:
    ./eval_results/<model_short_name>/results.json
"""

import os
import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_MODEL      = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_PATH    = "./output_peft/phi-3.5-mini/lora_adapter"
TEST_DATA_PATH  = "test_data.jsonl"
OUTPUT_DIR      = "./eval_results/phi-3.5-mini"
COMPARE_ZEROSHOT = True
MAX_NEW_TOKENS  = 1024

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

def load_model(use_adapter=True):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = True

    if use_adapter:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print(f"Adapter loaded from: {ADAPTER_PATH}")
    else:
        print("Running zero-shot (no adapter)")

    model.eval()
    return model, tokenizer

# ── INFERENCE ─────────────────────────────────────────────────────────────────

def generate(model, tokenizer, instruction, input_text=""):
    if input_text:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ── JSON EXTRACTION ───────────────────────────────────────────────────────────

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), None
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "No JSON found"

# ── SCORERS ───────────────────────────────────────────────────────────────────

def score_tc1(raw_output, expected):
    """
    Weighted accuracy for TC1 (config extraction).
    Acc = 0.25*target + 0.25*comparison + 0.20*F1_controls + 0.15*filter + 0.15*params
    """
    predicted, err = extract_json(raw_output)
    if predicted is None:
        return 0.0, {"parse_fail": True, "error": err}

    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except Exception:
            return 0.0, {"parse_fail": True, "error": "bad expected"}

    scores = {}
    scores["target"] = 1.0 if predicted.get("target_variable") == expected.get("target_variable") else 0.0
    scores["comparison"] = 1.0 if predicted.get("comparison_attribute") == expected.get("comparison_attribute") else 0.0

    pred_controls = set(predicted.get("control_variables", []))
    exp_controls = set(expected.get("control_variables", []))
    if pred_controls or exp_controls:
        tp = len(pred_controls & exp_controls)
        precision = tp / len(pred_controls) if pred_controls else 0.0
        recall = tp / len(exp_controls) if exp_controls else 0.0
        scores["f1_controls"] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    else:
        scores["f1_controls"] = 1.0

    pred_filters = predicted.get("filters", {})
    exp_filters = expected.get("filters", {})
    scores["filter"] = 1.0 if pred_filters == exp_filters else 0.0

    pred_metrics = set(predicted.get("metrics", []))
    exp_metrics = set(expected.get("metrics", []))
    scores["params"] = 1.0 if pred_metrics == exp_metrics else 0.0

    weighted = (
        0.25 * scores["target"] +
        0.25 * scores["comparison"] +
        0.20 * scores["f1_controls"] +
        0.15 * scores["filter"] +
        0.15 * scores["params"]
    )
    return round(weighted, 4), scores


def score_tc2(raw_output, expected):
    """
    Faithfulness for TC2 (narrative report) — text-based scoring.
    Faithfulness = 0.35*Numerical + 0.20*Citation + 0.25*(1-Hallucination) + 0.20*Structure
    No JSON parsing — model outputs plain text narrative.
    """
    text = raw_output.lower()
    scores = {}

    # Numerical: check that PLN figures from expected appear in output
    if isinstance(expected, str):
        pln_values = re.findall(r'PLN\s+[\d,]+\.?\d*', expected, re.IGNORECASE)
        if pln_values:
            found = sum(1 for v in pln_values if v.lower() in text)
            scores["numerical"] = found / len(pln_values)
        else:
            scores["numerical"] = 1.0
    else:
        scores["numerical"] = 0.5

    # Citation: statistical/directive terminology present
    citation_terms = ["mean gap", "median gap", "article 9", "directive", "threshold",
                      "controlling", "adjusted", "joint", "%"]
    found = sum(1 for t in citation_terms if t in text)
    scores["citation"] = min(1.0, found / 5)

    # Hallucination proxy: no obviously wrong fabrications
    # Check output is not empty and has reasonable length
    word_count = len(raw_output.split())
    if word_count < 50:
        scores["hallucination_ok"] = 0.0
    elif word_count > 600:
        scores["hallucination_ok"] = 0.5
    else:
        scores["hallucination_ok"] = 1.0

    # Structure: paragraphs, no bullet points, PLN mentioned
    has_pln = "pln" in text
    no_bullets = "•" not in raw_output and not re.search(r'^\s*[-*]\s', raw_output, re.MULTILINE)
    has_paragraphs = raw_output.count("\n\n") >= 1
    scores["structure"] = (int(has_pln) + int(no_bullets) + int(has_paragraphs)) / 3.0

    weighted = (
        0.35 * scores["numerical"] +
        0.20 * scores["citation"] +
        0.25 * scores["hallucination_ok"] +
        0.20 * scores["structure"]
    )
    return round(weighted, 4), scores


def score_tc3(raw_output, expected):
    """
    Coverage for TC3 (compliance checklist).
    Focuses on checklist boolean accuracy — lenient on violations wording.
    Coverage = correct_checklist_items / 6
    """
    predicted, err = extract_json(raw_output)
    if predicted is None:
        return 0.0, {"parse_fail": True, "error": err}

    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except Exception:
            return 0.0, {"parse_fail": True, "error": "bad expected"}

    scores = {}

    scores["compliant"] = 1.0 if predicted.get("compliant") == expected.get("compliant") else 0.0

    pred_checklist = predicted.get("checklist", {})
    exp_checklist = expected.get("checklist", {})
    checklist_keys = list(exp_checklist.keys())
    if checklist_keys:
        correct = sum(1 for k in checklist_keys if pred_checklist.get(k) == exp_checklist.get(k))
        scores["checklist_accuracy"] = correct / len(checklist_keys)
    else:
        scores["checklist_accuracy"] = 1.0

    # Violations: only count match (not exact text)
    pred_v = len(predicted.get("violations", []))
    exp_v = len(expected.get("violations", []))
    scores["violations_count"] = 1.0 if pred_v == exp_v else max(0.0, 1.0 - abs(pred_v - exp_v) * 0.25)

    weighted = (
        0.30 * scores["compliant"] +
        0.50 * scores["checklist_accuracy"] +
        0.20 * scores["violations_count"]
    )
    return round(weighted, 4), scores


SCORERS = {"TC1": score_tc1, "TC2": score_tc2, "TC3": score_tc3}

# ── EVALUATE ──────────────────────────────────────────────────────────────────

def evaluate(model, tokenizer, label="finetuned"):
    records = []
    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\nEvaluating {len(records)} test examples [{label}]...")

    results = []
    tc_scores = {"TC1": [], "TC2": [], "TC3": []}

    for i, rec in enumerate(records):
        tc = rec.get("task_category", "TC1")
        instruction = rec["instruction"]
        input_text = rec.get("input", "")
        expected = rec.get("expected_output", rec.get("output", ""))

        raw_output = generate(model, tokenizer, instruction, input_text)

        scorer = SCORERS.get(tc, score_tc1)
        score, breakdown = scorer(raw_output, expected)

        result = {
            "id": rec.get("id", i),
            "task_category": tc,
            "label": label,
            "score": score,
            "breakdown": breakdown,
            "raw_output": raw_output[:500],
        }
        results.append(result)
        tc_scores[tc].append(score)

        status = f"{score:.3f}" if not breakdown.get("parse_fail") else "PARSE_FAIL"
        print(f"  [{i+1:02d}] {tc} | {status}")

    print(f"\n── Results [{label}] ────────────────────")
    for tc, scores in tc_scores.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {tc}: {avg:.3f} (n={len(scores)})")
    overall = [s for scores in tc_scores.values() for s in scores]
    print(f"  Overall: {sum(overall)/len(overall):.3f} (n={len(overall)})")

    return results

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    if COMPARE_ZEROSHOT:
        model, tokenizer = load_model(use_adapter=False)
        all_results += evaluate(model, tokenizer, label="zeroshot")
        del model
        torch.cuda.empty_cache()

    model, tokenizer = load_model(use_adapter=True)
    all_results += evaluate(model, tokenizer, label="finetuned")

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
