"""
RAG Baseline for pay equity compliance evaluation.
Uses sentence-transformers for embeddings and FAISS for retrieval.
Retrieves k most similar training examples and injects them as few-shot context.

No fine-tuning — base model only.

Usage:
    python rag_baseline.py

Output:
    ./eval_results/rag/<model_short_name>/results.json
"""

import os
import re
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss

# ── CONFIG ────────────────────────────────────────────────────────────────────

BASE_MODEL       = "microsoft/Phi-3.5-mini-instruct"
TRAIN_DATA_PATH  = "training_data.jsonl"
TEST_DATA_PATH   = "test_data.jsonl"
OUTPUT_DIR       = "./eval_results/rag/phi-3.5-mini"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 3       # number of examples to retrieve per query
MAX_NEW_TOKENS   = 1024

# ── LOAD BASE MODEL (no adapter) ─────────────────────────────────────────────

def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.config.use_cache = True
    model.eval()
    print(f"Base model loaded: {BASE_MODEL}")
    return model, tokenizer

# ── BUILD VECTOR INDEX ────────────────────────────────────────────────────────

def build_index(train_records, embed_model):
    """Embed training inputs and build FAISS index per task category."""
    indices = {}
    for tc in ["TC1", "TC2", "TC3"]:
        records = [r for r in train_records if _get_tc(r) == tc]
        if not records:
            continue

        texts = [r.get("input", r.get("instruction", "")) for r in records]
        embeddings = embed_model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        indices[tc] = {"index": index, "records": records, "embeddings": embeddings}
        print(f"  {tc}: {len(records)} examples indexed")

    return indices


def _get_tc(record):
    """Infer task category from instruction."""
    instr = record.get("instruction", "")
    if "configuration object" in instr or "JSON configuration" in instr:
        return "TC1"
    if "auditor" in instr or "checklist" in instr:
        return "TC3"
    if "narrative report" in instr:
        return "TC2"
    return "TC1"


def retrieve(query_text, tc, indices, embed_model, k=TOP_K):
    """Retrieve k most similar training examples for a given query."""
    if tc not in indices:
        return []

    query_emb = embed_model.encode([query_text], show_progress_bar=False)
    query_emb = np.array(query_emb, dtype="float32")
    faiss.normalize_L2(query_emb)

    scores, idx = indices[tc]["index"].search(query_emb, k)
    results = []
    for i in idx[0]:
        if i >= 0:
            results.append(indices[tc]["records"][i])
    return results

# ── BUILD RAG PROMPT ──────────────────────────────────────────────────────────

def build_rag_prompt(instruction, input_text, examples):
    """Build few-shot prompt with retrieved examples."""
    prompt = ""

    for ex in examples:
        ex_input = ex.get("input", "")
        ex_output = ex.get("output", "")
        if ex_input:
            prompt += (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex_input}\n\n"
                f"### Response:\n{ex_output}\n\n"
            )
        else:
            prompt += (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n{ex_output}\n\n"
            )

    if input_text:
        prompt += (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        prompt += (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    return prompt

# ── INFERENCE ─────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Truncate if prompt too long (keep last 3000 tokens for context)
    if inputs["input_ids"].shape[1] > 3000:
        inputs["input_ids"] = inputs["input_ids"][:, -3000:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, -3000:]

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

# ── SCORERS ───────────────────────────────────────────────────────────────────

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group()), None
        except json.JSONDecodeError as e:
            return None, str(e)
    return None, "No JSON found"


def score_tc1(raw_output, expected):
    predicted, err = extract_json(raw_output)
    if predicted is None:
        return 0.0, {"parse_fail": True, "error": err}
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except Exception:
            return 0.0, {"parse_fail": True}

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

    scores["filter"] = 1.0 if predicted.get("filters", {}) == expected.get("filters", {}) else 0.0
    scores["params"] = 1.0 if set(predicted.get("metrics", [])) == set(expected.get("metrics", [])) else 0.0

    weighted = (
        0.25 * scores["target"] +
        0.25 * scores["comparison"] +
        0.20 * scores["f1_controls"] +
        0.15 * scores["filter"] +
        0.15 * scores["params"]
    )
    return round(weighted, 4), scores


def score_tc2(raw_output, expected):
    text = raw_output.lower()
    scores = {}

    if isinstance(expected, str):
        pln_values = re.findall(r'PLN\s+[\d,]+\.?\d*', expected, re.IGNORECASE)
        if pln_values:
            found = sum(1 for v in pln_values if v.lower() in text)
            scores["numerical"] = found / len(pln_values)
        else:
            scores["numerical"] = 1.0
    else:
        scores["numerical"] = 0.5

    citation_terms = ["mean gap", "median gap", "article 9", "directive", "threshold",
                      "controlling", "adjusted", "joint", "%"]
    found = sum(1 for t in citation_terms if t in text)
    scores["citation"] = min(1.0, found / 5)

    word_count = len(raw_output.split())
    if word_count < 50:
        scores["hallucination_ok"] = 0.0
    elif word_count > 600:
        scores["hallucination_ok"] = 0.5
    else:
        scores["hallucination_ok"] = 1.0

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
    predicted, err = extract_json(raw_output)
    if predicted is None:
        return 0.0, {"parse_fail": True, "error": err}
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except Exception:
            return 0.0, {"parse_fail": True}

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

def evaluate(model, tokenizer, test_records, indices, embed_model):
    print(f"\nEvaluating {len(test_records)} test examples [RAG k={TOP_K}]...")

    results = []
    tc_scores = {"TC1": [], "TC2": [], "TC3": []}

    for i, rec in enumerate(test_records):
        tc = rec.get("task_category", "TC1")
        instruction = rec["instruction"]
        input_text = rec.get("input", "")
        expected = rec.get("expected_output", rec.get("output", ""))

        query_text = input_text if input_text else instruction
        examples = retrieve(query_text, tc, indices, embed_model)

        prompt = build_rag_prompt(instruction, input_text, examples)
        raw_output = generate(model, tokenizer, prompt)

        scorer = SCORERS.get(tc, score_tc1)
        score, breakdown = scorer(raw_output, expected)

        result = {
            "id": rec.get("id", i),
            "task_category": tc,
            "label": "rag",
            "score": score,
            "breakdown": breakdown,
            "retrieved_ids": [e.get("id", "?") for e in examples],
            "raw_output": raw_output[:500],
        }
        results.append(result)
        tc_scores[tc].append(score)

        status = f"{score:.3f}" if not breakdown.get("parse_fail") else "PARSE_FAIL"
        print(f"  [{i+1:02d}] {tc} | {status}")

    print(f"\n── Results [RAG k={TOP_K}] ────────────────────")
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

    train_records = [json.loads(l) for l in open(TRAIN_DATA_PATH, encoding="utf-8")]
    test_records = [json.loads(l) for l in open(TEST_DATA_PATH, encoding="utf-8")]
    print(f"Train: {len(train_records)} | Test: {len(test_records)}")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Building FAISS index...")
    indices = build_index(train_records, embed_model)

    model, tokenizer = load_model()
    results = evaluate(model, tokenizer, test_records, indices, embed_model)

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
