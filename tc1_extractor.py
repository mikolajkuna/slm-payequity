"""
TC1 Extractor
Sends natural language analysis request to local Ollama.
Model returns config JSON. Script validates structure and optionally runs pipeline.

Usage:
    python tc1_extractor.py                        # runs all built-in test cases
    python tc1_extractor.py "your prompt here"     # single custom prompt
"""

import json
import sys
import urllib.request
import urllib.error

# ── SETTINGS ─────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3.5:latest"   # change to: llama3.1:8b | mistral:7b
DATASET_PATH = "salary_data_2009.csv"
RUN_PIPELINE = True     # set False to skip pipeline execution

SYSTEM_PROMPT = """You are a pay equity analysis assistant. Your task is to parse a natural language description of a pay equity analysis request and return a valid JSON configuration object. Do not explain your answer. Return only valid JSON.

Dataset schema:
- gender: categorical, values: ["F", "M"]
- income: integer, monthly gross salary in PLN, range: 1276-42883
- age: integer, range: 0-62
- education_level: ordinal integer, values: [1, 3, 4] (1=basic, 3=higher, 4=postgraduate)
- job_level: ordinal integer, values: [1, 2, 3, 4] (1=junior, 2=mid, 3=senior, 4=lead)
- experience_years: integer, years of experience
- distance_from_home: integer, km from home to workplace
- absence: integer, days absent per year
- child: integer, number of children, range: 0-4

Definitions:
- filters: restrict which rows are included in the analysis (e.g. only senior employees)
- control_variables: variables to condition on when computing the gap (e.g. to isolate the effect of gender)

Output JSON schema:
{
  "comparison_attribute": "<column name>",
  "target_variable": "income",
  "filters": {},
  "control_variables": ["<column names>"],
  "metrics": ["<subset of: mean_gap, median_gap, bonus_gap, quartile_distribution>"]
}

Return only valid JSON. No explanation."""

# ── TEST CASES ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "TC1-01",
        "prompt": "Analyze whether there is a gender pay gap among senior and lead employees, controlling for education level and years of experience.",
        "expected": {
            "comparison_attribute": "gender",
            "target_variable": "income",
            "filters": {"job_level": [3, 4]},
            "control_variables": ["education_level", "experience_years"],
            "metrics": ["mean_gap", "median_gap"]
        }
    },
    {
        "id": "TC1-02",
        "prompt": "Check if employees with children earn less than those without, across all job levels. Include quartile distribution in the results.",
        "expected": {
            "comparison_attribute": "child",
            "target_variable": "income",
            "filters": {},
            "control_variables": ["job_level", "gender"],
            "metrics": ["mean_gap", "median_gap", "quartile_distribution"]
        }
    },
    {
        "id": "TC1-03",
        "prompt": "Compare income by education level for junior and mid-level employees only, controlling for age and gender.",
        "expected": {
            "comparison_attribute": "education_level",
            "target_variable": "income",
            "filters": {"job_level": [1, 2]},
            "control_variables": ["age", "gender"],
            "metrics": ["mean_gap", "median_gap"]
        }
    },
    {
        "id": "TC1-04",
        "prompt": "Is there a pay gap between employees who live far from work versus those who live nearby? Control for job level and experience.",
        "expected": {
            "comparison_attribute": "distance_from_home",
            "target_variable": "income",
            "filters": {},
            "control_variables": ["job_level", "experience_years"],
            "metrics": ["mean_gap", "median_gap"]
        }
    },
]

# ── REQUIRED JSON FIELDS ─────────────────────────────────────────────────────

REQUIRED_FIELDS = {
    "comparison_attribute": str,
    "target_variable": str,
    "filters": dict,
    "control_variables": list,
    "metrics": list
}

VALID_COLUMNS = {
    "gender", "income", "age", "education_level", "job_level",
    "experience_years", "distance_from_home", "absence", "child"
}

VALID_METRICS = {"mean_gap", "median_gap", "bonus_gap", "quartile_distribution"}

# ── VALIDATION ────────────────────────────────────────────────────────────────

def validate_config(config: dict) -> list:
    errors = []

    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in config:
            errors.append(f"Missing field: '{field}'")
        elif not isinstance(config[field], expected_type):
            errors.append(f"'{field}' should be {expected_type.__name__}, got {type(config[field]).__name__}")

    if "comparison_attribute" in config:
        if config["comparison_attribute"] not in VALID_COLUMNS:
            errors.append(f"'comparison_attribute' value '{config['comparison_attribute']}' not in dataset columns")

    if "control_variables" in config and isinstance(config["control_variables"], list):
        for v in config["control_variables"]:
            if v not in VALID_COLUMNS:
                errors.append(f"control_variable '{v}' not in dataset columns")

    if "metrics" in config and isinstance(config["metrics"], list):
        for m in config["metrics"]:
            if m not in VALID_METRICS:
                errors.append(f"metric '{m}' not in valid metrics list")

    return errors

def score_against_expected(actual: dict, expected: dict) -> dict:
    scores = {}

    scores["comparison_attribute"] = (
        actual.get("comparison_attribute") == expected.get("comparison_attribute")
    )
    scores["target_variable"] = (
        actual.get("target_variable") == expected.get("target_variable")
    )
    scores["filters"] = (
        actual.get("filters") == expected.get("filters")
    )
    scores["control_variables"] = (
        set(actual.get("control_variables", [])) == set(expected.get("control_variables", []))
    )
    scores["metrics"] = (
        set(actual.get("metrics", [])) == set(expected.get("metrics", []))
    )

    total = sum(scores.values())
    scores["total"] = f"{total}/{len(scores)-1}"
    return scores

# ── PIPELINE (minimal inline) ─────────────────────────────────────────────────

def run_pipeline_check(config: dict) -> str:
    try:
        import pandas as pd
        df = pd.read_csv(DATASET_PATH, sep=";")

        filters = config.get("filters", {})
        attr = config["comparison_attribute"]

        for col, values in filters.items():
            if isinstance(values, list):
                df = df[df[col].isin(values)]
            else:
                df = df[df[col] == values]

        if len(df) == 0:
            return "PIPELINE: 0 rows after filter — invalid config"

        groups = df[attr].value_counts().to_dict()
        return f"PIPELINE OK — {len(df)} rows, groups: {groups}"
    except Exception as e:
        return f"PIPELINE ERROR: {e}"

# ── OLLAMA CALL ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"ERROR: {e}"

# ── PARSE JSON FROM RESPONSE ──────────────────────────────────────────────────

def parse_json_response(raw: str) -> tuple:
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as e:
        return None, str(e)

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single custom prompt mode
        test_cases = [{"id": "CUSTOM", "prompt": " ".join(sys.argv[1:]), "expected": None}]
    else:
        test_cases = TEST_CASES

    results_summary = []

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {tc['id']}")
        print(f"PROMPT: {tc['prompt']}")
        print(f"{'='*60}")

        raw_response = call_ollama(tc["prompt"])
        print(f"\nRAW MODEL OUTPUT:\n{raw_response}")

        config, parse_error = parse_json_response(raw_response)

        if parse_error:
            print(f"\n[FAIL] JSON parse error: {parse_error}")
            results_summary.append({"id": tc["id"], "status": "PARSE_FAIL", "score": "0/5"})
            continue

        print(f"\nPARSED CONFIG:\n{json.dumps(config, indent=2)}")

        errors = validate_config(config)
        if errors:
            print(f"\n[VALIDATION ERRORS]")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"\n[VALIDATION] OK")

        if tc.get("expected"):
            scores = score_against_expected(config, tc["expected"])
            print(f"\n[SCORE vs EXPECTED]")
            for k, v in scores.items():
                if k != "total":
                    status = "OK" if v else "FAIL"
                    print(f"  {k}: {status}")
            print(f"  TOTAL: {scores['total']}")
            results_summary.append({"id": tc["id"], "status": "OK" if not errors else "INVALID", "score": scores["total"]})

        if RUN_PIPELINE and not parse_error and not errors:
            pipeline_result = run_pipeline_check(config)
            print(f"\n[{pipeline_result}]")

    print(f"\n{'='*60}")
    print(f"SUMMARY — MODEL: {MODEL}")
    print(f"{'='*60}")
    for r in results_summary:
        print(f"  {r['id']}: {r['status']} | score {r['score']}")
