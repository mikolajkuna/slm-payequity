"""
TC2 Narrator
Sends pay equity pipeline output to local Ollama and returns narrative interpretation.
Usage:
    python tc2_narrator.py                        # runs default test config
    python tc2_narrator.py config.json            # runs with custom config JSON
    python tc2_narrator.py config.json data.csv   # custom config + custom dataset path
"""

import json
import sys
import urllib.request
import urllib.error

# ── SETTINGS ─────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b"
DATASET_PATH = "salary_data_2009.csv"

SYSTEM_PROMPT = """You are a pay equity compliance analyst. You will receive a JSON object containing the results of a pay equity analysis conducted under EU Directive 2023/970. Your task is to write a clear, professional narrative interpretation of the results in English.

Your report must cover:
1. The scope of the analysis (which group was compared, any filters applied)
2. Key findings: mean gap, median gap, and quartile distribution if present
3. The controlled gap result if present, and what it means
4. Directive 2023/970 compliance status and required actions if any

Rules:
- Be factual and precise — cite exact numbers from the JSON
- Do not invent data not present in the JSON
- Do not use bullet points — write in paragraphs
- Length: 150-250 words
- Currency: all monetary values are in PLN (Polish złoty), not euros"""

# ── PIPELINE (inline to keep single-file) ────────────────────────────────────

def run_pipeline(config: dict, dataset_path: str = DATASET_PATH) -> dict:
    import pandas as pd

    def apply_filters(df, filters):
        for col, values in filters.items():
            if isinstance(values, list):
                df = df[df[col].isin(values)]
            else:
                df = df[df[col] == values]
        return df

    def mean_gap(df, target, attr):
        groups = df.groupby(attr)[target].mean()
        result = {str(k): round(float(v), 2) for k, v in groups.items()}
        vals = list(groups.values)
        if len(vals) == 2:
            result["gap_pct"] = round(float((vals[1] - vals[0]) / vals[1] * 100), 2)
            result["gap_abs"] = round(float(vals[1] - vals[0]), 2)
        return result

    def median_gap(df, target, attr):
        groups = df.groupby(attr)[target].median()
        result = {str(k): round(float(v), 2) for k, v in groups.items()}
        vals = list(groups.values)
        if len(vals) == 2:
            result["gap_pct"] = round(float((vals[1] - vals[0]) / vals[1] * 100), 2)
            result["gap_abs"] = round(float(vals[1] - vals[0]), 2)
        return result

    def quartile_distribution(df, target, attr):
        result = {}
        for group, gdf in df.groupby(attr):
            q = gdf[target].quantile([0.25, 0.5, 0.75])
            result[str(group)] = {
                "Q1": round(float(q[0.25]), 2),
                "Q2_median": round(float(q[0.5]), 2),
                "Q3": round(float(q[0.75]), 2),
                "max": round(float(gdf[target].max()), 2),
                "min": round(float(gdf[target].min()), 2),
                "n": int(len(gdf))
            }
        return result

    def controlled_gap(df, target, attr, control_vars):
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            available = [c for c in control_vars if c in df.columns and c != attr]
            if not available:
                return {"note": "No valid control variables."}
            sub = df[[target, attr] + available].dropna().copy()
            X = pd.get_dummies(sub[available])
            y = sub[target]
            scaler = StandardScaler()
            model = LinearRegression()
            model.fit(scaler.fit_transform(X), y)
            sub["_residual"] = y.values - model.predict(scaler.transform(X))
            gap_by_group = sub.groupby(attr)["_residual"].mean()
            vals = list(gap_by_group.values)
            return {
                "residual_mean_by_group": {str(k): round(float(v), 2) for k, v in gap_by_group.items()},
                "controlled_gap_abs": round(float(vals[1] - vals[0]), 2) if len(vals) == 2 else None,
                "controls_used": available
            }
        except ImportError:
            return {"note": "scikit-learn not available."}

    def directive_check(metrics_result):
        flags = []
        for metric_name in ["mean_gap", "median_gap"]:
            m = metrics_result.get(metric_name, {})
            gap = m.get("gap_pct")
            if gap is not None and abs(gap) > 5.0:
                flags.append({
                    "metric": metric_name,
                    "gap_pct": float(gap),
                    "threshold_pct": 5.0,
                    "action_required": "Joint pay assessment required (Art. 9, Directive 2023/970)"
                })
        return {"compliant": len(flags) == 0, "flags": flags}

    df = pd.read_csv(dataset_path, sep=";")
    attr = config["comparison_attribute"]
    target = config.get("target_variable", "income")
    filters = config.get("filters", {})
    control_vars = config.get("control_variables", [])
    metrics_requested = config.get("metrics", ["mean_gap", "median_gap"])

    df_f = apply_filters(df, filters)
    if len(df_f) == 0:
        return {"error": "No rows after filters."}

    results = {
        "config": config,
        "dataset_info": {
            "total_rows": int(len(df)),
            "rows_after_filter": int(len(df_f)),
            "groups": {str(k): int(v) for k, v in df_f[attr].value_counts().items()}
        },
        "metrics": {}
    }

    if "mean_gap" in metrics_requested:
        results["metrics"]["mean_gap"] = mean_gap(df_f, target, attr)
    if "median_gap" in metrics_requested:
        results["metrics"]["median_gap"] = median_gap(df_f, target, attr)
    if "quartile_distribution" in metrics_requested:
        results["metrics"]["quartile_distribution"] = quartile_distribution(df_f, target, attr)
    if control_vars:
        results["metrics"]["controlled_gap"] = controlled_gap(df_f, target, attr, control_vars)

    results["directive_2023_970"] = directive_check(results["metrics"])
    return results

# ── OLLAMA CALL ───────────────────────────────────────────────────────────────

def call_ollama(analysis_json: dict) -> str:
    prompt = (
        "Here are the pay equity analysis results. Write your narrative report:\n\n"
        + json.dumps(analysis_json, indent=2)
    )

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
        return f"ERROR: Could not reach Ollama at {OLLAMA_URL}\nDetails: {e}"

# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load config
    if len(sys.argv) < 2:
        config = {
            "comparison_attribute": "gender",
            "target_variable": "income",
            "filters": {"job_level": [3, 4]},
            "control_variables": ["education_level", "experience_years"],
            "metrics": ["mean_gap", "median_gap", "quartile_distribution"]
        }
        print("[INFO] No config provided — using default test config.\n", file=sys.stderr)
    else:
        with open(sys.argv[1]) as f:
            config = json.load(f)

    dataset = sys.argv[2] if len(sys.argv) > 2 else DATASET_PATH

    # Run pipeline
    print("[1/2] Running pay equity pipeline...", file=sys.stderr)
    analysis = run_pipeline(config, dataset)
    print("[INFO] Pipeline output:", file=sys.stderr)
    print(json.dumps(analysis, indent=2), file=sys.stderr)

    # Call Ollama TC2
    print("\n[2/2] Sending to Ollama for TC2 narrative...\n", file=sys.stderr)
    narrative = call_ollama(analysis)

    print("=" * 60)
    print("TC2 NARRATIVE REPORT")
    print("=" * 60)
    print(narrative)
    print("=" * 60)
