"""
Pay Equity Analysis Pipeline
EU Directive 2023/970 metrics
Input:  config JSON (from LLM TC1 output)
Output: results JSON (input for LLM TC2/TC3)
"""

import json
import sys
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATASET_PATH = "salary_data_2009.csv"  # override via CLI: python pipeline.py config.json /path/to/data.csv

# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for col, values in filters.items():
        if col not in df.columns:
            raise ValueError(f"Filter column '{col}' not in dataset.")
        if isinstance(values, list):
            df = df[df[col].isin(values)]
        else:
            df = df[df[col] == values]
    return df

# ── METRICS ───────────────────────────────────────────────────────────────────

def mean_gap(df: pd.DataFrame, target: str, attr: str) -> dict:
    groups = df.groupby(attr)[target].mean()
    result = {str(k): round(float(v), 2) for k, v in groups.items()}
    vals = list(groups.values)
    if len(vals) == 2:
        result["gap_pct"] = round(float((vals[1] - vals[0]) / vals[1] * 100), 2)
        result["gap_abs"] = round(float(vals[1] - vals[0]), 2)
    return result

def median_gap(df: pd.DataFrame, target: str, attr: str) -> dict:
    groups = df.groupby(attr)[target].median()
    result = {str(k): round(float(v), 2) for k, v in groups.items()}
    vals = list(groups.values)
    if len(vals) == 2:
        result["gap_pct"] = round(float((vals[1] - vals[0]) / vals[1] * 100), 2)
        result["gap_abs"] = round(float(vals[1] - vals[0]), 2)
    return result

def quartile_distribution(df: pd.DataFrame, target: str, attr: str) -> dict:
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

def controlled_gap(df: pd.DataFrame, target: str, attr: str,
                   control_vars: list) -> dict:
    """
    Simple OLS residual-based controlled gap.
    Regresses target on control_vars, then computes mean gap on residuals by attr.
    """
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        available_controls = [c for c in control_vars if c in df.columns and c != attr]
        if not available_controls:
            return {"note": "No valid control variables available."}

        sub = df[[target, attr] + available_controls].dropna()

        # Encode attr if categorical
        if sub[attr].dtype == object:
            sub = pd.get_dummies(sub, columns=[attr], drop_first=False)
            attr_cols = [c for c in sub.columns if c.startswith(attr + "_")]
        else:
            attr_cols = [attr]

        X = sub[available_controls].copy()
        # Encode any remaining categoricals
        X = pd.get_dummies(X)
        y = sub[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)
        residuals = y - model.predict(X_scaled)

        # Add residuals back
        sub = sub.copy()
        sub["_residual"] = residuals.values

        # Recover original attr for grouping
        orig_attr = df.loc[sub.index, attr]
        gap_by_group = sub.groupby(orig_attr)["_residual"].mean().to_dict()

        vals = list(gap_by_group.values())
        controlled_gap_abs = round(vals[1] - vals[0], 2) if len(vals) == 2 else None

        return {
            "residual_mean_by_group": {str(k): round(float(v), 2) for k, v in gap_by_group.items()},
            "controlled_gap_abs": round(float(vals[1] - vals[0]), 2) if len(vals) == 2 else None,
            "controls_used": available_controls
        }
    except ImportError:
        return {"note": "scikit-learn not available — controlled gap skipped."}

# ── DIRECTIVE CHECK ───────────────────────────────────────────────────────────

def directive_check(metrics_result: dict) -> dict:
    """
    EU Directive 2023/970 Art. 9: gap >5% without justification
    requires joint pay assessment.
    """
    flags = []
    threshold = 5.0

    for metric_name in ["mean_gap", "median_gap"]:
        m = metrics_result.get(metric_name, {})
        gap = m.get("gap_pct")
        if gap is not None and abs(gap) > threshold:
            flags.append({
                "metric": metric_name,
                "gap_pct": gap,
                "threshold_pct": threshold,
                "action_required": "Joint pay assessment required (Art. 9, Directive 2023/970)"
            })

    return {
        "compliant": len(flags) == 0,
        "flags": flags
    }

# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_pipeline(config: dict, dataset_path: str = DATASET_PATH) -> dict:
    df = load_data(dataset_path)

    comparison_attr = config["comparison_attribute"]
    target_var = config.get("target_variable", "income")
    filters = config.get("filters", {})
    control_vars = config.get("control_variables", [])
    metrics_requested = config.get("metrics", ["mean_gap", "median_gap"])

    # Apply filters
    df_filtered = apply_filters(df, filters)
    n_total = len(df)
    n_filtered = len(df_filtered)

    if n_filtered == 0:
        return {"error": "No rows remaining after applying filters."}

    results = {
        "config": config,
        "dataset_info": {
            "total_rows": int(len(df)),
            "rows_after_filter": int(len(df_filtered)),
            "groups": {str(k): int(v) for k, v in df_filtered[comparison_attr].value_counts().items()}
        },
        "metrics": {}
    }

    # Compute requested metrics
    if "mean_gap" in metrics_requested:
        results["metrics"]["mean_gap"] = mean_gap(df_filtered, target_var, comparison_attr)

    if "median_gap" in metrics_requested:
        results["metrics"]["median_gap"] = median_gap(df_filtered, target_var, comparison_attr)

    if "quartile_distribution" in metrics_requested:
        results["metrics"]["quartile_distribution"] = quartile_distribution(
            df_filtered, target_var, comparison_attr)

    if "bonus_gap" in metrics_requested:
        results["metrics"]["bonus_gap"] = {"note": "No bonus column in dataset — skipped."}

    # Controlled gap (always computed if control_vars provided)
    if control_vars:
        results["metrics"]["controlled_gap"] = controlled_gap(
            df_filtered, target_var, comparison_attr, control_vars)

    # Directive compliance check
    results["directive_2023_970"] = directive_check(results["metrics"])

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        config = {
            "comparison_attribute": "gender",
            "target_variable": "income",
            "filters": {"job_level": [3, 4]},
            "control_variables": ["education_level", "experience_years"],
            "metrics": ["mean_gap", "median_gap", "quartile_distribution"]
        }
        print("No config provided — running default test config.", file=sys.stderr)
    else:
        with open(sys.argv[1]) as f:
            config = json.load(f)

    dataset = sys.argv[2] if len(sys.argv) > 2 else DATASET_PATH
    result = run_pipeline(config, dataset)
    print(json.dumps(result, indent=2, ensure_ascii=False))
