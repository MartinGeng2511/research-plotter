#!/usr/bin/env python3
import json
import sys
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except Exception as e:
    print(json.dumps({"error": f"missing dependency: {e}"}, ensure_ascii=False, indent=2))
    sys.exit(1)

TABULAR_EXTS = {".csv", ".xlsx", ".xls"}
SCRIPT_EXTS = {".py"}


def safe_read_table(path: Path):
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, nrows=20)
        return pd.read_excel(path, nrows=20)
    except Exception:
        return None


def classify_table(df):
    cols = {str(c).strip() for c in df.columns}
    numeric_count = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)

    roles = []
    if numeric_count >= 3:
        roles.append("tabular_dataset_candidate")
    if {"Model", "R2"}.issubset(cols) or {"Model", "RMSE"}.issubset(cols):
        roles.append("cv_or_bootstrap_metrics_candidate")
    if "Standardized_Residual" in cols or "Residual" in cols:
        roles.append("residual_diagnostics_candidate")
    if "Actual" in cols and any(str(c).startswith("Pred") for c in cols):
        roles.append("prediction_results_candidate")
    if any(str(c).startswith("Q5") for c in cols) and any(str(c).startswith("Q95") for c in cols):
        roles.append("prediction_interval_candidate")
    return roles


def detect(root: Path):
    out = {
        "root": str(root.resolve()),
        "project_layout": {k: (root / k).exists() for k in ["data", "results", "shap", "scripts", "figures"]},
        "files": [],
        "input_types_found": set(),
        "figure_families_feasible": set(),
    }

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rec = {"path": str(path.relative_to(root)), "suffix": path.suffix.lower(), "roles": []}

        if path.suffix.lower() in TABULAR_EXTS:
            df = safe_read_table(path)
            if df is not None:
                rec["columns"] = [str(c) for c in df.columns[:12]]
                rec["roles"].extend(classify_table(df))
        elif path.suffix.lower() == ".npy" and "shap" in path.name.lower():
            rec["roles"].append("shap_values_candidate")
        elif path.suffix.lower() == ".json" and "feature" in path.name.lower():
            rec["roles"].append("feature_metadata_candidate")
        elif path.suffix.lower() in SCRIPT_EXTS:
            rec["roles"].append("plotting_script_candidate")

        out["files"].append(rec)
        for role in rec["roles"]:
            out["input_types_found"].add(role)

    found = out["input_types_found"]
    if "tabular_dataset_candidate" in found:
        out["figure_families_feasible"].add("raw_data_figures")
    if any(x in found for x in ["cv_or_bootstrap_metrics_candidate", "residual_diagnostics_candidate", "prediction_interval_candidate", "prediction_results_candidate"]):
        out["figure_families_feasible"].add("model_evaluation_figures")
    if "plotting_script_candidate" in found:
        out["figure_families_feasible"].add("existing_script_execution")
        out["figure_families_feasible"].add("flowchart_figures")
    if {"shap_values_candidate", "feature_metadata_candidate"}.issubset(found):
        out["figure_families_feasible"].add("shap_figures")

    out["input_types_found"] = sorted(out["input_types_found"])
    out["figure_families_feasible"] = sorted(out["figure_families_feasible"])
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(json.dumps(detect(root), ensure_ascii=False, indent=2))
