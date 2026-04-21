#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_SUBDIRS = ["data", "results", "shap", "scripts", "figures"]
TABULAR_EXTS = {".csv", ".xlsx", ".xls"}


def resolve_project_root(root: Optional[str] = None) -> Path:
    base = Path(root).resolve() if root else Path.cwd().resolve()
    if all((base / name).exists() for name in ["data", "results", "shap"]):
        return base
    for parent in [base, *base.parents]:
        if all((parent / name).exists() for name in ["data", "results", "shap"]):
            return parent
    return base


def ensure_figures_dir(project_root: Path) -> Path:
    out = project_root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def _scan_for_tables(paths: Iterable[Path]) -> list[Path]:
    out = []
    for base in paths:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in TABULAR_EXTS:
                out.append(p)
    return out


def find_tabular_candidates(project_root: Path) -> list[Path]:
    search_roots = [project_root / "data", project_root / "results", project_root]
    return _scan_for_tables(search_roots)


def find_single_dataset(project_root: Path, explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (project_root / explicit)
    candidates = []
    for p in find_tabular_candidates(project_root):
        try:
            sample = read_table(p).head(10)
        except Exception:
            continue
        numeric_cols = sum(pd.api.types.is_numeric_dtype(sample[c]) for c in sample.columns)
        if numeric_cols >= 3:
            candidates.append((numeric_cols, p))
    if not candidates:
        raise FileNotFoundError("No compatible dataset table found.")
    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return candidates[0][1]


def find_table_by_columns(project_root: Path, required_any: list[set[str]] | None = None, required_all: set[str] | None = None, explicit: Optional[str] = None) -> Path:
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (project_root / explicit)
    required_all = required_all or set()
    required_any = required_any or []
    candidates = []
    for p in find_tabular_candidates(project_root):
        try:
            df = read_table(p).head(20)
        except Exception:
            continue
        cols = {str(c).strip() for c in df.columns}
        if required_all and not required_all.issubset(cols):
            continue
        if required_any and not all(any(col in cols or any(str(c).startswith(col.rstrip("*")) for c in cols) for col in reqset) for reqset in required_any):
            continue
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError("No table matched the required schema hints.")
    candidates.sort(key=lambda x: len(str(x)))
    return candidates[0]


def find_shap_bundle(project_root: Path) -> dict:
    shap_dir = project_root / "shap"
    if not shap_dir.exists():
        shap_dir = project_root
    bundle = {"shap_values": None, "feature_json": None, "x_matrix": None}
    for p in sorted(shap_dir.rglob("*.npy")):
        if "shap" in p.name.lower():
            bundle["shap_values"] = p
            break
    for p in sorted(shap_dir.rglob("*.json")):
        if "feature" in p.name.lower():
            bundle["feature_json"] = p
            break
    x_candidates = []
    for p in sorted(shap_dir.rglob("*.csv")):
        if "x_explain" in p.name.lower() or "explain" in p.name.lower() or p.stem.lower().startswith("x_"):
            x_candidates.append(p)
    if x_candidates:
        x_candidates.sort(key=lambda p: ("subset" not in p.name.lower(), len(p.name)))
        bundle["x_matrix"] = x_candidates[0]
    return bundle


def load_feature_names(feature_json: Path) -> list[str]:
    meta = json.loads(feature_json.read_text(encoding="utf-8"))
    if isinstance(meta, dict):
        return list(meta.get("features", []))
    return list(meta)


def ensure_2d_shap(shap_values: np.ndarray) -> np.ndarray:
    sv = np.asarray(shap_values)
    if sv.ndim == 2:
        return sv
    if sv.ndim == 3:
        return sv[1] if sv.shape[0] == 2 else sv[0]
    raise ValueError(f"Unsupported SHAP shape: {sv.shape}")


def save_png_svg(fig, out_dir: Path, stem: str, dpi: int = 300) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.05)
    return png, svg
