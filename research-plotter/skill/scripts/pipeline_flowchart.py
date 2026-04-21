#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""High-fidelity XGB-Optuna style pipeline flowchart."""

import argparse
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from plot_utils import ensure_figures_dir, resolve_project_root


def setup_times():
    paths = [
        r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\Times.ttf", r"C:\Windows\Fonts\TIMES.TTF",
        r"C:\Windows\Fonts\timesbd.ttf", r"C:\Windows\Fonts\timesi.ttf", r"C:\Windows\Fonts\timesbi.ttf",
    ]
    for fp in paths:
        if os.path.exists(fp):
            fe = fm.FontEntry(fname=fp, name="Times New Roman")
            fm.fontManager.ttflist.insert(0, fe)
            mpl.rcParams["font.family"] = "Times New Roman"
            mpl.rcParams["font.serif"] = ["Times New Roman"]
            break
    else:
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]


setup_times()
mpl.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 1.0, "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight"})
C = {"data": "#2166AC", "process": "#F4A582", "model": "#1A9641", "output": "#762A83", "arrow": "#444444", "text": "#1a1a1a", "light": "#f0f0f0", "border": "#666666"}

DEFAULT_SPEC = {
    "title": "XGB-Optuna: Automated Hyperparameter Optimization Pipeline",
    "footer": "FRP Bond Strength Prediction",
    "dataset_label": "Dataset.xlsx  (n = 292)",
    "train_label": "Training Set  (n = 230)",
    "test_label": "Test Set  (n = 58)",
    "final_label": "XGB-Optuna  (Final Model)",
    "metric_label": r"$R^2$ = 0.932  ·  RMSE = 1.304",
}


def box(ax, x, y, w, h, label, fc=C["light"], ec=C["border"], lw=0.9, radius=0.12, fontsize=9.5, sublabel="", subfontsize=8.0):
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad={radius}", facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.02, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", fontfamily="Times New Roman", color=C["text"], zorder=4)
        ax.text(x, y - 0.175, sublabel, ha="center", va="center", fontsize=subfontsize, fontfamily="Times New Roman", color=C["text"], zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", fontfamily="Times New Roman", color=("white" if fc in {C['data'], C['process'], C['model'], C['output']} else C['text']), zorder=4)


def arrow(ax, x1, y1, x2, y2, label="", color=C["arrow"], lw=1.1):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=11, shrinkA=5, shrinkB=5), zorder=5)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.04, my, label, ha="left", va="center", fontsize=7.5, fontfamily="Times New Roman", color=C["text"], style="italic", zorder=6)


def build_flowchart(spec: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(13.5, 10.5)); ax.set_xlim(0, 13.5); ax.set_ylim(0, 10.5); ax.axis("off"); fig.patch.set_facecolor("white")
    box(ax, 6.75, 9.70, 3.20, 0.65, spec["dataset_label"], fc=C["data"], ec=C["data"], lw=1.2, radius=0.10)
    box(ax, 3.20, 8.25, 2.40, 0.60, spec["train_label"], fc=C["light"], ec=C["border"], radius=0.10)
    box(ax, 10.30, 8.25, 2.40, 0.60, spec["test_label"], fc=C["light"], ec=C["border"], radius=0.10)
    arrow(ax, 5.15, 9.38, 3.20, 8.55); arrow(ax, 8.35, 9.38, 10.30, 8.55); ax.text(6.75, 8.95, "80 / 20 split", ha="center", va="center", fontsize=7.5, fontfamily="Times New Roman", color=C["text"], style="italic")
    box(ax, 3.20, 7.10, 2.70, 0.65, "Feature Engineering", fc=C["process"], ec=C["process"], lw=1.1, radius=0.10)
    box(ax, 3.20, 6.20, 2.70, 0.65, "Log / Sqrt Transforms", fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.5)
    box(ax, 3.20, 5.30, 2.70, 0.65, "Feature Interactions", fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.5)
    box(ax, 3.20, 4.40, 2.70, 0.65, "SMOTE  (n scanty samples)", fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.5)
    arrow(ax, 3.20, 7.88, 3.20, 7.42); arrow(ax, 3.20, 6.48, 3.20, 5.63); arrow(ax, 3.20, 5.63, 3.20, 4.77)
    box(ax, 9.50, 7.10, 3.20, 0.65, "Optuna Tuning  (60 trials)", fc=C["model"], ec=C["model"], lw=1.1, radius=0.10)
    box(ax, 9.50, 6.20, 3.20, 0.65, "10-Fold CV  ·  R² objective", fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.5)
    box(ax, 9.50, 5.30, 3.20, 0.65, "n_estimators · max_depth\nlearning_rate · subsample …", fc=C["light"], ec=C["border"], radius=0.10, fontsize=7.8)
    arrow(ax, 9.50, 7.88, 9.50, 7.42); arrow(ax, 9.50, 6.48, 9.50, 5.63)
    box(ax, 6.50, 4.40, 2.80, 0.65, "Best Hyperparameters", fc=C["model"], ec=C["model"], lw=1.1, radius=0.10)
    ax.annotate("", xy=(6.50, 4.72), xytext=(3.20, 4.07), arrowprops=dict(arrowstyle="-|>", color=C["border"], lw=0.9, linestyle="dashed", mutation_scale=9), zorder=5)
    ax.annotate("", xy=(6.50, 4.72), xytext=(9.50, 4.97), arrowprops=dict(arrowstyle="-|>", color=C["border"], lw=0.9, linestyle="dashed", mutation_scale=9), zorder=5)
    box(ax, 6.50, 3.30, 3.10, 0.65, "Multi-Seed Ensemble  (5 seeds)", fc=C["model"], ec=C["model"], lw=1.1, radius=0.10)
    box(ax, 6.50, 2.40, 3.10, 0.65, "Seed = 42, 123, 456, 789, 1024", fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.5)
    arrow(ax, 6.50, 4.07, 6.50, 3.63); arrow(ax, 6.50, 2.73, 6.50, 2.40)
    box(ax, 6.50, 1.50, 3.10, 0.70, spec["final_label"], fc=C["output"], ec=C["output"], lw=1.3, radius=0.12, fontsize=10.5); arrow(ax, 6.50, 2.07, 6.50, 1.85)
    box(ax, 2.40, 1.50, 2.20, 0.60, "Predictions", fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.8)
    box(ax, 6.50, 0.55, 2.20, 0.55, spec["metric_label"], fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.0)
    box(ax, 10.60, 1.50, 2.20, 0.60, "Feature Importance", fc=C["light"], ec=C["border"], radius=0.10, fontsize=8.8)
    arrow(ax, 5.00, 1.50, 2.40, 1.50); arrow(ax, 8.00, 1.50, 10.60, 1.50); arrow(ax, 6.50, 1.15, 6.50, 0.83, lw=0.8)
    legend_items = [mpatches.Patch(facecolor=C["data"], edgecolor=C["data"], label="Data"), mpatches.Patch(facecolor=C["process"], edgecolor=C["process"], label="Preprocessing"), mpatches.Patch(facecolor=C["model"], edgecolor=C["model"], label="Model / Tuning"), mpatches.Patch(facecolor=C["output"], edgecolor=C["output"], label="Output")]
    ax.legend(handles=legend_items, loc="lower right", bbox_to_anchor=(0.99, 0.01), frameon=True, edgecolor="black", fontsize=9, prop={"family": "Times New Roman"}, borderpad=0.5, labelspacing=0.4, facecolor="white", framealpha=0.95)
    ax.text(6.75, 10.22, spec["title"], ha="center", va="center", fontsize=13, fontweight="bold", fontfamily="Times New Roman", color=C["text"])
    ax.text(6.75, 0.12, spec["footer"], ha="center", va="center", fontsize=8.5, style="italic", fontfamily="Times New Roman", color="#555555")
    png = out_dir / "Fig0_Pipeline_Flowchart.png"; svg = out_dir / "Fig0_Pipeline_Flowchart.svg"
    fig.savefig(png, format="png", bbox_inches="tight", dpi=300); fig.savefig(svg, format="svg", bbox_inches="tight", dpi=300); plt.close(fig)
    return png, svg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--spec-json")
    args = ap.parse_args()
    project_root = resolve_project_root(args.project_root)
    out_dir = ensure_figures_dir(project_root)
    spec = DEFAULT_SPEC.copy()
    if args.spec_json:
        spec.update(json.loads(Path(args.spec_json).read_text(encoding="utf-8")))
    png, svg = build_flowchart(spec, out_dir)
    print(json.dumps({"output_png": str(png), "output_svg": str(svg)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
