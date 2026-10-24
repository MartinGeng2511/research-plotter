#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""Train/Test parity JointGrid figures matching the reference
FRP_Model_Visualization.py (CO2 JointGrid) style.

Outputs per model:  JointGrid_<model>.png/svg
Combined panel:     JointGrid_AllModels.png/svg
"""

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from plot_utils import (
    ensure_figures_dir,
    find_tabular_candidates,
    read_table,
    resolve_project_root,
    setup_times_new_roman,
)

setup_times_new_roman()

# ===================== Style (identical to the reference script) =====================
mpl.rcParams["axes.linewidth"] = 1.2
mpl.rcParams["savefig.dpi"] = 600

sns.set_style("white")
sns.set_context("talk")

PALETTE = {
    "Train": "#4C72B0",      # blue
    "Test": "#DD8452",       # orange

    "Boost": "#C44E52",      # red
    "XGB": "#55A868",        # green
    "CAT": "#8172B3",        # purple
    "LGB": "#64B5CD",        # cyan-blue
}
REF_COLOR = "black"

PREFERRED_MODEL_ORDER = ["Boost-Avg", "XGB-Optuna", "CAT-Optuna", "LGB-Optuna"]


# ===================== Data loading (same rules as the reference) =====================
def read_pred_csv(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_err = None

    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            df.columns = [str(c).strip() for c in df.columns]

            if "Actual" not in df.columns:
                raise ValueError(f"Need column: Actual. Got: {df.columns.tolist()}")

            df["Actual"] = pd.to_numeric(df["Actual"], errors="coerce")
            df = df.dropna(subset=["Actual"])

            for col in df.columns:
                if col.startswith("Pred"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if df.empty:
                raise ValueError(f"{csv_path.name} is empty after parsing.")

            return df

        except Exception as e:
            last_err = e

    raise ValueError(f"Cannot read file: {csv_path}, reason: {last_err}")


def r2_score(y_true, y_pred):
    """Same definition as sklearn.metrics.r2_score."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def _resolve(project_root: Path, explicit: str | None) -> Path | None:
    if not explicit:
        return None
    p = Path(explicit)
    return p if p.is_absolute() else (project_root / explicit)


def _has_prediction_schema(path: Path) -> bool:
    try:
        df = read_table(path).head(20)
    except Exception:
        return False
    cols = {str(c).strip() for c in df.columns}
    return "Actual" in cols and any(str(c).startswith("Pred") for c in cols)


def find_train_test_files(project_root: Path, explicit_train=None, explicit_test=None):
    """Locate the train/test prediction tables. Filenames containing
    'train'/'test' win; a single unlabelled prediction file is used as test."""
    train_p = _resolve(project_root, explicit_train)
    test_p = _resolve(project_root, explicit_test)
    if train_p and test_p:
        return train_p, test_p
    pred_files = [p for p in find_tabular_candidates(project_root) if _has_prediction_schema(p)]
    pred_files.sort(key=lambda p: len(str(p)))
    if train_p is None:
        named = [p for p in pred_files if "train" in p.stem.lower()]
        train_p = named[0] if named else None
    if test_p is None:
        named = [p for p in pred_files if "test" in p.stem.lower() and p != train_p]
        if named:
            test_p = named[0]
        else:
            rest = [p for p in pred_files if p != train_p]
            test_p = rest[0] if rest else None
    return train_p, test_p


def detect_models(test_df: pd.DataFrame) -> list[str]:
    pred_cols = [c for c in test_df.columns if str(c).startswith("Pred")]
    models = [str(c)[len("Pred_"):] if str(c).startswith("Pred_") else str(c)[len("Pred"):] for c in pred_cols]
    ordered = [m for m in PREFERRED_MODEL_ORDER if m in models]
    ordered += [m for m in models if m not in ordered]
    return ordered


def set_black_spines(ax):
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.0)


def force_times_ticks(ax, fontsize=12):
    ax.tick_params(axis="both", labelsize=fontsize)
    for lab in ax.get_xticklabels(which="both"):
        lab.set_fontname("Times New Roman")
    for lab in ax.get_yticklabels(which="both"):
        lab.set_fontname("Times New Roman")


# ===================== Single-model JointGrid =====================
def make_joint_plot(train_df, test_df, model_name, out_dir, target_name, x_label, y_label):
    """CO2-style JointGrid for one model: Train + Test on the same figure."""
    pred_col = f"Pred_{model_name}"

    if pred_col not in test_df.columns:
        print(f"[Skip] {model_name}: missing column {pred_col} in test file")
        return None
    has_train = train_df is not None and pred_col in train_df.columns

    train_sub = train_df[["Actual", pred_col]].dropna().copy() if has_train else pd.DataFrame(columns=["True", "Predicted"])
    train_sub.columns = ["True", "Predicted"]
    test_sub = test_df[["Actual", pred_col]].dropna().copy()
    test_sub.columns = ["True", "Predicted"]

    train_sub["Data Set"] = "Train"
    test_sub["Data Set"] = "Test"
    data = pd.concat([train_sub, test_sub], ignore_index=True)

    r2_train = r2_score(train_sub["True"], train_sub["Predicted"]) if len(train_sub) >= 2 else np.nan
    r2_test = r2_score(test_sub["True"], test_sub["Predicted"]) if len(test_sub) >= 2 else np.nan

    g = sns.JointGrid(data=data, x="True", y="Predicted", height=10, ratio=5, space=0)

    sns.scatterplot(
        data=train_sub, x="True", y="Predicted",
        color=PALETTE["Train"], alpha=0.40, s=85,
        ax=g.ax_joint, edgecolor=None, legend=False,
    )
    sns.scatterplot(
        data=test_sub, x="True", y="Predicted",
        color=PALETTE["Test"], alpha=0.40, s=85,
        ax=g.ax_joint, edgecolor=None, legend=False,
    )

    if len(train_sub) >= 3:
        sns.regplot(
            data=train_sub, x="True", y="Predicted",
            scatter=False, ax=g.ax_joint, color=PALETTE["Train"],
            ci=95, line_kws={"lw": 2.8, "alpha": 0.95},
        )
    if len(test_sub) >= 3:
        sns.regplot(
            data=test_sub, x="True", y="Predicted",
            scatter=False, ax=g.ax_joint, color=PALETTE["Test"],
            ci=95, line_kws={"lw": 2.8, "alpha": 0.95},
        )

    lim_min = float(min(data["True"].min(), data["Predicted"].min()))
    lim_max = float(max(data["True"].max(), data["Predicted"].max()))
    pad = (lim_max - lim_min) * 0.04 if lim_max > lim_min else 1.0
    lim_min_plot = lim_min - pad
    lim_max_plot = lim_max + pad

    g.ax_joint.plot(
        [lim_min_plot, lim_max_plot], [lim_min_plot, lim_max_plot],
        linestyle="--", color=REF_COLOR, lw=1.3,
    )
    g.ax_joint.set_xlim(lim_min_plot, lim_max_plot)
    g.ax_joint.set_ylim(lim_min_plot, lim_max_plot)
    g.ax_joint.set_aspect("equal", adjustable="box")

    sns.histplot(
        data=data, x="True", hue="Data Set",
        palette={"Train": PALETTE["Train"], "Test": PALETTE["Test"]},
        element="bars", multiple="stack", alpha=0.45, ax=g.ax_marg_x,
        bins=30, legend=False,
    )
    sns.histplot(
        data=data, y="Predicted", hue="Data Set",
        palette={"Train": PALETTE["Train"], "Test": PALETTE["Test"]},
        element="bars", multiple="stack", alpha=0.45, ax=g.ax_marg_y,
        bins=30, legend=False,
    )

    g.ax_joint.set_xlabel(x_label, fontfamily="Times New Roman", fontsize=15)
    g.ax_joint.set_ylabel(y_label, fontfamily="Times New Roman", fontsize=15)

    text_box_style = dict(boxstyle="round,pad=0.28", edgecolor="black", facecolor="white")
    g.ax_joint.text(
        0.93, 0.18, rf"Target = {target_name}",
        transform=g.ax_joint.transAxes, fontsize=12, fontfamily="Times New Roman",
        va="bottom", ha="right", bbox=text_box_style,
    )
    g.ax_joint.text(
        0.93, 0.12,
        rf"Train $R^2$ = {r2_train:.3f}" if np.isfinite(r2_train) else r"Train $R^2$ = n/a",
        transform=g.ax_joint.transAxes, fontsize=12, fontfamily="Times New Roman",
        va="bottom", ha="right", bbox=text_box_style,
    )
    g.ax_joint.text(
        0.93, 0.06,
        rf"Test  $R^2$ = {r2_test:.3f}" if np.isfinite(r2_test) else r"Test  $R^2$ = n/a",
        transform=g.ax_joint.transAxes, fontsize=12, fontfamily="Times New Roman",
        va="bottom", ha="right", bbox=text_box_style,
    )

    for ax in [g.ax_joint, g.ax_marg_x, g.ax_marg_y]:
        set_black_spines(ax)
        force_times_ticks(ax, fontsize=12)

    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=PALETTE["Train"], alpha=0.55, markersize=9, label="Train"),
        Line2D([0], [0], marker="o", linestyle="None", color=PALETTE["Test"], alpha=0.55, markersize=9, label="Test"),
    ]
    if len(train_sub):
        handles.append(Line2D([0], [0], color=PALETTE["Train"], lw=2.8, label="Train Regression Line"))
    handles.append(Line2D([0], [0], color=PALETTE["Test"], lw=2.8, label="Test Regression Line"))
    handles.append(Line2D([0], [0], color=REF_COLOR, linestyle="--", lw=1.3, label="x=y"))

    g.ax_joint.legend(
        handles=handles, loc="upper left", frameon=True, edgecolor="black",
        fontsize=10.5, prop={"family": "Times New Roman"},
        borderpad=0.5, labelspacing=0.5, handlelength=2.2,
    )

    safe = model_name.replace("-", "_")
    svg_path = out_dir / f"JointGrid_{safe}.svg"
    png_path = out_dir / f"JointGrid_{safe}.png"
    g.figure.savefig(svg_path, format="svg", bbox_inches="tight")
    g.figure.savefig(png_path, format="png", bbox_inches="tight")
    plt.close(g.figure)
    return png_path, svg_path


# ===================== All-models comparison panel =====================
def make_combined_joint_plot(train_df, test_df, models, out_dir, x_label, y_label):
    """All models in one figure, 2-column grid."""
    ncols = 2
    nrows = max(1, int(math.ceil(len(models) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = np.array(axes).reshape(-1)
    fig.subplots_adjust(wspace=0.25, hspace=0.28, top=0.92, bottom=0.08)

    for i, model in enumerate(models):
        ax = axes[i]
        model_key = model.split("-")[0]
        model_color = PALETTE.get(model_key, "#4C72B0")
        pred_col = f"Pred_{model}"

        if pred_col not in test_df.columns:
            ax.set_visible(False)
            continue

        train_sub = train_df[["Actual", pred_col]].dropna().copy() if train_df is not None and pred_col in train_df.columns else pd.DataFrame(columns=["True", "Predicted"])
        train_sub.columns = ["True", "Predicted"]
        test_sub = test_df[["Actual", pred_col]].dropna().copy()
        test_sub.columns = ["True", "Predicted"]

        if len(train_sub) > 0:
            ax.scatter(
                train_sub["True"], train_sub["Predicted"],
                s=50, color=PALETTE["Train"], alpha=0.35,
                edgecolor="white", linewidth=0.3, label="Train", zorder=2,
            )
            if len(train_sub) >= 3:
                sns.regplot(
                    x=train_sub["True"], y=train_sub["Predicted"],
                    scatter=False, ax=ax, color=PALETTE["Train"], ci=95,
                    line_kws={"lw": 2.0, "alpha": 0.80},
                )

        ax.scatter(
            test_sub["True"], test_sub["Predicted"],
            s=70, color=model_color, alpha=0.55,
            edgecolor="white", linewidth=0.4, label="Test", zorder=3,
        )
        if len(test_sub) >= 3:
            sns.regplot(
                x=test_sub["True"], y=test_sub["Predicted"],
                scatter=False, ax=ax, color=model_color, ci=95,
                line_kws={"lw": 2.5, "alpha": 0.95},
            )

        all_vals = pd.concat([train_sub["True"], test_sub["True"], train_sub["Predicted"], test_sub["Predicted"]])
        lim_min = float(all_vals.min())
        lim_max = float(all_vals.max())
        pad = (lim_max - lim_min) * 0.04 if lim_max > lim_min else 1.0
        line_range = [lim_min - pad, lim_max + pad]
        ax.plot(line_range, line_range, "k--", lw=1.2, alpha=0.50, label="x=y")
        ax.set_xlim(line_range)
        ax.set_ylim(line_range)
        ax.set_aspect("equal", adjustable="box")

        r2_test = r2_score(test_sub["True"], test_sub["Predicted"]) if len(test_sub) >= 2 else np.nan
        text_style = dict(boxstyle="round,pad=0.25", edgecolor="black", facecolor="white")
        ax.text(
            0.93, 0.08,
            rf"Test $R^2$ = {r2_test:.4f}" if np.isfinite(r2_test) else r"Test $R^2$ = n/a",
            transform=ax.transAxes, fontsize=11, fontfamily="Times New Roman",
            va="bottom", ha="right", bbox=text_style,
        )

        ax.set_xlabel(x_label, fontfamily="Times New Roman", fontsize=12, labelpad=4)
        ax.set_ylabel(y_label, fontfamily="Times New Roman", fontsize=12, labelpad=4)
        ax.legend(
            loc="upper left", frameon=True, edgecolor="black",
            fontsize=10, prop={"family": "Times New Roman"},
            borderpad=0.4, labelspacing=0.4, handlelength=1.8,
        )
        set_black_spines(ax)
        force_times_ticks(ax, fontsize=11)
        ax.set_title(model, fontsize=13, fontweight="bold",
                     fontfamily="Times New Roman", pad=8, color=model_color)

    for j in range(len(models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Visual Performance Assessment — All Models",
        y=0.98, fontsize=15, fontweight="bold",
        fontfamily="Times New Roman", color="#222222",
    )

    svg_path = out_dir / "JointGrid_AllModels.svg"
    png_path = out_dir / "JointGrid_AllModels.png"
    fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


# ===================== Main =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--train-file", help="train prediction CSV (Actual + Pred_<model> columns)")
    ap.add_argument("--test-file", help="test prediction CSV (Actual + Pred_<model> columns)")
    ap.add_argument("--target-name", default="Bond Strength")
    ap.add_argument("--x-label", default="Experimental bond stress (MPa)")
    ap.add_argument("--y-label", default="Predicted bond stress (MPa)")
    args = ap.parse_args()

    project_root = resolve_project_root(args.project_root)
    out_dir = ensure_figures_dir(project_root) / "model_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_p, test_p = find_train_test_files(project_root, args.train_file, args.test_file)
    out = {"train_source": str(train_p) if train_p else None,
           "test_source": str(test_p) if test_p else None}

    if test_p is None:
        out["error"] = ("missing input type: test prediction table with 'Actual' and 'Pred_<model>' columns"
                        + ("" if args.train_file else " (and a train prediction table for Train points)"))
        print(pd.Series(out, dtype=object).to_json(force_ascii=False, indent=2))
        return

    train_df = read_pred_csv(train_p) if train_p else None
    if train_df is None:
        print("[Warn] no train prediction file found; JointGrids will show Test only")
    test_df = read_pred_csv(test_p)

    models = detect_models(test_df)
    out["models"] = models

    singles = []
    for model in models:
        paths = make_joint_plot(train_df, test_df, model, out_dir, args.target_name, args.x_label, args.y_label)
        if paths:
            singles.append([str(p) for p in paths])
    out["joint_grids"] = singles

    if models:
        png, svg = make_combined_joint_plot(train_df, test_df, models, out_dir, args.x_label, args.y_label)
        out["all_models"] = [str(png), str(svg)]

    print(pd.Series(out, dtype=object).to_json(force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
