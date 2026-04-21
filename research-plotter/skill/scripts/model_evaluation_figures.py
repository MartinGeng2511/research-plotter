#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""High-fidelity model evaluation figures preserving the reference SCI style."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from plot_utils import ensure_figures_dir, find_table_by_columns, read_table, resolve_project_root

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.50,
    "grid.color": "#cccccc",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

C = {
    "blue": "#2166AC", "light_blue": "#92C5DE", "sky": "#D1E5F0", "red": "#D6604D",
    "light_red": "#F4A582", "green": "#1A9641", "light_green": "#A6D96A", "orange": "#E08214",
    "purple": "#762A83", "grey": "#636363", "light_grey": "#BABABA", "bg": "#F8F8F8",
    "white": "#FFFFFF", "dark": "#4D4D4D",
}
MODEL_COLORS = {"LGB-Optuna": C["blue"], "XGB-Optuna": C["red"], "CAT-Optuna": C["green"], "Boost-Avg": "#C44E52"}
MODEL_ORDER = ["LGB-Optuna", "XGB-Optuna", "CAT-Optuna"]


def _ensure_ticks_in(ax):
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def save_fig(fig: plt.Figure, out_dir: Path, stem: str):
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    for path in [png, svg]:
        fig.savefig(path, format=path.suffix.lstrip('.'))
    plt.close(fig)
    return png, svg


def fig1_cv_stability(df: pd.DataFrame, out_dir: Path):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    groups, labels, colors = [], [], []
    models = MODEL_ORDER + [m for m in df["Model"].astype(str).unique() if m not in MODEL_ORDER]
    for m in models:
        vals = pd.to_numeric(df.loc[df["Model"].astype(str).str.strip() == m, "R2"], errors="coerce").dropna().values
        if len(vals) == 0:
            continue
        groups.append(vals); labels.append(m); colors.append(MODEL_COLORS.get(m, C["purple"]))
    pos = np.arange(1, len(groups) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_facecolor(C["bg"]); fig.patch.set_facecolor(C["white"])
    vp = ax.violinplot(groups, positions=pos, widths=0.62, showmeans=False, showmedians=False, showextrema=False)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.22); body.set_linewidth(0.9)
    bp = ax.boxplot(groups, positions=pos, widths=0.17, patch_artist=True, notch=False, manage_ticks=False,
                    medianprops=dict(color=C["white"], linewidth=2.2), whiskerprops=dict(linewidth=1.1, color=C["grey"]),
                    capprops=dict(linewidth=1.1, color=C["grey"]), flierprops=dict(marker="o", markersize=3.5, markeredgewidth=0.5, alpha=0.6),
                    boxprops=dict(linewidth=0.8), zorder=3)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.88); patch.set_edgecolor(col)
    for flier, col in zip(bp["fliers"], colors):
        flier.set(markerfacecolor=col, markeredgecolor=col)
    rng = np.random.default_rng(42)
    for p, g, col in zip(pos, groups, colors):
        ax.plot(p, np.mean(g), marker="D", markersize=6, color=C["white"], markeredgecolor=col, markeredgewidth=1.5, zorder=7)
        jitter = rng.uniform(-0.08, 0.08, size=len(g))
        ax.scatter(p + jitter, g, s=16, color=col, alpha=0.55, linewidths=0, zorder=4)
        ax.annotate(f"$\\mu$={np.mean(g):.3f}\n$\\sigma$={np.std(g, ddof=1):.3f}", xy=(p, 0), xycoords=("data", "axes fraction"), xytext=(0, -44), textcoords="offset points", ha="center", va="top", fontsize=8.5, color=C["grey"], annotation_clip=False)
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel(r"Cross-Validation $R^{2}$", fontsize=12)
    ax.set_xlabel("Base Boosting Model", fontsize=12, labelpad=38)
    ax.set_title("Repeated K-Fold CV $R^{2}$ Stability", fontsize=13, pad=10)
    ax.set_xlim(0.3, len(groups) + 0.7)
    all_vals = np.concatenate(groups)
    ax.set_ylim(all_vals.min() - 0.018, all_vals.max() + 0.018)
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    handles = [mpatches.Patch(facecolor=MODEL_COLORS.get(m, C["purple"]), alpha=0.80, label=m) for m in labels]
    handles.append(Line2D([0], [0], marker="D", color="w", markerfacecolor=C["white"], markeredgecolor=C["grey"], markersize=6, linewidth=0, label="Mean"))
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.92, edgecolor="#cccccc", handlelength=1.2, fontsize=9.5, ncol=1)
    _ensure_ticks_in(ax)
    fig.subplots_adjust(bottom=0.18)
    return save_fig(fig, out_dir, "fig1_cv_r2_stability_boosting_models")


def fig2_bootstrap(df: pd.DataFrame, out_dir: Path):
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    model_col = "Model" if "Model" in df.columns else None
    if model_col and any(df[model_col].astype(str).str.strip() == "Boost-Avg"):
        df = df[df[model_col].astype(str).str.strip() == "Boost-Avg"].copy()
    r2 = pd.to_numeric(df["R2"], errors="coerce").dropna().values if "R2" in df.columns else np.array([])
    rmse = pd.to_numeric(df["RMSE"], errors="coerce").dropna().values if "RMSE" in df.columns else np.array([])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8)); fig.patch.set_facecolor(C["white"])

    def draw_panel(ax, data, ci, mean_val, xlabel, c_fill, c_line, unit="", legend_loc="upper left"):
        xs = np.linspace(data.min() * 0.995, data.max() * 1.005, 600)
        kde = stats.gaussian_kde(data, bw_method="scott")
        ys = kde(xs)
        ax.set_facecolor(C["bg"])
        ax.hist(data, bins=40, density=True, color=c_fill, alpha=0.22, edgecolor="none", zorder=1)
        ax.fill_between(xs, ys, alpha=0.15, color=c_fill, zorder=2)
        ax.plot(xs, ys, color=c_line, linewidth=1.9, zorder=3)
        mask = (xs >= ci[0]) & (xs <= ci[1])
        ax.fill_between(xs[mask], ys[mask], alpha=0.40, color=c_fill, zorder=2, label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]{unit}")
        for v in ci:
            ax.axvline(v, color=c_line, linewidth=1.0, linestyle=":", alpha=0.80, zorder=4)
        ax.axvline(mean_val, color=c_line, linewidth=1.7, linestyle="--", zorder=5, label=f"Mean = {mean_val:.4f}{unit}")
        ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel("Probability Density", fontsize=12)
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4)); ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
        ax.legend(frameon=True, framealpha=0.92, edgecolor="#cccccc", fontsize=9.5, loc=legend_loc)
        _ensure_ticks_in(ax)

    if len(r2):
        ci_r2 = np.percentile(r2, [2.5, 97.5]); draw_panel(axes[0], r2, ci_r2, np.mean(r2), xlabel=r"Bootstrap $R^{2}$", c_fill=C["light_blue"], c_line=C["blue"], legend_loc="upper left")
    axes[0].set_title(r"Bootstrap $R^{2}$ Distribution (n = 1000)", fontsize=13)
    if len(rmse):
        ci_rmse = np.percentile(rmse, [2.5, 97.5]); draw_panel(axes[1], rmse, ci_rmse, np.mean(rmse), xlabel="Bootstrap RMSE (MPa)", c_fill=C["light_red"], c_line=C["red"], unit=" MPa", legend_loc="upper right")
    axes[1].set_title("Bootstrap RMSE Distribution (n = 1000)", fontsize=13)
    fig.suptitle("Boost-Avg Bootstrap Confidence Interval Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, out_dir, "fig2_bootstrap_ci_boost_avg")


def fig3_qq_shapiro(df: pd.DataFrame, out_dir: Path):
    col = "Standardized_Residual" if "Standardized_Residual" in df.columns else "Residual"
    sr = pd.to_numeric(df[col], errors="coerce").dropna().values
    stat_sw, p_sw = stats.shapiro(sr)
    n = len(sr)
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theo = stats.norm.ppf(probs)
    emp = np.sort(sr)
    q25e, q75e = np.percentile(emp, 25), np.percentile(emp, 75)
    q25t, q75t = stats.norm.ppf(0.25), stats.norm.ppf(0.75)
    slope = (q75e - q25e) / (q75t - q25t)
    intercept = q25e - slope * q25t
    line_x = np.array([theo.min() - 0.15, theo.max() + 0.15]); line_y = slope * line_x + intercept
    sigma_rob = (q75e - q25e) / (q75t - q25t); phi_theo = stats.norm.pdf(theo)
    se_i = sigma_rob * np.sqrt(probs * (1 - probs) / (n * phi_theo ** 2))
    ci_lo = (slope * theo + intercept) - 1.96 * se_i; ci_hi = (slope * theo + intercept) + 1.96 * se_i
    dev = emp - (slope * theo + intercept); norm_dev = np.clip(np.abs(dev) / np.abs(dev).max(), 0, 1)
    fig, ax = plt.subplots(figsize=(6.8, 6.0)); fig.patch.set_facecolor(C["white"]); ax.set_facecolor(C["bg"])
    ax.fill_between(theo, ci_lo, ci_hi, color=C["light_blue"], alpha=0.28, label="95% Confidence Band", zorder=1)
    ax.plot(line_x, line_y, color=C["red"], linewidth=1.7, linestyle="--", label="Theoretical Normal Line", zorder=3)
    sc = ax.scatter(theo, emp, c=norm_dev, cmap="RdYlBu_r", vmin=0, vmax=0.85, s=42, linewidths=0.5, edgecolors="#888888", alpha=0.88, zorder=4)
    cbar = fig.colorbar(sc, ax=ax, pad=0.018, shrink=0.80, aspect=22)
    cbar.set_label("Normalised Deviation from\nTheoretical Quantile", fontsize=9.5, labelpad=6)
    cbar.ax.tick_params(labelsize=9)
    normality = "Yes" if p_sw > 0.05 else "No"
    sw_txt = f"Shapiro-Wilk Test\n$W$ = {stat_sw:.4f}\n$p$ = {p_sw:.4f}\nNormal ($\\alpha$=0.05): {normality}"
    ax.text(0.04, 0.97, sw_txt, transform=ax.transAxes, va="top", ha="left", fontsize=9.8, bbox=dict(boxstyle="round,pad=0.45", facecolor=C["white"], edgecolor=C["light_grey"], linewidth=0.8, alpha=0.94), zorder=6)
    ax.set_xlabel("Theoretical Quantiles", fontsize=12); ax.set_ylabel("Sample Quantiles (Standardised Residuals)", fontsize=12)
    ax.set_title("Normal Q-Q Plot of Standardised Residuals (Boost-Avg)", fontsize=13, pad=10)
    ax.set_xlim(line_x[0], line_x[1]); ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#cccccc", fontsize=9.5)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4)); ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4)); _ensure_ticks_in(ax); fig.tight_layout()
    return save_fig(fig, out_dir, "fig3_qq_shapiro_boost_avg")


def fig4_prediction_interval(df: pd.DataFrame, out_dir: Path):
    actual = pd.to_numeric(df["Actual"], errors="coerce").values
    pred_col = next(c for c in df.columns if str(c).startswith("Pred"))
    q5_col = next(c for c in df.columns if str(c).startswith("Q5") or str(c).lower().startswith("lower"))
    q95_col = next(c for c in df.columns if str(c).startswith("Q95") or str(c).lower().startswith("upper"))
    within_col = next((c for c in df.columns if "In_PI" in str(c)), None)
    pred = pd.to_numeric(df[pred_col], errors="coerce").values
    q5 = pd.to_numeric(df[q5_col], errors="coerce").values
    q95 = pd.to_numeric(df[q95_col], errors="coerce").values
    within = pd.to_numeric(df[within_col], errors="coerce").fillna(0).astype(int).values.astype(bool) if within_col else ((actual >= q5) & (actual <= q95))
    mask = ~(np.isnan(actual) | np.isnan(pred) | np.isnan(q5) | np.isnan(q95))
    actual, pred, q5, q95, within = actual[mask], pred[mask], q5[mask], q95[mask], within[mask]
    coverage = within.mean() * 100
    order = np.argsort(actual)
    actual_s, pred_s, q5_s, q95_s, within_s = actual[order], pred[order], q5[order], q95[order], within[order]
    idx = np.arange(len(actual_s)); pi_width = q95_s - q5_s; mean_w = np.mean(pi_width)
    fig = plt.figure(figsize=(11.5, 9.2)); gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.35); ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); fig.patch.set_facecolor(C["white"])
    ax1.set_facecolor(C["bg"])
    ax1.fill_between(idx, q5_s, q95_s, color=C["sky"], alpha=0.70, label="90% Prediction Interval", zorder=1)
    ax1.plot(idx, q5_s, color=C["light_blue"], linewidth=0.9, linestyle="--", alpha=0.65, zorder=2)
    ax1.plot(idx, q95_s, color=C["light_blue"], linewidth=0.9, linestyle="--", alpha=0.65, zorder=2)
    ax1.plot(idx, pred_s, color="#C44E52", linewidth=1.6, label="Predicted (Boost-Avg)", zorder=4)
    ax1.scatter(idx[within_s], actual_s[within_s], s=30, color=C["green"], linewidths=0.5, edgecolors="white", zorder=6, label=f"Actual (within PI, n={within_s.sum()})")
    ax1.scatter(idx[~within_s], actual_s[~within_s], s=55, marker="^", color=C["red"], linewidths=0.7, edgecolors="white", zorder=7, label=f"Actual (outside PI, n={(~within_s).sum()})")
    ax1.text(0.985, 0.04, f"90% PI Coverage: {coverage:.1f}%\nn(within) = {within_s.sum()} / {len(actual_s)}", transform=ax1.transAxes, ha="right", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.42", facecolor=C["white"], edgecolor=C["light_grey"], linewidth=0.8, alpha=0.95))
    ax1.set_ylabel("Bond Strength (MPa)", fontsize=12); ax1.set_xlabel("Sample Index (sorted by Actual)", fontsize=12); ax1.set_title("90% Prediction Interval Visualisation - Boost-Avg Model", fontsize=13, pad=10)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="#cccccc", fontsize=9.5, ncol=2, handlelength=1.5)
    ax1.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4)); ax1.set_ylim(q5_s.min() - 1.5, q95_s.max() + 1.5); ax1.set_xlim(-1, len(actual_s)); ax1.text(0.012, 0.975, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top"); _ensure_ticks_in(ax1)
    ax2.set_facecolor(C["bg"]); bar_colors = np.where(within_s, C["light_blue"], C["light_red"]); ax2.bar(idx, pi_width, width=0.85, color=bar_colors, edgecolor="none", zorder=2)
    ax2.axhline(mean_w, color=C["blue"], linewidth=1.3, linestyle="--", zorder=3)
    ax2.set_xlabel("Sample Index (sorted by Actual)", fontsize=12); ax2.set_ylabel("PI Width (MPa)", fontsize=12)
    ax2.legend([Line2D([0], [0], color=C["blue"], linestyle="--", linewidth=1.3)], [f"Mean PI Width = {mean_w:.2f} MPa"], loc="lower right", bbox_to_anchor=(1.0, 1.02), frameon=True, framealpha=0.92, edgecolor="#cccccc", fontsize=9.5)
    ax2.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4)); ax2.set_xlim(-1, len(actual_s)); ax2.set_ylim(0, pi_width.max() * 1.15); _ensure_ticks_in(ax2)
    return save_fig(fig, out_dir, "fig4_prediction_interval_boost_avg")


def choose_bootstrap_table(project_root: Path, explicit=None):
    if explicit:
        p = Path(explicit)
        return p if p.is_absolute() else (project_root / explicit)
    candidates = []
    results_dir = project_root / "results"
    for p in sorted(results_dir.rglob("*")) if results_dir.exists() else []:
        if not p.is_file() or p.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        try:
            df = read_table(p)
        except Exception:
            continue
        cols = {str(c).strip() for c in df.columns}
        if "Model" not in cols or not ({"R2", "RMSE", "MAE"} & cols):
            continue
        score = 0
        name = p.name.lower()
        if "bootstrap" in name:
            score += 100
        if "cv" in name:
            score -= 50
        score += min(len(df), 100)
        candidates.append((score, p))
    if not candidates:
        raise FileNotFoundError("No bootstrap-like metrics table found.")
    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return candidates[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--cv-file")
    ap.add_argument("--bootstrap-file")
    ap.add_argument("--residual-file")
    ap.add_argument("--prediction-file")
    args = ap.parse_args()
    project_root = resolve_project_root(args.project_root)
    out_dir = ensure_figures_dir(project_root)
    out = {}
    try:
        cv_path = find_table_by_columns(project_root, required_all={"Model", "R2"}, explicit=args.cv_file)
        out["cv_stability"] = [str(p) for p in fig1_cv_stability(read_table(cv_path), out_dir)]
        out["cv_source"] = str(cv_path)
    except Exception as e:
        out["cv_stability_error"] = str(e)
    try:
        boot_path = choose_bootstrap_table(project_root, explicit=args.bootstrap_file)
        out["bootstrap"] = [str(p) for p in fig2_bootstrap(read_table(boot_path), out_dir)]
        out["bootstrap_source"] = str(boot_path)
    except Exception as e:
        out["bootstrap_error"] = str(e)
    try:
        resid_path = find_table_by_columns(project_root, required_any=[{"Standardized_Residual", "Residual"}], explicit=args.residual_file)
        out["qq_plot"] = [str(p) for p in fig3_qq_shapiro(read_table(resid_path), out_dir)]
        out["residual_source"] = str(resid_path)
    except Exception as e:
        out["qq_plot_error"] = str(e)
    try:
        pred_path = find_table_by_columns(project_root, required_all={"Actual"}, required_any=[{"Pred_*"}, {"Q5_*"}, {"Q95_*"}], explicit=args.prediction_file)
        out["prediction_interval"] = [str(p) for p in fig4_prediction_interval(read_table(pred_path), out_dir)]
        out["prediction_source"] = str(pred_path)
    except Exception as e:
        out["prediction_interval_error"] = str(e)
    print(pd.Series(out, dtype=object).to_json(force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
