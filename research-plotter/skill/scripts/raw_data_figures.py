#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
High-fidelity raw-data publication figures.
Adapted to project layout while preserving the visual style of the reference script.
"""

import argparse
import math
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from plot_utils import ensure_figures_dir, find_single_dataset, read_table, resolve_project_root

warnings.filterwarnings("ignore", category=UserWarning)

PAL = {
    "bg": "#FFFFFF",
    "panel": "#FFFFFF",
    "dark": "#2F2F77",
    "grid": "#D7D2E8",
    "primary": "#77C6C0",
    "secondary": "#4F8971",
    "accent": "#F9A871",
    "pink": "#E3A7C1",
    "lightpurple": "#D7D2E8",
}
CMAP_CORR = LinearSegmentedColormap.from_list("elegant_div", ["#F9A871", "#FFFFFF", "#77C6C0"], N=256)

mpl.rcParams.update({
    "figure.facecolor": PAL["bg"],
    "axes.facecolor": PAL["panel"],
    "savefig.facecolor": PAL["bg"],
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Liberation Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "grid.alpha": 0.25,
    "grid.color": PAL["grid"],
    "grid.linestyle": "--",
    "axes.edgecolor": PAL["dark"],
    "text.color": PAL["dark"],
    "axes.labelcolor": PAL["dark"],
    "xtick.color": PAL["dark"],
    "ytick.color": PAL["dark"],
})


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 300):
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return png_path, svg_path


def panel_heading(ax, letter: str, title: str, fontsize: int = 11, weight: str = "bold"):
    ax.text(-0.06, 1.05, f"{letter}  {title}", transform=ax.transAxes, ha="left", va="bottom", fontsize=fontsize, fontweight=weight, color=PAL["dark"])


def add_y_grid(ax, alpha=0.20):
    ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=alpha, color=PAL["grid"], linestyle="--")


def draw_arrow_spines(ax, left=True, bottom=True, right=False, lw=1.0, ms=10, extend=0.0):
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
    if left:
        ax.add_patch(FancyArrowPatch((0.0, 0.0), (0.0, 1.0 + extend), transform=ax.transAxes, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=PAL["dark"], shrinkA=0, shrinkB=0, clip_on=False, zorder=10))
    if bottom:
        ax.add_patch(FancyArrowPatch((0.0, 0.0), (1.0 + extend, 0.0), transform=ax.transAxes, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=PAL["dark"], shrinkA=0, shrinkB=0, clip_on=False, zorder=10))


def fmt_sci_if_small(y, _):
    if y == 0:
        return "0"
    ay = abs(y)
    if ay < 0.01:
        return f"{y:.1e}"
    if ay >= 10:
        return f"{int(round(y))}"
    if ay >= 1:
        return f"{y:.1f}"
    return f"{y:.2f}"


def silverman_bandwidth(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = max(2, x.size)
    std = np.std(x, ddof=1) if n > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    h = 0.9 * sigma * (n ** (-1 / 5))
    if not np.isfinite(h) or h <= 0:
        h = max(1e-6, std * 0.1 if std > 0 else 1.0)
    return float(h)


def kde_gaussian_1d(x: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)
    bw = max(1e-9, float(bw))
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z * z).mean(axis=1) / (np.sqrt(2 * np.pi) * bw)


def compute_skewness(s: pd.Series) -> float:
    x = s.dropna().astype(float)
    return float(x.skew()) if len(x) >= 3 else 0.0


SKEW_THRESHOLD = 1.2


def should_log_display(s: pd.Series, skew_thr: float = SKEW_THRESHOLD) -> bool:
    x = s.dropna().astype(float)
    if len(x) < 5 or (x < 0).any():
        return False
    skew = abs(compute_skewness(x))
    q05, q95 = np.percentile(x, [5, 95])
    spread_ratio = (q95 / max(q05, 1e-9)) if q05 > 0 else np.inf
    return (skew > skew_thr) or (spread_ratio > 20)


def display_transform_series(s: pd.Series):
    if should_log_display(s):
        return np.log1p(s.astype(float)), True
    return s.astype(float), False


def robust_scale_series(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    med = np.nanmedian(x)
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        iqr = np.nanstd(x)
    if not np.isfinite(iqr) or iqr == 0:
        iqr = 1.0
    return (x - med) / iqr


def iqr_outliers_per_feature(df_num: pd.DataFrame, k: float = 1.5) -> pd.Series:
    counts = {}
    for c in df_num.columns:
        x = df_num[c].dropna()
        if len(x) == 0:
            counts[c] = 0
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        counts[c] = int(((df_num[c] < lo) | (df_num[c] > hi)).sum())
    return pd.Series(counts).sort_values(ascending=False)


def zscore_outliers_per_feature(df_num: pd.DataFrame, thr: float = 3.0) -> pd.Series:
    std = df_num.std(ddof=0).replace(0, 1)
    z = (df_num - df_num.mean()) / std
    return (z.abs() > thr).sum().sort_values(ascending=False)


def simple_smooth(y, window=3):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    out = y.copy()
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), i + window // 2 + 1)
        out[i] = np.mean(y[lo:hi])
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def detect_target(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit and explicit in df.columns:
        return explicit
    priority = [
        "bond_strength_mpa", "bond strength mpa", "bond_strength", "bond strength",
        "target", "response", "label", "y"
    ]
    lc = {c: str(c).strip().lower() for c in df.columns}
    for key in priority:
        for c, cl in lc.items():
            if cl == key:
                return c
    for c, cl in lc.items():
        if "bond" in cl and "strength" in cl:
            return c
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[-1]


def detect_categoricals(df: pd.DataFrame, explicit: list[str] | None = None) -> list[str]:
    if explicit:
        return [c for c in explicit if c in df.columns]
    candidates = []
    for c in df.columns:
        s = df[c]
        nunique = s.nunique(dropna=True)
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            if 2 <= nunique <= 12:
                candidates.append(c)
        elif nunique <= 8 and not pd.api.types.is_float_dtype(s):
            candidates.append(c)
    preferred = [c for c in candidates if any(k in c.lower() for k in ["type", "surface", "treatment", "class", "group"])]
    return (preferred + [c for c in candidates if c not in preferred])[:2]


def prepare_dataframe(dataset_path: Path):
    df = normalize_columns(read_table(dataset_path))
    df = df.dropna(axis=1, how="all")
    if len(df) > 1 and df.iloc[0].isna().mean() > 0.5:
        df = df.iloc[1:].reset_index(drop=True)
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() >= max(5, int(0.6 * len(df[c].dropna()))):
                df[c] = coerced
    target = detect_target(df)
    numeric_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = numeric_cols[:11]
    if target not in df.columns:
        raise ValueError("Target column could not be detected.")
    categorical_cols = detect_categoricals(df)
    return df, target, numeric_cols + [target], categorical_cols


def build_main_fig1_distributions(df: pd.DataFrame, numeric_cols: list[str], out_dir: Path):
    cols = [c for c in numeric_cols if c in df.columns]
    n = len(cols)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(15.2, max(9.5, 1.5 * nrows + 1.6)))
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.33, hspace=0.52)
    axes = [fig.add_subplot(gs[i, j]) for i in range(nrows) for j in range(ncols)]
    for i, c in enumerate(cols):
        ax = axes[i]
        x_raw = df[c].dropna().astype(float)
        x_disp, used_log = display_transform_series(x_raw)
        x = x_disp.values[np.isfinite(x_disp.values)]
        if len(x) == 0:
            ax.axis("off")
            continue
        xmin, xmax = float(np.min(x)), float(np.max(x))
        xr = xmax - xmin
        pad = 0.03 * (xr if xr > 0 else 1.0)
        xlo, xhi = xmin - pad, xmax + pad
        axr = ax.twinx()
        axr.hist(x, bins=24, density=False, color=PAL["lightpurple"], edgecolor="white", alpha=0.70)
        grid = np.linspace(xlo, xhi, 240)
        kde_y = kde_gaussian_1d(x, grid, bw=silverman_bandwidth(x))
        ax.plot(grid, kde_y, color=PAL["primary"], lw=1.8)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0, max(kde_y.max(), 1e-9) * 1.18)
        cnt_max = max([p.get_height() for p in axr.patches], default=1.0)
        axr.set_ylim(0, cnt_max * 1.18)
        add_y_grid(ax, alpha=0.15)
        ax.set_xlabel(f"{c} (log1p)" if used_log else c, fontsize=8.5, labelpad=5)
        if i == 0:
            ax.set_ylabel("Density", color=PAL["primary"], fontsize=8.5)
            axr.set_ylabel("Frequency", color=PAL["dark"], fontsize=8.5)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_sci_if_small))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axr.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(axis="y", labelsize=7)
        axr.tick_params(axis="y", labelsize=7, labelright=True, right=True)
        ax.tick_params(axis="x", labelsize=8)
        draw_arrow_spines(ax, left=True, bottom=True, right=False, extend=0.14)
        axr.spines["right"].set_visible(True)
        axr.spines["right"].set_edgecolor(PAL["dark"])
        axr.spines["right"].set_linewidth(1.0)
        axr.tick_params(right=True, labelright=True, direction="in")
        axr.add_patch(FancyArrowPatch((1.0, 0.0), (1.0, 1.0 + 0.14), transform=axr.transAxes, arrowstyle='-|>', mutation_scale=10, lw=1.0, color=PAL["dark"], shrinkA=0, shrinkB=0, clip_on=False, zorder=10))
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Distributions of numerical variables", y=0.965, fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.92)
    return save_figure(fig, out_dir, "Figure_1_Distributions", dpi=300)


def build_main_fig2_correlations(df: pd.DataFrame, numeric_cols: list[str], out_dir: Path):
    df_num = df[numeric_cols].copy().apply(pd.to_numeric, errors='coerce').dropna()
    pearson = df_num.corr(method="pearson")
    spearman = df_num.corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), constrained_layout=True)
    im0 = axes[0].imshow(pearson.values, cmap=CMAP_CORR, vmin=-1, vmax=1)
    axes[1].imshow(spearman.values, cmap=CMAP_CORR, vmin=-1, vmax=1)
    labels = list(pearson.columns)
    short_labels = [l.replace('_MPa', '').replace('_mm', '').replace('_GPa', '') for l in labels]
    for ax, mat, letter, title in [(axes[0], pearson, "a", "Pearson correlation"), (axes[1], spearman, "b", "Spearman correlation")]:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(short_labels, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(short_labels)
        ax.tick_params(axis="x", pad=2)
        ax.tick_params(axis="y", pad=2)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat.values[r, c]
                if np.isnan(v):
                    continue
                txt_col = "white" if abs(v) > 0.55 else PAL["dark"]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7.0, color=txt_col)
        panel_heading(ax, letter, title)
    cbar = fig.colorbar(im0, ax=axes, fraction=0.042, pad=0.015)
    cbar.set_label("Correlation coefficient")
    return save_figure(fig, out_dir, "Figure_2_Correlations", dpi=300)


def plot_binned_box_relationship(ax, x: pd.Series, y: pd.Series, x_label: str, target_label: str, letter: str):
    x_disp, used_log = display_transform_series(x.astype(float))
    xv = x_disp.values.astype(float)
    yv = y.values.astype(float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[mask]
    yv = yv[mask]
    bins = np.unique(np.quantile(xv, np.linspace(0, 1, 7)))
    if len(bins) < 4:
        ax.scatter(xv, yv, s=12, alpha=0.25, color=PAL["primary"], edgecolors="none")
        ax.set_xlabel(f"{x_label} (log1p)" if used_log else x_label)
        ax.set_ylabel(target_label)
        add_y_grid(ax, alpha=0.12)
        draw_arrow_spines(ax, left=True, bottom=True)
        panel_heading(ax, letter, f"{x_label} vs bond strength")
        return
    groups = []
    valid_bin_labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (xv >= lo) & (xv <= hi if hi == bins[-1] else xv < hi)
        vals = yv[m]
        if len(vals) >= 4:
            groups.append(vals)
            valid_bin_labels.append(f"Bin {len(valid_bin_labels) + 1}")
    if len(groups) < 3:
        ax.scatter(xv, yv, s=12, alpha=0.25, color=PAL["primary"], edgecolors="none")
        ax.set_xlabel(f"{x_label} (log1p)" if used_log else x_label)
        ax.set_ylabel(target_label)
        add_y_grid(ax, alpha=0.12)
        draw_arrow_spines(ax, left=True, bottom=True)
        panel_heading(ax, letter, f"{x_label} vs bond strength")
        return
    bp = ax.boxplot(groups, patch_artist=True, widths=0.62, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(PAL["lightpurple"])
        patch.set_edgecolor(PAL["dark"])
        patch.set_alpha(0.80)
        patch.set_linewidth(0.8)
    for item in ["whiskers", "caps", "medians"]:
        for line in bp[item]:
            line.set_color(PAL["dark"])
            line.set_linewidth(0.95)
    medians = np.array([np.median(g) for g in groups], dtype=float)
    ax.plot(np.arange(1, len(groups) + 1), simple_smooth(medians, window=3), color=PAL["accent"], lw=2.0, marker="o", markersize=3.8, label="Median trend")
    rng = np.random.default_rng(42)
    for i, vals in enumerate(groups, start=1):
        sample_n = min(60, len(vals))
        idx = rng.choice(len(vals), size=sample_n, replace=False)
        jitter = rng.normal(0, 0.055, size=sample_n)
        ax.scatter(np.full(sample_n, i) + jitter, vals[idx], s=10, alpha=0.15, color=PAL["primary"], edgecolors="none")
    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels(valid_bin_labels)
    xlabel = f"{x_label} (quantile bins; log1p)" if used_log else f"{x_label} (quantile bins)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(target_label)
    add_y_grid(ax, alpha=0.14)
    draw_arrow_spines(ax, left=True, bottom=True)
    panel_heading(ax, letter, f"{x_label} vs bond strength")
    if letter == "a":
        ax.legend(frameon=False, loc="upper left")


def build_main_fig3_relationships(df: pd.DataFrame, numeric_cols: list[str], target_col: str, out_dir: Path):
    df_num = df[numeric_cols].copy().apply(pd.to_numeric, errors='coerce')
    corr_with_target = df_num.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_features = corr_with_target.head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for i, feat in enumerate(top_features):
        plot_binned_box_relationship(axes[i], df_num[feat], df_num[target_col], feat, target_col, chr(ord("a") + i))
    for j in range(len(top_features), len(axes)):
        axes[j].axis("off")
    return save_figure(fig, out_dir, "Figure_3_KeyRelationships", dpi=300)


def build_main_fig4_categorical_effects(df: pd.DataFrame, categorical_cols: list[str], target_col: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), constrained_layout=True)
    for idx, ax in enumerate(axes):
        if idx >= len(categorical_cols):
            ax.axis("off")
            continue
        cat_col = categorical_cols[idx]
        df_copy = df.copy()
        stats = df_copy.groupby(cat_col)[target_col].agg(["median", "count"]).sort_values("median")
        order = stats.index.tolist()
        data = [df_copy.loc[df_copy[cat_col] == cat, target_col].dropna().values for cat in order]
        top2 = stats["median"].nlargest(min(2, len(order))).index.tolist()
        highlight_idx = [order.index(x) for x in top2 if x in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.60, showfliers=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PAL["primary"] if i in highlight_idx else PAL["lightpurple"])
            patch.set_alpha(0.85)
            patch.set_edgecolor(PAL["dark"])
            patch.set_linewidth(0.9)
        for item in ["whiskers", "caps", "medians"]:
            for line in bp[item]:
                line.set_color(PAL["dark"])
                line.set_linewidth(1.0)
        meds = stats["median"].values
        ax.scatter(np.arange(1, len(order) + 1), meds, s=26, color=PAL["accent"], zorder=4)
        labels = [f"{cat}\n(n={int(stats.loc[cat, 'count'])})" for cat in order]
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(target_col)
        add_y_grid(ax, alpha=0.14)
        draw_arrow_spines(ax, left=True, bottom=False)
        panel_heading(ax, chr(ord('a') + idx), f"Effect of {cat_col}")
    return save_figure(fig, out_dir, "Figure_4_CategoricalEffects", dpi=300)


def build_supp_figS1_outliers(df: pd.DataFrame, numeric_cols: list[str], out_dir: Path):
    df_num = df[numeric_cols].copy().apply(pd.to_numeric, errors='coerce')
    iqr_counts = iqr_outliers_per_feature(df_num, k=1.5)
    z_counts = zscore_outliers_per_feature(df_num, thr=3.0)
    fig = plt.figure(figsize=(13.8, 9.6))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1.5, 1.0], hspace=0.85, wspace=0.34)
    ax0 = fig.add_subplot(gs[0, :])
    data = [robust_scale_series(df_num[c].astype(float)).dropna().values for c in numeric_cols]
    bp = ax0.boxplot(data, patch_artist=True, widths=0.56, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(PAL["lightpurple"])
        patch.set_alpha(0.75)
        patch.set_edgecolor(PAL["dark"])
        patch.set_linewidth(0.8)
    for item in ["whiskers", "caps", "medians"]:
        for line in bp[item]:
            line.set_color(PAL["dark"])
            line.set_linewidth(1.0)
    for fl in bp["fliers"]:
        fl.set(marker='o', markersize=2.4, markerfacecolor=PAL["accent"], alpha=0.50, markeredgewidth=0)
    ax0.set_xticks(np.arange(1, len(numeric_cols) + 1))
    ax0.set_xticklabels(numeric_cols, rotation=35, ha="right")
    ax0.tick_params(axis="x", pad=8)
    ax0.set_ylabel("Robust-scaled value")
    add_y_grid(ax0, alpha=0.15)
    draw_arrow_spines(ax0, left=True, bottom=False)
    panel_heading(ax0, "a", "Robust-scaled boxplot overview")
    ax1 = fig.add_subplot(gs[1, 0])
    iqr_sorted = iqr_counts.sort_values(ascending=True)
    ax1.barh(list(iqr_sorted.index), iqr_sorted.values, color=PAL["primary"], edgecolor="white", linewidth=0.6)
    ax1.set_xlabel("Outlier count")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    draw_arrow_spines(ax1, left=True, bottom=True)
    panel_heading(ax1, "b", "IQR outliers (k=1.5)")
    ax2 = fig.add_subplot(gs[1, 1])
    z_sorted = z_counts.sort_values(ascending=True)
    ax2.barh(list(z_sorted.index), z_sorted.values, color=PAL["secondary"], edgecolor="white", linewidth=0.6)
    ax2.set_xlabel("Outlier count")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    draw_arrow_spines(ax2, left=True, bottom=True)
    panel_heading(ax2, "c", "Z-score outliers (|z| > 3)")
    return save_figure(fig, out_dir, "Figure_S1_Outliers", dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--data-file")
    ap.add_argument("--target")
    args = ap.parse_args()
    project_root = resolve_project_root(args.project_root)
    dataset_path = Path(args.data_file).resolve() if args.data_file else find_single_dataset(project_root)
    figures_root = ensure_figures_dir(project_root)
    main_dir = ensure_dir(figures_root / "main_figures")
    supp_dir = ensure_dir(figures_root / "supplementary_figures")
    df, target_col, numeric_cols, categorical_cols = prepare_dataframe(dataset_path)
    if args.target and args.target in df.columns:
        target_col = args.target
        numeric_cols = [c for c in numeric_cols if c != target_col] + [target_col]
    outputs = {
        "dataset": str(dataset_path),
        "target_col": target_col,
        "categorical_cols": categorical_cols,
        "figure_1": [str(p) for p in build_main_fig1_distributions(df, numeric_cols, main_dir)],
        "figure_2": [str(p) for p in build_main_fig2_correlations(df, numeric_cols, main_dir)],
        "figure_3": [str(p) for p in build_main_fig3_relationships(df, numeric_cols, target_col, main_dir)],
        "figure_4": [str(p) for p in build_main_fig4_categorical_effects(df, categorical_cols, target_col, main_dir)],
        "figure_s1": [str(p) for p in build_supp_figS1_outliers(df, numeric_cols, supp_dir)],
    }
    print(pd.Series(outputs, dtype=object).to_json(force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
