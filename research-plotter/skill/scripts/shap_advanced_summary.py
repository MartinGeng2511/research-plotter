#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""High-fidelity SHAP advanced composite figure."""

import argparse
import re
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_utils import ensure_2d_shap, ensure_figures_dir, find_shap_bundle, load_feature_names, read_table, resolve_project_root

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 11

COLOR_SCHEMES = {"viridis": plt.cm.viridis, "plasma": plt.cm.plasma, "coolwarm": plt.cm.coolwarm, "RdYlBu": plt.cm.RdYlBu, "RdBu_r": plt.cm.RdBu_r}
_DISPLAY_MAP = {
    "Concrete_Strength_MPa": "Concrete Strength (MPa)", "Cement_Binder_Ratio": "Cement/Binder Ratio",
    "Seawater_Binder_Ratio": "Seawater/Binder Ratio", "SeaSand_Binder_Ratio": "Sea Sand/Binder Ratio",
    "CoarseAggregate_Binder_Ratio": "Coarse Aggregate/Binder", "Max_CA_Size_mm": "Max CA Size (mm)",
    "Bar_Diameter_mm": "Bar Diameter (mm)", "Rib_Spacing_Diameter_Ratio": "Rib Spacing/Diam. Ratio",
    "Rib_Height_Diameter_Ratio": "Rib Height/Diam. Ratio", "Bar_Surface_Treatment": "Bar Surface Treatment",
    "Bonded_Length_Diameter_Ratio": "Bonded Length/Diam. Ratio", "Bar_Fiber_Type": "Bar Fiber Type",
    "Bar_Tensile_Modulus_GPa": "Bar Tensile Modulus (GPa)", "sqrt_fc": "sqrt(fc)", "frp_stiffness": "FRP Stiffness",
    "log_bond_length": "log(Bond Length)", "log_bond_ratio": "log(Bond Ratio)", "effective_bond_param": "Effective Bond Param",
    "bar_surface_area": "Bar Surface Area", "bar_surface_area_log": "log(Bar Surface Area)", "rib_factor": "Rib Factor",
    "rib_sp_x_height": "Rib Spacing x Height", "rib_density": "Rib Density", "stiffness_ratio": "Stiffness Ratio",
    "modulus_x_sqrtfc": "E x sqrt(fc)", "wb_ratio": "W/B Ratio", "agg_powder_ratio": "Agg/Powder Ratio",
    "total_powder": "Total Powder", "powder_agg_product": "Powder x Agg", "particle_bond_factor": "Particle Bond Factor",
    "particle_x_fc": "Particle x fc", "elastic_x_diameter": "Elastic x Diameter", "bond_len_x_diameter": "Bond Len x Diameter",
    "sand_x_aggregate": "Sand x Aggregate", "cement_x_seawater": "Cement x Seawater", "fc_x_lb": "fc x Lb",
    "fc_x_d": "fc x d", "fc_x_ri": "fc x ri", "sqrt_fc_sq": "sqrt(fc)^2", "modulus_sq": "E^2",
    "bond_ratio_sq": "(Lb/d)^2", "diameter_sq": "d^2", "fc_sq": "fc^2", "log_fc": "log(fc)",
    "log_E": "log(E)", "log_d": "log(d)", "log_lb": "log(Lb)", "sqrtfc_E_lb": "sqrt(fc)*E*Lb", "E_d_lb": "E*d*Lb", "fc_E_rib": "fc*E*Rib",
}


def pretty_name(raw: str) -> str:
    return _DISPLAY_MAP.get(raw, raw.replace("_", " "))


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\s-]", "_", str(s)).strip().replace(" ", "_")


def create_optimized_cmap(base_cmap, start=0.16, end=0.92):
    colors = base_cmap(np.linspace(start, end, 256))
    return LinearSegmentedColormap.from_list("optimized_cmap", colors)


def set_tnr(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname("Times New Roman")
    ax.xaxis.label.set_fontname("Times New Roman")
    ax.yaxis.label.set_fontname("Times New Roman")
    ax.title.set_fontname("Times New Roman")


def add_left_colorbar(fig, rect, cmap):
    ax_cbar = fig.add_axes(rect)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_cbar.imshow(gradient, aspect="auto", cmap=cmap, origin="lower", extent=[0, 1, 0, 1])
    ax_cbar.set_xlim(0, 1); ax_cbar.set_ylim(0, 1); ax_cbar.set_xticks([]); ax_cbar.set_yticks([]); ax_cbar.set_frame_on(False)
    ax_cbar.text(0.5, 1.03, "High", transform=ax_cbar.transAxes, ha="center", va="bottom", fontsize=11, fontweight="bold", fontname="Times New Roman")
    ax_cbar.text(0.5, -0.03, "Low", transform=ax_cbar.transAxes, ha="center", va="top", fontsize=11, fontweight="bold", fontname="Times New Roman")
    ax_cbar.text(-0.6, 0.50, "Mean |SHAP Value|", transform=ax_cbar.transAxes, rotation=90, ha="right", va="center", fontsize=11, fontweight="bold", fontname="Times New Roman")


def add_right_feature_value_colorbar(fig, host_ax, cmap_name):
    pos = host_ax.get_position(); cbar_left = pos.x1 + 0.015; cbar_bottom = pos.y0 + pos.height * 0.20; cbar_width = 0.012; cbar_height = pos.height * 0.55
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cmap = plt.get_cmap(cmap_name); gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    cax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower", extent=[0, 1, 0, 1])
    cax.set_xlim(0, 1); cax.set_ylim(0, 1); cax.set_xticks([]); cax.set_yticks([0, 1]); cax.set_yticklabels(["Low", "High"], fontsize=10, fontweight="bold"); cax.yaxis.tick_right(); cax.tick_params(axis="y", length=0, pad=4); cax.set_frame_on(False)
    for lab in cax.get_yticklabels():
        lab.set_fontname("Times New Roman"); lab.set_fontweight("bold")
    cax.text(3.5, 0.5, "Feature Value", transform=cax.transAxes, rotation=270, ha="center", va="center", fontsize=11, fontweight="bold", fontname="Times New Roman")


def plot_ring_rose_inset(ax_bar, sorted_vals, colors, max_label_num=10):
    num_vars = len(sorted_vals); percentages = sorted_vals / sorted_vals.sum() * 100; widths = (sorted_vals / sorted_vals.sum()) * 2 * np.pi; thetas = np.cumsum(np.r_[0, widths[:-1]])
    rose_ax = inset_axes(ax_bar, width="40%", height="40%", loc="lower left", borderpad=0.75, axes_class=plt.PolarAxes); rose_ax.patch.set_alpha(0)
    hole_r, gray_bottom, gray_height = 0.72, 0.72, 0.68; color_bottom = gray_bottom + gray_height; outer_heights = 0.45 + 0.95 * (sorted_vals / sorted_vals.max())
    gray_colors = ["#EFEFEF" if i % 2 == 0 else "#E5E5E5" for i in range(num_vars)]
    rose_ax.bar(thetas, np.full(num_vars, gray_height), width=widths, bottom=np.full(num_vars, gray_bottom), color=gray_colors, align="edge", edgecolor="white", linewidth=0.6)
    rose_ax.bar(thetas, outer_heights, width=widths, bottom=np.full(num_vars, color_bottom), color=colors, align="edge", edgecolor="white", linewidth=0.7)
    rose_ax.bar(0, hole_r, width=2 * np.pi, bottom=0, color="white", edgecolor="white", linewidth=0)
    top_n = min(max_label_num, num_vars)
    for i in range(top_n):
        angle = thetas[i] + widths[i] / 2; radius = color_bottom + outer_heights[i] + 0.22
        rose_ax.text(angle, radius, f"{percentages[i]:.2f}%", ha="center", va="center", fontsize=5.8, fontweight="bold", fontname="Times New Roman")
    rose_ax.set_title("Relative Contribution", fontsize=8, fontweight="bold", pad=8, fontname="Times New Roman")
    rose_ax.set_theta_zero_location("N"); rose_ax.set_theta_direction(-1); rose_ax.set_xticklabels([]); rose_ax.set_yticklabels([]); rose_ax.grid(False); rose_ax.spines["polar"].set_visible(False)
    rose_ax.set_ylim(0, color_bottom + outer_heights.max() + 0.35)


def save_triple_combined_plot(shap_values, X, display_names, model_tag, out_dir, scheme_name="viridis", max_display=None):
    shap_values = ensure_2d_shap(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0); idx = np.argsort(mean_abs)[::-1]
    sorted_features = [display_names[i] for i in idx]; sorted_vals = mean_abs[idx]; X_ordered = X[:, idx]; shap_ordered = shap_values[:, idx]; feature_names_ordered = [display_names[i] for i in idx]
    if max_display is not None:
        sorted_features = sorted_features[:max_display]; sorted_vals = sorted_vals[:max_display]; X_ordered = X_ordered[:, :max_display]; shap_ordered = shap_ordered[:, :max_display]; feature_names_ordered = feature_names_ordered[:max_display]
    num_vars = len(sorted_features)
    base_cmap = COLOR_SCHEMES.get(scheme_name, plt.cm.viridis); cmap = create_optimized_cmap(base_cmap); vmin = np.quantile(sorted_vals, 0.20); vmax = np.quantile(sorted_vals, 0.80)
    if np.isclose(vmin, vmax):
        vmin = sorted_vals.min(); vmax = sorted_vals.max() + 1e-12
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax); colors = cmap(norm(sorted_vals))
    fig_h = max(10.0, 0.44 * num_vars + 2.5); fig = plt.figure(figsize=(22, fig_h), facecolor="white")
    bottom_margin, top_margin = 0.10, 0.05; usable_h = 1 - bottom_margin - top_margin
    cbar_left, cbar_width, bar_left, bar_width, feat_label_width = 0.06, 0.012, 0.12, 0.28, 0.12; dot_left = bar_left + bar_width + feat_label_width; dot_width = max(0.90 - dot_left, 0.28)
    add_left_colorbar(fig, [cbar_left, bottom_margin, cbar_width, usable_h], cmap)
    ax_bar = fig.add_axes([bar_left, bottom_margin, bar_width, usable_h]); y_pos = np.arange(num_vars)
    ax_bar.barh(y_pos, sorted_vals, color=colors, height=0.58, edgecolor="white", linewidth=0.8); ax_bar.invert_xaxis(); ax_bar.invert_yaxis()
    xmax = sorted_vals.max(); ax_bar.set_xlim(xmax * 1.35, 0); ax_bar.set_ylim(num_vars - 0.5, -0.5); ax_bar.set_yticks([])
    ax_bar.set_xlabel("Mean|SHAP Value|", fontsize=13, fontweight="bold", labelpad=12, loc="center"); ax_bar.xaxis.label.set_fontname("Times New Roman")
    ax_bar.spines["top"].set_visible(False); ax_bar.spines["left"].set_visible(False); ax_bar.spines["bottom"].set_linewidth(2.2); ax_bar.spines["right"].set_linewidth(2.2); ax_bar.spines["bottom"].set_color("#333333"); ax_bar.spines["right"].set_color("#333333"); ax_bar.spines["right"].set_position(("data", 0)); ax_bar.grid(False)
    ax_bar.tick_params(axis="x", which="major", direction="in", length=7, width=1.6, labelsize=10, pad=5); ax_bar.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(5)); ax_bar.tick_params(axis="x", which="minor", direction="in", length=3.5, width=0.8); set_tnr(ax_bar)
    for i, val in enumerate(sorted_vals):
        ax_bar.text(val + xmax * 0.02, i, f"{val:.4f}", ha="right", va="center", fontsize=8, fontweight="bold", fontname="Times New Roman", color="#000000", clip_on=False)
    bar_right_fig = bar_left + bar_width; feat_center_x = bar_right_fig + feat_label_width * 0.50
    for i, feat in enumerate(sorted_features):
        y_frac = 1.0 - (i - (-0.5)) / (num_vars - 0.5 - (-0.5)); y_fig = bottom_margin + y_frac * usable_h
        fig.text(feat_center_x, y_fig, feat, ha="center", va="center", fontsize=9.5, fontweight="bold", fontname="Times New Roman")
    plot_ring_rose_inset(ax_bar, sorted_vals, colors, max_label_num=10)
    ax_dot = fig.add_axes([dot_left, bottom_margin, dot_width, usable_h]); plt.sca(ax_dot)
    shap.summary_plot(shap_ordered, X_ordered, feature_names=feature_names_ordered, plot_type="dot", show=False, color_bar=False, cmap=scheme_name, max_display=num_vars, plot_size=None)
    ax_dot.set_xlabel("SHAP Value", fontsize=13, fontweight="bold", labelpad=12, loc="center"); ax_dot.xaxis.label.set_fontname("Times New Roman"); ax_dot.set_ylabel("")
    ax_dot.tick_params(axis="y", labelleft=False, length=0, pad=4); ax_dot.axvline(0, color="#6B6B6B", linewidth=1.0, zorder=0)
    ax_dot.spines["top"].set_visible(False); ax_dot.spines["right"].set_visible(False); ax_dot.spines["left"].set_visible(False); ax_dot.spines["bottom"].set_linewidth(2.2); ax_dot.spines["bottom"].set_color("#333333")
    ax_dot.tick_params(axis="x", which="major", direction="in", length=7, width=1.6, labelsize=10, pad=5); ax_dot.grid(False); set_tnr(ax_dot); ax_dot.set_ylim(-0.5, num_vars - 0.5)
    add_right_feature_value_colorbar(fig, ax_dot, scheme_name)
    safe_tag = safe_filename(model_tag); svg_path = out_dir / f"{safe_tag}_triple_combined_v1.svg"; png_path = out_dir / f"{safe_tag}_triple_combined_v1.png"
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white"); plt.savefig(png_path, format="png", dpi=600, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return png_path, svg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--scheme", default="viridis")
    ap.add_argument("--max-display", type=int, default=0)
    ap.add_argument("--model-tag", default="Stacking")
    args = ap.parse_args()
    project_root = resolve_project_root(args.project_root)
    out_dir = ensure_figures_dir(project_root) / "shap_advanced_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = find_shap_bundle(project_root)
    if not all(bundle.values()):
        raise FileNotFoundError(f"Missing SHAP inputs: {[k for k, v in bundle.items() if not v]}")
    feat_names = load_feature_names(bundle["feature_json"])
    shap_values = ensure_2d_shap(np.load(bundle["shap_values"], allow_pickle=True))
    X = read_table(bundle["x_matrix"])[feat_names].to_numpy(dtype=float)
    display_names = [pretty_name(n) for n in feat_names]
    max_display = args.max_display if args.max_display and args.max_display > 0 else None
    png, svg = save_triple_combined_plot(shap_values, X, display_names, args.model_tag, out_dir, scheme_name=args.scheme, max_display=max_display)
    print({"output_png": str(png), "output_svg": str(svg), "output_dir": str(out_dir)})


if __name__ == "__main__":
    main()
