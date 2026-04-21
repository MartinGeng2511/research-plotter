#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""High-fidelity SHAP dependence plots preserving the uploaded layout and style."""

import argparse
import math
import warnings
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import shap

from plot_utils import ensure_2d_shap, ensure_figures_dir, find_shap_bundle, load_feature_names, read_table, resolve_project_root

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 0.9,
})

_DISPLAY_MAP = {
    "Concrete_Strength_MPa": "Concrete Strength (MPa)", "Cement_Binder_Ratio": "Cement/Binder Ratio",
    "Seawater_Binder_Ratio": "Seawater/Binder Ratio", "SeaSand_Binder_Ratio": "Sea Sand/Binder Ratio",
    "CoarseAggregate_Binder_Ratio": "Coarse Aggregate/Binder Ratio", "Max_CA_Size_mm": "Max CA Size (mm)",
    "Bar_Diameter_mm": "Bar Diameter (mm)", "Rib_Spacing_Diameter_Ratio": "Rib Spacing/Diameter Ratio",
    "Rib_Height_Diameter_Ratio": "Rib Height/Diameter Ratio", "Bar_Surface_Treatment": "Bar Surface Treatment",
    "Bonded_Length_Diameter_Ratio": "Bonded Length/Diameter Ratio", "Bar_Fiber_Type": "Bar Fiber Type",
    "Bar_Tensile_Modulus_GPa": "Bar Tensile Modulus (GPa)", "sqrt_fc": "sqrt(fc)", "frp_stiffness": "FRP Stiffness",
    "log_bond_length": "log(Bond Length)", "log_bond_ratio": "log(Bond Ratio)", "effective_bond_param": "Effective Bond Param",
    "bar_surface_area": "Bar Surface Area", "bar_surface_area_log": "log(Bar Surface Area)", "rib_factor": "Rib Factor",
    "rib_sp_x_height": "Rib Spacing x Height", "rib_density": "Rib Density", "stiffness_ratio": "Stiffness Ratio",
    "modulus_x_sqrtfc": "E x sqrt(fc)", "wb_ratio": "W/B Ratio", "agg_powder_ratio": "Agg/Powder Ratio",
    "total_powder": "Total Powder", "powder_agg_product": "Powder x Agg", "particle_bond_factor": "Particle Bond Factor",
    "particle_x_fc": "Particle x fc", "elastic_x_diameter": "Elastic x Diameter", "bond_len_x_diameter": "Bond Length x Diameter",
    "sand_x_aggregate": "Sand x Aggregate", "cement_x_seawater": "Cement x Seawater", "fc_x_lb": "fc x Lb",
    "fc_x_d": "fc x d", "fc_x_ri": "fc x ri", "sqrt_fc_sq": "sqrt(fc)^2", "modulus_sq": "E^2",
    "bond_ratio_sq": "(Lb/d)^2", "diameter_sq": "d^2", "fc_sq": "fc^2", "log_fc": "log(fc)",
    "log_E": "log(E)", "log_d": "log(d)", "log_lb": "log(Lb)", "sqrtfc_E_lb": "sqrt(fc)*E*Lb",
    "E_d_lb": "E*d*Lb", "fc_E_rib": "fc*E*Rib",
}


def pretty_name(raw: str) -> str:
    return _DISPLAY_MAP.get(raw, raw.replace("_", " "))


def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\s-]", "_", str(s)).strip().replace(" ", "_")


def format_axis(ax):
    for attr, get_lim in [("x", ax.get_xlim), ("y", ax.get_ylim)]:
        lo, hi = get_lim()
        if np.isfinite(lo) and np.isfinite(hi):
            span = abs(hi - lo)
            if span < 1e-3:
                fmt = ticker.FormatStrFormatter("%.1e")
            else:
                fmt = ticker.ScalarFormatter(useOffset=False)
                fmt.set_scientific(False)
            getattr(ax, f"{attr}axis").set_major_formatter(fmt)
    ax.tick_params(axis="both", direction="in", top=True, right=True, length=4, width=0.8)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.5)


def set_tnr(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname("Times New Roman")
    ax.xaxis.label.set_fontname("Times New Roman")
    ax.yaxis.label.set_fontname("Times New Roman")
    ax.title.set_fontname("Times New Roman")


def draw_page(shap_values, X, feat_names, feat_indices, ncols=4, rank_offset=0):
    real_k = len(feat_indices)
    nrows = int(np.ceil(real_k / ncols))
    display_names = [pretty_name(n) for n in feat_names]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.4, nrows * 4.0), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for i, feat_idx in enumerate(feat_indices):
        ax = axes[i]
        rank = rank_offset + i + 1
        shap.dependence_plot(int(feat_idx), shap_values, X, feature_names=display_names, interaction_index="auto", ax=ax, show=False, alpha=0.72, dot_size=28, cmap=plt.get_cmap("coolwarm"))
        dname = display_names[feat_idx]
        ax.set_xlabel(dname, labelpad=5, fontsize=9.5, fontweight="bold", fontname="Times New Roman")
        ax.set_ylabel(f"SHAP value\n({dname})", labelpad=6, fontsize=9, fontname="Times New Roman")
        ax.set_title(f"Rank {rank}", fontsize=9, fontweight="bold", fontname="Times New Roman", pad=4, color="#444444")
        format_axis(ax)
        set_tnr(ax)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
    for j in range(real_k, len(axes)):
        axes[j].axis("off")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--page-size", type=int, default=20)
    args = ap.parse_args()
    project_root = resolve_project_root(args.project_root)
    out_dir = ensure_figures_dir(project_root) / "dependence_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = find_shap_bundle(project_root)
    if not all(bundle.values()):
        raise FileNotFoundError(f"Missing SHAP inputs: {[k for k, v in bundle.items() if not v]}")
    feat_names = load_feature_names(bundle["feature_json"])
    shap_values = ensure_2d_shap(np.load(bundle["shap_values"], allow_pickle=True))
    X_df = read_table(bundle["x_matrix"])
    X = X_df[feat_names].to_numpy(dtype=float)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top_idx = sorted_idx[: args.top_k]
    fig = draw_page(shap_values, X, feat_names, top_idx, ncols=4, rank_offset=0)
    base = out_dir / safe_filename(f"Fig_Dependence_Top{args.top_k}_4x{int(np.ceil(args.top_k/4))}_v1")
    fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    for p in range(int(math.ceil(len(sorted_idx) / args.page_size))):
        chunk = sorted_idx[p * args.page_size: (p + 1) * args.page_size]
        fig = draw_page(shap_values, X, feat_names, chunk, ncols=4, rank_offset=p * args.page_size)
        base = out_dir / safe_filename(f"Fig_Dependence_p{p+1}_v1")
        fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
        fig.savefig(base.with_suffix(".png"), format="png", dpi=600, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print({"shap_values": str(bundle["shap_values"]), "feature_json": str(bundle["feature_json"]), "x_matrix": str(bundle["x_matrix"]), "output_dir": str(out_dir)})


if __name__ == "__main__":
    main()
