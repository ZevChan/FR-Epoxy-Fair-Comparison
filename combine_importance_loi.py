"""
================================================================================
 combine_importance_loi.py — LOI Feature Importance 1×2 Combined
 (a) Feature Importance Bar  (b) Importance Slope
 Global sorting: After model feature_importances_ descending
================================================================================
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fair_holdout_comparison import OUTPUT_DIR

# ══════════════════════════════════════════════════════════════════
FS = 14
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.labelsize": FS + 2, "axes.titlesize": FS + 2,
    "legend.fontsize": FS - 2, "figure.titlesize": FS + 4,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_BEFORE = "#5DA5DA";  COLOR_AFTER = "#C91511"
COLOR_INCREASE = "#C91511";  COLOR_DECREASE = "#5DA5DA"

SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
STAGE3_DIR = os.path.join(OUTPUT_DIR, "Stage3")
os.makedirs(SAVE_DIR, exist_ok=True)

def shorten(name, n=30):
    return name if len(name) <= n else name[:n-1] + "\u2026"

def bold_axes(ax):
    for lbl in ax.get_xticklabels(): lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels(): lbl.set_fontweight("bold")
    if ax.xaxis.label: ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label: ax.yaxis.label.set_fontweight("bold")

# ══════════════════════════════════════════════════════════════════
def combine():
    target = "LOI"
    imp_path = os.path.join(STAGE3_DIR, f"feature_importance_{target}.csv")
    df = pd.read_csv(imp_path)
    df = df.head(15).copy()
    df["ShortName"] = df["Feature"].apply(shorten)

    names_ord = df["ShortName"].tolist()
    wb = df["Weight_Before"].values
    wa = df["Weight_After"].values
    top_n = len(df)

    # ── Canvas ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8), dpi=300)
    outer = fig.add_gridspec(1, 2, figure=fig,
                              left=0.08, right=0.97, top=0.95, bottom=0.08,
                              wspace=0.02)

    # ═══ (a) Feature Importance Bar ═══
    ax_a = fig.add_subplot(outer[0, 0])
    y_pos = np.arange(top_n); height = 0.35
    ax_a.barh(y_pos + height/2, wb, height,
              color=COLOR_BEFORE, alpha=0.85, edgecolor="white", linewidth=0.3,
              label="Before (Literature only)")
    ax_a.barh(y_pos - height/2, wa, height,
              color=COLOR_AFTER, alpha=0.85, edgecolor="white", linewidth=0.3,
              label="After (Literature + Experiment)")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names_ord, fontsize=FS - 3)
    ax_a.set_xlabel("Feature Importance Weight", fontweight="bold")
    ax_a.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 2)
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", linestyle=":", alpha=0.3)
    bold_axes(ax_a)
    ax_a.text(-0.12, 1.03, "(a)", transform=ax_a.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # ═══ (b) Importance Slope ═══
    ax_b = fig.add_subplot(outer[0, 1])
    for i in range(top_n):
        delta = wa[i] - wb[i]
        color = COLOR_INCREASE if delta >= 0 else COLOR_DECREASE
        lw = 0.6 + abs(delta) * 15; lw = np.clip(lw, 0.8, 3.5)
        ax_b.plot([wb[i], wa[i]], [i, i], color=color, linewidth=lw, alpha=0.75,
                  solid_capstyle="round")
        ax_b.scatter([wb[i]], [i], color=COLOR_BEFORE, s=35, zorder=5,
                     edgecolors="white", linewidth=0.5)
        ax_b.scatter([wa[i]], [i], color=COLOR_AFTER, s=35, zorder=5,
                     edgecolors="white", linewidth=0.5)
    ax_b.scatter([], [], color=COLOR_BEFORE, s=40, label="Before (Literature only)")
    ax_b.scatter([], [], color=COLOR_AFTER, s=40, label="After (Literature + Experiment)")
    ax_b.set_yticks(np.arange(top_n))
    ax_b.set_yticklabels(names_ord, fontsize=FS - 3)
    ax_b.set_xlabel("Feature Importance Weight", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=FS - 2, loc="lower right", bbox_to_anchor=(0.98, 0.03))
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", linestyle=":", alpha=0.3)
    bold_axes(ax_b)
    ax_b.text(-0.12, 1.03, "(b)", transform=ax_b.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # ── Auto spacing ─────────────────────────────────────────────────
    fig.canvas.draw()
    def _ytick_width(ax, fig):
        r = fig.canvas.get_renderer(); m = 0.0
        for lbl in ax.get_yticklabels():
            bb = lbl.get_window_extent(r)
            if bb is not None: m = max(m, bb.width)
        return m / fig.dpi if fig.dpi else 0.0
    left_w = _ytick_width(ax_a, fig)
    fig_w = fig.get_size_inches()[0]
    usable = fig_w * (0.97 - 0.08)
    needed = 2.5 * left_w
    ws = needed / (usable - needed) if usable > needed else 0.35
    outer.update(wspace=max(ws, 0.02))

    # ── Save ─────────────────────────────────────────────────────
    for fmt in ["png", "pdf", "svg"]:
        out = os.path.join(SAVE_DIR, f"Combined_Importance_LOI.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print(f"  -> {fmt}")
    plt.close(fig)
    print("[OK] Combined_Importance_LOI saved (png/pdf/svg)")


if __name__ == "__main__":
    combine()
