"""
================================================================================
 plot_stage2_topk.py — Stage2 Top-K feature selection curve
================================================================================
 Reads Results/Stage2/topk_results_[Target].csv, plots R²/Accuracy vs Top-K line chart.
 Marks the best K value point. Before = #5DA5DA (Steel blue), After = #C91511 (Crimson)
 Outputs PNG / PDF / SVG.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ==================== NC Journal publication plot global settings ====================
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER  = "#C91511"
COLOR_BEST   = "#DE1916"
COLOR_GRAY   = "#8C92AC"

DISPLAY_NAMES = {
    "LOI": "LOI", "UL94_Rating": "UL-94", "pHRR": "pHRR",
    "THR": "THR", "TSP": "TSP", "Flexural_Strength": "Flexural Strength",
}

TARGETS = ["LOI", "UL94_Rating", "pHRR", "THR", "TSP", "Flexural_Strength"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR = os.path.join(BASE_DIR, "Results", "Stage2")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot():
    for target in TARGETS:
        topk_file = os.path.join(STAGE2_DIR, f"topk_scan_{target}.csv")
        summary_file = os.path.join(STAGE2_DIR, "stage2_topk_summary.csv")

        if not os.path.exists(topk_file):
            print(f"  ⚠ Skipping {target}: TopK file not found")
            continue

        df = pd.read_csv(topk_file).sort_values("top_k")
        display_name = DISPLAY_NAMES.get(target, target)
        is_classification = "before_accuracy" in df.columns

        # Read best K
        best_k = None
        if os.path.exists(summary_file):
            sdf = pd.read_csv(summary_file)
            srow = sdf[sdf["target"] == target]
            if not srow.empty:
                best_k = int(srow.iloc[0]["best_k"])

        # ---- Figure 1: Before vs After dual-line comparison ----
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=600)

        if is_classification:
            metric_col = "accuracy"
            metric_label = "Test Accuracy"
        else:
            metric_col = "r2"
            metric_label = "Test $R^2$"

        ax.plot(df["top_k"], df[f"before_{metric_col}"],
                color=COLOR_BEFORE, linewidth=1.4, marker="o", markersize=3.5,
                label="Before (Literature only)")
        ax.plot(df["top_k"], df[f"after_{metric_col}"],
                color=COLOR_AFTER, linewidth=1.4, marker="s", markersize=3.5,
                label="After (Literature + Experiment)")

        # Mark best K
        if best_k is not None and best_k in df["top_k"].values:
            row = df[df["top_k"] == best_k].iloc[0]
            best_val = row[f"after_{metric_col}"]
            ax.scatter([best_k], [best_val], color=COLOR_BEST, s=50, zorder=5,
                       edgecolors="white", linewidth=1.0)
            ax.annotate(f"K={best_k}\n{best_val:.3f}",
                        (best_k, best_val), textcoords="offset points",
                        xytext=(8, -12), fontsize=14, color=COLOR_BEST,
                        fontweight="bold")

        ax.set_xlabel("Number of Features (Top K)", fontsize=13)
        ax.set_ylabel(metric_label, fontsize=13)
        ax.set_title(f"Feature Selection: {display_name}", fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(linestyle=":", alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Stage2_TopK_{target}.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print(f"✅ Stage2_TopK_{target}")


if __name__ == "__main__":
    plot()
