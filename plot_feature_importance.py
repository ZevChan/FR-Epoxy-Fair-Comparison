"""
================================================================================
 plot_feature_importance.py — Feature importance shift (before/after experiment guidance)
================================================================================
 Reads feature_importance_[Target].csv, draws Slope Charts,
 showing how the model's reliance on features was re-ranked after introducing experimental data.

 Before = #5DA5DA (Steel blue), After = #C91511 (Crimson)
 Outputs PNG / PDF / SVG.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ==================== Color definitions ====================
COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER  = "#C91511"
COLOR_INCREASE = "#C91511"  # Importance increase → red
COLOR_DECREASE = "#5DA5DA"  # Importance decrease → blue
COLOR_GRAY = "#8C92AC"

TARGETS = ["LOI", "pHRR", "THR", "TSP", "Flexural_Strength", "UL94_Rating"]

DISPLAY_NAMES = {
    "LOI": "LOI",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
    "UL94_Rating": "UL-94",
}

# ==================== Paths ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_importance_slope():
    for target in TARGETS:
        imp_file = os.path.join(RESULTS_DIR, "Stage3", f"feature_importance_{target}.csv")
        if not os.path.exists(imp_file):
            print(f"  [WARN] Skipping {target}: feature importance file not found")
            continue

        df = pd.read_csv(imp_file)
        # Take top-15 (sorted by After weight)
        df = df.head(15).copy()
        # Short feature name
        df["ShortName"] = df["Feature"].str[:35]

        display_name = DISPLAY_NAMES.get(target, target)

        fig, ax = plt.subplots(figsize=(7, 6.5), dpi=600)

        y_pos = np.arange(len(df))

        for i, (_, row) in enumerate(df.iterrows()):
            w_b = row["Weight_Before"]
            w_a = row["Weight_After"]
            delta = w_a - w_b
            color = COLOR_INCREASE if delta >= 0 else COLOR_DECREASE
            lw = 0.6 + abs(delta) * 15  # Line width proportional to change magnitude
            lw = np.clip(lw, 0.8, 3.5)

            ax.plot([w_b, w_a], [i, i], color=color, linewidth=lw, alpha=0.75,
                    solid_capstyle="round")
            ax.scatter([w_b], [i], color=COLOR_BEFORE, s=35, zorder=5,
                       edgecolors="white", linewidth=0.5)
            ax.scatter([w_a], [i], color=COLOR_AFTER, s=35, zorder=5,
                       edgecolors="white", linewidth=0.5)

        # Decoration
        ax.scatter([], [], color=COLOR_BEFORE, s=40, label="Before (Literature only)")
        ax.scatter([], [], color=COLOR_AFTER, s=40, label="After (Literature + Experiment)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["ShortName"], fontsize=10)
        ax.set_xlabel("Feature Importance Weight", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12, width=0.8)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("bold")
        ax.set_title(f"Feature Importance Shift: {display_name}",
                     fontsize=14, fontweight="bold")
        ax.legend(frameon=False, fontsize=14, loc="lower right", bbox_to_anchor=(0.98, 0.05))
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Importance_Slope_{target}.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print(f"[OK] Importance_Slope_{target}")


if __name__ == "__main__":
    print("Generating feature importance slope chart ...")
    plot_importance_slope()
    print("Done.")
