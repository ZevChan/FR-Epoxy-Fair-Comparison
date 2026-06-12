"""
================================================================================
 plot_optuna_history.py — Optuna hyperparameter optimization history
================================================================================
 Reads optuna_history_before/after_[Target].csv, plots:
   1. Optimization history curve (Objective Value vs Trial)
   2. Hyperparameter parallel coordinates plot
 Outputs PNG / PDF / SVG.
 Before = #5DA5DA (Steel blue), After = #C91511 (Crimson)
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
COLOR_BEST_BEFORE = "#3A7CB8"
COLOR_BEST_AFTER  = "#A0100E"
COLOR_GRAY = "#8C92AC"

TARGETS = ["LOI", "UL94_Rating", "pHRR", "THR", "TSP", "Flexural_Strength"]

DISPLAY_NAMES = {
    "LOI": "LOI",
    "UL94_Rating": "UL-94",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
}

# ==================== Paths ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_optuna_history():
    for target in TARGETS:
        hist_before_p = os.path.join(RESULTS_DIR, "Stage3", f"optuna_history_before_{target}.csv")
        hist_after_p  = os.path.join(RESULTS_DIR, "Stage3", f"optuna_history_after_{target}.csv")

        if not os.path.exists(hist_before_p) or not os.path.exists(hist_after_p):
            print(f"  ⚠ Skipping {target}: history files missing")
            continue

        df_before = pd.read_csv(hist_before_p)
        df_after  = pd.read_csv(hist_after_p)

        display_name = DISPLAY_NAMES.get(target, target)

        value_col = "value"
        if value_col not in df_before.columns:
            # Optuna may use "values_0" or other naming
            val_cols = [c for c in df_before.columns if "value" in c.lower()]
            value_col = val_cols[0] if val_cols else df_before.columns[-1]

        # ---- Figure 1: Optimization history ----
        fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=600)

        trials_b = np.arange(1, len(df_before) + 1)
        trials_a = np.arange(1, len(df_after) + 1)

        ax.plot(trials_b, df_before[value_col], color=COLOR_BEFORE, linewidth=1.2,
                alpha=0.7, label="Before (Literature only)")
        ax.plot(trials_a, df_after[value_col], color=COLOR_AFTER, linewidth=1.2,
                alpha=0.7, label="After (Literature + Experiment)")

        # Best value horizontal line
        best_b = df_before[value_col].max()
        best_a = df_after[value_col].max()
        ax.axhline(best_b, color=COLOR_BEST_BEFORE, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(best_a, color=COLOR_BEST_AFTER, linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_xlabel("Trial Number", fontsize=13)
        ax.set_ylabel("Best CV Score", fontsize=13)
        ax.set_title(f"Optimization History: {display_name}",
                     fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Optuna_History_{target}.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print(f"✅ Optuna_History_{target}")

        # ---- Figure 2: Hyperparameter importance ----
        # Extract hyperparameter columns
        param_cols = [c for c in df_before.columns
                      if c.startswith("params_") and df_before[c].nunique() > 1]
        if len(param_cols) >= 3:
            # Compute correlation of each hyperparameter with the target value as "importance"
            fig, axes = plt.subplots(1, len(param_cols), figsize=(2.5 * len(param_cols), 3), dpi=600)
            if len(param_cols) == 1:
                axes = [axes]

            for i, col in enumerate(param_cols):
                ax_i = axes[i]
                param_name = col.replace("params_", "")
                corr_b = df_before[col].corr(df_before[value_col]) if df_before[col].dtype in ["float64", "int64"] else 0
                corr_a = df_after[col].corr(df_after[value_col]) if df_after[col].dtype in ["float64", "int64"] else 0

                x_pos = [0, 1]
                ax_i.bar(x_pos, [corr_b, corr_a], color=[COLOR_BEFORE, COLOR_AFTER],
                         edgecolor="white", linewidth=0.5, width=0.5)
                ax_i.set_xticks(x_pos)
                ax_i.set_xticklabels(["Before", "After"], fontsize=13, rotation=30)
                ax_i.set_title(param_name, fontsize=14)
                ax_i.axhline(0, color=COLOR_GRAY, linewidth=0.5)
                ax_i.grid(axis="y", linestyle=":", alpha=0.3)

            fig.suptitle(f"Hyperparameter–Score Correlation: {display_name}",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            for fmt in ["png", "pdf", "svg"]:
                plt.savefig(os.path.join(SAVE_DIR, f"Optuna_Params_{target}.{fmt}"),
                            dpi=600, bbox_inches="tight")
            plt.close()
            print(f"✅ Optuna_Params_{target}")


if __name__ == "__main__":
    print("Generating Optuna optimization history plot ...")
    plot_optuna_history()
    print("Done.")
