"""
================================================================================
 plot_pva.py — Predicted vs Actual scatter plot (before/after experiment guidance)
================================================================================
 Plots Before/After predicted vs actual scatter plots on the same fixed test set.
 Side-by-side comparison. Outputs PNG / PDF / SVG.
 Colors: Before = #5DA5DA (Steel blue), After = #C91511 (Crimson)
       y=x reference line = #8C92AC (Cool slate gray)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score
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
COLOR_LINE   = "#8C92AC"

REGRESSION_TARGETS = ["LOI", "pHRR", "THR", "TSP", "Flexural_Strength"]

DISPLAY_NAMES = {
    "LOI": "LOI",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
}

UNITS = {
    "LOI": "%",
    "pHRR": "kW/m^2",
    "THR": "MJ/m^2",
    "TSP": "m^2",
    "Flexural_Strength": "MPa",
}

# ==================== Path configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_pva():
    for target in REGRESSION_TARGETS:
        pred_file = os.path.join(RESULTS_DIR, "Stage3", f"predictions_{target}.csv")
        if not os.path.exists(pred_file):
            print(f"  [WARN] Skipping {target}: prediction file not found")
            continue

        df = pd.read_csv(pred_file)
        y_true = df["y_true"].values
        y_pred_before = df["y_pred_before"].values
        y_pred_after = df["y_pred_after"].values

        # Compute metrics
        r2_b = r2_score(y_true, y_pred_before)
        r2_a = r2_score(y_true, y_pred_after)
        rmse_b = np.sqrt(mean_squared_error(y_true, y_pred_before))
        rmse_a = np.sqrt(mean_squared_error(y_true, y_pred_after))

        unit = UNITS.get(target, "")
        display_name = DISPLAY_NAMES.get(target, target)

        # Unified axis range
        all_vals = np.concatenate([y_true, y_pred_before, y_pred_after])
        lim_min = all_vals.min() * 0.92 if all_vals.min() > 0 else all_vals.min() * 1.08
        lim_max = all_vals.max() * 1.08
        lims = [lim_min, lim_max]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=600)

        # ---- Before ----
        ax1.scatter(y_true, y_pred_before, c=COLOR_BEFORE, alpha=0.55,
                    s=22, edgecolors="none")
        ax1.plot(lims, lims, color=COLOR_LINE, linestyle="--", linewidth=1.0)
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_xlabel(f"Actual {display_name} ({unit})" if unit else f"Actual {display_name}",
                       fontsize=12)
        ax1.set_ylabel(f"Predicted {display_name} ({unit})" if unit else f"Predicted {display_name}",
                       fontsize=12)
        ax1.set_title("Before (Literature only)", fontsize=13, fontweight="bold",
                      color=COLOR_BEFORE)
        ax1.text(0.05, 0.93, f"$R^2$ = {r2_b:.3f}\nRMSE = {rmse_b:.2f}",
                 transform=ax1.transAxes, fontsize=14, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                           edgecolor="#cccccc"))
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.grid(linestyle=":", alpha=0.3)

        # ---- After ----
        ax2.scatter(y_true, y_pred_after, c=COLOR_AFTER, alpha=0.55,
                    s=22, edgecolors="none")
        ax2.plot(lims, lims, color=COLOR_LINE, linestyle="--", linewidth=1.0)
        ax2.set_xlim(lims)
        ax2.set_ylim(lims)
        ax2.set_xlabel(f"Actual {display_name} ({unit})" if unit else f"Actual {display_name}",
                       fontsize=12)
        ax2.set_ylabel(f"Predicted {display_name} ({unit})" if unit else f"Predicted {display_name}",
                       fontsize=12)
        ax2.set_title("After (Literature + Experiment)", fontsize=13, fontweight="bold",
                      color=COLOR_AFTER)
        ax2.text(0.05, 0.93, f"$R^2$ = {r2_a:.3f}\nRMSE = {rmse_a:.2f}",
                 transform=ax2.transAxes, fontsize=14, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85,
                           edgecolor="#cccccc"))
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.grid(linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"PvA_{target}.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print(f"[OK] PvA_{target} saved (png/pdf/svg)")

    # ---- UL94 binary classification: Confusion matrix comparison ----
    ul94_file = os.path.join(RESULTS_DIR, "Stage3", "predictions_UL94_Rating.csv")
    if os.path.exists(ul94_file):
        df = pd.read_csv(ul94_file)
        y_true = df["y_true"].values
        y_pred_before = df["y_pred_before"].values
        y_pred_after = df["y_pred_after"].values

        cm_b = confusion_matrix(y_true, y_pred_before, labels=[0, 1])
        cm_a = confusion_matrix(y_true, y_pred_after, labels=[0, 1])
        acc_b = accuracy_score(y_true, y_pred_before)
        acc_a = accuracy_score(y_true, y_pred_after)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6), dpi=600)

        for ax, cm, acc, title, color in [
            (ax1, cm_b, acc_b, "Before (Literature only)", COLOR_BEFORE),
            (ax2, cm_a, acc_a, "After (Literature + Experiment)", COLOR_AFTER),
        ]:
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=np.max([cm_b, cm_a]))
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred V-0", "Pred Non V-0"], fontsize=11)
            ax.set_yticklabels(["True V-0", "True Non V-0"], fontsize=11)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            fontsize=16, fontweight="bold",
                            color="white" if cm[i, j] > np.max(cm) / 2 else "#333333")
            ax.set_title(f"{title}\nAccuracy = {acc:.3f}", fontsize=12,
                         fontweight="bold", color=color, pad=8)

        fig.suptitle("UL-94 Classification (V-0 vs Non V-0): Confusion Matrix", fontsize=14,
                     fontweight="bold", y=1.02)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"PvA_UL94_Rating.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] PvA_UL94_Rating saved (png/pdf/svg)")
    else:
        print("[WARN] Skipping UL94_Rating: prediction file not found")


if __name__ == "__main__":
    print("Generating Predicted vs Actual scatter plot ...")
    plot_pva()
    print("Done.")
