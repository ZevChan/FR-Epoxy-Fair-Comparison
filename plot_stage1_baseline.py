"""
================================================================================
 plot_stage1_baseline.py — Stage1 default parameter XGBoost baseline comparison
================================================================================
 Reads Results/Stage1/stage1_baseline_results.csv, draws Before vs After bar chart.
 Before = #5DA5DA (Steel blue), After = #C91511 (Crimson)
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

DISPLAY_NAMES = {
    "LOI": "LOI", "UL94_Rating": "UL-94", "pHRR": "pHRR",
    "THR": "THR", "TSP": "TSP", "Flexural_Strength": "Flexural Strength",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "Results", "Stage1", "stage1_baseline_results.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot():
    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        print("No data. Please run fair_holdout_comparison.py first.")
        return

    reg_df = df[df["target"] != "UL94_Rating"].copy()
    cls_df = df[df["target"] == "UL94_Rating"].copy()

    # ---- R² ----
    if not reg_df.empty:
        targets = reg_df["target"].tolist()
        labels = [DISPLAY_NAMES.get(t, t) for t in targets]
        x = np.arange(len(targets))
        width = 0.32

        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=600)
        b1 = ax.bar(x - width/2, reg_df["before_test_r2"], width,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                    label="Before (Literature only)")
        b2 = ax.bar(x + width/2, reg_df["after_test_r2"], width,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                    label="After (Literature + Experiment)")
        for b in b1:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=13, color=COLOR_BEFORE)
        for b in b2:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=13, color=COLOR_AFTER)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Test $R^2$ (Default XGBoost)", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Stage1_Baseline_R2.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Stage1_Baseline_R2")

        # ---- R² no pHRR ----
        reg_np_r2 = reg_df[reg_df["target"] != "pHRR"]
        t_np_r2 = reg_np_r2["target"].tolist()
        l_np_r2 = [DISPLAY_NAMES.get(t, t) for t in t_np_r2]
        xn_r2 = np.arange(len(t_np_r2))
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)
        b1 = ax.bar(xn_r2 - width/2, reg_np_r2["before_test_r2"], width,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                    label="Before (Literature only)")
        b2 = ax.bar(xn_r2 + width/2, reg_np_r2["after_test_r2"], width,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                    label="After (Literature + Experiment)")
        for b in b1:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=13, color=COLOR_BEFORE)
        for b in b2:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=13, color=COLOR_AFTER)
        ax.set_xticks(xn_r2)
        ax.set_xticklabels(l_np_r2)
        ax.set_ylabel("Test $R^2$ (Default XGBoost)", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Stage1_Baseline_R2_no_pHRR.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Stage1_Baseline_R2_no_pHRR")

        # ---- RMSE ----
        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=600)
        b1 = ax.bar(x - width/2, reg_df["before_test_rmse"], width,
               color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
               label="Before (Literature only)")
        b2 = ax.bar(x + width/2, reg_df["after_test_rmse"], width,
               color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
               label="After (Literature + Experiment)")
        for b in b1:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9, color=COLOR_BEFORE)
        for b in b2:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9, color=COLOR_AFTER)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Test RMSE (Default XGBoost)", fontsize=13)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.margins(y=0.12)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Stage1_Baseline_RMSE.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Stage1_Baseline_RMSE")

        # ---- RMSE no pHRR ----
        reg_np = reg_df[reg_df["target"] != "pHRR"]
        t_np = reg_np["target"].tolist()
        l_np = [DISPLAY_NAMES.get(t, t) for t in t_np]
        xn = np.arange(len(t_np))
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)
        b1 = ax.bar(xn - width/2, reg_np["before_test_rmse"], width,
               color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
               label="Before (Literature only)")
        b2 = ax.bar(xn + width/2, reg_np["after_test_rmse"], width,
               color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
               label="After (Literature + Experiment)")
        for b in b1:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=12, color=COLOR_BEFORE)
        for b in b2:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=12, color=COLOR_AFTER)
        ax.set_xticks(xn)
        ax.set_xticklabels(l_np)
        ax.set_ylabel("Test RMSE (Default XGBoost)", fontsize=13)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        # Expand Y-axis upper limit to leave room for value labels
        all_h = [b.get_height() for b in b1] + [b.get_height() for b in b2]
        top = max(all_h) + 0.2
        ax.set_ylim(0, top * 1.45)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Stage1_Baseline_RMSE_no_pHRR.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Stage1_Baseline_RMSE_no_pHRR")

    # ---- UL94 ----
    if not cls_df.empty:
        metrics = ["Accuracy", "F1"]
        before_vals = [cls_df.iloc[0]["before_test_accuracy"],
                       cls_df.iloc[0]["before_test_f1"]]
        after_vals = [cls_df.iloc[0]["after_test_accuracy"],
                      cls_df.iloc[0]["after_test_f1"]]
        x = np.arange(2)
        width = 0.32
        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=600)
        ax.bar(x - width/2, before_vals, width, color=COLOR_BEFORE,
               edgecolor="white", linewidth=0.5, label="Before")
        ax.bar(x + width/2, after_vals, width, color=COLOR_AFTER,
               edgecolor="white", linewidth=0.5, label="After")
        for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
            ax.text(i - width/2, bv + 0.015, f"{bv:.3f}", ha="center", fontsize=14,
                    color=COLOR_BEFORE)
            ax.text(i + width/2, av + 0.015, f"{av:.3f}", ha="center", fontsize=14,
                    color=COLOR_AFTER)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_title("UL-94 (Default XGBoost)", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Stage1_Baseline_UL94.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Stage1_Baseline_UL94")


if __name__ == "__main__":
    plot()
