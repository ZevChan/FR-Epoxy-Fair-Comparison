"""
================================================================================
 plot_metrics_bar.py — Performance comparison bar chart (before/after experiment guidance)
================================================================================
 Reads results from fair_holdout_comparison.py, draws NC journal-grade grouped bar charts.
 Outputs PNG / PDF / SVG.
 Colors: Before = #5DA5DA (Steel blue), After = #C91511 (Crimson)
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

# ==================== Color definitions ====================
COLOR_BEFORE = "#5DA5DA"   # Steel blue — Literature baseline
COLOR_AFTER  = "#C91511"   # Crimson — After experiment guidance

# Target display name mapping
DISPLAY_NAMES = {
    "LOI": "LOI",
    "UL94_Rating": "UL-94",
    "pHRR": "pHRR",
    "THR": "THR",
    "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength",
}

# ==================== Path configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BASE_DIR, "Results", "Stage3", "stage3_optuna_results.csv")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_metrics_bar():
    df = pd.read_csv(RESULTS_FILE)
    if df.empty:
        print("No result data. Please run fair_holdout_comparison.py first.")
        return

    # Process regression and classification separately
    reg_df = df[df["target"] != "UL94_Rating"].copy()
    cls_df = df[df["target"] == "UL94_Rating"].copy()

    # ---- Figure 1: R² comparison (regression targets) ----
    if not reg_df.empty:
        targets = reg_df["target"].tolist()
        labels = [DISPLAY_NAMES.get(t, t) for t in targets]
        x = np.arange(len(targets))
        width = 0.32

        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=600)

        bars1 = ax.bar(x - width / 2, reg_df["before_test_r2"], width,
                       color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                       label="Before (Literature only)")
        bars2 = ax.bar(x + width / 2, reg_df["after_test_r2"], width,
                       color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                       label="After (Literature + Experiment)")

        # Value annotations
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=13, color=COLOR_BEFORE)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=13, color=COLOR_AFTER)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Test $R^2$", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=14, loc="lower right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Metrics_Bar_R2.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Metrics_Bar_R2 saved (png/pdf/svg)")

        # ---- Figure 1b: R² comparison (excluding pHRR) ----
        reg_no_phrr = reg_df[reg_df["target"] != "pHRR"]
        targets_np = reg_no_phrr["target"].tolist()
        labels_np = [DISPLAY_NAMES.get(t, t) for t in targets_np]
        x_np = np.arange(len(targets_np))

        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)
        b1 = ax.bar(x_np - width/2, reg_no_phrr["before_test_r2"], width,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                    label="Before (Literature only)")
        b2 = ax.bar(x_np + width/2, reg_no_phrr["after_test_r2"], width,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                    label="After (Literature + Experiment)")
        for b in b1:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=13, color=COLOR_BEFORE)
        for b in b2:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.012, f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=13, color=COLOR_AFTER)
        ax.set_xticks(x_np)
        ax.set_xticklabels(labels_np)
        ax.set_ylabel("Test $R^2$", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Metrics_Bar_R2_no_pHRR.{fmt}"), dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Metrics_Bar_R2_no_pHRR saved (png/pdf/svg)")

        # ---- Figure 2: RMSE comparison (regression targets) ----
        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=600)

        bars1 = ax.bar(x - width / 2, reg_df["before_test_rmse"], width,
                       color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                       label="Before (Literature only)")
        bars2 = ax.bar(x + width / 2, reg_df["after_test_rmse"], width,
                       color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                       label="After (Literature + Experiment)")

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=COLOR_BEFORE)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=COLOR_AFTER)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Test RMSE", fontsize=13)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=14, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Metrics_Bar_RMSE.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Metrics_Bar_RMSE saved (png/pdf/svg)")

        # ---- Figure 2b: RMSE comparison (excluding pHRR) ----
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)
        b1 = ax.bar(x_np - width/2, reg_no_phrr["before_test_rmse"], width,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                    label="Before (Literature only)")
        b2 = ax.bar(x_np + width/2, reg_no_phrr["after_test_rmse"], width,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                    label="After (Literature + Experiment)")
        for b in b1:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=12, color=COLOR_BEFORE)
        for b in b2:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=12, color=COLOR_AFTER)
        ax.set_xticks(x_np)
        ax.set_xticklabels(labels_np)
        ax.set_ylabel("Test RMSE", fontsize=13)
        all_rmse_h = [b.get_height() for b in b1] + [b.get_height() for b in b2]
        top_rmse = max(all_rmse_h) + 0.3
        ax.set_ylim(0, top_rmse * 1.45)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=14, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Metrics_Bar_RMSE_no_pHRR.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Metrics_Bar_RMSE_no_pHRR saved (png/pdf/svg)")

        # ---- Figure 3: MAE comparison (regression targets) ----
        fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=600)

        bars1 = ax.bar(x - width / 2, reg_df["before_test_mae"], width,
                       color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                       label="Before (Literature only)")
        bars2 = ax.bar(x + width / 2, reg_df["after_test_mae"], width,
                       color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                       label="After (Literature + Experiment)")

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=COLOR_BEFORE)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9, color=COLOR_AFTER)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Test MAE", fontsize=13)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=14, loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.margins(y=0.15)  # Leave space for legend

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Metrics_Bar_MAE.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Metrics_Bar_MAE saved (png/pdf/svg)")

        # ---- Figure 3b: MAE comparison (excluding pHRR) ----
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)
        b1 = ax.bar(x_np - width/2, reg_no_phrr["before_test_mae"], width,
                    color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
                    label="Before (Literature only)")
        b2 = ax.bar(x_np + width/2, reg_no_phrr["after_test_mae"], width,
                    color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
                    label="After (Literature + Experiment)")
        for b in b1:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=12, color=COLOR_BEFORE)
        for b in b2:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=12, color=COLOR_AFTER)
        ax.set_xticks(x_np)
        ax.set_xticklabels(labels_np)
        ax.set_ylabel("Test MAE", fontsize=13)
        # Expand Y-axis upper limit to leave room for legend and value labels
        all_heights = [b.get_height() for b in b1] + [b.get_height() for b in b2]
        top_val = max(all_heights) + 0.3
        ax.set_ylim(0, top_val * 1.45)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14, loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Metrics_Bar_MAE_no_pHRR.{fmt}"), dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Metrics_Bar_MAE_no_pHRR saved (png/pdf/svg)")

    # ---- Figure 4: UL94 classification metrics ----
    if not cls_df.empty:
        metrics = ["Accuracy", "F1"]
        before_vals = [cls_df.iloc[0]["before_test_accuracy"],
                       cls_df.iloc[0]["before_test_f1"]]
        after_vals = [cls_df.iloc[0]["after_test_accuracy"],
                      cls_df.iloc[0]["after_test_f1"]]

        x = np.arange(len(metrics))
        width = 0.32

        fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=600)

        ax.bar(x - width / 2, before_vals, width,
               color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
               label="Before")
        ax.bar(x + width / 2, after_vals, width,
               color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
               label="After")

        for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
            ax.text(i - width / 2, bv + 0.015, f"{bv:.3f}",
                    ha="center", fontsize=14, color=COLOR_BEFORE)
            ax.text(i + width / 2, av + 0.015, f"{av:.3f}",
                    ha="center", fontsize=14, color=COLOR_AFTER)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Score", fontsize=13)
        ax.set_title("UL-94 Rating", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.3)

        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Metrics_Bar_UL94.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print("[OK] Metrics_Bar_UL94 saved (png/pdf/svg)")


if __name__ == "__main__":
    plot_metrics_bar()
