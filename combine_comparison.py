"""
================================================================================
 combine_comparison.py — 3-Row Comprehensive Comparison Chart
 Row 1: (a) Metrics_Bar_R2_no_pHRR | (b) Metrics_Bar_RMSE_no_pHRR | (c) UL94
 Row 2: (d) PvA_LOI                   | (e) PvA_UL94_Rating
 Row 3: (f) Error_Density_LOI | (g) THR | (h) TSP | (i) Flexural Strength
================================================================================
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix

# ══════════════════════════════════════════════════════════════════
FS = 22
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.labelsize": FS + 1, "axes.titlesize": FS + 1,
    "legend.fontsize": FS - 4, "figure.titlesize": FS + 4,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.labelsize": FS - 2, "ytick.labelsize": FS - 2,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_BEFORE = "#5DA5DA";  COLOR_AFTER = "#C91511"
COLOR_LINE   = "#8C92AC"

SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
STAGE1_DIR = os.path.join(RESULTS_DIR, "Stage1")
STAGE3_DIR = os.path.join(RESULTS_DIR, "Stage3")
os.makedirs(SAVE_DIR, exist_ok=True)

DISPLAY_NAMES = {"LOI":"LOI","pHRR":"pHRR","THR":"THR","TSP":"TSP",
                 "Flexural_Strength":"Flexural Strength","UL94_Rating":"UL-94"}
UNITS = {"LOI":"%","pHRR":"kW/m\u00b2","THR":"MJ/m\u00b2",
         "TSP":"m\u00b2","Flexural_Strength":"MPa"}

def bold_axes(ax):
    for lbl in ax.get_xticklabels(): lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels(): lbl.set_fontweight("bold")
    if ax.xaxis.label: ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label: ax.yaxis.label.set_fontweight("bold")

def panel_label(ax, letter):
    ax.text(-0.12, 1.03, f"({letter})", transform=ax.transAxes,
            fontsize=FS + 6, fontweight="bold", va="bottom", ha="right")

# ══════════════════════════════════════════════════════════════════
def combine():
    # ── Load data ─────────────────────────────────────────────────
    # Stage 3 metrics (for R², RMSE, UL94 bar charts)
    mdf = pd.read_csv(os.path.join(STAGE3_DIR, "stage3_optuna_results.csv"))
    reg_df = mdf[mdf["target"] != "UL94_Rating"]
    r2no = reg_df[reg_df["target"] != "pHRR"]
    cls_row = mdf[mdf["target"] == "UL94_Rating"].iloc[0]
    r2_targets = r2no["target"].tolist()
    r2_labels = [DISPLAY_NAMES.get(t, t) for t in r2_targets]
    w_bar = 0.32

    # PvA predictions
    pva_loi = pd.read_csv(os.path.join(STAGE3_DIR, "predictions_LOI.csv"))
    pva_ul94 = pd.read_csv(os.path.join(STAGE3_DIR, "predictions_UL94_Rating.csv"))
    yt_loi = pva_loi["y_true"].values
    ypb_loi = pva_loi["y_pred_before"].values
    ypa_loi = pva_loi["y_pred_after"].values
    all_loi = np.concatenate([yt_loi, ypb_loi, ypa_loi])
    lmin, lmax = all_loi.min() * 0.92, all_loi.max() * 1.08
    lims_loi = [lmin, lmax]
    r2b_loi = r2_score(yt_loi, ypb_loi); r2a_loi = r2_score(yt_loi, ypa_loi)
    rmb_loi = np.sqrt(mean_squared_error(yt_loi, ypb_loi))
    rma_loi = np.sqrt(mean_squared_error(yt_loi, ypa_loi))

    yt_ul = pva_ul94["y_true"].values
    ypb_ul = pva_ul94["y_pred_before"].values
    ypa_ul = pva_ul94["y_pred_after"].values
    cm_b = confusion_matrix(yt_ul, ypb_ul, labels=[0, 1])
    cm_a = confusion_matrix(yt_ul, ypa_ul, labels=[0, 1])
    ul_labels = ["TN\n(True V-0)", "FP\n(False\nNon V-0)",
                 "FN\n(False V-0)", "TP\n(True\nNon V-0)"]
    cb = [cm_b[0, 0], cm_b[0, 1], cm_b[1, 0], cm_b[1, 1]]
    ca = [cm_a[0, 0], cm_a[0, 1], cm_a[1, 0], cm_a[1, 1]]

    # Error density targets
    error_targets = ["LOI", "THR", "TSP", "Flexural_Strength"]

    # ═══════════ Figure ═══════════════════════════════════════════
    fig = plt.figure(figsize=(24, 18), dpi=150)
    outer = fig.add_gridspec(3, 1, figure=fig,
                              hspace=0.42, left=0.06, right=0.98,
                              top=0.97, bottom=0.04,
                              height_ratios=[1.0, 1.2, 1.0])

    # ====== ROW 1: (a) R² no pHRR | (b) RMSE no pHRR | (c) UL94 ======
    row1 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0],
                                   width_ratios=[4, 4, 2.5], wspace=0.28)
    xn = np.arange(len(r2_targets))

    # (a) R² no pHRR
    ax_a = fig.add_subplot(row1[0])
    b1a = ax_a.bar(xn - w_bar/2, r2no["before_test_r2"], w_bar, color=COLOR_BEFORE,
                   edgecolor="white", linewidth=0.5,
                   label="Before (Literature only)")
    b2a = ax_a.bar(xn + w_bar/2, r2no["after_test_r2"], w_bar, color=COLOR_AFTER,
                   edgecolor="white", linewidth=0.5,
                   label="After (Literature + Experiment)")
    for bars, c in [(b1a, COLOR_BEFORE), (b2a, COLOR_AFTER)]:
        for b in bars:
            ax_a.text(b.get_x() + b.get_width()/2, b.get_height() + 0.012,
                      f"{b.get_height():.3f}", ha="center", va="bottom",
                      fontsize=FS - 4, color=c, fontweight="bold")
    ax_a.set_xticks(xn)
    ax_a.set_xticklabels(r2_labels, rotation=30, ha="right", fontsize=FS - 4)
    ax_a.set_ylabel("Test $R^2$", fontweight="bold")
    ax_a.set_ylim(0, 1.35)
    ax_a.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 4,
                loc="upper right")
    ax_a.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_a); panel_label(ax_a, "a")

    # (b) RMSE no pHRR
    ax_b = fig.add_subplot(row1[1])
    b1b = ax_b.bar(xn - w_bar/2, r2no["before_test_rmse"], w_bar, color=COLOR_BEFORE,
                   edgecolor="white", linewidth=0.5,
                   label="Before (Literature only)")
    b2b = ax_b.bar(xn + w_bar/2, r2no["after_test_rmse"], w_bar, color=COLOR_AFTER,
                   edgecolor="white", linewidth=0.5,
                   label="After (Literature + Experiment)")
    for bars, c in [(b1b, COLOR_BEFORE), (b2b, COLOR_AFTER)]:
        for b in bars:
            ax_b.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                      f"{b.get_height():.1f}", ha="center", va="bottom",
                      fontsize=FS - 4, color=c, fontweight="bold")
    ax_b.set_xticks(xn)
    ax_b.set_xticklabels(r2_labels, rotation=30, ha="right", fontsize=FS - 4)
    ax_b.set_ylabel("Test RMSE", fontweight="bold")
    all_h_b = [b.get_height() for b in b1b] + [b.get_height() for b in b2b]
    ax_b.set_ylim(0, (max(all_h_b) + 0.3) * 1.45)
    ax_b.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 4,
                loc="upper right")
    ax_b.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_b); panel_label(ax_b, "b")

    # (c) UL94
    ax_c = fig.add_subplot(row1[2])
    m_bv = [cls_row["before_test_accuracy"], cls_row["before_test_f1"]]
    m_av = [cls_row["after_test_accuracy"], cls_row["after_test_f1"]]
    xm = np.arange(2)
    ax_c.bar(xm - w_bar/2, m_bv, w_bar, color=COLOR_BEFORE, edgecolor="white",
             linewidth=0.5, label="Before (Literature only)")
    ax_c.bar(xm + w_bar/2, m_av, w_bar, color=COLOR_AFTER, edgecolor="white",
             linewidth=0.5, label="After (Literature + Experiment)")
    for i, (bv, av) in enumerate(zip(m_bv, m_av)):
        ax_c.text(i - w_bar/2, bv + 0.015, f"{bv:.3f}", ha="center",
                  fontsize=FS - 3, color=COLOR_BEFORE, fontweight="bold")
        ax_c.text(i + w_bar/2, av + 0.015, f"{av:.3f}", ha="center",
                  fontsize=FS - 3, color=COLOR_AFTER, fontweight="bold")
    ax_c.set_xticks(xm); ax_c.set_xticklabels(["Accuracy", "F1"])
    ax_c.set_ylabel("Score", fontweight="bold")
    ax_c.set_ylim(0, 1.35)
    ax_c.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 4)
    ax_c.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_c); panel_label(ax_c, "c")

    # ====== ROW 2: (d) PvA_LOI | (e) PvA_UL94  ======
    row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.30)

    # (d) PvA_LOI — split internally into Before/After
    gs_d = GridSpecFromSubplotSpec(1, 2, subplot_spec=row2[0], wspace=0.20)
    for col_idx, (yp, r2v, rmse, color, tag) in enumerate([
        (ypb_loi, r2b_loi, rmb_loi, COLOR_BEFORE, "Before (Literature only)"),
        (ypa_loi, r2a_loi, rma_loi, COLOR_AFTER, "After (Literature + Experiment)"),
    ]):
        ax_di = fig.add_subplot(gs_d[col_idx])
        ax_di.scatter(yt_loi, yp, c=color, alpha=0.55, s=18, edgecolors="none")
        ax_di.plot(lims_loi, lims_loi, color=COLOR_LINE, linestyle="--", linewidth=1.0)
        ax_di.set_xlim(lims_loi); ax_di.set_ylim(lims_loi)
        ax_di.set_xlabel("Actual LOI (%)", fontweight="bold")
        ax_di.set_ylabel("Predicted LOI (%)", fontweight="bold")
        ax_di.set_title(tag, fontsize=FS - 2, fontweight="bold", color=color, pad=10)
        ax_di.text(0.05, 0.93, f"$R^2$ = {r2v:.3f}\nRMSE = {rmse:.2f}",
                   transform=ax_di.transAxes, fontsize=FS - 4, va="top",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                             alpha=0.85, edgecolor="#cccccc"))
        ax_di.grid(linestyle=":", alpha=0.3)
        bold_axes(ax_di)
        if col_idx == 0:
            ax_d_list = []
        ax_d_list.append(ax_di)
    panel_label(ax_d_list[0], "d")

    # (e) PvA_UL94 — confusion matrix bars
    max_c_val = max(max(cb), max(ca))
    gs_e = GridSpecFromSubplotSpec(1, 2, subplot_spec=row2[1], wspace=0.30)
    for col_idx, (counts, color, tag) in enumerate([
        (cb, COLOR_BEFORE, "Before (Literature only)"),
        (ca, COLOR_AFTER, "After (Literature + Experiment)"),
    ]):
        ax_ei = fig.add_subplot(gs_e[col_idx])
        xc = np.arange(4)
        ax_ei.bar(xc, counts, color=color, edgecolor="white", linewidth=0.5, width=0.65)
        for i, v in enumerate(counts):
            ax_ei.text(i, v + max_c_val*0.02, str(v), ha="center", va="bottom",
                       fontsize=FS - 5, color=color, fontweight="bold")
        ax_ei.set_xticks(xc)
        ax_ei.set_xticklabels(ul_labels, fontsize=FS - 7, fontweight="bold")
        ax_ei.set_ylabel("Count", fontweight="bold")
        ax_ei.set_title(tag, fontsize=FS - 2, fontweight="bold", color=color, pad=10)
        ax_ei.set_ylim(0, max_c_val * 1.20)
        ax_ei.grid(axis="y", linestyle=":", alpha=0.3)
        bold_axes(ax_ei)
        if col_idx == 0:
            ax_e_list = []
        ax_e_list.append(ax_ei)
    panel_label(ax_e_list[0], "e")

    # ====== ROW 3: (f) Error_LOI | (g) Error_THR | (h) Error_TSP | (i) Error_FS ======
    row3 = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[2], wspace=0.32)
    for col_idx, (letter, target) in enumerate(
        zip(["f", "g", "h", "i"], error_targets)
    ):
        df_ed = pd.read_csv(os.path.join(STAGE3_DIR, f"predictions_{target}.csv"))
        yt = df_ed["y_true"].values
        ypb = df_ed["y_pred_before"].values
        ypa = df_ed["y_pred_after"].values
        rb = yt - ypb; ra = yt - ypa
        dname = DISPLAY_NAMES.get(target, target)
        unit = UNITS.get(target, "")

        ax = fig.add_subplot(row3[col_idx])
        try:
            kb = gaussian_kde(rb); ka = gaussian_kde(ra)
            ar = np.concatenate([rb, ra])
            xr = np.linspace(ar.min() * 1.15, ar.max() * 1.15, 300)
            ax.fill_between(xr, kb(xr), alpha=0.25, color=COLOR_BEFORE)
            ax.plot(xr, kb(xr), color=COLOR_BEFORE, linewidth=1.8,
                    label="Before (Literature only)")
            ax.fill_between(xr, ka(xr), alpha=0.35, color=COLOR_AFTER)
            ax.plot(xr, ka(xr), color=COLOR_AFTER, linewidth=1.8,
                    label="After (Literature + Experiment)")
        except Exception:
            ax.hist(rb, bins=20, density=True, alpha=0.35, color=COLOR_BEFORE,
                    edgecolor="white", linewidth=0.3)
            ax.hist(ra, bins=20, density=True, alpha=0.45, color=COLOR_AFTER,
                    edgecolor="white", linewidth=0.3)
        ax.axvline(0, color=COLOR_LINE, linestyle="--", linewidth=1.0, alpha=0.8)
        xl = f"Residual ({unit})" if unit else "Residual"
        ax.set_xlabel(xl, fontweight="bold")
        ax.set_ylabel("Probability Density", fontweight="bold")
        ax.set_title(dname, fontsize=FS, fontweight="bold", pad=12)
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=FS - 5, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
        bold_axes(ax)
        panel_label(ax, letter)

    # ── Save ──────────────────────────────────────────────────────
    print("Saving ...")
    for fmt in ["png", "pdf", "svg"]:
        out = os.path.join(SAVE_DIR, f"Combined_Comparison.{fmt}")
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.08)
        print(f"  -> {fmt}")
    plt.close(fig)
    print("[OK] Combined_Comparison saved (png/pdf/svg)")


if __name__ == "__main__":
    combine()
