"""
================================================================================
 combine_panels.py — 3-Row Multi-Panel Combined Figure (Large Font, No Ghost Axes)
================================================================================
 Figure (a) UMAP Descriptors (No title)
 Figure (b) Metrics_Bar_R2_no_pHRR
 Figure (c) Metrics_Bar_UL94 (No title)
 Figure (d) PvA_LOI (Before + After, Square)
 Figure (e) PvA_UL94_Rating (Before + After Bar Chart)
 Figure (f) Error_Density_LOI (Title embedded)
 Figure (g) Error_Density_TSP (Title embedded)
 Figure (h) Error_Density_Flexural_Strength (Title embedded)
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
import matplotlib.ticker as ticker
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix
import umap, seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

# ══════════════════════════════════════════════════════════════════
FS = 16   # Global font size — clear and moderate in Word
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FS,
    "axes.labelsize": FS + 2,
    "axes.titlesize": FS + 2,
    "legend.fontsize": FS - 2,
    "figure.titlesize": FS + 4,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_BEFORE = "#5DA5DA";  COLOR_AFTER = "#C91511"
COLOR_LIT    = "#5DA5DA";  COLOR_EXP   = "#C91511"
COLOR_LINE   = "#8C92AC"

SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")
STAGE3_DIR = os.path.join(OUTPUT_DIR, "Stage3")
os.makedirs(SAVE_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(OUTPUT_DIR, "Stage3", "stage3_optuna_results.csv")

DISPLAY_NAMES = {
    "LOI": "LOI", "pHRR": "pHRR", "THR": "THR", "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength", "UL94_Rating": "UL-94",
}
UNITS = {
    "LOI": "%", "pHRR": "kW m\u207b\u00b2", "THR": "MJ m\u207b\u00b2",
    "TSP": "m\u00b2", "Flexural_Strength": "MPa",
}

SMILES_COLS = ["EPOXY STRUCTURE", "Flame_retardant", "Curing_agent ",
               "Other_Material_1", "Other_Material_2"]
_cache = {}

def _smiles_to_vec(smiles):
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None
    s = smiles.strip()
    if s in _cache: return _cache[s]
    mol = Chem.MolFromSmiles(s)
    if mol is None: _cache[s] = None; return None
    d = Descriptors.CalcMolDescriptors(mol)
    vec = np.array(list(d.values()), dtype=np.float64)
    _cache[s] = vec; return vec

def _encode_col(series):
    vecs, none_idx = [], []
    for i, v in enumerate(series):
        arr = _smiles_to_vec(v)
        if arr is not None: vecs.append(arr)
        else: vecs.append(None); none_idx.append(i)
    valid = [x for x in vecs if x is not None]
    if not valid:
        sample = next((v for v in _cache.values() if v is not None), np.zeros(217))
        fill = np.zeros_like(sample)
    else:
        fill = np.mean(valid, axis=0)
    for i in none_idx: vecs[i] = fill
    return np.vstack(vecs)

def compute_umap():
    exp = pd.read_csv(os.path.join(BASE_DIR, "experimental_data.csv"))
    lit = pd.read_csv(os.path.join(BASE_DIR, "literature_data.csv"))
    if "Unnamed: 0" in exp.columns: exp = exp.drop(columns=["Unnamed: 0"])
    exp["Source"] = "Experiment"; lit["Source"] = "Literature"
    df = pd.concat([lit, exp], ignore_index=True)
    X = np.hstack([_encode_col(df[c]) for c in SMILES_COLS])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    li = (df["Source"] == "Literature").values
    ei = (df["Source"] == "Experiment").values
    sel = VarianceThreshold()
    Xl = sel.fit_transform(X[li]); Xe = sel.transform(X[ei])
    sc = StandardScaler()
    Xls = sc.fit_transform(Xl); Xes = sc.transform(Xe)
    red = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, verbose=False)
    return red.fit_transform(Xls), red.transform(Xes), len(Xls), len(Xes)

def bold_axes(ax):
    for lbl in ax.get_xticklabels(): lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels(): lbl.set_fontweight("bold")
    if ax.xaxis.label: ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label: ax.yaxis.label.set_fontweight("bold")

def add_panel_label(ax, letter):
    ax.text(-0.12, 1.03, f"({letter})", transform=ax.transAxes,
            fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

def combine():
    print("Loading data ...")
    lit_emb, exp_emb, n_lit, n_exp = compute_umap()
    mdf = pd.read_csv(RESULTS_FILE)
    reg_df = mdf[mdf["target"] != "UL94_Rating"]
    r2no = reg_df[reg_df["target"] != "pHRR"]
    cls_row = mdf[mdf["target"] == "UL94_Rating"].iloc[0]

    r2_targets = r2no["target"].tolist()
    r2_labels  = [DISPLAY_NAMES.get(t, t) for t in r2_targets]

    pva_loi  = pd.read_csv(os.path.join(STAGE3_DIR, "predictions_LOI.csv"))
    pva_ul94 = pd.read_csv(os.path.join(STAGE3_DIR, "predictions_UL94_Rating.csv"))

    fig = plt.figure(figsize=(19, 12), dpi=300)
    outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1],
                             hspace=0.35, left=0.07, right=0.97,
                             top=0.95, bottom=0.06)

    # ======== ROW 1 ========
    row1 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[0],
        width_ratios=[1.4, 1.1, 0.9], wspace=0.28)

    # (a) UMAP
    ax_a = fig.add_subplot(row1[0])
    sns.kdeplot(x=lit_emb[:, 0], y=lit_emb[:, 1], fill=True, cmap="Blues",
                alpha=0.28, levels=10, thresh=0.02, ax=ax_a)
    ax_a.scatter(lit_emb[:, 0], lit_emb[:, 1], c=COLOR_LIT, s=12, alpha=0.25,
                 edgecolors="none", label=f"Literature (n={n_lit})")
    ax_a.scatter(exp_emb[:, 0], exp_emb[:, 1], c=COLOR_EXP, s=80, alpha=0.75,
                 marker="o", edgecolors="black", linewidth=0.6,
                 label=f"Experiment (n={n_exp})")
    ax_a.set_xlabel("UMAP Dimension 1 (a.u.)", fontweight="bold")
    ax_a.set_ylabel("UMAP Dimension 2 (a.u.)", fontweight="bold")
    ax_a.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333",
                fontsize=FS - 3).get_frame().set_linewidth(0.6)
    bold_axes(ax_a); add_panel_label(ax_a, "a")

    # (b) R² no pHRR
    ax_b = fig.add_subplot(row1[1])
    xn = np.arange(len(r2_targets)); w = 0.32
    b1 = ax_b.bar(xn - w/2, r2no["before_test_r2"], w, color=COLOR_BEFORE,
                  edgecolor="white", linewidth=0.5, label="Before (Literature only)")
    b2 = ax_b.bar(xn + w/2, r2no["after_test_r2"], w, color=COLOR_AFTER,
                  edgecolor="white", linewidth=0.5, label="After (Literature + Experiment)")
    for bars, c in [(b1, COLOR_BEFORE), (b2, COLOR_AFTER)]:
        for b in bars:
            ax_b.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                      f"{b.get_height():.3f}", ha="center", va="bottom",
                      fontsize=FS - 4, color=c, fontweight="bold")
    ax_b.set_xticks(xn); ax_b.set_xticklabels(r2_labels)
    ax_b.set_ylabel("Test $R^2$", fontweight="bold")
    ax_b.set_ylim(0, 1.35)
    ax_b.legend(frameon=True, fancybox=False, edgecolor="#333333",
                fontsize=FS - 3, loc="upper right")
    ax_b.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_b); add_panel_label(ax_b, "b")

    # (c) UL94
    ax_c = fig.add_subplot(row1[2])
    m_b = [cls_row["before_test_accuracy"], cls_row["before_test_f1"]]
    m_a = [cls_row["after_test_accuracy"],  cls_row["after_test_f1"]]
    xm = np.arange(2)
    ax_c.bar(xm - w/2, m_b, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
             label="Before (Literature only)")
    ax_c.bar(xm + w/2, m_a, w, color=COLOR_AFTER,  edgecolor="white", linewidth=0.5,
             label="After (Literature + Experiment)")
    for i, (bv, av) in enumerate(zip(m_b, m_a)):
        ax_c.text(i - w/2, bv + 0.015, f"{bv:.3f}", ha="center", fontsize=FS - 4,
                  color=COLOR_BEFORE, fontweight="bold")
        ax_c.text(i + w/2, av + 0.015, f"{av:.3f}", ha="center", fontsize=FS - 4,
                  color=COLOR_AFTER,  fontweight="bold")
    ax_c.set_xticks(xm); ax_c.set_xticklabels(["Accuracy", "F1"])
    ax_c.set_ylabel("Score", fontweight="bold")
    ax_c.set_ylim(0, 1.35)
    ax_c.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 3,
                loc="upper right")
    ax_c.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_c); add_panel_label(ax_c, "c")

    # ======== ROW 2 ========
    row2 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        width_ratios=[0.80, 1.20], wspace=0.28)

    # (d) PvA_LOI
    gs_d = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=row2[0],
        width_ratios=[1, 1], wspace=0.28)
    ax_d1 = fig.add_subplot(gs_d[0])
    ax_d2 = fig.add_subplot(gs_d[1])
    yt_loi = pva_loi["y_true"].values
    ypb_loi = pva_loi["y_pred_before"].values
    ypa_loi = pva_loi["y_pred_after"].values
    all_loi = np.concatenate([yt_loi, ypb_loi, ypa_loi])
    lmin, lmax = all_loi.min() * 0.92, all_loi.max() * 1.08
    lims_loi = [lmin, lmax]
    r2b = r2_score(yt_loi, ypb_loi); r2a = r2_score(yt_loi, ypa_loi)
    rmb = np.sqrt(mean_squared_error(yt_loi, ypb_loi))
    rma = np.sqrt(mean_squared_error(yt_loi, ypa_loi))

    for ax, yp, r2, rmse, color, tag in zip(
        [ax_d1, ax_d2], [ypb_loi, ypa_loi], [r2b, r2a], [rmb, rma],
        [COLOR_BEFORE, COLOR_AFTER],
        ["Before (Literature only)", "After (Literature + Experiment)"],
    ):
        ax.scatter(yt_loi, yp, c=color, alpha=0.55, s=22, edgecolors="none")
        ax.plot(lims_loi, lims_loi, color=COLOR_LINE, linestyle="--", linewidth=1.0)
        ax.set_xlim(lims_loi); ax.set_ylim(lims_loi)
        ax.set_xlabel("Actual LOI (%)", fontweight="bold")
        ax.set_ylabel("Predicted LOI (%)", fontweight="bold")
        ax.set_title(tag, fontsize=FS - 2, fontweight="bold", color=color, pad=12)
        ax.text(0.05, 0.93, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, fontsize=FS - 4, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#cccccc"))
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.grid(linestyle=":", alpha=0.3)
        bold_axes(ax)

    add_panel_label(ax_d1, "d")

    # (e) PvA_UL94
    gs_e = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=row2[1],
        width_ratios=[1, 1], wspace=0.22)
    ax_e1 = fig.add_subplot(gs_e[0])
    ax_e2 = fig.add_subplot(gs_e[1])
    yt_ul = pva_ul94["y_true"].values
    ypb_ul = pva_ul94["y_pred_before"].values
    ypa_ul = pva_ul94["y_pred_after"].values
    cm_b = confusion_matrix(yt_ul, ypb_ul, labels=[0, 1])
    cm_a = confusion_matrix(yt_ul, ypa_ul, labels=[0, 1])
    ul_labels = ["TN\n(True V-0)", "FP\n(False\nNon V-0)",
                 "FN\n(False V-0)", "TP\n(True\nNon V-0)"]
    cb = [cm_b[0,0], cm_b[0,1], cm_b[1,0], cm_b[1,1]]
    ca = [cm_a[0,0], cm_a[0,1], cm_a[1,0], cm_a[1,1]]

    for ax, counts, color, tag in zip(
        [ax_e1, ax_e2], [cb, ca],
        [COLOR_BEFORE, COLOR_AFTER],
        ["Before (Literature only)", "After (Literature + Experiment)"],
    ):
        xc = np.arange(4)
        ax.bar(xc, counts, color=color, edgecolor="white", linewidth=0.5, width=0.65)
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts)*0.02, str(v), ha="center", va="bottom",
                    fontsize=FS - 5, color=color, fontweight="bold")
        ax.set_xticks(xc); ax.set_xticklabels(ul_labels, fontsize=FS - 8, fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title(tag, fontsize=FS - 2, fontweight="bold", color=color, pad=12)
        ax.set_ylim(0, max(counts) * 1.20)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        bold_axes(ax)

    add_panel_label(ax_e1, "e")

    # ======== ROW 3 ========
    row3 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[2],
        width_ratios=[1, 1, 1], wspace=0.30)

    for col_idx, (letter, target) in enumerate([
        ("f", "LOI"), ("g", "TSP"), ("h", "Flexural_Strength"),
    ]):
        df_ed = pd.read_csv(os.path.join(STAGE3_DIR, f"predictions_{target}.csv"))
        yt = df_ed["y_true"].values
        ypb = df_ed["y_pred_before"].values
        ypa = df_ed["y_pred_after"].values
        rb, ra = yt - ypb, yt - ypa
        dname = DISPLAY_NAMES.get(target, target)
        unit  = UNITS.get(target, "")

        ax = fig.add_subplot(row3[col_idx])
        try:
            kb, ka = gaussian_kde(rb), gaussian_kde(ra)
            ar = np.concatenate([rb, ra])
            xr = np.linspace(ar.min() * 1.15, ar.max() * 1.15, 300)
            ax.fill_between(xr, kb(xr), alpha=0.25, color=COLOR_BEFORE)
            ax.plot(xr, kb(xr), color=COLOR_BEFORE, linewidth=1.8,
                    label="Before (Literature only)")
            ax.fill_between(xr, ka(xr), alpha=0.35, color=COLOR_AFTER)
            ax.plot(xr, ka(xr), color=COLOR_AFTER,  linewidth=1.8,
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
        ax.text(0.04, 0.93, dname, transform=ax.transAxes,
                fontsize=FS, fontweight="bold", va="top")
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333",
                  fontsize=FS - 4, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.35)
        bold_axes(ax)
        add_panel_label(ax, letter)

    print("Saving ...")
    for fmt in ["png", "pdf", "svg"]:
        out = os.path.join(SAVE_DIR, f"Combined_Panels.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print(f"  -> {fmt}")
    plt.close(fig)
    print("[OK] Combined_Panels saved (png/pdf/svg)")


if __name__ == "__main__":
    combine()
