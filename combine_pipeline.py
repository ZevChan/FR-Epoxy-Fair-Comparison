"""
================================================================================
 combine_pipeline.py — 4-Row Comprehensive Layout Figure
 Row 1: (a) UMAP_Descriptors_Scatter Centered
 Row 2: (b) Stage1_Baseline_R2_no_pHRR | (c) RMSE_no_pHRR | (d) UL94
 Row 3: (e) Stage2_TopK_LOI    |   (f) Optuna_History_LOI
 Row 4: (g) Optuna_Params_LOI  Full width
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

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap, seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors

# ══════════════════════════════════════════════════════════════════
FS = 22  # global font base (enlarged for Word readability)
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.labelsize": FS + 1, "axes.titlesize": FS + 1,
    "legend.fontsize": FS - 3, "figure.titlesize": FS + 4,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_BEFORE = "#5DA5DA"; COLOR_AFTER = "#C91511"
COLOR_LIT = "#5DA5DA"; COLOR_EXP = "#C91511"
COLOR_BEST = "#DE1916"; COLOR_GRAY = "#8C92AC"
COLOR_BEST_BEFORE = "#3A7CB8"; COLOR_BEST_AFTER = "#A0100E"

SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
STAGE1_DIR = os.path.join(RESULTS_DIR, "Stage1")
STAGE2_DIR = os.path.join(RESULTS_DIR, "Stage2")
STAGE3_DIR = os.path.join(RESULTS_DIR, "Stage3")
os.makedirs(SAVE_DIR, exist_ok=True)

DISPLAY_NAMES = {"LOI":"LOI","pHRR":"pHRR","THR":"THR","TSP":"TSP",
                 "Flexural_Strength":"Flexural Strength","UL94_Rating":"UL-94"}

# ══════════════════════════════════════════════════════════════════
#  UMAP Data
# ══════════════════════════════════════════════════════════════════
SMILES_COLS = ["EPOXY STRUCTURE","Flame_retardant","Curing_agent ",
               "Other_Material_1","Other_Material_2"]
_cache = {}

def _smiles_to_vec(smiles):
    if pd.isna(smiles) or not isinstance(smiles,str) or smiles.strip()=="": return None
    s = smiles.strip()
    if s in _cache: return _cache[s]
    mol = Chem.MolFromSmiles(s)
    if mol is None: _cache[s]=None; return None
    d = Descriptors.CalcMolDescriptors(mol)
    v = np.array(list(d.values()), dtype=np.float64)
    _cache[s]=v; return v

def _encode_col(series):
    vecs, ni = [], []
    for i, v in enumerate(series):
        a = _smiles_to_vec(v)
        if a is not None: vecs.append(a)
        else: vecs.append(None); ni.append(i)
    valid = [x for x in vecs if x is not None]
    if not valid:
        sample = next((v for v in _cache.values() if v is not None), np.zeros(217))
        fill = np.zeros_like(sample)
    else: fill = np.mean(valid, axis=0)
    for i in ni: vecs[i] = fill
    return np.vstack(vecs)

def compute_umap():
    exp = pd.read_csv(os.path.join(BASE_DIR,"experimental_data.csv"))
    lit = pd.read_csv(os.path.join(BASE_DIR,"literature_data.csv"))
    if "Unnamed: 0" in exp.columns: exp = exp.drop(columns=["Unnamed: 0"])
    exp["Source"]="Experiment"; lit["Source"]="Literature"
    df = pd.concat([lit,exp], ignore_index=True)
    X = np.hstack([_encode_col(df[c]) for c in SMILES_COLS])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    li = (df["Source"]=="Literature").values
    ei = (df["Source"]=="Experiment").values
    sel = VarianceThreshold(); Xl = sel.fit_transform(X[li]); Xe = sel.transform(X[ei])
    sc = StandardScaler(); Xls = sc.fit_transform(Xl); Xes = sc.transform(Xe)
    red = umap.UMAP(n_neighbors=15,min_dist=0.1,n_components=2,random_state=42,verbose=False)
    return red.fit_transform(Xls), red.transform(Xes), len(Xls), len(Xes)

# ══════════════════════════════════════════════════════════════════
def bold_axes(ax):
    for lbl in ax.get_xticklabels(): lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels(): lbl.set_fontweight("bold")
    if ax.xaxis.label: ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label: ax.yaxis.label.set_fontweight("bold")

def panel_label(ax, letter, y_off=1.04):
    ax.text(-0.14, y_off, f"({letter})", transform=ax.transAxes,
            fontsize=FS+4, fontweight="bold", va="bottom", ha="right")

# ══════════════════════════════════════════════════════════════════
def combine():
    print("Loading data ...")
    lit_emb, exp_emb, n_lit, n_exp = compute_umap()

    # Stage 1
    s1 = pd.read_csv(os.path.join(STAGE1_DIR,"stage1_baseline_results.csv"))
    reg_df = s1[s1["target"]!="UL94_Rating"]
    r2no = reg_df[reg_df["target"]!="pHRR"]
    cls_row = s1[s1["target"]=="UL94_Rating"].iloc[0]

    r2_targets = r2no["target"].tolist()
    r2_labels = [DISPLAY_NAMES.get(t,t) for t in r2_targets]
    reg_all = s1[s1["target"]!="UL94_Rating"]
    reg_labels = [DISPLAY_NAMES.get(t,t) for t in reg_all["target"]]

    # Stage 2
    topk_df = pd.read_csv(os.path.join(STAGE2_DIR,"topk_scan_LOI.csv")).sort_values("top_k")
    topk_summary = pd.read_csv(os.path.join(STAGE2_DIR,"stage2_topk_summary.csv"))
    best_k_row = topk_summary[topk_summary["target"]=="LOI"]
    best_k_loi = int(best_k_row.iloc[0]["best_k"]) if not best_k_row.empty else None

    # Stage 3
    hb = pd.read_csv(os.path.join(STAGE3_DIR,"optuna_history_before_LOI.csv"))
    ha = pd.read_csv(os.path.join(STAGE3_DIR,"optuna_history_after_LOI.csv"))
    vcol = "value"
    if vcol not in hb.columns:
        val_cols = [c for c in hb.columns if "value" in c.lower()]
        vcol = val_cols[0] if val_cols else hb.columns[-1]
    param_cols = [c for c in hb.columns
                  if c.startswith("params_") and hb[c].nunique()>1]

    # ═══════════ Canvas 4 Rows ════════════════════════════════════════
    fig = plt.figure(figsize=(20, 22), dpi=150)
    outer = fig.add_gridspec(4, 1, figure=fig,
                              hspace=0.40, left=0.07, right=0.97,
                              top=0.97, bottom=0.03,
                              height_ratios=[1.0, 0.85, 0.85, 0.42])

    # ====== ROW 1: (a) UMAP Full width ======
    ax_a = fig.add_subplot(outer[0])
    sns.kdeplot(x=lit_emb[:,0], y=lit_emb[:,1], fill=True, cmap="Blues",
                alpha=0.28, levels=10, thresh=0.02, ax=ax_a)
    ax_a.scatter(lit_emb[:,0], lit_emb[:,1], c=COLOR_LIT, s=10, alpha=0.25,
                 edgecolors="none", label=f"Literature (n={n_lit})")
    ax_a.scatter(exp_emb[:,0], exp_emb[:,1], c=COLOR_EXP, s=70, alpha=0.75,
                 marker="o", edgecolors="black", linewidth=0.6,
                 label=f"Experiment (n={n_exp})")
    ax_a.set_xlabel("UMAP Dimension 1 (a.u.)", fontweight="bold")
    ax_a.set_ylabel("UMAP Dimension 2 (a.u.)", fontweight="bold")
    ax_a.legend(loc="best", frameon=True, fancybox=False, edgecolor="#333333",
                fontsize=FS-3).get_frame().set_linewidth(0.6)
    bold_axes(ax_a); panel_label(ax_a, "a")

    # ====== ROW 2: (b) R² | (c) RMSE | (d) UL94 ======
    row2 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1], wspace=0.40)

    # (b) R² no pHRR
    ax_b = fig.add_subplot(row2[0])
    xn = np.arange(len(r2_targets)); w = 0.32
    b1 = ax_b.bar(xn-w/2, r2no["before_test_r2"], w, color=COLOR_BEFORE,
                  edgecolor="white", linewidth=0.5, label="Before (Literature only)")
    b2 = ax_b.bar(xn+w/2, r2no["after_test_r2"], w, color=COLOR_AFTER,
                  edgecolor="white", linewidth=0.5, label="After (Literature + Experiment)")
    for bars, c in [(b1,COLOR_BEFORE),(b2,COLOR_AFTER)]:
        for b in bars:
            ax_b.text(b.get_x()+b.get_width()/2, b.get_height()+0.012,
                      f"{b.get_height():.3f}", ha="center", va="bottom",
                      fontsize=FS-4, color=c, fontweight="bold")
    ax_b.set_xticks(xn); ax_b.set_xticklabels(r2_labels, rotation=25, ha="right")
    ax_b.set_ylabel("Test $R^2$ (Default XGBoost)", fontweight="bold")
    ax_b.set_ylim(0, 1.30)
    ax_b.legend(frameon=True, fancybox=False, edgecolor="#333333",
                fontsize=FS-3, loc="upper right")
    ax_b.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_b); panel_label(ax_b, "b")

    # (c) RMSE no pHRR
    ax_c = fig.add_subplot(row2[1])
    b1c = ax_c.bar(xn-w/2, r2no["before_test_rmse"], w, color=COLOR_BEFORE,
                   edgecolor="white", linewidth=0.5, label="Before (Literature only)")
    b2c = ax_c.bar(xn+w/2, r2no["after_test_rmse"], w, color=COLOR_AFTER,
                   edgecolor="white", linewidth=0.5, label="After (Literature + Experiment)")
    for bars, c in [(b1c,COLOR_BEFORE),(b2c,COLOR_AFTER)]:
        for b in bars:
            ax_c.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                      f"{b.get_height():.1f}", ha="center", va="bottom",
                      fontsize=FS-4, color=c, fontweight="bold")
    ax_c.set_xticks(xn); ax_c.set_xticklabels(r2_labels, rotation=25, ha="right")
    ax_c.set_ylabel("Test RMSE (Default XGBoost)", fontweight="bold")
    all_h = [b.get_height() for b in b1c] + [b.get_height() for b in b2c]
    top_c = max(all_h) + 0.2
    ax_c.set_ylim(0, top_c * 1.45)
    ax_c.legend(frameon=True, fancybox=False, edgecolor="#333333",
                fontsize=FS-3, loc="upper right")
    ax_c.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_c); panel_label(ax_c, "c")

    # (d) UL94
    ax_d = fig.add_subplot(row2[2])
    mv_b = [cls_row["before_test_accuracy"], cls_row["before_test_f1"]]
    mv_a = [cls_row["after_test_accuracy"], cls_row["after_test_f1"]]
    xm = np.arange(2)
    ax_d.bar(xm-w/2, mv_b, w, color=COLOR_BEFORE, edgecolor="white", linewidth=0.5,
             label="Before (Literature only)")
    ax_d.bar(xm+w/2, mv_a, w, color=COLOR_AFTER, edgecolor="white", linewidth=0.5,
             label="After (Literature + Experiment)")
    for i, (bv, av) in enumerate(zip(mv_b, mv_a)):
        ax_d.text(i-w/2, bv+0.015, f"{bv:.3f}", ha="center", fontsize=FS-3,
                  color=COLOR_BEFORE, fontweight="bold")
        ax_d.text(i+w/2, av+0.015, f"{av:.3f}", ha="center", fontsize=FS-3,
                  color=COLOR_AFTER, fontweight="bold")
    ax_d.set_xticks(xm); ax_d.set_xticklabels(["Accuracy","F1"])
    ax_d.set_ylabel("Score (Default XGBoost)", fontweight="bold")
    ax_d.set_ylim(0, 1.30)
    ax_d.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS-3)
    ax_d.grid(axis="y", linestyle=":", alpha=0.3)
    bold_axes(ax_d); panel_label(ax_d, "d")

    # ====== ROW 3: (e) TopK | (f) Optuna History ======
    row3 = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2], wspace=0.30)

    # (e) Stage2 TopK LOI
    ax_e = fig.add_subplot(row3[0])
    metric_col = "r2"; metric_label = "Test $R^2$"
    ax_e.plot(topk_df["top_k"], topk_df[f"before_{metric_col}"],
              color=COLOR_BEFORE, linewidth=1.4, marker="o", markersize=3.5,
              label="Before (Literature only)")
    ax_e.plot(topk_df["top_k"], topk_df[f"after_{metric_col}"],
              color=COLOR_AFTER, linewidth=1.4, marker="s", markersize=3.5,
              label="After (Literature + Experiment)")
    if best_k_loi is not None and best_k_loi in topk_df["top_k"].values:
        row = topk_df[topk_df["top_k"]==best_k_loi].iloc[0]
        bv = row[f"after_{metric_col}"]
        ax_e.scatter([best_k_loi], [bv], color=COLOR_BEST, s=50, zorder=5,
                     edgecolors="white", linewidth=1.0)
        ax_e.annotate(f"K={best_k_loi}\n{bv:.3f}", (best_k_loi, bv),
                      textcoords="offset points", xytext=(8,-12),
                      fontsize=FS-2, color=COLOR_BEST, fontweight="bold")
    ax_e.set_xlabel("Number of Features (Top K)", fontweight="bold")
    ax_e.set_ylabel(metric_label, fontweight="bold")
    ax_e.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS-3)
    ax_e.grid(linestyle=":", alpha=0.3)
    ax_e.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    bold_axes(ax_e); panel_label(ax_e, "e")

    # (f) Optuna History LOI
    ax_f = fig.add_subplot(row3[1])
    tr_b = np.arange(1, len(hb)+1); tr_a = np.arange(1, len(ha)+1)
    ax_f.plot(tr_b, hb[vcol], color=COLOR_BEFORE, linewidth=1.2, alpha=0.7,
              label="Before (Literature only)")
    ax_f.plot(tr_a, ha[vcol], color=COLOR_AFTER, linewidth=1.2, alpha=0.7,
              label="After (Literature + Experiment)")
    best_b = hb[vcol].max(); best_a = ha[vcol].max()
    ax_f.axhline(best_b, color=COLOR_BEST_BEFORE, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_f.axhline(best_a, color=COLOR_BEST_AFTER, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_f.set_xlabel("Trial Number", fontweight="bold")
    ax_f.set_ylabel("Best CV Score", fontweight="bold")
    ax_f.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS-3)
    ax_f.grid(linestyle=":", alpha=0.3)
    bold_axes(ax_f); panel_label(ax_f, "f")

    # ====== ROW 4: (g) Optuna Params LOI ======
    n_params = len(param_cols)
    if n_params >= 1:
        row4 = gridspec.GridSpecFromSubplotSpec(
            1, n_params, subplot_spec=outer[3], wspace=0.38)
        for i, col in enumerate(param_cols):
            ax_g = fig.add_subplot(row4[i])
            pname = col.replace("params_", "")
            corr_b = hb[col].corr(hb[vcol]) if hb[col].dtype in ["float64","int64"] else 0
            corr_a = ha[col].corr(ha[vcol]) if ha[col].dtype in ["float64","int64"] else 0
            ax_g.bar([0,1], [corr_b, corr_a], color=[COLOR_BEFORE, COLOR_AFTER],
                     edgecolor="white", linewidth=0.5, width=0.55)
            ax_g.set_xticks([0,1])
            ax_g.set_xticklabels(["Before","After"], fontsize=FS-3, rotation=30)
            ax_g.set_title(pname, fontsize=FS, fontweight="bold")
            ax_g.axhline(0, color=COLOR_GRAY, linewidth=0.5)
            ax_g.grid(axis="y", linestyle=":", alpha=0.3)
            ax_g.set_box_aspect(0.7)  # aspect ratio ~w:h ≈ 1:1.4, refer to Optuna_Params figsize (17.5,3) each cell ≈2.5×3 inches
            bold_axes(ax_g)
        # Attach to the first param plot, do not create ghost axes
        if n_params >= 1:
            first_param_ax = fig.axes[-n_params]
            panel_label(first_param_ax, "g", y_off=1.04)

    # ── Save ─────────────────────────────────────────────────────
    print("Saving ...")
    for fmt in ["png","pdf","svg"]:
        out = os.path.join(SAVE_DIR, f"Combined_Pipeline.{fmt}")
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.08)
        print(f"  -> {fmt}")
    plt.close(fig)
    print("[OK] Combined_Pipeline saved (png/pdf/svg)")


if __name__ == "__main__":
    combine()
