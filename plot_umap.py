"""
================================================================================
 plot_umap.py — Chemical Space UMAP Projection
================================================================================
 Reads umap_data_MASTER.csv saved by fair_holdout_comparison.py,
 distinguishes Literature / Experiment by Source column, reduces to 2D.
 Outputs PNG / PDF / SVG.

 Colors:
   Literature data — #5DA5DA (Steel blue)
   Experimental data — #C91511 (Crimson, highlighted)
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import umap

# ==================== NC Journal Publication Global Settings ====================
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

# ==================== Color Definitions ====================
COLOR_LIT = "#5DA5DA"  # Steel blue — Literature data
COLOR_EXP = "#C91511"  # Crimson — Experimental data highlight

# ==================== Path Configuration ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_umap():
    umap_file = os.path.join(RESULTS_DIR, "umap_data_MASTER.csv")
    if not os.path.exists(umap_file):
        print("[WARN] umap_data_MASTER.csv not found. Please run fair_holdout_comparison.py first")
        return

    print("Loading UMAP data ...")
    df = pd.read_csv(umap_file)

    # Separate feature columns (exclude Target, Target_Value, Source)
    meta_cols = ["Target", "Target_Value", "Source"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Deduplicate (same row may appear in multiple targets)
    df_unique = df.drop_duplicates(subset=feature_cols).copy()

    n_lit = (df_unique["Source"] == "Literature").sum()
    n_exp = (df_unique["Source"] == "Experiment").sum()
    print(f"  Literature samples: {n_lit}")
    print(f"  Experimental samples: {n_exp}")

    X = df_unique[feature_cols].values
    lit_mask = df_unique["Source"] == "Literature"
    exp_mask = df_unique["Source"] == "Experiment"

    # ---- Fit on literature data only, transform experimental data ----
    print("Performing UMAP dimensionality reduction (fit on literature, embed experimental)...")
    selector = VarianceThreshold()
    X_lit_sel = selector.fit_transform(X[lit_mask])
    X_exp_sel = selector.transform(X[exp_mask])

    scaler = StandardScaler()
    X_lit_scaled = scaler.fit_transform(X_lit_sel)
    X_exp_scaled = scaler.transform(X_exp_sel)

    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        random_state=42, verbose=False,
    )
    lit_embedding = reducer.fit_transform(X_lit_scaled)
    exp_embedding = reducer.transform(X_exp_scaled)

    df_unique["UMAP-1"] = np.nan
    df_unique["UMAP-2"] = np.nan
    df_unique.loc[lit_mask, "UMAP-1"] = lit_embedding[:, 0]
    df_unique.loc[lit_mask, "UMAP-2"] = lit_embedding[:, 1]
    df_unique.loc[exp_mask, "UMAP-1"] = exp_embedding[:, 0]
    df_unique.loc[exp_mask, "UMAP-2"] = exp_embedding[:, 1]

    # ---- Main plot ----
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=600)

    # KDE background
    sns.kdeplot(
        data=df_unique[lit_mask],
        x="UMAP-1", y="UMAP-2",
        fill=True, alpha=0.12, color="#7f8c8d",
        levels=8, thresh=0.05, ax=ax,
    )

    # Literature scatter points
    ax.scatter(
        df_unique.loc[lit_mask, "UMAP-1"],
        df_unique.loc[lit_mask, "UMAP-2"],
        c=COLOR_LIT, s=10, alpha=0.3, edgecolors="none",
        label=f"Literature Data (n={n_lit})",
    )

    # Experimental scatter points highlighted
    ax.scatter(
        df_unique.loc[exp_mask, "UMAP-1"],
        df_unique.loc[exp_mask, "UMAP-2"],
        c=COLOR_EXP, s=55, alpha=0.95, marker="o",
        edgecolors="black", linewidth=0.6,
        label=f"Experimental Data (n={n_exp})",
    )

    ax.set_xlabel("UMAP Dimension 1 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2 (a.u.)", fontsize=13, fontweight="bold")
    ax.set_title("Chemical Space Distribution of EP/FR Composites",
                 fontsize=14, fontweight="bold", pad=12)

    legend = ax.legend(loc="best", frameon=True, fancybox=False,
                       edgecolor="#333333", fontsize=14)
    legend.get_frame().set_linewidth(0.6)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    for fmt in ["png", "pdf", "svg"]:
        plt.savefig(os.path.join(SAVE_DIR, f"UMAP_Chemical_Space.{fmt}"),
                    dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("[OK] UMAP_Chemical_Space saved (png/pdf/svg)")


if __name__ == "__main__":
    plot_umap()
