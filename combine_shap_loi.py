"""
================================================================================
 combine_shap_loi.py — LOI SHAP + Feature Importance 2×2 Combined Figure
================================================================================
 Global sorting benchmark: SHAP After mean(|SHAP|) descending.
 Top-left (a): Feature Importance (Before vs After) Bar Chart
 Top-right (b): Feature Importance Shift Slope Chart (Hide Y-axis labels)
 Bottom-left (c): SHAP Beeswarm Before
 Bottom-right (d): SHAP Beeswarm After (Hide Y-axis labels)
================================================================================
"""
import os, sys, json, re, warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import joblib, shap

# ── Monkey-patch XGBoost 3.x for SHAP 0.49 ──────────────────────
import shap.explainers._tree as _shap_tree
_orig = _shap_tree.XGBTreeModelLoader.__init__

def _patched(self, xgb_model):
    import io, scipy, xgboost as xgb
    from shap.explainers._tree import _check_xgboost_version, decode_ubjson_buffer
    _check_xgboost_version(xgb.__version__)
    model = xgb_model
    raw = model.save_raw(raw_format="ubj")
    with io.BytesIO(raw) as fd:
        jmodel = decode_ubjson_buffer(fd)
    learner = jmodel["learner"]; lm = learner["learner_model_param"]
    obj = learner["objective"]
    booster = learner["gradient_booster"]
    nc = max(int(lm["num_class"]),1); nt = max(int(lm["num_target"]),1)
    nt = max(nt, nc)
    if "gbtree" in booster and "model" not in booster: booster = booster["gbtree"]
    if booster["model"].get("iteration_indptr",None) is not None:
        ii = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
        diff = np.diff(ii)
    else:
        npt = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
        diff = np.repeat(nt*npt, model.num_boosted_rounds())
    if np.any(diff != diff[0]): raise ValueError("vector-leaf")
    self.n_trees_per_iter = int(diff[0]); self.n_targets = nt
    def _p(v):
        if isinstance(v,str): nums = re.findall(r'[\d.]+',v); return float(nums[0]) if nums else 0.5
        return float(v)
    self.base_score = _p(lm["base_score"])
    self.name_obj = obj["name"]; self.name_gbm = booster["name"]
    bs = _p(lm["base_score"])
    if self.name_obj in ("binary:logistic","reg:logistic"): self.base_score = scipy.special.logit(bs)
    elif self.name_obj in ("reg:gamma","reg:tweedie","count:poisson","survival:cox","survival:aft"): self.base_score = np.log(self.base_score)
    else: self.base_score = bs
    self.num_feature = int(lm["num_feature"]); self.num_class = int(lm["num_class"])
    trees = booster["model"]["trees"]; self.num_trees = len(trees)
    self.node_parents=[]; self.node_cleft=[]; self.node_cright=[]; self.node_sindex=[]
    self.children_default=[]; self.sum_hess=[]
    self.values=[]; self.thresholds=[]; self.threshold_types=[]; self.features=[]
    self.split_types=[]; self.categories=[]
    ft = model.feature_types
    if ft is not None:
        ci = np.where(np.asarray(ft)=="c")[0]
        self.cat_feature_indices = ci if len(ci)>0 else None
    else: self.cat_feature_indices = None
    def ti(d): assert isinstance(d,list); return np.asanyarray(d,dtype=np.uint8)
    for i in range(self.num_trees):
        tree = trees[i]
        parents=np.asarray(tree["parents"]); self.node_parents.append(parents)
        self.node_cleft.append(np.asarray(tree["left_children"],dtype=np.int32))
        self.node_cright.append(np.asarray(tree["right_children"],dtype=np.int32))
        self.node_sindex.append(np.asarray(tree["split_indices"],dtype=np.uint32))
        bw=np.asarray(tree["base_weights"],dtype=np.float32)
        if bw.size != self.node_cleft[-1].size: raise ValueError("vector-leaf")
        dl = ti(tree["default_left"])
        dc = np.where(dl==1,self.node_cleft[-1],self.node_cright[-1]).astype(np.int64)
        self.children_default.append(dc)
        self.sum_hess.append(np.asarray(tree["sum_hessian"],dtype=np.float64))
        is_leaf = self.node_cleft[-1]==-1
        sc2 = np.asarray(tree["split_conditions"],dtype=np.float32)
        thresholds = np.where(is_leaf,0.0, np.nextafter(sc2,-np.float32(np.inf)))
        self.values.append(np.where(is_leaf,sc2,0.0).reshape(-1,1))
        self.thresholds.append(thresholds)
        self.threshold_types.append(np.zeros_like(thresholds,dtype=np.int32))
        self.features.append(np.asarray(tree["split_indices"],dtype=np.int64))
        self.split_types.append(ti(tree["split_type"]))
        csg=tree["categories_segments"]; csz=tree["categories_sizes"]
        cnd=tree["categories_nodes"]; cats=tree["categories"]
        assert len(csg)==len(csz)==len(cnd)
        tc = self.parse_categories(cnd, csg, csz, cats, self.node_cleft[-1])
        self.categories.append(tc)
_shap_tree.XGBTreeModelLoader.__init__ = _patched

from fair_holdout_comparison import (
    load_features, load_targets, process_ul94, WITH_DATA_DIR,
    WITHOUT_DATA_DIR, OUTPUT_DIR,
)
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════
FS = 14  # Global font size
plt.rcParams.update({
    "font.family": "Arial", "font.size": FS,
    "axes.labelsize": FS + 2, "axes.titlesize": FS + 2,
    "legend.fontsize": FS - 2, "figure.titlesize": FS + 4,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

COLOR_BEFORE = "#5DA5DA";  COLOR_AFTER = "#C91511"
COLOR_INCREASE = "#C91511";  COLOR_DECREASE = "#5DA5DA"
SHAP_CMAP_B = LinearSegmentedColormap.from_list("b",["#6EAA5E","#E8E8E8","#C91511"],N=256)
SHAP_CMAP_A = LinearSegmentedColormap.from_list("a",["#E8923B","#E8E8E8","#5DA5DA"],N=256)

SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
STAGE3_DIR = os.path.join(OUTPUT_DIR, "Stage3")
os.makedirs(SAVE_DIR, exist_ok=True)

def shorten(name, n=30):
    return name if len(name) <= n else name[:n-1] + "\u2026"

def bold_axes(ax):
    for lbl in ax.get_xticklabels(): lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels(): lbl.set_fontweight("bold")
    if ax.xaxis.label: ax.xaxis.label.set_fontweight("bold")
    if ax.yaxis.label: ax.yaxis.label.set_fontweight("bold")

# ══════════════════════════════════════════════════════════════════
def combine():
    target = "LOI"
    print(f"Loading {target} data ...")

    # ── Load model & features ──
    model_b = joblib.load(os.path.join(STAGE3_DIR, f"model_before_{target}.pkl"))
    model_a = joblib.load(os.path.join(STAGE3_DIR, f"model_after_{target}.pkl"))
    sc_b = joblib.load(os.path.join(STAGE3_DIR, f"scaler_before_{target}.pkl"))
    sc_a = joblib.load(os.path.join(STAGE3_DIR, f"scaler_after_{target}.pkl"))
    with open(os.path.join(STAGE3_DIR, f"best_features_{target}.json")) as f:
        best_feats = json.load(f)

    # ── Rebuild test set ──
    X_all = load_features(WITH_DATA_DIR)
    y_all = load_targets(WITH_DATA_DIR)
    X_wo = load_features(WITHOUT_DATA_DIR)
    N_LIT = len(X_wo)
    is_lit = np.arange(len(X_all)) < N_LIT
    valid = y_all[target].notna().values
    X_v = X_all.loc[valid].copy(); y_v = y_all.loc[valid, target].copy()
    is_lit_v = is_lit[valid]
    X_lit = X_v[is_lit_v].reset_index(drop=True)
    y_lit = y_v[is_lit_v].reset_index(drop=True)
    _, X_test, _, _ = train_test_split(X_lit, y_lit, test_size=0.2, random_state=42)
    best_feats = [c for c in best_feats if c in X_test.columns]
    feat_list = best_feats
    X_test_b_s = sc_b.transform(X_test[feat_list])
    X_test_a_s = sc_a.transform(X_test[feat_list])
    n_feat = len(feat_list)
    short_names = [shorten(f) for f in feat_list]
    print(f"  Test set: {len(X_test)} samples, {n_feat} features")

    # ── Feature importance ──
    imp_b = model_b.feature_importances_
    imp_a = model_a.feature_importances_

    # ── SHAP ──
    print("  Computing SHAP ...")
    explainer_b = shap.TreeExplainer(model_b)
    shap_b_vals = explainer_b(X_test_b_s).values
    explainer_a = shap.TreeExplainer(model_a)
    shap_a_vals = explainer_a(X_test_a_s).values

    # ═══════════ Global sort index: SHAP After mean(|SHAP|) ═══════════
    mean_abs_shap_a = np.mean(np.abs(shap_a_vals), axis=0)
    global_order = np.argsort(mean_abs_shap_a)[::-1]  # Descending
    top_n = min(20, n_feat)
    order_idx = global_order[:top_n]

    imp_b_ord = imp_b[order_idx]
    imp_a_ord = imp_a[order_idx]
    names_ord  = [short_names[i] for i in order_idx]
    feat_ord   = [feat_list[i] for i in order_idx]
    shap_b_ord = shap_b_vals[:, order_idx]
    shap_a_ord = shap_a_vals[:, order_idx]
    X_b_ord = X_test_b_s[:, order_idx]
    X_a_ord = X_test_a_s[:, order_idx]

    # ═══════════ Canvas ═══════════════════════════════════════════
    fig = plt.figure(figsize=(16, 13), dpi=300, constrained_layout=False)
    outer = fig.add_gridspec(2, 2, figure=fig,
                              left=0.08, right=0.97, top=0.95, bottom=0.06,
                              hspace=0.40, wspace=0.02)

    # ──── (a) Feature Importance Bar ────
    ax_a = fig.add_subplot(outer[0, 0])
    y_pos = np.arange(top_n); height = 0.35
    ax_a.barh(y_pos + height/2, imp_b_ord, height,
              color=COLOR_BEFORE, alpha=0.85, edgecolor="white", linewidth=0.3,
              label="Before (Literature only)")
    ax_a.barh(y_pos - height/2, imp_a_ord, height,
              color=COLOR_AFTER, alpha=0.85, edgecolor="white", linewidth=0.3,
              label="After (Literature + Experiment)")
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(names_ord, fontsize=FS - 4)
    ax_a.set_xlabel("Feature Importance Weight", fontweight="bold")
    ax_a.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=FS - 2)
    ax_a.invert_yaxis()
    ax_a.grid(axis="x", linestyle=":", alpha=0.3)
    bold_axes(ax_a)

    # Panel label
    ax_a.text(-0.12, 1.03, "(a)", transform=ax_a.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # ──── (b) Importance Slope ────
    ax_b = fig.add_subplot(outer[0, 1])
    for i in range(top_n):
        wb = imp_b_ord[i]; wa = imp_a_ord[i]
        delta = wa - wb
        color = COLOR_INCREASE if delta >= 0 else COLOR_DECREASE
        lw = 0.6 + abs(delta) * 15; lw = np.clip(lw, 0.8, 3.5)
        ax_b.plot([wb, wa], [i, i], color=color, linewidth=lw, alpha=0.75,
                  solid_capstyle="round")
        ax_b.scatter([wb], [i], color=COLOR_BEFORE, s=35, zorder=5,
                     edgecolors="white", linewidth=0.5)
        ax_b.scatter([wa], [i], color=COLOR_AFTER, s=35, zorder=5,
                     edgecolors="white", linewidth=0.5)
    ax_b.scatter([], [], color=COLOR_BEFORE, s=40, label="Before (Literature only)")
    ax_b.scatter([], [], color=COLOR_AFTER, s=40, label="After (Literature + Experiment)")
    ax_b.set_yticks(np.arange(top_n))
    ax_b.set_yticklabels(names_ord, fontsize=FS - 4)
    ax_b.set_xlabel("Feature Importance Weight", fontweight="bold")
    ax_b.legend(frameon=False, fontsize=FS - 2, loc="lower right", bbox_to_anchor=(0.98, 0.03))
    ax_b.invert_yaxis()
    ax_b.grid(axis="x", linestyle=":", alpha=0.3)
    bold_axes(ax_b)

    ax_b.text(-0.12, 1.03, "(b)", transform=ax_b.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # ──── (c) SHAP Beeswarm Before ────
    ax_c = fig.add_subplot(outer[1, 0])
    shap.summary_plot(shap_b_ord, X_b_ord, feature_names=names_ord,
                      plot_type="dot", cmap=SHAP_CMAP_B, max_display=top_n,
                      show=False)
    ax_c = plt.gca()
    for lbl in ax_c.get_yticklabels(): lbl.set_fontsize(FS - 4)
    bold_axes(ax_c)

    ax_c.text(-0.12, 1.03, "(c)", transform=ax_c.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # ──── (d) SHAP Beeswarm After ────
    ax_d = fig.add_subplot(outer[1, 1])
    shap.summary_plot(shap_a_ord, X_a_ord, feature_names=names_ord,
                      plot_type="dot", cmap=SHAP_CMAP_A, max_display=top_n,
                      show=False)
    ax_d = plt.gca()
    for lbl in ax_d.get_yticklabels(): lbl.set_fontsize(FS - 4)
    bold_axes(ax_d)

    ax_d.text(-0.12, 1.03, "(d)", transform=ax_d.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # ── Auto-calculate left-right column spacing — measure max Y-axis label width ──────────────
    # Render once to make tick label bounding boxes active
    fig.canvas.draw()
    def _ytick_max_width_inches(ax, fig):
        """Returns the width in canvas inches of the widest Y tick label of ax."""
        renderer = fig.canvas.get_renderer()
        max_w = 0.0
        for lbl in ax.get_yticklabels():
            bb = lbl.get_window_extent(renderer)
            if bb is not None:
                max_w = max(max_w, bb.width)
        # window extent is in pixels, convert to inches
        dpi = fig.dpi
        return max_w / dpi if dpi else 0.0

    left_max = max(_ytick_max_width_inches(ax_a, fig),
                   _ytick_max_width_inches(ax_c, fig))
    fig_w = fig.get_size_inches()[0]
    # wspace is defined as column spacing / average column width.
    # Single column width ≈ (fig_w * (right-left) - gap) / 2
    usable = fig_w * (0.97 - 0.08)  # total usable width
    col_w  = (usable - 0.02 * fig_w) / 2  # approximate, since wspace is fraction of avg col width
    # More precisely: outer's wspace is a fraction of the average column width. Let's solve:
    # Total width of left and right columns = usable, gap gap_inches = wspace * (col_w)
    # col_w = (usable - wspace * col_w) / 2 => col_w = usable / (2 + wspace)
    # We need gap_inches >= 2 * left_max (one Y-axis on each side)
    # Solve: wspace * (usable / (2 + wspace)) >= 2 * left_max
    # wspace * usable >= 2 * left_max * (2 + wspace)
    # wspace * usable >= 4*left_max + 2*left_max*wspace
    # wspace * (usable - 2*left_max) >= 4*left_max
    # wspace >= 4*left_max / (usable - 2*left_max)
    needed = 2.5 * left_max  # Add some margin
    if usable - needed > 0:
        wspace_auto = needed / (usable - needed)
    else:
        wspace_auto = 0.35
    wspace_auto = max(wspace_auto, 0.02)
    outer.update(wspace=wspace_auto)

    # ── Save ────────────────────────────────────────────────────
    print("Saving ...")
    for fmt in ["png", "pdf", "svg"]:
        out = os.path.join(SAVE_DIR, f"Combined_SHAP_LOI.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print(f"  -> {fmt}")
    plt.close(fig)
    print("[OK] Combined_SHAP_LOI saved (png/pdf/svg)")


if __name__ == "__main__":
    combine()
