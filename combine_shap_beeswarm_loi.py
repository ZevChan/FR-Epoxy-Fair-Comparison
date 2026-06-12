"""
================================================================================
 combine_shap_beeswarm_loi.py — LOI SHAP Beeswarm 1×2 Combined
 (a) SHAP Beeswarm Before  (b) SHAP Beeswarm After
 Global sorting: SHAP After mean(|SHAP|) descending
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
from matplotlib.colors import LinearSegmentedColormap
import joblib, shap

# ── Monkey-patch XGBoost 3.x for SHAP 0.49 ──────────────────────
import shap.explainers._tree as _shap_tree
def _patched(self, xgb_model):
    import io, scipy, xgboost as xgb
    from shap.explainers._tree import _check_xgboost_version, decode_ubjson_buffer
    _check_xgboost_version(xgb.__version__)
    model = xgb_model
    raw = model.save_raw(raw_format="ubj")
    with io.BytesIO(raw) as fd: jmodel = decode_ubjson_buffer(fd)
    learner = jmodel["learner"]; lm = learner["learner_model_param"]
    obj = learner["objective"]
    booster = learner["gradient_booster"]
    nc = max(int(lm["num_class"]),1); nt = max(int(lm["num_target"]),1); nt = max(nt,nc)
    if "gbtree" in booster and "model" not in booster: booster = booster["gbtree"]
    if booster["model"].get("iteration_indptr",None) is not None:
        ii = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32); diff = np.diff(ii)
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
        self.node_parents.append(np.asarray(tree["parents"]))
        self.node_cleft.append(np.asarray(tree["left_children"],dtype=np.int32))
        self.node_cright.append(np.asarray(tree["right_children"],dtype=np.int32))
        self.node_sindex.append(np.asarray(tree["split_indices"],dtype=np.uint32))
        bw=np.asarray(tree["base_weights"],dtype=np.float32)
        if bw.size != self.node_cleft[-1].size: raise ValueError("vector-leaf")
        dl = ti(tree["default_left"])
        self.children_default.append(np.where(dl==1,self.node_cleft[-1],self.node_cright[-1]).astype(np.int64))
        self.sum_hess.append(np.asarray(tree["sum_hessian"],dtype=np.float64))
        is_leaf = self.node_cleft[-1]==-1
        sc2 = np.asarray(tree["split_conditions"],dtype=np.float32)
        self.values.append(np.where(is_leaf,sc2,0.0).reshape(-1,1))
        self.thresholds.append(np.where(is_leaf,0.0,np.nextafter(sc2,-np.float32(np.inf))))
        self.threshold_types.append(np.zeros_like(self.thresholds[-1],dtype=np.int32))
        self.features.append(np.asarray(tree["split_indices"],dtype=np.int64))
        self.split_types.append(ti(tree["split_type"]))
        csg=tree["categories_segments"]; csz=tree["categories_sizes"]
        cnd=tree["categories_nodes"]; cats=tree["categories"]
        self.categories.append(self.parse_categories(cnd,csg,csz,cats,self.node_cleft[-1]))
_shap_tree.XGBTreeModelLoader.__init__ = _patched

from fair_holdout_comparison import (
    load_features, load_targets, WITH_DATA_DIR, WITHOUT_DATA_DIR, OUTPUT_DIR,
)
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════
FS = 14
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

    model_b = joblib.load(os.path.join(STAGE3_DIR, f"model_before_{target}.pkl"))
    model_a = joblib.load(os.path.join(STAGE3_DIR, f"model_after_{target}.pkl"))
    sc_b = joblib.load(os.path.join(STAGE3_DIR, f"scaler_before_{target}.pkl"))
    sc_a = joblib.load(os.path.join(STAGE3_DIR, f"scaler_after_{target}.pkl"))
    with open(os.path.join(STAGE3_DIR, f"best_features_{target}.json")) as f:
        best_feats = json.load(f)

    X_all = load_features(WITH_DATA_DIR); y_all = load_targets(WITH_DATA_DIR)
    X_wo = load_features(WITHOUT_DATA_DIR)
    N_LIT = len(X_wo); is_lit = np.arange(len(X_all)) < N_LIT
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

    print("  Computing SHAP ...")
    explainer_b = shap.TreeExplainer(model_b)
    shap_b_vals = explainer_b(X_test_b_s).values
    explainer_a = shap.TreeExplainer(model_a)
    shap_a_vals = explainer_a(X_test_a_s).values

    # Global sorting: SHAP After mean(|SHAP|) descending
    mean_abs = np.mean(np.abs(shap_a_vals), axis=0)
    order_idx = np.argsort(mean_abs)[::-1]
    top_n = min(20, n_feat)
    order_idx = order_idx[:top_n]

    shap_b_ord = shap_b_vals[:, order_idx]
    shap_a_ord = shap_a_vals[:, order_idx]
    X_b_ord = X_test_b_s[:, order_idx]
    X_a_ord = X_test_a_s[:, order_idx]
    names_ord = [short_names[i] for i in order_idx]

    # ── Canvas ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(28, 4), dpi=300)
    outer = fig.add_gridspec(1, 2, figure=fig,
                              left=0.08, right=0.97, top=0.95, bottom=0.08,
                              wspace=0.02)

    # (a) SHAP Before
    ax_a = fig.add_subplot(outer[0, 0])
    shap.summary_plot(shap_b_ord, X_b_ord, feature_names=names_ord,
                      plot_type="dot", cmap=SHAP_CMAP_B, max_display=top_n,
                      show=False)
    ax_a = plt.gca()
    for lbl in ax_a.get_yticklabels(): lbl.set_fontsize(FS - 4)
    bold_axes(ax_a)
    ax_a.text(-0.12, 1.03, "(a)", transform=ax_a.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # (b) SHAP After
    ax_b = fig.add_subplot(outer[0, 1])
    shap.summary_plot(shap_a_ord, X_a_ord, feature_names=names_ord,
                      plot_type="dot", cmap=SHAP_CMAP_A, max_display=top_n,
                      show=False)
    ax_b = plt.gca()
    for lbl in ax_b.get_yticklabels(): lbl.set_fontsize(FS - 4)
    bold_axes(ax_b)
    ax_b.text(-0.12, 1.03, "(b)", transform=ax_b.transAxes,
              fontsize=FS + 4, fontweight="bold", va="bottom", ha="right")

    # ── Auto spacing ─────────────────────────────────────────────────
    fig.canvas.draw()
    def _ytick_width(ax, fig):
        r = fig.canvas.get_renderer(); m = 0.0
        for lbl in ax.get_yticklabels():
            bb = lbl.get_window_extent(r)
            if bb is not None: m = max(m, bb.width)
        return m / fig.dpi if fig.dpi else 0.0
    left_w = _ytick_width(ax_a, fig)
    fig_w = fig.get_size_inches()[0]
    usable = fig_w * (0.97 - 0.08)
    needed = 2.5 * left_w
    ws = needed / (usable - needed) if usable > needed else 0.35
    outer.update(wspace=max(ws, 0.02))

    # ── Save ─────────────────────────────────────────────────────
    for fmt in ["png", "pdf", "svg"]:
        out = os.path.join(SAVE_DIR, f"Combined_SHAP_Beeswarm_LOI.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
        print(f"  -> {fmt}")
    plt.close(fig)
    print("[OK] Combined_SHAP_Beeswarm_LOI saved (png/pdf/svg)")


if __name__ == "__main__":
    combine()
