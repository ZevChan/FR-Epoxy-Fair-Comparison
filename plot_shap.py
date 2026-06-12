"""
================================================================================
 plot_shap.py — SHAP beeswarm plot + feature importance (full test set)
================================================================================
 Directly loads Stage 3 model / Scaler / feature list,
 Rebuilds the fixed test set from raw data (consistent with fair_holdout_comparison),
 Computes SHAP values for the full test set using shap.TreeExplainer, outputs beeswarm plot.
================================================================================
"""
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import json
import re
import warnings
from matplotlib.colors import LinearSegmentedColormap

# Monkey-patch: fix XGBoost 3.x base_score format for SHAP 0.49
import shap.explainers._tree as _shap_tree
_orig_xgb_loader_init = _shap_tree.XGBTreeModelLoader.__init__

def _patched_xgb_loader_init(self, xgb_model):
    """Patched __init__ that handles XGBoost 3.x base_score array format."""
    import io
    import xgboost as xgb
    from shap.explainers._tree import _check_xgboost_version, decode_ubjson_buffer

    _check_xgboost_version(xgb.__version__)
    model: xgb.Booster = xgb_model

    raw = xgb_model.save_raw(raw_format="ubj")
    with io.BytesIO(raw) as fd:
        jmodel = decode_ubjson_buffer(fd)

    learner = jmodel["learner"]
    learner_model_param = learner["learner_model_param"]
    objective = learner["objective"]

    booster = learner["gradient_booster"]
    n_classes = max(int(learner_model_param["num_class"]), 1)
    n_targets = max(int(learner_model_param["num_target"]), 1)
    n_targets = max(n_targets, n_classes)

    if "gbtree" in booster and "model" not in booster:
        booster = booster["gbtree"]
    if booster["model"].get("iteration_indptr", None) is not None:
        iteration_indptr = np.asarray(booster["model"]["iteration_indptr"], dtype=np.int32)
        diff = np.diff(iteration_indptr)
    else:
        n_parallel_trees = int(booster["model"]["gbtree_model_param"]["num_parallel_tree"])
        diff = np.repeat(n_targets * n_parallel_trees, model.num_boosted_rounds())
    if np.any(diff != diff[0]):
        raise ValueError("vector-leaf is not yet supported.:", diff)

    self.n_trees_per_iter = int(diff[0])
    self.n_targets = n_targets

    # ---- PATCH: handle XGBoost 3.x array-formatted base_score ----
    def _parse_base_score(val):
        if isinstance(val, str):
            nums = re.findall(r'[\d.]+', val)
            return float(nums[0]) if nums else 0.5
        return float(val)

    self.base_score = _parse_base_score(learner_model_param["base_score"])
    assert self.n_trees_per_iter > 0

    self.name_obj = objective["name"]
    self.name_gbm = booster["name"]
    base_score = _parse_base_score(learner_model_param["base_score"])
    import scipy
    if self.name_obj in ("binary:logistic", "reg:logistic"):
        self.base_score = scipy.special.logit(base_score)
    elif self.name_obj in (
        "reg:gamma", "reg:tweedie", "count:poisson",
        "survival:cox", "survival:aft",
    ):
        self.base_score = np.log(self.base_score)
    else:
        self.base_score = base_score

    self.num_feature = int(learner_model_param["num_feature"])
    self.num_class = int(learner_model_param["num_class"])

    trees = booster["model"]["trees"]
    self.num_trees = len(trees)

    self.node_parents = []
    self.node_cleft = []
    self.node_cright = []
    self.node_sindex = []
    self.children_default: list = []
    self.sum_hess = []

    self.values = []
    self.thresholds = []
    self.threshold_types = []
    self.features = []

    self.split_types = []
    self.categories = []

    feature_types = model.feature_types
    if feature_types is not None:
        cat_feature_indices: np.ndarray = np.where(np.asarray(feature_types) == "c")[0]
        if len(cat_feature_indices) == 0:
            self.cat_feature_indices = None
        else:
            self.cat_feature_indices = cat_feature_indices
    else:
        self.cat_feature_indices = None

    def to_integers(data):
        assert isinstance(data, list)
        return np.asanyarray(data, dtype=np.uint8)

    for i in range(self.num_trees):
        tree = trees[i]
        parents = np.asarray(tree["parents"])
        self.node_parents.append(parents)
        self.node_cleft.append(np.asarray(tree["left_children"], dtype=np.int32))
        self.node_cright.append(np.asarray(tree["right_children"], dtype=np.int32))
        self.node_sindex.append(np.asarray(tree["split_indices"], dtype=np.uint32))

        base_weight = np.asarray(tree["base_weights"], dtype=np.float32)
        if base_weight.size != self.node_cleft[-1].size:
            raise ValueError("vector-leaf is not yet supported.")

        default_left = to_integers(tree["default_left"])
        default_child = np.where(
            default_left == 1, self.node_cleft[-1], self.node_cright[-1]
        ).astype(np.int64)
        self.children_default.append(default_child)
        self.sum_hess.append(np.asarray(tree["sum_hessian"], dtype=np.float64))

        is_leaf = self.node_cleft[-1] == -1
        split_cond = np.asarray(tree["split_conditions"], dtype=np.float32)
        thresholds = np.where(is_leaf, 0.0, split_cond)
        thresholds = np.where(is_leaf, 0.0, np.nextafter(thresholds, -np.float32(np.inf)))
        threshold_types = np.zeros_like(thresholds, dtype=np.int32)

        self.values.append(np.where(is_leaf, split_cond, 0.0).reshape(-1, 1))
        self.thresholds.append(thresholds)
        self.threshold_types.append(threshold_types)

        split_idx = np.asarray(tree["split_indices"], dtype=np.int64)
        self.features.append(split_idx)

        split_types = to_integers(tree["split_type"])
        self.split_types.append(split_types)
        cat_segments = tree["categories_segments"]
        cat_sizes = tree["categories_sizes"]
        cat_nodes = tree["categories_nodes"]
        assert len(cat_segments) == len(cat_sizes) == len(cat_nodes)
        cats = tree["categories"]
        tree_categories = self.parse_categories(
            cat_nodes, cat_segments, cat_sizes, cats, self.node_cleft[-1]
        )
        self.categories.append(tree_categories)

_shap_tree.XGBTreeModelLoader.__init__ = _patched_xgb_loader_init

# ==================== Monkey-patch complete ====================

from fair_holdout_comparison import (
    load_features, load_targets, process_ul94,
    WITH_DATA_DIR, WITHOUT_DATA_DIR, OUTPUT_DIR
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ==================== Plot settings ====================
plt.rcParams.update({
    "font.family": "Arial", "font.size": 12,
    "axes.unicode_minus": False, "axes.linewidth": 0.8,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

SHAP_CMAP_BEFORE = LinearSegmentedColormap.from_list(
    "shap_before", ["#6EAA5E", "#E8E8E8", "#C91511"], N=256
)
SHAP_CMAP_AFTER = LinearSegmentedColormap.from_list(
    "shap_after", ["#E8923B", "#E8E8E8", "#5DA5DA"], N=256
)
COLOR_BEFORE = "#5DA5DA"
COLOR_AFTER  = "#C91511"

TARGETS = ["LOI", "pHRR", "THR", "TSP", "Flexural_Strength", "UL94_Rating"]

DISPLAY_NAMES = {
    "LOI": "LOI", "pHRR": "pHRR", "THR": "THR", "TSP": "TSP",
    "Flexural_Strength": "Flexural Strength", "UL94_Rating": "UL-94",
}

SAVE_DIR = os.path.join(BASE_DIR, "Graphs")
STAGE3_DIR = os.path.join(OUTPUT_DIR, "Stage3")
os.makedirs(SAVE_DIR, exist_ok=True)


def shorten_name(name, max_len=28):
    return name if len(name) <= max_len else name[:max_len - 1] + "\u2026"


def prepare_split():
    """Rebuild the fixed split that is completely consistent with fair_holdout_comparison."""
    X_all = load_features(WITH_DATA_DIR)
    y_all = load_targets(WITH_DATA_DIR)
    X_wo = load_features(WITHOUT_DATA_DIR)
    N_LIT = len(X_wo)
    is_literature = np.arange(len(X_all)) < N_LIT
    return X_all, y_all, is_literature


def get_test_data(X_all, y_all, is_literature, target, best_feats):
    """Rebuild the fixed test set for the target column and return scaled X_test."""
    valid = y_all[target].notna().values
    X_v = X_all.loc[valid].copy()
    y_v = y_all.loc[valid, target].copy()
    is_lit_v = is_literature[valid]

    is_cls = "UL94" in target
    if is_cls:
        y_v = process_ul94(y_v)

    X_lit = X_v[is_lit_v].reset_index(drop=True)
    y_lit = y_v[is_lit_v].reset_index(drop=True)

    if len(X_lit) < 10:
        return None, None, None

    strat = y_lit if (is_cls and y_lit.nunique() >= 2) else None
    _, X_test, _, y_test = train_test_split(
        X_lit, y_lit, test_size=0.2, random_state=42, stratify=strat)

    best_feats = [c for c in best_feats if c in X_test.columns]
    return X_test[best_feats].copy(), y_test, best_feats


def plot():
    print("Loading raw data & rebuilding test set split ...")
    X_all, y_all, is_literature = prepare_split()
    print(f"  Total samples: {len(X_all)}  |  Literature: {is_literature.sum()}  |  Experiment: {(~is_literature).sum()}")

    for target in TARGETS:
        is_cls = "UL94" in target

        # ---- Load Stage 3 outputs ----
        model_b_path = os.path.join(STAGE3_DIR, f"model_before_{target}.pkl")
        model_a_path = os.path.join(STAGE3_DIR, f"model_after_{target}.pkl")
        sc_b_path = os.path.join(STAGE3_DIR, f"scaler_before_{target}.pkl")
        sc_a_path = os.path.join(STAGE3_DIR, f"scaler_after_{target}.pkl")
        feats_path = os.path.join(STAGE3_DIR, f"best_features_{target}.json")

        missing = []
        for p, label in [(model_b_path, "model_b"), (model_a_path, "model_a"),
                         (sc_b_path, "sc_b"), (sc_a_path, "sc_a"), (feats_path, "feats")]:
            if not os.path.exists(p):
                missing.append(label)
        if missing:
            print(f"  [WARN] Skipping {target}: missing {', '.join(missing)}")
            continue

        model_b = joblib.load(model_b_path)
        model_a = joblib.load(model_a_path)
        sc_b = joblib.load(sc_b_path)
        sc_a = joblib.load(sc_a_path)
        with open(feats_path, "r") as f:
            best_feats = json.load(f)

        # ---- Rebuild test set ----
        X_test, y_test, feat_list = get_test_data(X_all, y_all, is_literature,
                                                   target, best_feats)
        if X_test is None:
            print(f"  [WARN] Skipping {target}: test set rebuild failed")
            continue
        print(f"\n  {target}: test set {len(X_test)} samples, {len(feat_list)} features")

        X_test_b_s = sc_b.transform(X_test)
        X_test_a_s = sc_a.transform(X_test)

        short_names = [shorten_name(f) for f in feat_list]
        display_name = DISPLAY_NAMES.get(target, target)

        # ---- SHAP TreeExplainer (patched for XGBoost 3.x) ----
        print(f"    Computing SHAP (Before) ...")
        explainer_b = shap.TreeExplainer(model_b)
        shap_b = explainer_b(X_test_b_s)
        if is_cls and isinstance(shap_b.values, list):
            shap_b_vals = shap_b.values[1]
        else:
            shap_b_vals = shap_b.values

        print(f"    Computing SHAP (After) ...")
        explainer_a = shap.TreeExplainer(model_a)
        shap_a = explainer_a(X_test_a_s)
        if is_cls and isinstance(shap_a.values, list):
            shap_a_vals = shap_a.values[1]
        else:
            shap_a_vals = shap_a.values

        # ========== Figure 1: Beeswarm plot Before ==========
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=600)
        shap.summary_plot(shap_b_vals, X_test_b_s, feature_names=short_names,
                          plot_type="dot", cmap=SHAP_CMAP_BEFORE, max_display=20,
                          show=False)
        ax = plt.gca()
        ax.set_title(f"SHAP Beeswarm: {display_name} \u2014 Before", fontsize=14,
                     fontweight="bold")
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"SHAP_Beeswarm_{target}_Before.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()

        # ========== Figure 2: Beeswarm plot After ==========
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=600)
        shap.summary_plot(shap_a_vals, X_test_a_s, feature_names=short_names,
                          plot_type="dot", cmap=SHAP_CMAP_AFTER, max_display=20,
                          show=False)
        ax = plt.gca()
        ax.set_title(f"SHAP Beeswarm: {display_name} \u2014 After", fontsize=14,
                     fontweight="bold")
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"SHAP_Beeswarm_{target}_After.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()

        print(f"    [OK] SHAP_Beeswarm_{target}")

        # ========== Figure 3: Feature importance bar chart ==========
        imp_b = model_b.feature_importances_
        imp_a = model_a.feature_importances_

        top_n = min(15, len(feat_list))
        top_idx = np.argsort(imp_a)[-top_n:][::-1]
        top_names = [short_names[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(7, 5), dpi=600)
        y_pos = np.arange(top_n)
        height = 0.35

        ax.barh(y_pos + height / 2, imp_b[top_idx], height,
                color=COLOR_BEFORE, alpha=0.85, edgecolor="white", linewidth=0.3,
                label="Before (Literature only)")
        ax.barh(y_pos - height / 2, imp_a[top_idx], height,
                color=COLOR_AFTER, alpha=0.85, edgecolor="white", linewidth=0.3,
                label="After (Literature + Experiment)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names, fontsize=10)
        ax.set_xlabel("Feature Importance Weight", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12, width=0.8)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight("bold")
        ax.set_title(f"Feature Importance: {display_name}", fontsize=14, fontweight="bold")
        ax.legend(frameon=True, fancybox=False, edgecolor="#333333", fontsize=14)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.3)
        plt.tight_layout()
        for fmt in ["png", "pdf", "svg"]:
            plt.savefig(os.path.join(SAVE_DIR, f"Feature_Importance_{target}.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close()
        print(f"    [OK] Feature_Importance_{target}")


if __name__ == "__main__":
    print("Generating SHAP beeswarm plot & feature importance (full test set) ...")
    plot()
    print("\nDone.")
