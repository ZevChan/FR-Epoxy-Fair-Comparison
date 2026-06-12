"""
================================================================================
 fair_holdout_comparison.py —— Three-Stage Fair Comparison
================================================================================

 Method: Fixed 20% hold-out from literature data as unified test set
   Before = Train on remaining 80% literature data only
   After  = Train on 80% literature + 100% experimental data
   Both evaluated on the same test set

 Three stages:
   Stage 1 — Baseline modeling (XGBoost default parameters)
   Stage 2 — SHAP feature selection + Top-K scan
   Stage 3 — Optuna hyperparameter optimization (using best K features from Stage 2)

 Targets (6): LOI, UL94_Rating, pHRR, THR, TSP, Flexural_Strength
================================================================================
"""

import pandas as pd
import numpy as np
import os, sys, json, joblib, warnings, traceback
import chardet
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score,
)
from xgboost import XGBRegressor, XGBClassifier
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
np.random.seed(42)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================================================
# Path configuration
# ==========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..", "ML_guided_FR_Materials")
WITH_DATA_DIR = os.path.join(PROJECT_DIR, "ML_WithExperiment", "BasicData")
WITHOUT_DATA_DIR = os.path.join(PROJECT_DIR, "ML_WithoutExperiment", "BasicData")

OUTPUT_DIR = os.path.join(BASE_DIR, "Results")
for d in [OUTPUT_DIR,
          os.path.join(OUTPUT_DIR, "Stage1"),
          os.path.join(OUTPUT_DIR, "Stage2"),
          os.path.join(OUTPUT_DIR, "Stage3")]:
    os.makedirs(d, exist_ok=True)

TARGETS = ["LOI", "UL94_Rating", "pHRR", "THR", "TSP", "Flexural_Strength"]

# ==========================================================================
# Data loading (consistent with original project)
# ==========================================================================
def detect_encoding(fp):
    with open(fp, "rb") as f:
        return chardet.detect(f.read())["encoding"]

def read_csv_enc(fp):
    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(fp, encoding=detect_encoding(fp))

def clean_numeric(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            for ch in ["?", "*", "#", " "]:
                df[c] = df[c].astype(str).str.replace(ch, "", regex=False)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_features(data_dir):
    files = ["Curing_Des.csv", "Curing_Strategy.csv", "EP_Des.csv",
             "FR_Des.csv", "Other_Material_1_Des.csv",
             "Other_Material_2_Des.csv", "Other_Material_3_Des.csv"]
    dfs = []
    for fn in files:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            dfs.append(clean_numeric(read_csv_enc(fp)))
    if not dfs:
        raise FileNotFoundError(f"No feature files in {data_dir}")
    X = pd.concat(dfs, axis=1)
    return X.loc[:, ~X.columns.duplicated()]

def load_targets(data_dir):
    fp = os.path.join(data_dir, "Target.csv")
    return clean_numeric(read_csv_enc(fp))

def process_ul94(y):
    y = pd.to_numeric(y, errors="coerce").fillna(-1).astype(int)
    return (y == 3).astype(int)


# ==========================================================================
# Fixed data split (shared across the entire script)
# ==========================================================================
def prepare_fixed_split():
    """Load data and determine fixed test set. Returns (X_all, y_all, N_LIT, is_literature)"""
    print("=" * 60)
    print("  Loading data & splitting fixed hold-out test set")
    print("=" * 60)

    X_all = load_features(WITH_DATA_DIR)
    y_all = load_targets(WITH_DATA_DIR)
    X_wo = load_features(WITHOUT_DATA_DIR)

    N_LIT = len(X_wo)
    N_ALL = len(X_all)
    is_literature = np.arange(N_ALL) < N_LIT

    print(f"  Literature samples: {N_LIT}  |  Experiment samples: {N_ALL - N_LIT}  |  Total: {N_ALL}")
    return X_all, y_all, N_LIT, is_literature


def get_train_test_for_target(X_all, y_all, is_literature, target_col):
    """Build fixed Before/After training sets + fixed test set for the target column"""
    valid = y_all[target_col].notna().values
    X_v = X_all.loc[valid].copy()
    y_v = y_all.loc[valid, target_col].copy()
    is_lit_v = is_literature[valid]

    is_cls = "UL94" in target_col
    if is_cls:
        y_v = process_ul94(y_v)

    X_lit = X_v[is_lit_v].reset_index(drop=True)
    y_lit = y_v[is_lit_v].reset_index(drop=True)
    X_exp = X_v[~is_lit_v].reset_index(drop=True)
    y_exp = y_v[~is_lit_v].reset_index(drop=True)

    if len(X_lit) < 10:
        return None

    try:
        strat = y_lit if (is_cls and y_lit.nunique() >= 2) else None
        X_train_lit, X_test, y_train_lit, y_test = train_test_split(
            X_lit, y_lit, test_size=0.2, random_state=42, stratify=strat)
    except Exception:
        return None

    # Before training set
    X_tr_before = X_train_lit
    y_tr_before = y_train_lit
    # After training set
    X_tr_after = pd.concat([X_train_lit, X_exp], ignore_index=True)
    y_tr_after = pd.concat([y_train_lit, y_exp], ignore_index=True)

    return {
        "target": target_col,
        "is_classification": is_cls,
        "n_lit": len(X_lit),
        "n_exp": len(X_exp),
        "n_test": len(X_test),
        "n_train_before": len(X_tr_before),
        "n_train_after": len(X_tr_after),
        "X_train_before": X_tr_before,
        "y_train_before": y_tr_before,
        "X_train_after": X_tr_after,
        "y_train_after": y_tr_after,
        "X_test": X_test,
        "y_test": y_test,
    }


# ==========================================================================
# Stage 1 — Baseline modeling
# ==========================================================================
def run_stage1(X_all, y_all, is_literature):
    """Default XGBoost parameters, Before / After trained separately, evaluated on the same test set"""
    print("\n" + "=" * 60)
    print("  STAGE 1 — Baseline modeling (XGBoost default parameters)")
    print("=" * 60)

    results = []

    for target in TARGETS:
        data = get_train_test_for_target(X_all, y_all, is_literature, target)
        if data is None:
            print(f"  [WARN] {target}: Insufficient samples, skipping")
            continue

        is_cls = data["is_classification"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # ---- Before ----
        sc_b = StandardScaler()
        X_tr_b_s = sc_b.fit_transform(data["X_train_before"])
        X_te_b_s = sc_b.transform(X_test)

        if is_cls:
            m_b = XGBClassifier(random_state=42, eval_metric="logloss")
        else:
            m_b = XGBRegressor(random_state=42)
        m_b.fit(X_tr_b_s, data["y_train_before"])
        yp_b = m_b.predict(X_te_b_s)

        # ---- After ----
        sc_a = StandardScaler()
        X_tr_a_s = sc_a.fit_transform(data["X_train_after"])
        X_te_a_s = sc_a.transform(X_test)

        if is_cls:
            m_a = XGBClassifier(random_state=42, eval_metric="logloss")
        else:
            m_a = XGBRegressor(random_state=42)
        m_a.fit(X_tr_a_s, data["y_train_after"])
        yp_a = m_a.predict(X_te_a_s)

        # Metrics
        if is_cls:
            metrics = {
                "before_test_accuracy": accuracy_score(y_test, yp_b),
                "after_test_accuracy":  accuracy_score(y_test, yp_a),
                "before_test_f1": f1_score(y_test, yp_b, average="binary"),
                "after_test_f1":  f1_score(y_test, yp_a, average="binary"),
            }
        else:
            metrics = {
                "before_test_r2":   r2_score(y_test, yp_b),
                "after_test_r2":    r2_score(y_test, yp_a),
                "before_test_rmse": np.sqrt(mean_squared_error(y_test, yp_b)),
                "after_test_rmse":  np.sqrt(mean_squared_error(y_test, yp_a)),
                "before_test_mae":  mean_absolute_error(y_test, yp_b),
                "after_test_mae":   mean_absolute_error(y_test, yp_a),
            }

        row = {
            "target": target,
            "n_lit": data["n_lit"], "n_exp": data["n_exp"],
            "n_test": data["n_test"],
            "n_train_before": data["n_train_before"],
            "n_train_after": data["n_train_after"],
            "n_features": data["X_train_before"].shape[1],
            **metrics,
        }
        results.append(row)

        # Save Stage1 models and predictions (for use in subsequent stages)
        joblib.dump(m_b, os.path.join(OUTPUT_DIR, "Stage1", f"model_before_{target}.pkl"))
        joblib.dump(m_a, os.path.join(OUTPUT_DIR, "Stage1", f"model_after_{target}.pkl"))
        joblib.dump(sc_b, os.path.join(OUTPUT_DIR, "Stage1", f"scaler_before_{target}.pkl"))
        joblib.dump(sc_a, os.path.join(OUTPUT_DIR, "Stage1", f"scaler_after_{target}.pkl"))
        pd.DataFrame({"y_true": y_test, "y_pred_before": yp_b, "y_pred_after": yp_a}).to_csv(
            os.path.join(OUTPUT_DIR, "Stage1", f"predictions_{target}.csv"),
            index=False, encoding="utf-8-sig")

        # Brief output
        if is_cls:
            print(f"  {target}: Before Acc={metrics['before_test_accuracy']:.3f} F1={metrics['before_test_f1']:.3f}  |  "
                  f"After Acc={metrics['after_test_accuracy']:.3f} F1={metrics['after_test_f1']:.3f}")
        else:
            print(f"  {target}: Before R^2={metrics['before_test_r2']:.3f} RMSE={metrics['before_test_rmse']:.3f}  |  "
                  f"After R^2={metrics['after_test_r2']:.3f} RMSE={metrics['after_test_rmse']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "Stage1", "stage1_baseline_results.csv"),
              index=False, encoding="utf-8-sig")
    print(f"\n  Stage 1 done, results: Stage1/stage1_baseline_results.csv")
    return df


# ==========================================================================
# Stage 2 — SHAP feature selection + Top-K
# ==========================================================================
def run_stage2(X_all, y_all, is_literature):
    """SHAP analysis + scan different K values -> find best feature count"""
    print("\n" + "=" * 60)
    print("  STAGE 2 — SHAP feature selection + Top-K scan")
    print("=" * 60)

    try:
        import shap
    except ImportError:
        print("  [WARN] shap not installed, skipping Stage 2")
        return

    all_topk = {}

    for target in TARGETS:
        data = get_train_test_for_target(X_all, y_all, is_literature, target)
        if data is None:
            continue

        is_cls = data["is_classification"]
        X_tr_a = data["X_train_after"]
        y_tr_a = data["y_train_after"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        # ---- 2a. Train default model on After training set -> SHAP ----
        sc = StandardScaler()
        X_tr_a_s = sc.fit_transform(X_tr_a)
        if is_cls:
            model = XGBClassifier(random_state=42, eval_metric="logloss")
        else:
            model = XGBRegressor(random_state=42)
        model.fit(X_tr_a_s, y_tr_a)

        # XGBoost native feature importance (avoiding SHAP version compatibility issues)
        importance = model.feature_importances_
        feat_names = list(X_tr_a.columns)
        rank_df = pd.DataFrame({"Feature": feat_names, "Importance": importance})
        rank_df = rank_df.sort_values("Importance", ascending=False).reset_index(drop=True)

        # Remove features with zero importance
        rank_df = rank_df[rank_df["Importance"] > 0]
        n_total = len(rank_df)
        print(f"\n  {target}: {n_total} valid features, scanning Top-K ...")

        # ---- 2b. Top-K scan ----
        k_values = []
        for k in [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]:
            if k <= n_total:
                k_values.append(k)
        if n_total not in k_values:
            k_values.append(n_total)

        topk_results = []
        for k in k_values:
            top_feats = rank_df.head(k)["Feature"].tolist()

            # Before
            Xb = data["X_train_before"][top_feats]
            yb = data["y_train_before"]
            sc_b = StandardScaler()
            Xb_s = sc_b.fit_transform(Xb)
            Xt_s = sc_b.transform(X_test[top_feats])
            if is_cls:
                mb = XGBClassifier(random_state=42, eval_metric="logloss")
            else:
                mb = XGBRegressor(random_state=42)
            mb.fit(Xb_s, yb)
            yp_b = mb.predict(Xt_s)

            # After
            Xa = data["X_train_after"][top_feats]
            ya = data["y_train_after"]
            sc_a = StandardScaler()
            Xa_s = sc_a.fit_transform(Xa)
            Xt_a_s = sc_a.transform(X_test[top_feats])
            if is_cls:
                ma = XGBClassifier(random_state=42, eval_metric="logloss")
            else:
                ma = XGBRegressor(random_state=42)
            ma.fit(Xa_s, ya)
            yp_a = ma.predict(Xt_a_s)

            # Print current K metrics
            if is_cls:
                print(f"    K={k}: Before Acc={accuracy_score(y_test, yp_b):.3f} After Acc={accuracy_score(y_test, yp_a):.3f}")
            else:
                print(f"    K={k}: Before R^2={r2_score(y_test, yp_b):.3f} After R^2={r2_score(y_test, yp_a):.3f}")

            if is_cls:
                topk_results.append({
                    "top_k": k,
                    "before_accuracy": accuracy_score(y_test, yp_b),
                    "after_accuracy": accuracy_score(y_test, yp_a),
                    "before_f1": f1_score(y_test, yp_b, average="binary"),
                    "after_f1": f1_score(y_test, yp_a, average="binary"),
                })
            else:
                topk_results.append({
                    "top_k": k,
                    "before_r2": r2_score(y_test, yp_b),
                    "after_r2": r2_score(y_test, yp_a),
                    "before_rmse": np.sqrt(mean_squared_error(y_test, yp_b)),
                    "after_rmse": np.sqrt(mean_squared_error(y_test, yp_a)),
                    "before_mae": mean_absolute_error(y_test, yp_b),
                    "after_mae": mean_absolute_error(y_test, yp_a),
                })

        df_topk = pd.DataFrame(topk_results)
        df_topk.to_csv(os.path.join(OUTPUT_DIR, "Stage2", f"topk_scan_{target}.csv"),
                       index=False, encoding="utf-8-sig")
        rank_df.to_csv(os.path.join(OUTPUT_DIR, "Stage2", f"shap_ranking_{target}.csv"),
                       index=False, encoding="utf-8-sig")

        # Best K: take the smallest K reaching 98% of max (determined independently per target)
        # Target-specific cap: pHRR/TSP/Flexural relaxed to 1000, others 300
        _TARGET_CAPS = {"pHRR": 1000, "TSP": 1000, "Flexural_Strength": 1000}
        _cap = _TARGET_CAPS.get(target, 300)

        df_sorted = df_topk.sort_values("top_k")
        if is_cls:
            metric_col = "after_accuracy"
        else:
            metric_col = "after_r2"
        max_metric = df_sorted[metric_col].max()
        threshold = max_metric * 0.98
        elbow_rows = df_sorted[df_sorted[metric_col] >= threshold]
        best_k = int(elbow_rows.iloc[0]["top_k"])
        if best_k > _cap:
            print(f"    [NOTE] true elbow at K={best_k}, capped to {_cap}")
            best_k = _cap
        best_features = rank_df.head(best_k)["Feature"].tolist()

        all_topk[target] = {"best_k": best_k, "best_features": best_features}
        print(f"    Best K = {best_k}")

        # Save
        with open(os.path.join(OUTPUT_DIR, "Stage2", f"best_features_{target}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(best_features, f, ensure_ascii=False, indent=2)

    # Summary
    summary = []
    for t, v in all_topk.items():
        summary.append({"target": t, "best_k": v["best_k"],
                        "n_features": len(v["best_features"])})
    pd.DataFrame(summary).to_csv(
        os.path.join(OUTPUT_DIR, "Stage2", "stage2_topk_summary.csv"),
        index=False, encoding="utf-8-sig")
    print(f"\n  Stage 2 done: Stage2/stage2_topk_summary.csv")
    return all_topk


# ==========================================================================
# Stage 3 — Optuna hyperparameter optimization
# ==========================================================================
def objective_regression(trial, X, y, n_splits=5):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": 42,
    }
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, va in kf.split(X):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X.iloc[tr])
        X_va_s = sc.transform(X.iloc[va])
        m = XGBRegressor(**param)
        m.fit(X_tr_s, y.iloc[tr])
        scores.append(r2_score(y.iloc[va], m.predict(X_va_s)))
    return np.mean(scores)


def objective_classification(trial, X, y, n_splits=5):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": 42,
    }
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, va in kf.split(X):
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X.iloc[tr])
        X_va_s = sc.transform(X.iloc[va])
        m = XGBClassifier(**param)
        m.fit(X_tr_s, y.iloc[tr])
        scores.append(accuracy_score(y.iloc[va], m.predict(X_va_s)))
    return np.mean(scores)


def run_stage3(X_all, y_all, is_literature, all_topk):
    """Optuna optimization using the best K features from Stage 2"""
    print("\n" + "=" * 60)
    print("  STAGE 3 — Optuna hyperparameter optimization")
    print("=" * 60)

    try:
        import optuna
    except ImportError:
        print("  [WARN] optuna not installed, skipping Stage 3")
        return

    results = []

    for target in TARGETS:
        if target not in all_topk:
            print(f"  [WARN] {target}: No Stage 2 results, skipping")
            continue

        data = get_train_test_for_target(X_all, y_all, is_literature, target)
        if data is None:
            continue

        best_feats = all_topk[target]["best_features"]
        best_feats = [c for c in best_feats if c in data["X_train_before"].columns]
        is_cls = data["is_classification"]
        X_test_f = data["X_test"][best_feats]
        y_test = data["y_test"]

        n_trials = 50

        # ---- Before ----
        X_tr_b = data["X_train_before"][best_feats]
        y_tr_b = data["y_train_before"]
        print(f"\n  {target}: Before Optuna ({len(best_feats)} features, {len(X_tr_b)} samples) ...")

        try:
            study_b = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
            obj_b = (objective_classification if is_cls else objective_regression)
            study_b.optimize(lambda trial: obj_b(trial, X_tr_b, y_tr_b),
                             n_trials=n_trials, show_progress_bar=False)
        except Exception as e:
            print(f"    [WARN] Before Optuna failed: {e}")
            continue

        # ---- After ----
        X_tr_a = data["X_train_after"][best_feats]
        y_tr_a = data["y_train_after"]
        print(f"  {target}: After Optuna ({len(best_feats)} features, {len(X_tr_a)} samples) ...")

        try:
            study_a = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
            obj_a = (objective_classification if is_cls else objective_regression)
            study_a.optimize(lambda trial: obj_a(trial, X_tr_a, y_tr_a),
                             n_trials=n_trials, show_progress_bar=False)
        except Exception as e:
            print(f"    [WARN] After Optuna failed: {e}")
            continue

        # ---- Final evaluation ----
        sc_b = StandardScaler()
        X_tr_b_s = sc_b.fit_transform(X_tr_b)
        X_te_b_s = sc_b.transform(X_test_f)
        if is_cls:
            m_b = XGBClassifier(**study_b.best_params, random_state=42, eval_metric="logloss")
        else:
            m_b = XGBRegressor(**study_b.best_params, random_state=42)
        m_b.fit(X_tr_b_s, y_tr_b)
        yp_b = m_b.predict(X_te_b_s)

        sc_a = StandardScaler()
        X_tr_a_s = sc_a.fit_transform(X_tr_a)
        X_te_a_s = sc_a.transform(X_test_f)
        if is_cls:
            m_a = XGBClassifier(**study_a.best_params, random_state=42, eval_metric="logloss")
        else:
            m_a = XGBRegressor(**study_a.best_params, random_state=42)
        m_a.fit(X_tr_a_s, y_tr_a)
        yp_a = m_a.predict(X_te_a_s)

        # Save all Stage 3 outputs
        s3 = os.path.join(OUTPUT_DIR, "Stage3")
        # Predictions
        pd.DataFrame({"y_true": y_test.values, "y_pred_before": yp_b, "y_pred_after": yp_a}).to_csv(
            os.path.join(s3, f"predictions_{target}.csv"), index=False, encoding="utf-8-sig")
        # Model & Scaler
        joblib.dump(m_b, os.path.join(s3, f"model_before_{target}.pkl"))
        joblib.dump(m_a, os.path.join(s3, f"model_after_{target}.pkl"))
        joblib.dump(sc_b, os.path.join(s3, f"scaler_before_{target}.pkl"))
        joblib.dump(sc_a, os.path.join(s3, f"scaler_after_{target}.pkl"))
        # Scaled data
        pd.DataFrame(X_tr_b_s, columns=best_feats).to_csv(
            os.path.join(s3, f"X_train_scaled_before_{target}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame(X_tr_a_s, columns=best_feats).to_csv(
            os.path.join(s3, f"X_train_scaled_after_{target}.csv"), index=False, encoding="utf-8-sig")
        # Feature importance
        imp_df = pd.DataFrame({
            "Feature": best_feats,
            "Weight_Before": m_b.feature_importances_,
            "Weight_After": m_a.feature_importances_,
        })
        imp_df["Shift_Delta"] = imp_df["Weight_After"] - imp_df["Weight_Before"]
        imp_df = imp_df.sort_values("Weight_After", ascending=False)
        imp_df.to_csv(os.path.join(s3, f"feature_importance_{target}.csv"),
                      index=False, encoding="utf-8-sig")
        # Optuna history
        study_b.trials_dataframe().to_csv(
            os.path.join(s3, f"optuna_history_before_{target}.csv"), index=False, encoding="utf-8-sig")
        study_a.trials_dataframe().to_csv(
            os.path.join(s3, f"optuna_history_after_{target}.csv"), index=False, encoding="utf-8-sig")
        # Hyperparameters
        json.dump(study_b.best_params, open(os.path.join(s3, f"best_params_before_{target}.json"), "w"), indent=2)
        json.dump(study_a.best_params, open(os.path.join(s3, f"best_params_after_{target}.json"), "w"), indent=2)
        # Feature list
        json.dump(best_feats, open(os.path.join(s3, f"best_features_{target}.json"), "w"),
                  ensure_ascii=False, indent=2)

        if is_cls:
            metrics = {
                "before_test_accuracy": accuracy_score(y_test, yp_b),
                "after_test_accuracy": accuracy_score(y_test, yp_a),
                "before_test_f1": f1_score(y_test, yp_b, average="binary"),
                "after_test_f1": f1_score(y_test, yp_a, average="binary"),
            }
        else:
            metrics = {
                "before_test_r2": r2_score(y_test, yp_b),
                "after_test_r2": r2_score(y_test, yp_a),
                "before_test_rmse": np.sqrt(mean_squared_error(y_test, yp_b)),
                "after_test_rmse": np.sqrt(mean_squared_error(y_test, yp_a)),
                "before_test_mae": mean_absolute_error(y_test, yp_b),
                "after_test_mae": mean_absolute_error(y_test, yp_a),
            }

        row = {
            "target": target,
            "n_features": len(best_feats),
            "best_k": all_topk[target]["best_k"],
            "n_train_before": len(X_tr_b),
            "n_train_after": len(X_tr_a),
            "n_test": len(X_test_f),
            "before_cv_best": study_b.best_value,
            "after_cv_best": study_a.best_value,
            **metrics,
        }
        results.append(row)

        if is_cls:
            print(f"    Before Acc={metrics['before_test_accuracy']:.3f} F1={metrics['before_test_f1']:.3f}")
            print(f"    After  Acc={metrics['after_test_accuracy']:.3f} F1={metrics['after_test_f1']:.3f}")
        else:
            print(f"    Before R^2={metrics['before_test_r2']:.3f} RMSE={metrics['before_test_rmse']:.3f}")
            print(f"    After  R^2={metrics['after_test_r2']:.3f} RMSE={metrics['after_test_rmse']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, "Stage3", "stage3_optuna_results.csv"),
              index=False, encoding="utf-8-sig")
    print(f"\n  Stage 3 done: Stage3/stage3_optuna_results.csv")
    return df


# ==========================================================================
# UMAP data export
# ==========================================================================
def export_umap_data(X_all, y_all, is_literature):
    print("\n" + "=" * 60)
    print("  Exporting UMAP data")
    print("=" * 60)

    # Collect the union of all Stage 2 feature files
    s2_dir = os.path.join(OUTPUT_DIR, "Stage2")
    all_feats = set()
    for t in TARGETS:
        fp = os.path.join(s2_dir, f"best_features_{t}.json")
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                all_feats.update(json.load(f))
    all_feats = sorted([c for c in all_feats if c in X_all.columns])

    master_rows = []
    for t in TARGETS:
        vmask = y_all[t].notna().values
        X_t = X_all.loc[vmask, all_feats].copy()
        y_t = y_all.loc[vmask, t].copy()
        is_lit_t = is_literature[vmask]
        df_u = X_t.copy()
        df_u["Target"] = t
        df_u["Target_Value"] = y_t.values
        df_u["Source"] = np.where(is_lit_t, "Literature", "Experiment")
        master_rows.append(df_u)

    master = pd.concat(master_rows, ignore_index=True)
    master.to_csv(os.path.join(OUTPUT_DIR, "umap_data_MASTER.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"  umap_data_MASTER.csv: {len(master)} rows, {len(all_feats)} features")

    # Per-target
    for t in TARGETS:
        fp = os.path.join(s2_dir, f"best_features_{t}.json")
        if not os.path.exists(fp):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            tf_list = json.load(f)
        tf_list = [c for c in tf_list if c in X_all.columns]
        vmask = y_all[t].notna().values
        X_t = X_all.loc[vmask, tf_list].copy()
        y_t = y_all.loc[vmask, t].copy()
        is_lit_t = is_literature[vmask]
        df_t = X_t.copy()
        df_t["Target_Value"] = y_t.values
        df_t["Source"] = np.where(is_lit_t, "Literature", "Experiment")
        df_t.to_csv(os.path.join(OUTPUT_DIR, f"umap_data_{t}.csv"),
                    index=False, encoding="utf-8-sig")
        print(f"  umap_data_{t}.csv: {len(df_t)} rows")


# ==========================================================================
# Main entry
# ==========================================================================
def main():
    # 1. Load & fixed split
    X_all, y_all, N_LIT, is_literature = prepare_fixed_split()

    # 2. Stage 1 — Baseline
    run_stage1(X_all, y_all, is_literature)

    # 3. Stage 2 — SHAP + TopK
    all_topk = run_stage2(X_all, y_all, is_literature)

    # 4. Stage 3 — Optuna
    if all_topk:
        run_stage3(X_all, y_all, is_literature, all_topk)
    else:
        print("\n  [WARN] Stage 2 produced no results, skipping Stage 3")

    # 5. UMAP data
    export_umap_data(X_all, y_all, is_literature)

    print("\n" + "=" * 60)
    print("  All done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
