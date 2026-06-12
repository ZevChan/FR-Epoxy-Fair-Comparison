# Fair_Comparison — Three-Stage Fair Comparison Framework

Comparing the performance improvement of ML models in flame-retardant epoxy property prediction before and after incorporating experimental data.

## Project Structure

```
Fair_Comparison_GitHub/
├── fair_holdout_comparison.py   # Three-stage core pipeline (Baseline → SHAP feature selection → Optuna optimization)
├── combine_pipeline.py          # 4-row combined publication figure
├── combine_panels.py            # Multi-panel comparison figure
├── combine_comparison.py        # Comprehensive comparison figure
├── combine_shap_loi.py          # SHAP stacked figure (LOI)
├── combine_shap_beeswarm_loi.py # SHAP Beeswarm stacked figure (LOI)
├── combine_importance_loi.py    # Feature importance combined figure (LOI)
├── plot_*.py                    # Individual visualization scripts
├── experimental_data.csv        # Experimental dataset
├── literature_data.csv          # Literature dataset
├── .gitignore
└── README.md
```

## Dependencies

Use the Conda environment `DFT_FR_GNN_transformer`:

```bash
conda activate DFT_FR_GNN_transformer
```

Main dependencies:
- Python 3.10+
- pandas, numpy, scikit-learn
- xgboost
- optuna
- matplotlib, seaborn
- umap-learn
- rdkit
- shap
- chardet, joblib

## Usage

### 1. Run the Three-Stage Pipeline

```bash
python fair_holdout_comparison.py
```

> ⚠️ **Note**: `fair_holdout_comparison.py` requires feature files (Curing_Des.csv, FR_Des.csv, EP_Des.csv, Target.csv, etc.). These files are not included in this repository — please contact the author if needed (zhongw_chen@njtech.edu.cn) .

### 2. Generate Visualizations

```bash
python plot_umap_raw.py          # UMAP raw data dimensionality reduction
python plot_stage1_baseline.py   # Stage 1 baseline bar chart
python plot_stage2_topk.py       # Stage 2 Top-K scan
python plot_optuna_history.py    # Optuna optimization history
python plot_pva.py               # Predicted vs Actual
python plot_shap.py              # SHAP analysis
# ... and other plot_*.py scripts
```

### 3. Generate Combined Figures

```bash
python combine_pipeline.py       # 4-row publication figure
python combine_panels.py         # Multi-panel comparison
python combine_comparison.py     # Comprehensive comparison
```

## Three Stages Overview

| Stage | Description | Details |
|-------|-------------|---------|
| 1 | Baseline modeling | XGBoost default params, Before (literature only) vs After (literature + experiment) |
| 2 | SHAP feature selection | SHAP ranking + Top-K scan to determine optimal feature count |
| 3 | Optuna hyperparameter optimization | Hyperparameter search on optimal feature subset |

## Target Properties

LOI, UL94_Rating, pHRR, THR, TSP, Flexural_Strength
