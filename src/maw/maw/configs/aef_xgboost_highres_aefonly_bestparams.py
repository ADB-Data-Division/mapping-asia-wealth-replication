from pathlib import Path

# --- Paths
ROOT = Path.home() / "Projects/mapping-asia-wealth"
MODEL_OUTDIR = ROOT / "data-clean/models/xgboost/"
PREDICTIONS_OUTDIR = ROOT / "data-clean/outputs/predictions/xgboost/"

LABELS_PATH = ROOT / "data-raw/covariates/country/phl/phl_country_labels.csv"
COVARIATES_PATH = ROOT / "data-raw/covariates/country/phl/phl_country_covs.parquet"
EMBEDDINGS_PATH = ROOT / "data-raw/covariates/country/phl/phl_country_aef.parquet"


# --- Training Params
SEED = 42
FIXED_PARAMS = {
    "tree_method": "hist",
    "random_state": 42
}

PARAM_BASE = {
    "learning_rate": [0.005],
    "subsample": [0.7],
    "colsample_bytree": [0.6],
    "min_child_weight": [3],
    "reg_lambda": [2.0],
    "reg_alpha": [0.5],
    "gamma": [1.0],
}

PARAM_DEPTHWISE = {"max_depth": [10]}
PARAM_LOSSGUIDE = {"grow_policy": ["lossguide"], "max_leaves": [63, 127, 255, 511], "max_depth": [0]}
