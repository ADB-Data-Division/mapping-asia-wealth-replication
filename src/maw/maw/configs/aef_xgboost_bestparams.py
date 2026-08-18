from pathlib import Path


# --- Paths
ROOT = Path.home() / "Projects/mapping-asia-wealth"
MODEL_OUTDIR = ROOT / "data-clean/models/xgboost/"
PREDICTIONS_OUTDIR = ROOT / "data-clean/outputs/predictions/xgboost/"

LABELS_PATH = ROOT / "data-clean/dhs/asia-training-labels-sentinel.csv"
COVARIATES_PATH = ROOT / "data-raw/covariates/training/sentinel-covariates.parquet"
EMBEDDINGS_PATH = ROOT / "data-raw/covariates/training/alphaearth-foundation-embeddings.parquet"


# --- Training Params
SEED = 42
FIXED_PARAMS = {
    "tree_method": "hist",
    "random_state": 42
}

PARAM_BASE = {
    "learning_rate": [0.0025],
    "subsample": [0.7],
    "colsample_bytree":  [0.7],
    "min_child_weight": [5],
    "reg_lambda": [0.0],
    "reg_alpha": [1.0],
    "gamma": [0.0],
}


PARAM_DEPTHWISE = {"max_depth": [10]}
PARAM_LOSSGUIDE = {"grow_policy": ["lossguide"], "max_leaves": [63, 127, 255, 511], "max_depth": [0]}
