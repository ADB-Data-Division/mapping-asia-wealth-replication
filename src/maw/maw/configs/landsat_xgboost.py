from pathlib import Path


# --- Paths
ROOT = Path.home() / "Projects/mapping-asia-wealth"
MODEL_OUTDIR = ROOT / "data-clean/models/xgboost/"
PREDICTIONS_OUTDIR = ROOT / "data-clean/outputs/predictions/xgboost/"

LABELS_PATH = ROOT / "data-clean/dhs/asia-training-labels-landsat.csv"
COVARIATES_PATH = ROOT / "data-raw/covariates/training/landsat-covariates.parquet"
EMBEDDINGS_PATH = ROOT / "data-raw/embeddings/training/landsat-resnet50-embeddings.parquet"


# --- Training Params
SEED = 42
FIXED_PARAMS = {
    "tree_method": "hist",
    "random_state": 42
}

PARAM_BASE = {
    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1],
    "subsample": [0.4, 0.6, 0.7, 0.8, 0.9, 1],
    "colsample_bytree": [0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7, 10],
    "reg_lambda": [0.0, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "gamma": [0.0, 0.1, 0.5, 1.0],
}

PARAM_DEPTHWISE = {"max_depth": [4, 6, 8, 10]}
PARAM_LOSSGUIDE = {"grow_policy": ["lossguide"], "max_leaves": [63, 127, 255, 511], "max_depth": [0]}
