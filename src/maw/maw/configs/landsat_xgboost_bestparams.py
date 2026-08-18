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
    "learning_rate": [0.01],
    "subsample": [0.4],
    "colsample_bytree": [0.3],
    "min_child_weight": [5],
    "reg_lambda": [1.0],
    "reg_alpha": [0.0],
    "gamma": [0.0],
}

PARAM_DEPTHWISE = {"max_depth": [6]}
PARAM_LOSSGUIDE = None
