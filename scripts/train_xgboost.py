# train_xgboost.py
import argparse
import os
import json
import math
import random
import importlib
from typing import Sequence
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timezone
from maw.training.metrics import compute_metrics
from maw.utils.loaders import load_features
from maw.utils.helpers import sample_params


def append_ndjson(path: str, record: dict):
    """Append one JSON line to `path`."""
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def random_search_xgboost(
        model_param_space: dict,
        model_fixed_params: dict,
        n_trials: int,
        feature_type: str,
        paths_dict: dict,
        label_name: str,
        id_name: str,
        fold_name: str,
        model_outdir: str,
        eval_metric: str = "mae",
        max_estimators: int = 3000,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
        run_id: str | None = None,
        test_folds: Sequence[int] | None = None):

    if eval_metric not in ["mae", "rmse"]:
        raise ValueError(f"Invalid eval_metric: {eval_metric}. Must be one of ['mae', 'rmse']")

    available_folds = sorted(
        pd.read_csv(paths_dict["labels-csv"], usecols=[fold_name])[fold_name].dropna().unique().tolist()
    )
    if len(available_folds) < 3:
        raise ValueError(
            f"Expected at least 3 folds in '{fold_name}', found {len(available_folds)}: {available_folds}"
        )

    if test_folds is None:
        test_folds = available_folds
    else:
        test_folds = list(test_folds)
        invalid_folds = sorted(set(test_folds) - set(available_folds))
        if invalid_folds:
            logging.info(
                f"Requested split IDs: {test_folds}; available dataset folds: {available_folds}."
            )
            raise ValueError(
                f"Invalid test fold(s): {invalid_folds}. "
                f"Available folds in '{fold_name}': {available_folds}."
            )

    # Setup output paths
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for test_fold in tqdm(test_folds, desc="Split", total=len(test_folds), leave=False):

        # Paths for saving training run metrics and models
        os.makedirs(model_outdir, exist_ok=True)
        history_file = os.path.join(model_outdir, f"{run_id}_split{test_fold}_{feature_type}_history.ndjson")
        trial_results_file = os.path.join(model_outdir, f"{run_id}_split{test_fold}_{feature_type}_results.ndjson")

        val_fold = random.choice([i for i in available_folds if i != test_fold])
        train_folds = [i for i in available_folds if i not in [test_fold, val_fold]]
        trainval_folds = train_folds + [val_fold]

        LOADER_CONFIG = {
            "target_col": label_name,
            "id_col": id_name,
            "fold_col": fold_name,
            "feature_type": feature_type,
            "paths_dict": paths_dict,
            "embed_backbone_heldoutset": test_fold,
            "as_numpy": False
        }

        # Load targets and features
        X_train, y_train = load_features(folds=train_folds, **LOADER_CONFIG)
        X_val, y_val = load_features(folds=[val_fold], **LOADER_CONFIG)
        y_train, y_val = y_train.to_numpy(), y_val.to_numpy()
        if len(y_train) == 0 or len(y_val) == 0:
            raise ValueError(
                f"Empty train/validation data for split {test_fold}. "
                f"train_folds={train_folds}, val_fold={val_fold}, available_folds={available_folds}."
            )

        # Randomly search over combinations of params defined in model_param_space
        best = {"params": dict(), "score": np.inf, "best_n_estimators": None}
        for t in tqdm(range(1, n_trials + 1), desc=f"Training {feature_type}", leave=False):

            # Build model with sampled params
            sampled_params = sample_params(model_param_space)
            model = xgb.XGBRegressor(**sampled_params, **model_fixed_params, n_estimators=max_estimators,
                                     eval_metric=eval_metric, early_stopping_rounds=early_stopping_rounds)

            # Fit model on train/val data
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=verbose)

            # Use MAE/RMSE on the same validation to chose best params
            val_pred = model.predict(X_val, validate_features=True)
            val_mae = mean_absolute_error(y_true=y_val, y_pred=val_pred)
            val_rmse = math.sqrt(mean_squared_error(y_true=y_val, y_pred=val_pred))
            val_r2 = r2_score(y_true=y_val, y_pred=val_pred)
            final_score = val_mae if eval_metric == "mae" else val_rmse

            best_iter = getattr(model, "best_iteration", None)
            best_n_estimators = (int(best_iter) + 1) if best_iter is not None else model.n_estimators

            if final_score < best["score"]:
                best.update(params=sampled_params, score=final_score, best_n_estimators=best_n_estimators)

            trial_record = {
                "stage": "trial",
                "run_id": run_id,
                "trial": t,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_fold": test_fold,
                "val_fold": val_fold,
                "train_folds": train_folds,
                "val_mae": float(val_mae),
                "val_rmse": float(val_rmse),
                "val_r2": float(val_r2),
                "params": sampled_params,
                "model_fixed_params": model_fixed_params,
                "best_iteration": best_iter,
                "best_n_estimators": best_n_estimators,
            }
            append_ndjson(history_file, trial_record)

            logging.info(
                f"\nFeatures: {feature_type} Trial: {t:02d}"
                f"\nMAE: {val_mae:.5f} RMSE: {val_rmse:.5f} R2: {val_r2:.5f}"
                f"\nScore={final_score:.5f} Best N: {best_n_estimators} \n"
            )

        # Params and stack train + val sets
        del X_train, y_train, X_val, y_val
        X_trainval, y_trainval = load_features(folds=trainval_folds, **LOADER_CONFIG)
        y_trainval = y_trainval.to_numpy()

        # Refit model using best-performing params and all data
        final_model = xgb.XGBRegressor(**best["params"], **model_fixed_params, n_estimators=best["best_n_estimators"],
                                       eval_metric=eval_metric)
        final_model.fit(X_trainval, y_trainval, verbose=verbose)

        # Evaluate on held-out test set
        X_test, y_test = load_features(folds=[test_fold], **LOADER_CONFIG)
        y_test = y_test.to_numpy()
        if len(y_test) == 0:
            raise ValueError(
                f"Empty test data for split {test_fold}. "
                f"Available folds in '{fold_name}': {available_folds}."
            )

        test_pred = final_model.predict(X_test, validate_features=True)
        test_mae, test_rmse, test_r2 = compute_metrics(y_true=y_test, y_pred=test_pred)

        logging.info(
            f"\nRun: {run_id} Test Fold: {test_fold}"
            f"\nFeatures: {feature_type}"
            f"\nTest RMSE: {test_rmse}"
            f"\nTest MAE: {test_mae}"
            f"\nTest R2: {test_r2}\n"
            f"\nBest Params: {best['params']}"
        )

        # Save best-performing model
        model_stem = run_id if run_id is not None else f"XGBOOST-{feature_type.upper()}"
        model_name = f"{model_stem}-SPLIT{test_fold}.json"
        final_model.save_model(os.path.join(model_outdir, model_name))

        result_record = {
            "run_id": run_id,
            "stage": "result",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_fold": test_fold,
            "trainval_folds": trainval_folds,
            "feature_type": feature_type,
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
            "max_estimators": max_estimators,
            "early_stopping_rounds": early_stopping_rounds,
            **best
        }
        append_ndjson(trial_results_file, result_record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--feature-type', type=str, choices=["embeddings-only", "spatial-only", "all-features"], required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n-trials', type=int, default=60)
    parser.add_argument('--max-estimators', type=int, default=3000)
    parser.add_argument('--early-stopping-rounds', type=int, default=50)
    parser.add_argument('--label-name', type=str, default="wealth_index")
    parser.add_argument('--fold-name', type=str, default="fold_kmeans")
    parser.add_argument("--eval-metric", default="mae")
    parser.add_argument("--njobs", type=int, default=6)
    parser.add_argument("--lossguide", action="store_true", default=False)
    parser.add_argument(
        '--splits',
        nargs='+',
        type=int,
        default=None,
        help="Test fold IDs. Omit to use all folds found in labels.",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--train-gpu", action="store_true", default=False, help="Use GPU for training (sets device='cuda')")
    args = parser.parse_args()

    # ------------------------
    # XGBoost Experiment Config
    # ------------------------

    mod = importlib.import_module(f'maw.configs.{args.config}')
    mod.FIXED_PARAMS['n_jobs'] = args.njobs
    if args.train_gpu:
        mod.FIXED_PARAMS['device'] = 'cuda'
    PARAM_SPACE = {**mod.PARAM_BASE, **(mod.PARAM_LOSSGUIDE if args.lossguide else mod.PARAM_DEPTHWISE)}

    # Seed for reproducibility
    random.seed(mod.SEED)
    np.random.seed(mod.SEED)

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Print cleanly to console
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_h)

    # Log timestamps for file printing
    os.makedirs(mod.MODEL_OUTDIR, exist_ok=True)
    file_h = logging.FileHandler(os.path.join(mod.MODEL_OUTDIR, "training.log"))
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_h)

    logging.info(f"\n-------- Initiating Training: {args.run_name} --------")
    logging.info(f"Label Name:            {args.label_name}")
    logging.info(f"Fold Name:             {args.fold_name}")
    logging.info(f"Feature Type:          {args.feature_type}")
    logging.info(f"Number of Trials:      {args.n_trials}")
    logging.info(f"Splits (CV):           {args.splits if args.splits is not None else 'all available folds'}")
    logging.info(f"Loss Criterion:        {args.eval_metric}")
    logging.info(f"Grow Policy:           {'lossguide' if args.lossguide else 'depthwise'}")
    logging.info(f"Tree Method:           {mod.FIXED_PARAMS['tree_method']}")
    logging.info(f"Max Estimators:        {args.max_estimators}")
    logging.info(f"Early Stopping Rounds: {args.early_stopping_rounds}")
    logging.info(f"Learning Rate:         {PARAM_SPACE['learning_rate']}")
    logging.info(f"Lambda:                {PARAM_SPACE['reg_lambda']}")
    logging.info(f"Alpha:                 {PARAM_SPACE['reg_alpha']}")
    logging.info(f"Gamma:                 {PARAM_SPACE['gamma']}")
    logging.info(f"Subsample:             {PARAM_SPACE['subsample']}")
    logging.info(f"Colsample By Tree:     {PARAM_SPACE['colsample_bytree']}")
    logging.info(f"Max Depth:             {PARAM_SPACE['max_depth']}")
    logging.info(f"Max Leaves:            {PARAM_SPACE['max_leaves'] if args.lossguide else None}")

    PATHS_DICT = {
        "labels-csv": mod.LABELS_PATH,
        "spatial-features-parquet": getattr(mod, "COVARIATES_PATH", None),
        "embeddings-parquet": getattr(mod, "EMBEDDINGS_PATH", None),
    }

    random_search_xgboost(
        n_trials=args.n_trials,
        label_name=args.label_name,
        id_name=getattr(mod, "ID_NAME", "patch_id"),
        fold_name=args.fold_name,
        model_fixed_params=mod.FIXED_PARAMS,
        model_param_space=PARAM_SPACE,
        model_outdir=mod.MODEL_OUTDIR,
        eval_metric=args.eval_metric,
        max_estimators=args.max_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=args.verbose,
        feature_type=args.feature_type,
        run_id=args.run_name,
        paths_dict = PATHS_DICT,
        test_folds=args.splits,
    )
