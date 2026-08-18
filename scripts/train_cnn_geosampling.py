import gc
import os

# Cap GDAL's per-process block cache before rasterio/GDAL is imported.
# Default is 5% of RAM; with many DataLoader workers each holding their own
# GDAL cache this can exhaust memory on large datasets.
os.environ.setdefault("GDAL_CACHEMAX", "256")

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import importlib
import copy
import random
import logging
from dataclasses import asdict, fields
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples

from maw.training.trainer import Trainer, TrainConfig
from maw.models.resnet import build_resnet50
from maw.utils.samplers import PointGeoSampler, RandomPointGeoSampler
from maw.datasets.rasters import Landsat


NODATA = -9999


def geo_collate(samples):
    """Stack torchgeo samples and compute per-patch mean label from mask."""
    batch = stack_samples(samples)
    mask = batch["mask"].float().masked_fill(batch["mask"] == NODATA, float("nan"))
    batch["label"] = mask.nanmean(dim=(1, 2))   # (B,)
    return batch


def build_loaders(dataset, points_gdf, train_folds, val_fold, fold_col, patch_size, epoch_size=None, val_epoch_size=None, skip_validation=False, **kwargs):
    """Build train/val DataLoaders using PointGeoSampler over fixed point locations."""
    crs = dataset.crs
    train_pts = points_gdf[points_gdf[fold_col].isin(train_folds)].to_crs(crs)
    val_pts = points_gdf[points_gdf[fold_col] == val_fold].to_crs(crs)

    train_xy = list(zip(train_pts.geometry.x, train_pts.geometry.y))
    val_xy = list(zip(val_pts.geometry.x, val_pts.geometry.y))

    if epoch_size is not None:
        train_sampler = RandomPointGeoSampler(dataset, points_xy=train_xy, points_crs=crs, size=patch_size, epoch_size=epoch_size, skip_validation=skip_validation)
    else:
        train_sampler = PointGeoSampler(dataset, points_xy=train_xy, points_crs=crs, size=patch_size, skip_validation=skip_validation)

    n_val = len(val_pts)
    if val_epoch_size is not None and n_val < val_epoch_size:
        logging.warning(
            f"Val fold {val_fold} has only {n_val} points, fewer than --n-samples-validation={val_epoch_size}; "
            f"using all {n_val} val points."
        )
    effective_val_size = min(n_val, val_epoch_size) if val_epoch_size is not None else n_val
    if effective_val_size < n_val:
        val_sampler = RandomPointGeoSampler(dataset, points_xy=val_xy, points_crs=crs, size=patch_size, epoch_size=effective_val_size, skip_validation=skip_validation, fixed=True)
    else:
        val_sampler = PointGeoSampler(dataset, points_xy=val_xy, points_crs=crs, size=patch_size, skip_validation=skip_validation)

    return (DataLoader(dataset, sampler=train_sampler, collate_fn=geo_collate, **kwargs),
            DataLoader(dataset, sampler=val_sampler, collate_fn=geo_collate, **kwargs))



def run_one_experiment(cfg, dataset, test_folds, num_workers, prefetch_factor, persistent_workers,
                       patch_size, final_training,
                       points_gdf, fold_col="fold", epoch_size=None, val_epoch_size=None, skip_validation=False):

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    split_histories = []
    for held_out in test_folds:
        
        # Get folds
        train_folds = list(range(1, cfg.n_folds + 1))
        train_folds.remove(held_out)
        if final_training:
            val_fold = held_out
        else:
            val_fold = train_folds[0]
            train_folds.remove(val_fold)

        # Build loaders
        train_loader, val_loader = build_loaders(
            points_gdf=points_gdf,
            dataset=dataset,
            train_folds=train_folds,
            val_fold=val_fold,
            fold_col=fold_col,
            patch_size=patch_size,
            epoch_size=epoch_size,
            val_epoch_size=val_epoch_size,
            skip_validation=skip_validation,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory=(cfg.device.type == "cuda")
        )
       
        # Build model
        model = build_resnet50(
            hidden_dim=cfg.fc_hidden, dropout=cfg.dropout,
            frozen_backbone=cfg.freeze_backbone, weights=cfg.initial_weights,
        ).to(cfg.device)

        logging.info(f"---- Training - Fold {held_out} ----")
        logging.info(f"LR: {cfg.learning_rate}  WD: {cfg.weight_decay}  DP: {cfg.dropout}")

        train_fields = {f.name for f in fields(TrainConfig)}
        tcfg = TrainConfig(**{k: v for k, v in asdict(cfg).items() if k in train_fields})

        model_name = f"{cfg.run_name}-SPLIT{held_out}"
        if final_training:
            tcfg.early_stopping_params = None
        elif tcfg.early_stopping_params:
            tcfg.early_stopping_params["filename"] = f"{model_name}-CKPT.pt"

        tb_subdir = os.path.join(cfg.metrics_dir, "tb", model_name)
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, cfg=tcfg, log_dir=tb_subdir)
        history = trainer.fit()
        trainer.save_history(
            os.path.join(cfg.metrics_dir, f"{model_name.lower()}-metrics.csv"),
            dp=cfg.dropout, held_out_fold=held_out, model_name=model_name,
        )
        history["dp"] = cfg.dropout
        history["held_out_fold"] = held_out
        history["model_name"] = model_name
        split_histories.append(history)

        if trainer.early_stopper is not None:
            trainer.early_stopper.restore_best(trainer.model)
        torch.save(trainer.model.state_dict(), os.path.join(cfg.checkpoint_dir, f"{model_name}-FINAL.pt"))

        del trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    return pd.concat(split_histories)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Hyperparameter grid
    parser.add_argument("--lr",  nargs="+", type=float)
    parser.add_argument("--wd",  nargs="+", type=float)
    parser.add_argument("--dp",  nargs="+", type=float)
    
    # Training config
    parser.add_argument("--config",           type=str, required=True)
    parser.add_argument("--base-run-name",    type=str, required=True)
    parser.add_argument("--epochs",           type=int)
    parser.add_argument("--batch-size",       type=int)
    parser.add_argument("--test-folds",       nargs="+", type=int)
    parser.add_argument("--num-workers",      type=int, default=4)
    parser.add_argument("--prefetch-factor",  type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--skip-validation",    action="store_true",
                        help="Skip PointGeoSampler's per-point patch validity check. "
                             "Only safe with point sets pre-filtered by validate_points.py.")
    parser.add_argument("--final-training",   action="store_true",
                        help="Train on train+val sets, evaluate on test fold. "
                             "Requires single values for --lr, --wd, --dp.")
    
    # Geosampling data
    parser.add_argument("--imagery-year",  type=int, required=True)
    parser.add_argument("--n-samples-epoch", type=int, default=None)
    parser.add_argument("--n-samples-validation", type=int, default=5120)
    parser.add_argument("--patch-size",    type=int, default=224)
    parser.add_argument("--fold-column",   type=str, default="fold")
    args = parser.parse_args()

    # Load config
    mod = importlib.import_module(f"maw.configs.{args.config}")
    base_cfg = mod.cfg
    if args.epochs is not None:
        base_cfg.epochs = args.epochs
    if args.batch_size is not None:
        base_cfg.batch_size = args.batch_size
    test_folds = tuple(args.test_folds) if args.test_folds else tuple(range(1, base_cfg.n_folds + 1))

    os.makedirs(base_cfg.metrics_dir, exist_ok=True)
    os.makedirs(os.path.join(base_cfg.metrics_dir, "tb"), exist_ok=True)
    os.makedirs(base_cfg.checkpoint_dir, exist_ok=True)

    # Logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(base_cfg.metrics_dir, "training.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    if args.final_training:
        if not (args.lr and args.wd and args.dp):
            raise ValueError("--final-training requires --lr, --wd, and --dp")
        if len(args.lr) != 1 or len(args.wd) != 1 or len(args.dp) != 1:
            raise ValueError("--final-training requires single values for --lr, --wd, --dp")
        logging.info("\nFINAL TRAINING MODE: train+val sets, early stopping disabled.")

    param_grid = {
        "learning_rate": args.lr if args.lr else base_cfg.learning_rate,
        "weight_decay":  args.wd if args.wd else base_cfg.weight_decay,
        "dropout":       args.dp if args.dp else base_cfg.dropout,
    }
    for k, v in param_grid.items():
        if not isinstance(v, list):
            param_grid[k] = [v]

    logging.info(f"\n-------- Initiating Training: {args.base_run_name} --------")
    logging.info(f"Imagery Year:  {args.imagery_year}")
    logging.info(f"Imagery Dir:   {base_cfg.images_root}")
    logging.info(f"Labels Dir:    {base_cfg.mask_root}")
    logging.info(f"Points File:   {base_cfg.sample_pts_path}")
    logging.info(f"N Samples/Epoch: {args.n_samples_epoch or 'all'}")
    logging.info(f"N Samples/Val:   {args.n_samples_validation}")
    logging.info(f"Epochs:        {base_cfg.epochs}")
    logging.info(f"Batch Size:    {base_cfg.batch_size}")
    logging.info(f"Test Folds:    {test_folds}")
    logging.info(f"LR:            {param_grid['learning_rate']}")
    logging.info(f"WD:            {param_grid['weight_decay']}")
    logging.info(f"DP:            {param_grid['dropout']}\n")

    # Build dataset once (shared across all experiments)
    class Wealth(RasterDataset):
        filename_glob = f"phl_census_wealth_{args.imagery_year}.tif"
        filename_regex = rf"phl_census_wealth_(?P<date>\d{{4}})"
        date_format = "%Y"
        is_image = False
        dtype = torch.float32

    landsat_dataset = Landsat(
        base_cfg.images_root,
        filename_glob=f"landsat-phl-{args.imagery_year}.tif",
        filename_regex=rf"landsat-phl-(?P<date>\d{{4}})",
    )
    dataset = landsat_dataset & Wealth(base_cfg.mask_root)

    # Load sampling points
    pts_path = str(base_cfg.sample_pts_path)
    if pts_path.endswith(".parquet") or pts_path.endswith(".geoparquet"):
        points_gdf = gpd.read_parquet(pts_path)
    else:
        points_gdf = gpd.read_file(pts_path)

    all_results = []
    for params in ParameterGrid(param_grid):
        exp_cfg = copy.deepcopy(base_cfg)
        exp_cfg.learning_rate = params["learning_rate"]
        exp_cfg.weight_decay = params["weight_decay"]
        exp_cfg.dropout = params["dropout"]
        exp_cfg.run_name = (
            f"{args.base_run_name}-LR{exp_cfg.learning_rate}-WD{exp_cfg.weight_decay}-DP{exp_cfg.dropout}"
        )

        history = run_one_experiment(
            cfg=exp_cfg, dataset=dataset, test_folds=test_folds,
            num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
            patch_size=args.patch_size, final_training=args.final_training,
            points_gdf=points_gdf, fold_col=args.fold_column,
            epoch_size=args.n_samples_epoch,
            val_epoch_size=args.n_samples_validation,
            skip_validation=args.skip_validation,
        )
        all_results.append(history)

    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(os.path.join(base_cfg.metrics_dir, "grid_search_summary.csv"), index=False)
