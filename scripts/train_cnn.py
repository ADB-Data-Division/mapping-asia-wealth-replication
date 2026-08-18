import gc
import torch
import numpy as np
import pandas as pd
import os
import argparse
import importlib
import copy
import random
import logging
from dataclasses import asdict, fields
from sklearn.model_selection import ParameterGrid
from maw.training.trainer import Trainer, TrainConfig
from maw.utils.loaders import build_loaders
from maw.models.resnet import build_resnet50
from maw.models.swinv2 import MultimodalSwinV2TRegressor, ModelConfig


def run_one_experiment(cfg,
                       test_folds: tuple[int] = (1, 2, 3, 4, 5),
                       num_workers: int = 4,
                       prefetch_factor: int = 2,
                       persistent_workers: bool = False,
                       final_training: bool = False,
                       init_checkpoint: str | None = None):

    """Train a model using configs in cfg on specified CV splits."""

    # Seed RNGs for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    split_histories = []
    for held_out in test_folds:

        # ---- BUILD CV SPLIT AND LOADERS -----
        train_folds = list(range(1, cfg.n_folds + 1))
        train_folds.remove(held_out)
        if final_training:
            test_fold = held_out
        else:
            test_fold = train_folds[0]
            train_folds.remove(test_fold)

        train_loader, val_loader = build_loaders(
            train_folds=train_folds, test_fold=test_fold, cfg=cfg, num_workers=num_workers,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, weighted=True
        )

        # ---- BUILD MODEL ----
        # Create a fresh instance of the model and freeze backbone

        if cfg.model_type == 'resnet50':
            model = build_resnet50(hidden_dim=cfg.fc_hidden, dropout=cfg.dropout, frozen_backbone=cfg.freeze_backbone,
                                   weights=cfg.initial_weights)
        elif cfg.model_type == 'swinv2':
            model_config = ModelConfig(head_dropout=cfg.dropout)
            model = MultimodalSwinV2TRegressor(freeze_backbone=cfg.freeze_backbone, cfg=model_config)
        else: 
            raise NotImplementedError("Only 'resnet50' and 'swinv2' are currently supported")
        
        model = model.to(cfg.device)

        if init_checkpoint is not None:
            ckpt_path = init_checkpoint.replace("{held_out}", str(held_out))
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Could not find --init-checkpoint for fold {held_out}: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device, weights_only=True))
            logging.info(f"Initialised model weights from checkpoint: {ckpt_path}")

        # ---- LOG TRAINING RUN INFO ----
        logging.info(f"---- Training - Fold {held_out} ----")
        logging.info(f"LR: {cfg.learning_rate}")
        logging.info(f"WD: {cfg.weight_decay}")
        logging.info(f"DP: {cfg.dropout}")

        # ---- TRAIN MODEL ----
        # Get configs needed for training
        train_fields = {f.name for f in fields(TrainConfig)}
        tcfg = TrainConfig(**{k: v for k, v in asdict(cfg).items() if k in train_fields})

        # Configure Trainer
        model_name = f"{cfg.run_name}-SPLIT{held_out}"   
        if final_training:
            # Disable early stopping for final training to avoid data leakage
            tcfg.early_stopping_params = None
        elif tcfg.early_stopping_params:
            tcfg.early_stopping_params["filename"] = f"{model_name}-CKPT.pt"
        
        tb_subdir = os.path.join(cfg.metrics_dir, 'tb', model_name)

        # Train model and save metrics
        trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, cfg=tcfg, log_dir=tb_subdir)
        history = trainer.fit()
        trainer.save_history(os.path.join(cfg.metrics_dir, f"{model_name.lower()}-metrics.csv"),
                                dp = cfg.dropout, held_out_fold=held_out, model_name=model_name)
        history["dp"] = cfg.dropout
        history["held_out_fold"] = held_out
        history["model_name"] = model_name
        split_histories.append(history)

        # Save model state after training terminates
        torch.save(trainer.model.state_dict(), os.path.join(cfg.checkpoint_dir, f"{model_name}-FINAL.pt"))

        # Potential culprit of file-descriptor exhaustion on long runs
        del trainer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    return pd.concat(split_histories)


if __name__ == '__main__':

    # ---- CLI ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',         nargs='+', type=float)
    parser.add_argument('--wd',         nargs='+', type=float)
    parser.add_argument('--dp',    nargs='+', type=float)
    parser.add_argument('--test-folds',      nargs='+', type=int, help='Which folds to use as test sets (1-5)')
    parser.add_argument('--num-workers',        type=int)
    parser.add_argument('--prefetch-factor',    type=int)
    parser.add_argument('--batch-size',         type=int)
    parser.add_argument('--epochs',             type=int)
    parser.add_argument('--base-run-name',      type=str, required=True)
    parser.add_argument('--config',             type=str, required=True)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--final-training', action='store_true', 
                        help='Train on train+val sets for each CV fold, using test fold for evaluation. '
                             'Requires single values for --lr, --wd, --dp (no grid search).')
    parser.add_argument('--init-checkpoint', type=str, default=None,
                        help="Optional checkpoint to initialize weights before training. "
                             "Use '{held_out}' placeholder for per-fold checkpoints.")
    args = parser.parse_args()

    # Load config file
    mod = importlib.import_module(f'maw.configs.{args.config}')
    base_cfg = mod.cfg
    if args.epochs is not None: base_cfg.epochs = args.epochs
    if args.batch_size is not None: base_cfg.batch_size = args.batch_size
    test_folds = tuple(args.test_folds) if args.test_folds is not None else tuple(range(1, base_cfg.n_folds + 1))
    num_workers = args.num_workers if args.num_workers is not None else 4
    prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else 2
    persistent_workers = args.persistent_workers

    # Build output directories
    os.makedirs(base_cfg.metrics_dir, exist_ok=True)
    os.makedirs(os.path.join(base_cfg.metrics_dir, 'tb'), exist_ok=True)
    os.makedirs(base_cfg.checkpoint_dir, exist_ok=True)

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Print cleanly to console
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_h)

    # Log timestamps for file printing
    file_h = logging.FileHandler(os.path.join(base_cfg.metrics_dir, "training.log"))
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_h)

    # ---- VALIDATE FINAL TRAINING MODE ----
    if args.final_training:
        # Assert that single values are provided for hyperparameters
        if args.lr is None or args.wd is None or args.dp is None:
            raise ValueError("--final-training requires --lr, --wd, and --dp arguments to be specified")
        if len(args.lr) != 1 or len(args.wd) != 1 or len(args.dp) != 1:
            raise ValueError("--final-training requires single values for --lr, --wd, --dp (no grid search)")
        
        logging.info("\nFINAL TRAINING MODE: Training on train+val sets, evaluating on test folds.")
        logging.info("Early stopping disabled to avoid data leakage.")

    # ---- PREPARE GRIDSEARCH ----
    # Build parameter grid to search over, giving priority to arguments supplied to CLI
    param_grid = {
        'learning_rate':  args.lr if args.lr is not None else base_cfg.learning_rate,
        'weight_decay':   args.wd if args.wd is not None else base_cfg.weight_decay,
        'dropout':        args.dp if args.dp is not None else base_cfg.dropout,
    }

    # ParameterGrid expects single-values to be wrapped in a list
    for key, value in param_grid.items():
        if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
            param_grid[key] = [value]

    logging.info(f"\n-------- Initiating Training: {args.base_run_name} --------")
    logging.info(f"Label Name:        {base_cfg.label_name}")
    logging.info(f"Epochs:            {base_cfg.epochs}")
    logging.info(f"Batch Size:        {base_cfg.batch_size}")
    logging.info(f"Test Folds (CV):   {test_folds}")
    logging.info(f"Learning Rate:     {param_grid['learning_rate']}")
    logging.info(f"Weight Decay:      {param_grid['weight_decay']}")
    logging.info(f"Dropout:           {param_grid['dropout']}")
    logging.info(f"Loss Criterion:    {base_cfg.criterion}")
    logging.info(f"Optimizer:         {base_cfg.optim_class}")
    logging.info(f"LR Schedule:       {base_cfg.scheduler_params}")
    logging.info(f"Training Schedule: {base_cfg.training_schedule}")
    logging.info(f"Initial Weights:   {base_cfg.initial_weights}")
    logging.info(f"Init Checkpoint:   {args.init_checkpoint}")
    logging.info(f"India S. Weights:  {base_cfg.india_weights}")
    logging.info(f"See configs/{args.config}.py for further details.\n")

    # ---- TRAIN MODELS ----
    all_results = []
    for params in ParameterGrid(param_grid):
        exp_cfg = copy.deepcopy(base_cfg)
        exp_cfg.learning_rate = params['learning_rate']
        exp_cfg.weight_decay = params['weight_decay']
        exp_cfg.dropout = params['dropout']
        exp_cfg.run_name = f"{args.base_run_name}-LR{exp_cfg.learning_rate}-WD{exp_cfg.weight_decay}-DP{exp_cfg.dropout}"

        history = run_one_experiment(exp_cfg, test_folds=test_folds, num_workers=num_workers, prefetch_factor=prefetch_factor,
                                     persistent_workers=persistent_workers, final_training=args.final_training,
                                     init_checkpoint=args.init_checkpoint)
        all_results.append(history)

    summary = pd.concat(all_results, ignore_index=True)
    summary.to_csv(os.path.join(base_cfg.metrics_dir, 'grid_search_summary.csv'), index=False)
