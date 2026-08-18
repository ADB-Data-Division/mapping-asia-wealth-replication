from dataclasses import replace
from .base import BaseConfig, base_cfg, SENTINEL_IMAGES_DIR, SENTINEL_LABELS_PATH, SENTINEL_COVARIATES_PATH
from ..datasets.sentinel import TRANSFORMS_TRAIN, TRANSFORMS_VAL

METRICS_DIR = base_cfg.metrics_dir / 'SENTINEL-SWINV2'
CHECKPOINT_DIR = base_cfg.checkpoint_dir / 'SENTINEL-SWINV2'

# Model-specific overrides of base config
cfg = replace(BaseConfig(),
              run_name               = 'SENTINEL-SWINV2-BASE',
              model_type             = 'swinv2',
              imagery_type           = 'sentinel',
              initial_weights        = "SENTINEL2_SI_MS_SATLAS",                  # currently not implemented...
              images_root            = SENTINEL_IMAGES_DIR,
              covariates_path        = SENTINEL_COVARIATES_PATH,
              labels_csv             = SENTINEL_LABELS_PATH,
              transforms_train       = TRANSFORMS_TRAIN.to(base_cfg.device),
              transforms_val         = TRANSFORMS_VAL.to(base_cfg.device),
              checkpoint_dir         = CHECKPOINT_DIR,
              metrics_dir            = METRICS_DIR,
              early_stopping_params  = {"patience": 7, "min_delta": 0.001, "checkpoint_dir": CHECKPOINT_DIR},
              scheduler_params       = None
              )
