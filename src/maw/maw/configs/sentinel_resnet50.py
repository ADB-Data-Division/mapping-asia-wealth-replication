from dataclasses import replace
from .base import BaseConfig, base_cfg, SENTINEL_IMAGES_DIR, SENTINEL_LABELS_PATH
from ..datasets.sentinel import TRANSFORMS_TRAIN, TRANSFORMS_VAL

METRICS_DIR = base_cfg.metrics_dir / 'SENTINEL-RESNET50'
CHECKPOINT_DIR = base_cfg.checkpoint_dir / 'SENTINEL-RESNET50'

# Model-specific overrides of base config
cfg = replace(BaseConfig(),
              run_name               = 'SENTINEL-RESNET50-BASE',
              model_type             = 'resnet50',
              imagery_type           = 'sentinel',
              initial_weights        = "SENTINEL2_SI_MS_SATLAS",
              images_root            = SENTINEL_IMAGES_DIR,
              labels_csv             = SENTINEL_LABELS_PATH,
              transforms_train       = TRANSFORMS_TRAIN.to(base_cfg.device),
              transforms_val         = TRANSFORMS_VAL.to(base_cfg.device),
              checkpoint_dir         = CHECKPOINT_DIR,
              metrics_dir            = METRICS_DIR,
              optim_sf               = {"fc": (1, 1), "layer4": (0.3, 0.1), "layer3": (0.1, 0.05)},
              early_stopping_params  = {"patience": 5, "min_delta": 0.001, "checkpoint_dir": CHECKPOINT_DIR},
              scheduler_params       = {"warmup_epochs": 3, "decay_rate": 0.97},
              training_schedule      = {10: ["layer4"], 25: ["layer3"]}
              )
