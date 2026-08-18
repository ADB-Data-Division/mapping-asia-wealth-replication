from dataclasses import replace
from .base import BaseConfig, base_cfg, LANDSAT_IMAGES_DIR, LANDSAT_LABELS_PATH
from ..datasets.landsat import TRANSFORMS_TRAIN, TRANSFORMS_VAL

METRICS_DIR = base_cfg.metrics_dir / 'LANDSAT-RESNET50'
CHECKPOINT_DIR = base_cfg.checkpoint_dir / 'LANDSAT-RESNET50'


cfg = BaseConfig()

# Model-specific overrides of base config
cfg = replace(BaseConfig(),
              run_name              = 'LANDSAT-RESNET50-BASE',
              imagery_type          = 'landsat',
              model_type            = 'resnet50',
              images_root           = LANDSAT_IMAGES_DIR,
              labels_csv            = LANDSAT_LABELS_PATH,
              transforms_train      = TRANSFORMS_TRAIN.to(base_cfg.device),
              transforms_val        = TRANSFORMS_VAL.to(base_cfg.device),
              initial_weights       = "LANDSAT_ETM_SR_MOCO",
              checkpoint_dir        = CHECKPOINT_DIR,
              metrics_dir           = METRICS_DIR,
              early_stopping_params = {"patience": 5, "min_delta": 0.001, "checkpoint_dir": CHECKPOINT_DIR},
              scheduler_params      = {"warmup_epochs": 3, "decay_rate": 0.97},
              optim_sf              = {"fc": (1, 1), "layer4": (0.3, 0.1), "layer3": (0.1, 0.05)},
              training_schedule     = {10: ["layer4"]}
              )
