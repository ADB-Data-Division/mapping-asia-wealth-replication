from pathlib import Path
from dataclasses import replace
from .base import BaseConfig, base_cfg, LANDSAT_IMAGES_DIR, LANDSAT_LABELS_PATH, ROOT
from ..datasets.landsat import TRANSFORMS_TRAIN, TRANSFORMS_VAL

METRICS_DIR = base_cfg.metrics_dir / 'LANDSAT-RESNET50-GEOSAMPLING'
CHECKPOINT_DIR = base_cfg.checkpoint_dir / 'LANDSAT-RESNET50-GEOSAMPLING'


cfg = BaseConfig()

# Model-specific overrides of base config
cfg = replace(BaseConfig(),
              run_name              = 'LANDSAT-RESNET50-GEOSAMPLING-BASE',
              imagery_type          = 'landsat',
              model_type            = 'resnet50',
              images_root           = ROOT / 'data-raw/imagery/country/PHL',
              mask_root             = ROOT / 'data-raw/census/PHL/',
              sample_pts_path       = ROOT / 'data-raw/census/PHL/sample_pts_with_labels_2020_filtered.parquet',
              transforms_train      = TRANSFORMS_TRAIN.to(base_cfg.device),
              transforms_val        = TRANSFORMS_VAL.to(base_cfg.device),
              initial_weights       = 'LANDSAT_ETM_SR_MOCO',
              checkpoint_dir        = CHECKPOINT_DIR,
              metrics_dir           = METRICS_DIR,
              min_epochs            = 15,
              early_stopping_params = {"patience": 7, "min_delta": 0.001, "checkpoint_dir": CHECKPOINT_DIR},
              scheduler_params      = {"warmup_epochs": 3, "decay_rate": 0.97},
              optim_sf              = {"fc": (1, 1), "layer4": (0.3, 0.1), "layer3": (0.1, 0.05)},
              training_schedule     = {10: ["layer4"]}
              )
