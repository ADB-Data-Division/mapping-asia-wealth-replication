from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn, optim
import kornia.augmentation as K


ROOT = Path.home() / "Projects/mapping-asia-wealth"

LANDSAT_IMAGES_DIR = ROOT / 'data-raw/imagery/training/landsat/'
LANDSAT_LABELS_PATH = ROOT / "data-clean/dhs/asia-training-labels-landsat.csv"
LANDSAT_COVARIATES_PATH = ROOT / "data-raw/covariates/training/landsat-covariates.parquet"

SENTINEL_IMAGES_DIR = ROOT / 'data-raw/imagery/training/sentinel/'
SENTINEL_LABELS_PATH = ROOT / "data-clean/dhs/asia-training-labels-sentinel.csv"
SENTINEL_COVARIATES_PATH = ROOT / "data-raw/covariates/training/sentinel-covariates.parquet"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


@dataclass
class BaseConfig:

    # -------------------- Paths ----------------------------
    run_name:        str = 'BASE-RUN-CONFIG'
    label_name:      str = 'wealth_index'
    id_name:         str = 'patch_id'
    dataset:         str = 'asia-rwi'
    imagery_type:    str = None
    model_type:      str = None

    images_root:     str = None
    mask_root:       str = None                                       # root directory for wealth mask, used for geosampling
    sample_pts_path: str = None                                       # path to sample pts, used for geosampling
    covariates_path: str = None
    labels_csv:      str = None
    metrics_dir:     str = ROOT / 'data-clean/metrics/'
    checkpoint_dir:  str = ROOT / 'data-clean/models/'

    # ----------------- Training Defaults ---------------------
    batch_size:             int          = 32
    epochs:                 int          = 50
    min_epochs:             int          = 0
    n_samples_epoch:        int          = 50000                       # max number of samples, only used if down-weighting India
    india_weights:          float | None = None                         # weights for obs from India, for WeightedRandomSampler
    training_schedule:      dict | None  = None                        # dict mapping layer names to period to unfreeze layer
    early_stopping_params:  dict | None  = None                        # arguments for EarlyStopper (e.g. patience, min_delta)
    transforms_train:       K.AugmentationSequential | None = None
    transforms_val:         K.AugmentationSequential | None = None

    criterion:              nn.Module = nn.HuberLoss(delta=1)           # loss criterion
    optim_class:            optim.Optimizer = optim.AdamW               # class of optimizer used for training
    optim_sf:               dict | None = None                          # dict mapping layer names to factors to tuples of (lr_scale, wd_scale)
    scheduler_params:       dict | None = None                          # params required to build scheduler for learning rate

    n_folds:                int = 5                                     # number of folds for k-fold cross-validation
    fold_column:            str = "fold_kmeans"                         # column name containing fold assignments

    # -------------------- Model Defaults -----------------------
    learning_rate:          float = 1e-3
    weight_decay:           float = 1e-3
    dropout:                float = 0.3
    freeze_backbone:        bool = True
    fc_hidden:              int = 1024
    initial_weights:        str = None

    # ------------------------- Misc --------------------------
    seed:   int = 42
    device: str = DEVICE


base_cfg = BaseConfig()
