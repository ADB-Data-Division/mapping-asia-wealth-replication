# Absolute and Relative Wealth Index Models

We train a series of convolutional neural networks to infer absolute and relative wealth indices from freely available satellite imagery from Landsat 7-9 and Sentinel-2. In addition, we train a series of models that integrate both satellite imagery (in the form of embeddings) and spatial covariate summary statistics using XGBoost to predict asset weatlh. 

This README explains how to reproduce our analysis from raw imagery and survey archives, to trained models.

---

## Table of contents

1. [Requirements](#requirements)
2. [Getting Started](#getting-started)
3. [Model Configurations](#model-configurations)
4. [Acknowledgements](#acknowledgements)

---
<a id="requirements"></a>
## Requirements
|            | Tested Using:                         |
|------------|---------------------------------------| 
| **OS**     | Ubuntu 24.04LTS                       | 
| **Python** | 3.12                                  | 
| **GPU**    | Nvidia Tesla T4 (16GB)                | 
| **Disk**   | 1TB SSD                               | 
| **RAM**    | 32GB                                  |
| **Other**  | Additional packages listed in env.yml | 


(Optional) Create a VM Instance:
```bash
# Example: Google Cloud instance with T4 GPU and 1TB SSD
gcloud compute instances create adb \
    --machine-type n1-standard-16 \
    --zone europe-west1-b \
    --boot-disk-size 1000GB \
    --boot-disk-type pd-ssd \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --image-family ubuntu-2404-lts \
    --image-project ubuntu-os-cloud \
    --maintenance-policy TERMINATE
```
Install the drivers for the GPU by following these [instructions.](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)

---

<a id="getting-started"></a>
## Getting Started

> All commands assume your working directory is `~/Projects/mapping-asia-wealth`.

### 1. Create the Environment:

**Python.** Create a Conda environment and install all necessary dependencies listed in the `env.yml` file by running:
```bash
conda env create -f env.yml
conda activate maw
```
Install the local Python packages needed by the preprocessing scripts:
```bash
# Install maw (sources live in src/maw)
python -m pip install -e src
# Install the Earth Engine utilities package
python -m pip install git+https://github.com/jordanimahori/ee-utils.git
```

**R.** `env.yml` only covers the Python side. Separately install R (e.g. `conda install -c conda-forge r-base` or a system install) and the CRAN packages used across `R/`:
```r
install.packages(c(
  "haven", "dplyr", "readr", "sf", "stringr", "purrr", "ggplot2", "arrow",
  "dbscan", "tidyr", "terra", "sfarrow", "tidyverse", "broom", "yardstick", "knitr"
))
```

### 2. Download DHS Survey Archives:
DHS data is proprietary but free after registration.

1. Register at [https://dhsprogram.com](https://dhsprogram.com) and request access to every DHS5+ survey in Asia in DTA format.
2. Place each ZIP file at

   ```
   data-raw/dhs/<COUNTRY_ISO>/<SURVEY>.zip
   ```

*Note*: Survey archives for India are large. For devices with limited RAM, you may need to extract the relevant survey files using STATA and drop unused columns to avoid memory exhaustion issues. 

 
### 3. Clean & Pool Survey Data: 
The following script unzips each archive, joins the location information with the Household Recode file, and calculates AWI and RWI indices for each country.

Run: `R/preprocessing/preprocess_dhs.R`

The script will save the outputs to `data-clean/dhs/`.


### 4. Download Image Patches:

To train the model, we first need to extract image patches that correspond to each cluster in the pooled DHS Clusters dataset, for the year in which the surveys were conducted. You'll need a (free) Earth Engine account to download Landsat imagery. 

Run: `notebook/preprocessing/export_training_inputs.ipynb`

Depending on which model(s) you wish to train, you may wish to only download Landsat/Sentinel 2 imagery, which you can do by running the relevant sections. The notebook authenticates to Google Earth Engine, creates cloud-free composites from all available Landsat (L2) or Sentinel (L2A) images, and extracts and saves the patches as GeoTIFFs. It also exports the spatial covariates and AlphaEarth Foundations (AEF) embeddings used by the XGBoost models. *Ensure that the cluster centroids CSV has been successfully created by Step 3 and filepath parameters are updated, as necessary.* 

### 5. Train CNN Models:
Training can be launched via the `scripts/train_cnn.py` script. To train a model, first create a model config under `src/maw/maw/configs/` with desired parameters to override those in `src/maw/maw/configs/base.py`. You can supply additional arguments via the CLI to override values in your config. By default, only fully-connected layers are trained. Additional layers can be trained by specifying this in the `training_schedule` parameter in the config file. Our final models followed a progressive unfreezing schedule where the FC layers were trained on a frozen backbone with MOCO-pretrained weights for 10 epochs. After 10 epochs, Layer 4 was unfrozen and both the FC layers and Layer 4 were trained for an additional 10 epochs. After Epoch 20, Layer 3 was unfrozen and training continued for up to a maximum of 80 epochs with FC, Layer 4, and Layer 3 unfrozen. Lower-level layers were never unfrozen to protect the representations learned during pre-training and minimize the risk of overfitting to our training set.

To launch a grid search over various choices for the learning rate, weight decay, and FC layer dropouts, run:

```bash
# Train 3x2x2 models learning_rate=lr, weight_decay=wd, and dropouts=dp
python scripts/train_cnn.py \
    --base-run-name RESNET50-LANDSAT-BESTPARAMS \
    --config landsat_resnet50 \
    --lr 1e-3 1e-4 5e-5 --wd 1e-2 1e-3 --dp 0.5 0.3 \
    --epochs 50 --batch-size 32 \
    --num-workers 10 --prefetch-factor 2
```
Note: 
* Default behaviour is 5‑fold spatial CV (fold indices 1‑5).
* Restrict splits via e.g., `--test-folds 1 2` to accelerate debugging.
* All outputs are written to the directories specified in your model config.


### 6. Train XGBoost Models:
We train a series of XGBoost models to predict asset wealth using the same spatial CV folds as the CNN models. We train three models: one using spatial covariates only, one using spatial covariates and satellite imagery embeddings (i.e. activations from our trained Landsat/Sentinel CNNs), and one using both. 

As the search space for hyperparameters is larger, we identify the best-performing choice of hyperparameters using Random Search. Similar to the training procedure for CNNs, training runs are launched via `scripts/train_xgboost.py` and supplying at a minimum the run name, a feature type, and a config file. In contrast to `train_cnn.py`, hyperparameters are specified directly in the config file. We also provide a config file with the set of best-performing hyperparameters for each model (`*_bestparams.py`).

To launch a random search over various choices of hyperparameters, run:

```bash
python scripts/train_xgboost.py \
    --run-name XGB-SPATIAL-ONLY \
    --feature-type spatial-only \
    --config sentinel_xgboost_bestparams \
```

### 7. Evaluate Best-Performing Models: 
We evaluate the performance of the best-performing models from Steps 5–6 using the notebooks in `notebook/preprocessing` and `notebook/analysis`.

To train our XGBoost models that combine satellite imagery and spatial covariates, we first need a lower-dimension 'summary' of each image patch. This is produced by `notebook/preprocessing/extract_training_embeddings.ipynb`, which runs each trained CNN's held-out-fold checkpoints over the training set and saves per-patch embeddings to `data-raw/embeddings/training/` (the path each `*_xgboost*.py` config's `EMBEDDINGS_PATH` points at) alongside its own copy of the predictions.

Once the embeddings exist, run the following notebooks in `notebook/analysis/` to generate the out-of-fold predictions and metrics used by the `R/analysis/*.R` scripts:

* `evaluate_model_cnn.ipynb` — out-of-fold ResNet50/SwinV2T CNN predictions, saved to `data-clean/outputs/predictions/`.
* `evaluate_model_xgboost.ipynb` — out-of-fold predictions for a given XGBoost model/feature-type, saved to `data-clean/outputs/predictions/`.

Repeat `evaluate_model_xgboost.ipynb` for each `feature_type` (`spatial-only`, `embeddings-only`, `all-features`) × imagery source (`landsat`, `sentinel`, `aef`) combination you trained in Step 6 — `R/analysis/cross_sectional_performance_dhs.R` expects predictions files for all of them.


### 8. Create AWI & RWI Wealth Maps: 

During training, we created five models each trained on a different split of our sample. Our final predictions used for our wealth maps are the average of the predicted values across our five models. We then re-assemble these emsemble predictions into a single raster in which each pixel matches the footprint of each patch of input imagery. 

Run: `notebook/analysis/create_wealth_maps.ipynb`

This notebook builds one shared sampling grid over an area of interest -- a country's ADM0 boundary by default, or an arbitrary polygon (e.g. a coastal buffer) via `BOUNDARY_PATH` -- and produces both CNN/XGBoost maps at the requested resolution from that same grid.

* `MAP_COUNTRY` / `BOUNDARY_PATH` -- the area of interest.
* `BUILD_CNN_MAP` / `BUILD_XGB_MAP` -- which product(s) to build this run (independent; XGBoost can also consume CNN embeddings for the `resnet-*` configs if both are enabled, since both branches share the same grid/`patch_id`s).
* `OUTPUT_MAP_RESOLUTION` -- the shared grid's spacing. At a covariates/AlphaEarth COG's native resolution (e.g. 500m) that source is point-sampled; at a coarser resolution it's a nodata-masked spatial mean over each point's footprint instead. If `BUILD_CNN_MAP` is set, `OUTPUT_MAP_RESOLUTION` must exactly equal `PATCH_SIZE * IMAGERY_SCALE[...]` (asserted) so each CNN patch covers exactly one grid cell.

**Prerequisite**: both branches read directly from the per-country COG mosaics built by `notebook/preprocessing/export_country_composites.ipynb` (spatial covariates and AlphaEarth embeddings for the XGBoost branch; Sentinel/Landsat for the CNN branch). That notebook's `IMAGERY_TYPES` must include the relevant type(s).

The notebook loads the relevant trained model(s) from Step 6, runs inference, and averages over the five spatial-CV folds. It stores the final raster(s) of predictions under `data-clean/outputs/maps/`.

Once you have out-of-fold predictions from Step 7, you can also reproduce the paper's summary tables and figures with the remaining `R/analysis/` scripts:

* `R/analysis/summary_stats_dhs.R` — basic counts of pooled DHS survey responses/clusters.
* `R/analysis/cross_sectional_performance_dhs.R` — R²/MAE/RMSE by fold, country, rural/urban, and wealth quintile across every model, plus the observed-vs-predicted scatter plots.

---

<a id="model-configurations"></a>
## Model Configurations

Experiments are fully specified by Python config files stored in `src/maw/maw/configs/`. 

For CNNs, each config file extends `src/maw/maw/configs/base.py` and is simply imported at runtime (as `maw.configs.<name>`), so you can use any Python logic.

A minimal config looks like:

```python
# src/maw/maw/configs/landsat_resnet50_experiment.py
from dataclasses import replace
from .base import BaseConfig, base_cfg, ROOT

# ---- PATHS
METRICS_DIR = ROOT / 'data-clean/outputs/metrics/RESNET50-PROGRESSIVE'
CHECKPOINT_DIR = ROOT / 'data-clean/outputs/checkpoints/RESNET50-PROGRESSIVE'

# ---- OVERRIDE BASE CONFIGS
cfg = replace(BaseConfig(),
              run_name='ASIA-RESNET50-PROGRESSIVE-BASE',
              checkpoint_dir=CHECKPOINT_DIR,
              metrics_dir=METRICS_DIR,
              label_name='wealth_index'
              )
```

CLI arguments **override** the values in the selected config, which makes grid searches straightforward (see example in Step 5).

For XGBoost models, configs are simpler and there is no base config. They are primarily used to specify paths to necessary input files and the search space for hyperparameters. A minimal config looks like:

```python
# src/maw/maw/configs/sentinel_xgboost_experiment.py
from pathlib import Path

SEED = 42

# --- Paths (see "A note on paths" above re: Path.home())
ROOT = Path.home() / 'Projects/mapping-asia-wealth'
MODEL_OUTDIR = ROOT / "data-clean/models/xgboost/"
COVARIATES_PATH = ROOT / "data-raw/covariates/training/sentinel-covariates.parquet"
EMBEDDINGS_PATH = ROOT / "data-raw/embeddings/training/sentinel-resnet50-training-embeddings.parquet"
LABELS_PATH = ROOT / "data-clean/dhs/asia-training-labels-sentinel.csv"

# --- Training Params
FIXED_PARAMS = {
    "tree_method": "hist",
    "random_state": 42
}

PARAM_BASE = {
    "learning_rate": [0.001, 0.01, 0.1],
    "subsample": [0.4, 0.6, 0.8],
    "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "reg_lambda": [0.0, 0.5, 1.0, 5.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "gamma": [0.0, 0.1, 0.5, 1.0],
}

PARAM_DEPTHWISE = {"max_depth": [4, 6, 8, 10]}
PARAM_LOSSGUIDE = {"grow_policy": ["lossguide"], "max_leaves": [63, 127, 255, 511], "max_depth": [0]}

```

---

<a id="acknowledgements"></a>
## Acknowledgements

This work relies on survey microdata from the [DHS Program](https://dhsprogram.com), satellite
imagery and derived products made available through [Google Earth Engine](https://earthengine.google.com)
(including Landsat, Sentinel-2, Dynamic World, VIIRS, and the [AlphaEarth Foundations](https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/)
embeddings), and population data from [WorldPop](https://www.worldpop.org). 
