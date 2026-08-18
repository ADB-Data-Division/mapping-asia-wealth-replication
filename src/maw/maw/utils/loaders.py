import numpy as np
import torch
import pandas as pd
import pyarrow.dataset as ds
from dataclasses import dataclass
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from ..datasets.landsat import LandsatDataset
from ..datasets.sentinel import SentinelDataset


def build_loaders(
    train_folds: list[int],
    test_fold: int,
    cfg: dataclass,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    weighted: bool = False
):
    """
    Create train and test DataLoaders for a spatial CV split.

    Args:
        train_folds: Tuple of train fold numbers.
        test_fold: Test fold number.
        cfg: Dataclass containing dataset paths and training params.
        num_workers: Number of workers to spawn to load data.
        prefetch_factor: Number of batches to load ahead of time.
        persistent_workers: Keep workers alive between epochs.
        weighted: Whether to use weights from cfg.india_weights.

    Returns:
        Tuple of (train_loader, test_loader).

    Example:
        - split = SpatialCVSplit.from_test_fold(test_fold=1)
        - train_loader, test_loader = build_loaders(split, cfg)
    """

    # Check inputs
    if not isinstance(test_fold, int) or not (1 <= test_fold <= cfg.n_folds):
        raise TypeError(f"test_fold must be an integer between 1 and {cfg.n_folds}")

    if not all([isinstance(f, int) for f in train_folds]) or not all([1 <= f <= cfg.n_folds for f in train_folds]):
        raise TypeError("train_folds must contain only integers between 1 and n_folds")

    if test_fold in train_folds:
        raise ValueError("train_folds and test_fold must be disjoint")

    # Build query strings for train/test subsets
    label_filter = f"{cfg.label_name}.notnull()"
    train_query = f"{label_filter} & ({cfg.fold_column} in {train_folds})"
    test_query = f"{label_filter} & ({cfg.fold_column} == {test_fold})"

    # Common dataset args
    base_args = {
        "obs_csv": cfg.labels_csv,
        "root_dir": cfg.images_root,
        "label_name": cfg.label_name,
        "covariate_path": cfg.covariates_path
    }

    # Create dataset according to imagery type
    if cfg.imagery_type == "landsat":
        train_set = LandsatDataset(**base_args, subset_query=train_query)
        test_set = LandsatDataset(**base_args, subset_query=test_query)
    elif cfg.imagery_type == "sentinel":
        train_set = SentinelDataset(**base_args, subset_query=train_query)
        test_set = SentinelDataset(**base_args, subset_query=test_query)
    else:
        raise NotImplementedError(f"Unknown imagery_type: {cfg.imagery_type}")

    # Handle India down-weighting if configured
    if cfg.india_weights and weighted:
        weights_train = [cfg.india_weights if cc == "IA" else 1 for cc in train_set.obs_table["country_code"]]
        sampler_train = WeightedRandomSampler(
            weights=torch.tensor(weights_train, dtype=torch.float32),
            num_samples=int(min(cfg.n_samples_epoch, len(train_set))),
            replacement=False
        )

        # Subsample India in test set
        india_idxs = [i for i, cc in enumerate(test_set.obs_table["country_code"]) if cc == "IA"]
        other_idxs = [i for i in range(len(test_set.obs_table)) if i not in india_idxs]
        test_idxs = other_idxs + np.random.choice(india_idxs, size=max(1, int(len(india_idxs)*cfg.india_weights)), replace=False).tolist()
        test_set = Subset(test_set, test_idxs)

        train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size, sampler=sampler_train,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers, pin_memory=(cfg.device.type == "cuda")
        )
        test_loader = DataLoader(
            test_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers, pin_memory=(cfg.device.type == "cuda")
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers, pin_memory=(cfg.device.type == "cuda")
        )
        test_loader = DataLoader(
            test_set, batch_size=cfg.batch_size, shuffle=False,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers, pin_memory=(cfg.device.type == "cuda")
        )

    return train_loader, test_loader


def load_features(target_col: str,
                  id_col: str,
                  folds: list,
                  feature_type: str,
                  paths_dict: dict,
                  fold_col: str = "fold_kmeans",
                  embed_backbone_heldoutset: int | None = None,
                  return_meta: bool = False,
                  as_numpy: bool = True
                  ):
    """
    Load feature matrices for a specified set of CV folds by merging labels,
    optional spatial covariates, and optional embedding features.
    target_col : str
        Name of the target (label) column to return as the response vector `y`.
    folds : list of int
        Fold identifiers to retain; rows whose `fold_col` is not in this list
        are dropped.
    feature_type : {"spatial-only", "embeddings-only", "all-features"}
        Controls which feature sources to load:
        - "spatial-only": load spatial covariates only
        - "embeddings-only": load embedding features only
        - "all-features": load both spatial and embedding features
    paths_dict : dict
        Dictionary containing paths to:
        - 'labels-csv'
        - 'spatial-features-parquet' (if spatial features are used)
        - 'embeddings-parquet' (if embeddings are used)
    embed_backbone_heldoutset : int or None
        Value used to filter the embeddings parquet column
        `model_backbone_heldoutset` when that column exists.

    Returns
    X : np.ndarray (float32)
        Feature matrix containing the selected spatial and/or embedding columns.
    y : np.ndarray (float32)
        Target vector aligned to the rows of `X`.
    meta : pd.DataFrame or None
        Associated label data, if `return_meta` is True.

    Notes
    - Embedding columns are inferred from columns matching ``^[A-Za-z][0-9]{1,4}$``
      (e.g., X1..X2048 or A1..A64), with a numeric-column fallback.
    - All merges are performed on `id_col` using an inner join,
      so rows missing in any source table will be dropped.
    - Raises an assertion error if no features are found or required paths
      are missing.
    """

    base_cols = [id_col, fold_col, target_col, "survey_id", "country_code", "survey_year", "rural", "lat", "lon"]
    base = pd.read_csv(paths_dict['labels-csv'], usecols=base_cols)
    base = base[base[fold_col].isin(folds)].reset_index(drop=True)

    assert feature_type in ["spatial-only", "embeddings-only", "all-features"]
    feature_cols = []
    LOAD_SPATIAL = True if feature_type in ["spatial-only", "all-features"] else False
    LOAD_EMBEDDINGS = True if feature_type in ["embeddings-only", "all-features"] else False

    if LOAD_SPATIAL:
        assert "spatial-features-parquet" in paths_dict.keys(), "spatial-features-parquet path must be supplied in paths_dict if features is 'spatial-only' or 'all-features'"
        spatial_covariates = pd.read_parquet(
            path=paths_dict["spatial-features-parquet"],
            engine="pyarrow"
        )
        feature_cols += [c for c in spatial_covariates.columns.tolist() if c != id_col]
        base = base.merge(spatial_covariates, on=[id_col], how="inner")

    if LOAD_EMBEDDINGS:
        embeddings_path = paths_dict.get("embeddings-parquet")
        assert embeddings_path is not None, (
            "embeddings-parquet path must be supplied in paths_dict if features include embeddings"
        )

        embeddings_schema = ds.dataset(embeddings_path, format="parquet").schema
        has_heldout_col = "model_backbone_heldoutset" in embeddings_schema.names

        read_kwargs = {
            "path": embeddings_path,
            "engine": "pyarrow",
        }
        if has_heldout_col:
            assert embed_backbone_heldoutset is not None, (
                "embed_backbone_heldoutset must be supplied when embeddings parquet contains "
                "'model_backbone_heldoutset'"
            )
            read_kwargs["filters"] = [("model_backbone_heldoutset", "==", embed_backbone_heldoutset)]

        embeddings = pd.read_parquet(**read_kwargs)

        embed_cols = embeddings.filter(regex=r"^[A-Za-z][0-9]{1,4}$").columns.tolist()
        if len(embed_cols) == 0:
            metadata_cols = {id_col, "model_backbone_heldoutset"}
            embed_cols = [
                c for c in embeddings.columns
                if c not in metadata_cols and pd.api.types.is_numeric_dtype(embeddings[c])
            ]
        assert len(embed_cols) > 0, "Could not find embedding feature columns in embeddings parquet."

        embeddings = embeddings[[id_col, *embed_cols]]
        base = base.merge(embeddings, on=[id_col], how="inner")
        feature_cols += embed_cols

    assert len(feature_cols) > 0, "Could not find any features to load."
    X = base[feature_cols]
    y = base[target_col]

    if as_numpy:
        X, y = X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32)

    if return_meta:
        return X, y, base[base_cols]

    return X, y
