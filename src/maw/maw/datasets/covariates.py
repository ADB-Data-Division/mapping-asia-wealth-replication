"""Shared covariate loading utilities for image datasets."""

import torch
import pandas as pd
import numpy as np
import warnings
from typing import Sequence
from pathlib import Path


# Expected covariate column names and order
EXPECTED_COVARIATES: tuple[str, ...] = (
    'lulc_prob_water', 'lulc_prob_trees', 'lulc_prob_grass',
       'lulc_prob_flooded_vegetation', 'lulc_prob_crops',
       'lulc_prob_shrub_and_scrub', 'lulc_prob_built', 'lulc_prob_bare',
       'lulc_prob_snow_and_ice', 'viirs_mean', 'viirs_max', 'viirs_min',
       'viirs_stdDev', 'population_count', 'built_surface',
       'built_surface_nres', 'tmmx_mean', 'tmmx_max', 'tmmx_min',
       'tmmx_stdDev', 'tmmn_mean', 'tmmn_max', 'tmmn_min', 'tmmn_stdDev',
       'precipitation_mean', 'precipitation_max', 'precipitation_min',
       'precipitation_stdDev', 'soil_ph_0cm'
)


class CovariateLoader:
    """
    Loads and indexes spatial covariates from a parquet file.

    Covariates are loaded once at initialization and indexed by patch_id for
    efficient lookup during training. Validates that covariate columns match
    EXPECTED_COVARIATES in name and order.
    """

    def __init__(
            self,
            covariate_path: str | Path,
            patch_ids: Sequence[str],
            patch_id_name: str = "patch_id",
            fill_na: bool = True,
            normalise_values: bool = True,
            expected_covariates: tuple[str, ...] = EXPECTED_COVARIATES,
    ):
        """
        Args:
            covariate_path: Path to parquet file containing covariates.
            patch_id_name: Column name used as the patch identifier.
            patch_ids: Sequence of patch IDs to filter covariates to.
            gap_fill: Bool indicating if NAs should be replaced with country-year means values.
            normalise_values: Bool indicating if covariates should be standardised to mean zero unit var.
            expected_covariates: Tuple of expected covariate column names in order.
        """
        self.patch_id_name = patch_id_name
        self.expected_covariates = list(expected_covariates)

        # Load parquet and filter to relevant patch IDs
        covs_df = pd.read_parquet(covariate_path, engine="pyarrow")
        covs_df = covs_df[covs_df[patch_id_name].isin(patch_ids)].copy()

        # Validate id and covariate columns
        feature_cols = [c for c in covs_df.columns if c != patch_id_name]
        self._validate_columns(feature_cols)
        dup_mask = covs_df[patch_id_name].duplicated()
        if dup_mask.any():
            dup_ids = covs_df.loc[dup_mask, patch_id_name].unique()
            raise ValueError(f"Duplicate patch_ids found: {dup_ids[:10]}")

        if fill_na:
            covs_df = self._gap_fill(covs_df)

        if normalise_values: 
            covs_df = self._normalise(covs_df)

        # Build lookup: patch_id -> tensor of covariates
        covs_df = covs_df.set_index(patch_id_name)
        values = covs_df[self.expected_covariates].astype(np.float32)
        self._covariate_lookup = {
            pid: torch.from_numpy(row.values.copy())
            for pid, row in values.iterrows()
        }

    def _validate_columns(self, feature_cols: list[str]) -> None:
        """Validate that feature columns match expected covariates in name and order."""
        if feature_cols != self.expected_covariates:
            missing = set(self.expected_covariates) - set(feature_cols)
            extra = set(feature_cols) - set(self.expected_covariates)
            msg = f"Covariate columns do not match expected. Missing: {missing}, Extra: {extra}"
            if not missing and not extra:
                msg = f"Covariate column order mismatch. Expected: {self.expected_covariates}, Got: {feature_cols}"
            raise ValueError(msg)

    def _gap_fill(self, data):
        
        data = data.copy()
        data['country_year'] = data[self.patch_id_name].str[:6]
        cols = data.columns.drop([self.patch_id_name, 'country_year'])
    
        grouped = data.groupby('country_year')[cols]
        means = grouped.transform('mean')
        data[cols] = data[cols].fillna(means).fillna(0)
        
        # Warn if all obs for country-year are NAs
        na_mean_mask = grouped.mean().isna()
        if na_mean_mask.any().any():
            problematic = {
                col: na_mean_mask.index[na_mean_mask[col]].tolist()
                for col in na_mean_mask.columns if na_mean_mask[col].any()
            }
            warnings.warn(
                f"All-NA groups detected (filled with 0). Details: {problematic}",
                RuntimeWarning
            )

        return data.drop(columns=['country_year'])

    def _normalise(self, data):
        
        data = data.copy()
        cols = data.columns.drop([self.patch_id_name])
        mean = data[cols].mean()
        std = data[cols].std()
    
        zero_std_cols = std[std == 0].index.tolist()
        if zero_std_cols:
            raise ValueError(f"Zero-variance columns detected: {zero_std_cols}")
    
        data[cols] = (data[cols] - mean) / std
        return data

    def __getitem__(self, patch_id: str) -> torch.Tensor:
        """Return covariate tensor for given patch_id."""
        return self._covariate_lookup[patch_id]

    def __contains__(self, patch_id: str) -> bool:
        return patch_id in self._covariate_lookup
