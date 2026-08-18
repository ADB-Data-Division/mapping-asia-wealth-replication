import os
import torch
import rasterio
import logging
import random
import pandas as pd
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from collections.abc import Sequence
import kornia.augmentation as K
from .covariates import CovariateLoader, EXPECTED_COVARIATES
from ..preprocessing.augmentation import ClampAndRescale


# Suppress rasterio warnings
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

# Normalisation Constants for SATLAS Pretrained Models
# NOTE: Satlas normalises INT16 values for B5-12 by dividing by 8160; uses TCI values for B2-4 (which saturate at SR=0.25)
# See also ResNet50_Weights.SENTINEL2_SI_MS_SATLAS.transforms
SATLAS_BAND_ORDER = [3, 2, 1, 4, 5, 6, 7, 8, 9]                     # SATLAS expects RGB bands ordered as B4, B3, B2...
SATLAS_MIN_INT = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
SATLAS_MAX_INT = torch.tensor([2500, 2500, 2500, 8160, 8160, 8160, 8160, 8160, 8160], dtype=torch.float32)

TRANSFORMS_VAL = K.AugmentationSequential(
    ClampAndRescale(mins=SATLAS_MIN_INT, maxs=SATLAS_MAX_INT)
)

TRANSFORMS_TRAIN = K.AugmentationSequential(
    ClampAndRescale(mins=SATLAS_MIN_INT, maxs=SATLAS_MAX_INT),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation90(times=(0, 3), p=1),
)


def to_satlas_band_order(x: torch.Tensor) -> torch.Tensor:
    """Reorder a Sentinel-2 tensor with bands [B2,B3,B4,B5,B6,B7,B8,B11,B12] (...,C,H,W)
    to SATLAS's expected [B4,B3,B2,B5,B6,B7,B8,B11,B12] order.

    Mirrors the band reordering SentinelDataset applies when reading patch GeoTIFFs
    directly via rasterio (indexes=SATLAS_BAND_ORDER); needed here because sampling
    imagery from a COG via a raw RasterDataset/PointGeoSampler does not reorder bands.
    """
    band_idx = [i - 1 for i in SATLAS_BAND_ORDER]  # SATLAS_BAND_ORDER is 1-indexed (rasterio convention)
    return x[..., band_idx, :, :]


# ---- SENTINEL DATASET CLASS ----

class SentinelDataset(Dataset):
    """Dataset consisting of Sentinel patches centred on DHS clusters with corresponding wealth index labels."""
    def __init__(self,
                 obs_csv: str,
                 root_dir: str,
                 patch_id_name: str = "patch_id",
                 label_name: str | None = None,
                 transform: K.AugmentationSequential | None = None,
                 subset_query: str | None = None,
                 covariate_path: str | None = None,
                 expected_covariates: tuple[str, ...] = EXPECTED_COVARIATES
                 ):
        """
        Arguments:
            obs_csv: Path to the csv file with labels and additional patch info.
            root_dir: Directory with all the images.
            patch_id_name: Name of column containing patch identifiers.
            label_name: Name of column containing label values.
            transform: Transform to be applied on sample.
            subset_query: Pandas query string to subset observations (e.g., "fold in [1,2,3]").
            image_crop_size: Size of the center crop to extract from the image.
            image_width: Width of the source image.
            image_height: Height of the source image.
            covariate_path: Path to parquet file containing spatial covariates.
            expected_covariates: Tuple of expected covariate column names in order.
        """
        self.obs_table = pd.read_csv(obs_csv)
        if subset_query:
            self.obs_table = self.obs_table.query(subset_query).reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, path) for path in self.obs_table['image_name']]
        self.patch_ids = list(self.obs_table[patch_id_name])
        self.labels = torch.tensor(self.obs_table[label_name].values, dtype=torch.float32) if label_name else None

        # Initialize covariate loader if path provided
        self.covariate_loader = None
        if covariate_path is not None:
            self.covariate_loader = CovariateLoader(
                covariate_path=covariate_path,
                patch_id_name=patch_id_name,
                patch_ids=self.patch_ids,
                expected_covariates=expected_covariates,
            )


    def __len__(self):
        return len(self.obs_table)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with rasterio.open(path, sharing=False) as src:
            # reorder channels to [B4,B3,B2,B5,B6,B7,B8,B11,B12] to match SATLAS input order
            image = torch.from_numpy(src.read(indexes=SATLAS_BAND_ORDER)).float()

        if self.transform:
            image = self.transform(image.unsqueeze(0)).squeeze(0)

        patch_id = self.patch_ids[idx]
        label = self.labels[idx] if self.labels is not None else torch.tensor(-1.0)

        result = {"image": image, "label": label, "patch_id": patch_id}
        if self.covariate_loader:
            result["covariates"] = self.covariate_loader[patch_id]
        return result

    def plot_random_16(self, bands: Sequence[int] = (0, 1, 2), show_patch_id: bool = False, **kwargs):
        """Randomly select and plot 16 images from a SentinelDataset."""
        num_samples = min(16, len(self.image_paths))
        random_indices = random.sample(range(len(self.image_paths)), num_samples)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        for i, idx in enumerate(random_indices):

            sample = self.__getitem__(idx)
            image = sample["image"].squeeze().numpy()
            label = sample["label"]
            patch_id = sample["patch_id"]
            rgb_image = image[bands, :, :]

            ax = axes[i // 4, i % 4]
            ax.imshow(rgb_image.transpose(1, 2, 0), **kwargs)  # transpose to (H, W, C) for plotting
            ax.axis("off")
            title = f"ID: {patch_id}" if show_patch_id else f"Wealth Index: {label.item():.2f}"
            ax.set_title(title)

        plt.tight_layout()
        plt.show()
