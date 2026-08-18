import torch
from torchgeo.datasets import RasterDataset


class SpatialCovariates(RasterDataset):
    """Spatial covariate composite."""
    date_format = "%Y"
    is_image = False   # lets this be combined with imagery datasets via IntersectionDataset
    dtype = torch.float32

    def __init__(self, paths, filename_glob: str, filename_regex: str, **kwargs):
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        super().__init__(paths=paths, **kwargs)


class AlphaEarth(RasterDataset):
    """Quantised AlphaEarth Foundations embedding composite."""
    date_format = "%Y"
    is_image = True
    dtype = torch.int16              # widen from the COG's int8 storage so de-quantisation math doesn't overflow
    all_bands = tuple(f"A{i:02d}" for i in range(64))
    rgb_bands = ("A01", "A04", "A03")

    def __init__(self, paths, filename_glob: str, filename_regex: str, **kwargs):
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        super().__init__(paths=paths, **kwargs)

    def plot(self, sample):
        import matplotlib.pyplot as plt

        rgb_indices = [self.all_bands.index(band) for band in self.rgb_bands]
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = torch.clamp(image + 0.3, min=0, max=0.6).numpy()

        fig, ax = plt.subplots()
        ax.imshow(image)
        return fig


class Sentinel(RasterDataset):
    """Sentinel-2 SR composite mosaic."""
    date_format = "%Y"
    is_image = True

    def __init__(self, paths, filename_glob: str, filename_regex: str, **kwargs):
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        super().__init__(paths=paths, **kwargs)


class Landsat(RasterDataset):
    """Landsat SR composite mosaic."""
    date_format = "%Y"
    is_image = True

    def __init__(self, paths, filename_glob: str, filename_regex: str, **kwargs):
        self.filename_glob = filename_glob
        self.filename_regex = filename_regex
        super().__init__(paths=paths, **kwargs)
