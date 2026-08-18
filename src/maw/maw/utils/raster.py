import math
import torch


def reduce_spatial_mean(x: torch.Tensor, nodata: float | None = None) -> torch.Tensor:
    """Collapse the trailing (H, W) dims of a (..., C, H, W) tensor to a per-band mean.

    Used to turn a raster window sampled around a coarse grid point (e.g. via
    `PointGeoSampler(units=Units.CRS, size=resolution_m)`) into a single "zonal mean"
    feature vector per point, without a separate zonal-stats step.

    Args:
        x: tensor of shape (..., C, H, W).
        nodata: value (or NaN) marking invalid pixels, excluded from the mean. If None,
            all pixels are treated as valid.

    Returns:
        Tensor of shape (..., C). A channel with no valid pixels in a window is NaN.
    """
    x = x.float()
    if nodata is not None:
        valid = torch.isnan(x) if math.isnan(nodata) else (x == nodata)
        x = torch.where(valid, torch.nan, x)
    return torch.nanmean(x, dim=(-2, -1))
