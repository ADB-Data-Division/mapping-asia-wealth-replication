import torch
from torch import nn
import kornia.augmentation as K


class ClampToRange(K.IntensityAugmentationBase2D):
    """Clamp to supplied min/max."""
    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor):
        super().__init__(p=1)
        self.flags = {'mins': mins.view(1, -1, 1, 1), 'maxs': maxs.view(1, -1, 1, 1)}

    def apply_transform(self, input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, torch.Tensor],
                        transform: K.AugmentationBase2D | None = None
                        ) -> torch.Tensor:
        return torch.clamp(input, min=flags['mins'], max=flags['maxs']) # clip to min/max


class ClampAndRescale(K.IntensityAugmentationBase2D):
    """Normalize channels to [0, 1] by clamping to supplied min/max and rescaling."""
    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor):
        super().__init__(p=1.0)

        if mins.ndim != 1 or maxs.ndim != 1:
            raise ValueError("mins and maxs must be 1D tensors of length C")
        if mins.shape != maxs.shape:
            raise ValueError("mins and maxs must have the same shape")
        if torch.any(maxs <= mins):
            raise ValueError("All maxs must be strictly greater than mins")

        self.register_buffer("mins", mins.view(1, -1, 1, 1))
        self.register_buffer("maxs", maxs.view(1, -1, 1, 1))

    def apply_transform(self, input: torch.Tensor,
                        params: dict[str, torch.Tensor],
                        flags: dict[str, torch.Tensor],
                        transform: K.AugmentationBase2D | None = None
                        ) -> torch.Tensor:
        clipped = torch.clamp(input, min=self.mins, max=self.maxs)
        return (clipped - self.mins) / (self.maxs - self.mins)


class RescaleAndClamp(K.IntensityAugmentationBase2D):
    """Normalize channels to [0, 1] by rescaling using the supplied scale and offset, and clamping."""
    def __init__(self, scale: torch.Tensor, offset: torch.Tensor, min: torch.Tensor, max: torch.Tensor):
        super().__init__(p=1.0)
        if scale.ndim == 0:
            scale = scale.view(1)
        if offset.ndim == 0:
            offset = offset.view(1)
        if min.ndim == 0:
            min = min.view(1)
        if max.ndim == 0:
            max = max.view(1)

        self.register_buffer('scale', scale.view(1, -1, 1, 1))
        self.register_buffer('offset', offset.view(1, -1, 1, 1))
        self.register_buffer('min', min.view(1, -1, 1, 1))
        self.register_buffer('max', max.view(1, -1, 1, 1))

    def apply_transform(self,
                        input: torch.Tensor,
                        params: dict,
                        flags: dict,
                        transform: K.AugmentationBase2D | None = None) -> torch.Tensor:
        # input expected shape: (B, C, H, W)
        x = input.to(torch.float32)
        rescaled = x * self.scale + self.offset
        return rescaled.clamp(min=self.min, max=self.max)


class DequantiseAEF(nn.Module):
    """De-quantise AlphaEarth Foundations embeddings stored as signed int8.

    Google's raw AEF asset is float32; export_country_composites.ipynb re-quantises it
    to int8 before export to shrink COG size (see its `quantise_aef`). This inverts that
    step. Must be applied per-pixel before any spatial averaging, since the transform is
    nonlinear (averaging quantised ints then de-quantising is not equivalent to
    de-quantising then averaging).
    """
    def __init__(self, output_nodata: float = float("nan")):
        super().__init__()
        self.register_buffer("nodata_val", torch.tensor(-128, dtype=torch.int8))
        self.output_nodata = output_nodata

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: int8 (or wider) tensor containing raw quantised AEF values, any shape.

        Returns:
            float32 tensor of de-quantised AEF values, same shape as `x`.
        """
        x_float = x.to(torch.float32)
        mask = (x == self.nodata_val)
        sign = torch.sign(x_float)
        mag = torch.abs(x_float) / 127.5            # undo scaling
        mag = mag ** 2                              # undo sqrt
        out = sign * mag                            # restore sign
        out[mask] = self.output_nodata               # reapply nodata
        return out
