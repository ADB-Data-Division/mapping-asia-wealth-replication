import numpy as np
import torch
import pyproj
import pandas as pd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from torchgeo.datasets import GeoDataset
from torchgeo.samplers import GeoSampler, Units
from collections.abc import Iterator, Sequence
from typing import Optional, Tuple, Union


def _to_tuple(x):
    return (x, x) if isinstance(x, (int, float)) else x


class PointGeoSampler(GeoSampler):
    """
    Deterministic sampler that yields patches centered at supplied points.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        points_xy: Sequence[Tuple[float, float]],
        points_crs: Union[str, int, pyproj.CRS],
        size: Union[float, Tuple[float, float]],
        *,
        roi: Optional[Polygon] = None,
        toi: Optional[pd.Interval] = None,
        units: Units = Units.PIXELS,
        drop_invalid: bool = False,
        skip_validation: bool = False,
    ) -> None:

        super().__init__(dataset=dataset, roi=roi, toi=toi)

        if not skip_validation and getattr(self.index, "sindex", None) is None:
            raise RuntimeError(
                "GeoDataFrame.sindex is unavailable. "
                "Install spatial index dependencies (e.g., pygeos/rtree)."
            )

        self.size = _to_tuple(size)
        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self._half_h = 0.5 * self.size[0]
        self._half_w = 0.5 * self.size[1]
        self._t_slice = slice(self.toi.left, self.toi.right)

        transformer = pyproj.Transformer.from_crs(
            crs_from=pyproj.CRS.from_user_input(points_crs),
            crs_to=pyproj.CRS.from_user_input(dataset.crs),
            always_xy=True,
        )

        # Transform all points in one vectorised call, then filter valid ones.
        # Storing coords as float64 array (~16 bytes/point) instead of slice
        # tuples (~200 bytes/point) cuts memory ~12x for large point sets.
        pts_in = np.asarray(points_xy, dtype=np.float64)
        if pts_in.size == 0:
            pts_in = pts_in.reshape(0, 2)
        elif pts_in.ndim == 1:
            if pts_in.shape[0] != 2:
                raise ValueError("points_xy must be a sequence of (x, y) pairs")
            pts_in = pts_in.reshape(1, 2)
        elif pts_in.ndim != 2 or pts_in.shape[1] != 2:
            raise ValueError("points_xy must be a sequence of (x, y) pairs")

        if len(pts_in):
            xs_ds, ys_ds = transformer.transform(pts_in[:, 0], pts_in[:, 1])
            xs_ds = np.asarray(xs_ds, dtype=np.float64)
            ys_ds = np.asarray(ys_ds, dtype=np.float64)
        else:
            xs_ds = np.empty(0, dtype=np.float64)
            ys_ds = np.empty(0, dtype=np.float64)

        if skip_validation:
            self._coords = np.stack([xs_ds, ys_ds], axis=1)
        else:
            valid_mask = np.zeros(len(pts_in), dtype=bool)
            for i, (x_ds, y_ds) in enumerate(zip(xs_ds, ys_ds)):
                patch_poly = box(
                    x_ds - self._half_w, y_ds - self._half_h,
                    x_ds + self._half_w, y_ds + self._half_h,
                )
                if self._is_valid_patch(patch_poly):
                    valid_mask[i] = True
                elif not drop_invalid:
                    x_in, y_in = pts_in[i]
                    raise ValueError(
                        f"Invalid patch for point ({x_in}, {y_in}) -> ({x_ds}, {y_ds})."
                    )
            self._coords = np.stack([xs_ds[valid_mask], ys_ds[valid_mask]], axis=1)  # (N_valid, 2)

    def _is_valid_patch(self, patch_poly: Polygon) -> bool:
        hits = self.index.sindex.query(patch_poly, predicate="intersects")
        return len(hits) > 0

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        half_h, half_w, t_slice = self._half_h, self._half_w, self._t_slice
        for x_ds, y_ds in self._coords:
            yield (
                slice(x_ds - half_w, x_ds + half_w),
                slice(y_ds - half_h, y_ds + half_h),
                t_slice,
            )

    def __len__(self) -> int:
        return len(self._coords)


class RandomPointGeoSampler(PointGeoSampler):
    """
    Like PointGeoSampler but randomly samples a fixed number of patches per epoch.

    All points are validated upfront at construction time. Each call to __iter__
    draws epoch_size points without replacement (using torch.randperm so it
    respects torch.manual_seed).
    """

    def __init__(self, *args, epoch_size: int, fixed: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if epoch_size > len(self._coords):
            raise ValueError(
                f"epoch_size ({epoch_size}) exceeds number of valid points ({len(self._coords)})"
            )
        self.epoch_size = epoch_size
        # Fixed subset: draw once at construction so early stopping monitors a stable target.
        self._fixed_indices = torch.randperm(len(self._coords))[:epoch_size].tolist() if fixed else None

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        indices = self._fixed_indices if self._fixed_indices is not None else torch.randperm(len(self._coords))[:self.epoch_size].tolist()
        half_h, half_w, t_slice = self._half_h, self._half_w, self._t_slice
        for i in indices:
            x_ds, y_ds = self._coords[i]
            yield (
                slice(x_ds - half_w, x_ds + half_w),
                slice(y_ds - half_h, y_ds + half_h),
                t_slice,
            )

    def __len__(self) -> int:
        return self.epoch_size
