import pandas as pd
import pyproj
import pytest
import torch
from geopandas import GeoDataFrame
from shapely.geometry import box
from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.datasets.utils import stack_samples

from maw.utils.samplers import PointGeoSampler


class DummyGeoDataset(GeoDataset):
    def __init__(self) -> None:
        super().__init__()
        self._res = (1, 1)
        self._bounds = BoundingBox(0, 6000, 0, 6000, 0, 1)
        # torchgeo's GeoDataset.index is a GeoDataFrame keyed by a datetime IntervalIndex
        # (see RasterDataset.__init__), not the older rtree-based index.
        interval = pd.IntervalIndex.from_tuples(
            [(pd.Timestamp(self._bounds.mint), pd.Timestamp(self._bounds.maxt))],
            closed="both", name="datetime",
        )
        geometry = [box(self._bounds.minx, self._bounds.miny, self._bounds.maxx, self._bounds.maxy)]
        self.index = GeoDataFrame(
            {"filepath": ["dummy"]}, index=interval, geometry=geometry,
            crs=pyproj.CRS.from_epsg(3857),
        )

    def __getitem__(self, query: tuple[slice, slice, slice]) -> dict[str, torch.Tensor]:
        x_slice, y_slice, _t_slice = query
        width = int((x_slice.stop - x_slice.start) / self.res[0])
        height = int((y_slice.stop - y_slice.start) / self.res[1])
        image = torch.zeros((6, height, width), dtype=torch.float32)
        bounds_tensor = torch.tensor(
            [
                x_slice.start,
                x_slice.stop,
                self.res[0],
                y_slice.start,
                y_slice.stop,
                self.res[1],
                0.0,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )
        return {"image": image, "bounds": bounds_tensor}


def test_sampler_returns_expected_patches() -> None:
    dataset = DummyGeoDataset()
    pts = [(0.01, 0.01), (0.03, 0.02)]
    sampler = PointGeoSampler(
        dataset, points_xy=pts, points_crs="EPSG:4326", size=(500, 500)
    )

    loader = DataLoader(dataset, sampler=sampler, collate_fn=stack_samples)
    transformer = pyproj.Transformer.from_crs(
        pyproj.CRS.from_epsg(3857),
        pyproj.CRS.from_epsg(4326),
        always_xy=True,
    )

    lon_lats = []
    x_scales = []
    y_scales = []
    shapes = []

    for sample in loader:
        xmin, xmax, xscale, ymin, ymax, yscale, *_ = sample["bounds"].squeeze()
        x = float((xmin + xmax) / 2)
        y = float((ymin + ymax) / 2)
        lon, lat = transformer.transform(x, y)
        lon_lats.append((lon, lat))
        x_scales.append(xscale.item())
        y_scales.append(yscale.item())
        shapes.append(tuple(sample["image"].shape))

    assert len(sampler) == len(pts)
    for (lon, lat), (exp_lon, exp_lat) in zip(lon_lats, pts):
        assert abs(lon - exp_lon) < 1e-4
        assert abs(lat - exp_lat) < 1e-4
    assert all(scale == 1 for scale in x_scales)
    assert all(scale == 1 for scale in y_scales)
    assert all(shape == (1, 6, 500, 500) for shape in shapes)


def test_sampler_respects_drop_invalid_flag() -> None:
    dataset = DummyGeoDataset()
    pts = [(0.01, 0.01), (1.0, 1.0)]

    with pytest.raises(ValueError):
        PointGeoSampler(
            dataset, points_xy=pts, points_crs="EPSG:4326", size=(500, 500)
        )

    sampler = PointGeoSampler(
        dataset,
        points_xy=pts,
        points_crs="EPSG:4326",
        size=(500, 500),
        drop_invalid=True,
    )

    assert len(sampler) == 1
    x_slice, y_slice, _ = next(iter(sampler))
    bounds_x, bounds_y, _ = dataset.bounds
    assert x_slice.start >= bounds_x.start
    assert x_slice.stop <= bounds_x.stop
    assert y_slice.start >= bounds_y.start
    assert y_slice.stop <= bounds_y.stop


def test_sampler_accepts_empty_points() -> None:
    dataset = DummyGeoDataset()
    sampler = PointGeoSampler(
        dataset, points_xy=[], points_crs="EPSG:4326", size=(500, 500)
    )
    assert len(sampler) == 0
    assert list(iter(sampler)) == []


def test_sampler_accepts_single_xy_tuple() -> None:
    dataset = DummyGeoDataset()
    sampler = PointGeoSampler(
        dataset, points_xy=(0.01, 0.01), points_crs="EPSG:4326", size=(500, 500)
    )
    assert len(sampler) == 1
