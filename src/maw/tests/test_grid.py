import math

import pytest
import rasterio
from shapely.geometry import Polygon, box

from maw.utils.grid import build_grid, get_utm_epsg, grid_transform
from maw.utils.raster import reduce_spatial_mean

import torch


def test_get_utm_epsg_northern_and_southern_hemisphere():
    # Manila, Philippines (~121E, 14.6N) -> UTM zone 51N
    assert get_utm_epsg(120.9842, 14.5995) == 32651
    # Honiara, Solomon Islands (~159.9E, 9.4S) -> UTM zone 57S
    assert get_utm_epsg(159.95, -9.43) == 32757


def test_get_utm_epsg_clamps_zone_to_valid_range():
    assert get_utm_epsg(180.0, 0.0) == 32660
    assert get_utm_epsg(180.00001, 0.0) == 32660


def test_build_grid_fresh_bounds_covers_polygon_and_snaps_to_stride():
    # Right triangle: same bounding box as box(105, 205, 495, 495), but only covers
    # half of it, so the filtered grid should be a strict subset of the full grid.
    polygon = Polygon([(105, 205), (495, 205), (105, 495)])
    full_grid, filtered_grid = build_grid(polygon, crs="EPSG:32643", resolution_m=100.0)

    # Fresh-bounds grid origin is snapped to multiples of the stride from 0.
    assert (full_grid["utm_x"] % 100 == 50).all()
    assert (full_grid["utm_y"] % 100 == 50).all()

    # Every grid *cell* (centroid +/- half-stride) must cover the polygon's bounding box.
    assert full_grid["utm_x"].min() - 50 <= polygon.bounds[0]
    assert full_grid["utm_x"].max() + 50 >= polygon.bounds[2]

    # Filtered grid is a strict subset that actually intersects the polygon.
    assert len(filtered_grid) < len(full_grid)
    assert set(filtered_grid["patch_id"]) <= set(full_grid["patch_id"])
    cells = filtered_grid.geometry.buffer(50, cap_style=3)
    assert cells.intersects(polygon).all()

    # patch_id is unique and keyed off grid_x/grid_y.
    assert full_grid["patch_id"].is_unique
    assert (full_grid["patch_id"] == "PT_" + full_grid["grid_x"].astype(str) + "_" + full_grid["grid_y"].astype(str)).all()


def test_build_grid_align_to_snaps_to_source_transform():
    # A COG with 500m pixels, origin at (200000, 9000000), covering a 10x10 pixel tile.
    transform = rasterio.Affine(500, 0, 200000, 0, -500, 9000000)
    polygon = box(200750, 8997750, 202250, 8999250)  # interior of the tile, off-grid bounds

    full_grid, filtered_grid = build_grid(
        polygon, crs="EPSG:32643", resolution_m=500.0, align_to=transform
    )

    # Grid points must land exactly on the source transform's pixel centers.
    assert ((full_grid["utm_x"] - transform.c) % transform.a == transform.a / 2).all()
    assert len(filtered_grid) > 0
    assert len(filtered_grid) <= len(full_grid)


def test_build_grid_align_to_rejects_non_multiple_resolution():
    transform = rasterio.Affine(500, 0, 0, 0, -500, 0)
    polygon = box(0, -1000, 1000, 0)

    with pytest.raises(ValueError):
        build_grid(polygon, crs="EPSG:32643", resolution_m=100.0, align_to=transform)


def test_build_grid_align_to_supports_coarser_patch_grid():
    # 10m-pixel COG (e.g. a Sentinel COG); request a patch-centroid grid at
    # PATCH_SIZE(=4) * 10m = 40m stride, still snapped to the COG's own pixel grid.
    transform = rasterio.Affine(10, 0, 200000, 0, -10, 9000000)
    polygon = box(200050, 8999600, 200350, 8999950)

    full_grid, filtered_grid = build_grid(
        polygon, crs="EPSG:32643", resolution_m=40.0, align_to=transform
    )

    assert ((full_grid["utm_x"] - transform.c) % 40 == 20).all()
    assert ((transform.f - full_grid["utm_y"]) % 40 == 20).all()
    assert len(filtered_grid) > 0


def test_grid_transform_reproduces_align_to_origin():
    transform = rasterio.Affine(500, 0, 200000, 0, -500, 9000000)
    polygon = box(200750, 8997750, 202250, 8999250)

    full_grid, _ = build_grid(polygon, crs="EPSG:32643", resolution_m=500.0, align_to=transform)
    recovered = grid_transform(full_grid, resolution_m=500.0)

    assert math.isclose(recovered.a, transform.a)
    assert math.isclose((recovered.c - transform.c) % transform.a, 0, abs_tol=1e-6)
    assert math.isclose((transform.f - recovered.f) % transform.a, 0, abs_tol=1e-6)


def test_reduce_spatial_mean_masks_nodata():
    # 1 sample, 2 bands, 2x2 window; band 0 has one nodata pixel, band 1 is all nodata.
    x = torch.tensor(
        [[[[1.0, 3.0], [-9999.0, 5.0]],
          [[-9999.0, -9999.0], [-9999.0, -9999.0]]]]
    )
    out = reduce_spatial_mean(x, nodata=-9999.0)

    assert out.shape == (1, 2)
    assert math.isclose(out[0, 0].item(), (1.0 + 3.0 + 5.0) / 3)
    assert torch.isnan(out[0, 1])


def test_reduce_spatial_mean_single_pixel_is_identity():
    x = torch.tensor([[[[7.0]], [[2.0]]]])  # (1, 2, 1, 1)
    out = reduce_spatial_mean(x)
    assert out.shape == (1, 2)
    assert out.tolist() == [[7.0, 2.0]]
