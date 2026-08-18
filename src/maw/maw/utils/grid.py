import math
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point


def get_utm_epsg(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) // 6) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def build_grid(
    polygon,
    crs,
    resolution_m: float,
    buffer: float = 0.0,
    align_to: rasterio.Affine | None = None,
    id_prefix: str = "PT",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build a regular point grid over `polygon`'s (optionally buffered) bounding box.

    Grid spacing is `resolution_m` unless `align_to` is given, in which case the grid's
    *origin* is snapped to that raster transform's own pixel grid so points land exactly
    on multiples of its pixel boundaries -- `resolution_m` must then be a positive integer
    multiple of the transform's own pixel size (checked). Passing `resolution_m` equal to
    the transform's pixel size yields one grid point per source pixel (e.g. for point-
    sampling covariates/AEF at their native resolution); passing a larger multiple (e.g.
    `patch_size * pixel_size`) yields a coarser patch-centroid grid whose patch windows
    still land on exact pixel boundaries of the source raster (e.g. for CNN patch
    extraction from a Sentinel/Landsat COG).

    Args:
        polygon: shapely polygon (in `crs`) to build the grid over.
        crs: CRS of `polygon`, used for the returned GeoDataFrames.
        resolution_m: grid spacing in `crs` units (typically meters).
        buffer: distance to buffer `polygon` by before computing bounds/filtering.
        align_to: optional affine transform of an existing raster to snap the grid to.
        id_prefix: prefix used to build each point's `patch_id`.

    Returns:
        (full_bbox_grid, polygon_filtered_grid) -- GeoDataFrames in `crs` with columns
        patch_id, grid_x, grid_y, utm_x, utm_y, epsg_crs, geometry. `full_bbox_grid`
        covers the entire snapped bounding box (needed to size the output raster);
        `polygon_filtered_grid` is restricted to cells intersecting the buffered polygon.
    """
    buffered = polygon.buffer(buffer) if buffer else polygon
    xmin, ymin, xmax, ymax = buffered.bounds

    if align_to is not None:
        origin_x, origin_y = align_to.c, align_to.f
        px_x, px_y = align_to.a, -align_to.e
        if px_x <= 0 or resolution_m <= 0:
            raise ValueError("resolution_m and align_to's pixel size must both be positive")
        ratio = resolution_m / px_x
        if not math.isclose(ratio, round(ratio), rel_tol=1e-3) or round(ratio) < 1:
            raise ValueError(
                f"resolution_m ({resolution_m}) is not a positive integer multiple of "
                f"align_to's pixel size ({px_x}) -- grid points would not land on exact "
                "pixel boundaries of the source raster."
            )
        stride_x = stride_y = resolution_m
    else:
        origin_x, origin_y = 0.0, 0.0
        stride_x = stride_y = resolution_m

    col0 = math.floor((xmin - origin_x) / stride_x)
    col1 = math.ceil((xmax - origin_x) / stride_x)
    row0 = math.floor((origin_y - ymax) / stride_y)
    row1 = math.ceil((origin_y - ymin) / stride_y)

    grid_xmin = origin_x + col0 * stride_x
    grid_ymax = origin_y - row0 * stride_y
    n_cols, n_rows = col1 - col0, row1 - row0

    x_centroids = grid_xmin + (np.arange(n_cols) + 0.5) * stride_x
    y_centroids = grid_ymax - (np.arange(n_rows) + 0.5) * stride_y

    grid_pts = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x in x_centroids for y in y_centroids],
        crs=crs,
    )
    grid_pts["grid_x"] = np.rint((grid_pts.geometry.x - grid_xmin) / stride_x - 0.5).astype(int)
    grid_pts["grid_y"] = np.rint((grid_ymax - grid_pts.geometry.y) / stride_y - 0.5).astype(int)
    grid_pts["utm_x"] = grid_pts.geometry.x
    grid_pts["utm_y"] = grid_pts.geometry.y
    epsg = gpd.GeoSeries([polygon], crs=crs).crs.to_epsg()
    grid_pts["epsg_crs"] = f"EPSG:{epsg}"
    grid_pts["patch_id"] = (
        id_prefix + "_" + grid_pts["grid_x"].astype(str) + "_" + grid_pts["grid_y"].astype(str)
    )

    grid_cells = grid_pts.geometry.buffer(min(stride_x, stride_y) / 2, cap_style=3)
    filtered = grid_pts.loc[grid_cells.intersects(buffered)].reset_index(drop=True)

    return grid_pts.reset_index(drop=True), filtered


def grid_transform(full_grid: gpd.GeoDataFrame, resolution_m: float) -> rasterio.Affine:
    """Recover the north-up affine transform for a grid built by `build_grid`.

    `full_grid` must be the *first* (unfiltered bounding-box) element returned by
    `build_grid`, so its extent matches the raster this transform will describe.
    """
    xmin = full_grid["utm_x"].min() - resolution_m / 2
    ymax = full_grid["utm_y"].max() + resolution_m / 2
    return rasterio.transform.from_origin(xmin, ymax, resolution_m, resolution_m)
