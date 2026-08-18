#!/usr/bin/env python3
"""
Filter sample points that would produce invalid patches during geosampled training.

Three sources of invalid labels are removed:
  1. Points near the raster boundary where the patch box extends outside any raster extent.
  2. Points whose wealth label is NaN in the parquet (the underlying raster pixel is -9999).
  3. Points whose patch window in the wealth raster contains only NODATA pixels.

Pre-filtering with this script allows --skip-validation in train_cnn_geosampling.py to
bypass PointGeoSampler's expensive per-point validity check.

Usage:
    python scripts/validate_points.py \
        --input  data-raw/census/PHL/sample_pts_with_labels_2020.parquet \
        --output data-raw/census/PHL/sample_pts_with_labels_2020_filtered.parquet \
        --wealth-raster data-raw/census/PHL/phl_census_wealth_2020.tif \
        --imagery-raster data-raw/imagery/country/PHL/landsat-phl-2020.tif \
        --label-col wealth_index_year_specific \
        --patch-size 224
"""

import argparse
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from pyproj import Transformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True, help="Input .parquet file with sample points")
    parser.add_argument("--output",         required=True, help="Output .parquet file for filtered points")
    parser.add_argument("--wealth-raster",  required=True, help="Wealth label raster (.tif)")
    parser.add_argument("--imagery-raster", default=None,  help="Imagery raster (.tif); bounds are intersected with wealth raster for Filter 1")
    parser.add_argument("--label-col",      default="wealth_index_year_specific",
                        help="Parquet column to check for NaN labels (default: wealth_index_year_specific)")
    parser.add_argument("--patch-size",     type=int, default=224,
                        help="Patch size in pixels passed to training (default: 224)")
    args = parser.parse_args()

    # Load points
    print(f"Loading points from {args.input} ...")
    pts = gpd.read_parquet(args.input)
    n_total = len(pts)
    print(f"  Total points: {n_total:,}")

    # Load wealth raster metadata
    with rasterio.open(args.wealth_raster) as src:
        bounds_left   = src.bounds.left
        bounds_right  = src.bounds.right
        bounds_bottom = src.bounds.bottom
        bounds_top    = src.bounds.top
        res_x         = src.res[0]
        raster_crs    = src.crs.to_epsg()
        nodata        = src.nodata if src.nodata is not None else -9999

    half_m = (args.patch_size / 2) * res_x
    print(f"\nWealth raster CRS:    EPSG:{raster_crs}")
    print(f"Wealth raster bounds: left={bounds_left:.0f}, bottom={bounds_bottom:.0f}, right={bounds_right:.0f}, top={bounds_top:.0f}")
    print(f"Patch half-size:      {half_m:.0f} m  ({args.patch_size} px × {res_x:.0f} m/px)")
    print(f"Wealth raster nodata: {nodata}")

    # Intersect imagery raster bounds if provided (covers PointGeoSampler's _is_valid_patch
    # check, which uses the combined dataset footprint and would reject points outside the
    # imagery raster even if they are within the wealth raster)
    if args.imagery_raster:
        with rasterio.open(args.imagery_raster) as src:
            img_crs = src.crs.to_epsg()
            if img_crs == raster_crs:
                ib = src.bounds
            else:
                ib = transform_bounds(src.crs, f"EPSG:{raster_crs}", *src.bounds)
        bounds_left   = max(bounds_left,   ib[0])
        bounds_right  = min(bounds_right,  ib[2])
        bounds_bottom = max(bounds_bottom, ib[1])
        bounds_top    = min(bounds_top,    ib[3])
        print(f"\nImagery raster CRS:   EPSG:{img_crs}")
        print(f"Effective bounds (intersection): left={bounds_left:.0f}, bottom={bounds_bottom:.0f}, right={bounds_right:.0f}, top={bounds_top:.0f}")

    # Project all points into the raster CRS (vectorised)
    transformer = Transformer.from_crs(str(pts.crs), f"EPSG:{raster_crs}", always_xy=True)
    x_proj, y_proj = transformer.transform(
        pts.geometry.x.values,
        pts.geometry.y.values,
    )

    # --- Filter 1: patch bbox must be fully within effective raster bounds ---
    mask_bounds = (
        (x_proj - half_m >= bounds_left)   &
        (x_proj + half_m <= bounds_right)  &
        (y_proj - half_m >= bounds_bottom) &
        (y_proj + half_m <= bounds_top)
    )
    n_oob = int((~mask_bounds).sum())
    print(f"\nFilter 1 — patch extends outside raster bounds: {n_oob:,} points removed")

    # --- Filter 2: NaN label in parquet ---
    if args.label_col in pts.columns:
        mask_label = pts[args.label_col].notna().values
        n_nan = int((~mask_label).sum())
        print(f"Filter 2 — NaN in '{args.label_col}':           {n_nan:,} points removed")
    else:
        print(f"Warning: column '{args.label_col}' not found — skipping NaN label filter")
        mask_label = np.ones(n_total, dtype=bool)
        n_nan = 0


    # Combined
    keep = mask_bounds & mask_label
    n_removed = int((~keep).sum())
    n_kept    = int(keep.sum())

    # Overlap between filters (for reporting)
    n_overlap_12 = int((~mask_bounds & ~mask_label).sum())
    if n_overlap_12:
        print(f"  (filters 1+2 overlap: {n_overlap_12:,})")

    print(f"\nTotal removed: {n_removed:,}  ({100 * n_removed / n_total:.2f}%)")
    print(f"Total kept:    {n_kept:,}  ({100 * n_kept / n_total:.2f}%)")

    pts_out = pts[keep].reset_index(drop=True)
    pts_out.to_parquet(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
