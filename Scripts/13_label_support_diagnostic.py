#!/usr/bin/env python3
"""Quantify survey-label heterogeneity and aggregation sensitivity per HLS pixel.

These diagnostics use the supplied sounding coordinates and the 30 m HLS grid.
They quantify within-pixel seabed variability and sensitivity to soundings
duplicated in the CAD export. They are not instrument accuracy, vertical-datum
uncertainty, or a substitute for hydrographic total vertical uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol


ROOT = Path(__file__).resolve().parents[1]
SUBCELLS_PER_AXIS = 3  # 3 x 3 ten-metre support diagnostic within a 30 m HLS pixel
MIN_OCCUPIED_SUBCELLS_FOR_SHIFT = 3


def read_survey(site_id: str, site: dict) -> pd.DataFrame:
    path = ROOT / site["survey_points"]
    if path.suffix.lower() == ".csv":
        points = pd.read_csv(path).rename(columns={"z_source": "z"})
    else:
        points = pd.read_csv(path, sep=r"\s+", header=None, names=["point_id", "x", "y", "z"])
    for column in ("x", "y", "z"):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    points = points.dropna(subset=["x", "y", "z"]).copy()
    points = points.loc[points["z"] <= 0].copy()
    points["depth_m"] = -points["z"]
    to_image = Transformer.from_crs(site["survey_crs"], site["image_crs"], always_xy=True)
    points["x_img"], points["y_img"] = to_image.transform(points.x.to_numpy(), points.y.to_numpy())
    return points


def aggregate_site(site_id: str, site: dict, inventory: dict) -> tuple[pd.DataFrame, dict]:
    items = sorted(
        [item for item in inventory["items"] if item["site"] == site_id and item["status"] == "complete"],
        key=lambda item: item["datetime_utc"],
    )
    if not items:
        raise ValueError(f"No completed HLS scenes for {site_id}")

    reference = ROOT / items[0]["assets"]["blue"]["file"]
    with rasterio.open(reference) as ds:
        transform = ds.transform
        shape = (ds.height, ds.width)
        image_crs = ds.crs
    if str(image_crs) != site["image_crs"]:
        raise ValueError(f"Configured image CRS and reference raster CRS differ for {site_id}")

    points = read_survey(site_id, site)
    rows, cols = rowcol(transform, points.x_img.to_numpy(), points.y_img.to_numpy())
    points["row"] = np.asarray(rows, dtype=int)
    points["col"] = np.asarray(cols, dtype=int)
    inside = (
        (points.row >= 0) & (points.row < shape[0]) &
        (points.col >= 0) & (points.col < shape[1])
    )
    points = points.loc[inside].copy()

    # Exact duplicate XYZ records are repeat exports, not independent samples.
    # Keep the original point-weighted median for comparison with the current
    # pipeline, and calculate a second median after removing exact XYZ copies.
    points["_is_exact_xyz_duplicate"] = points.duplicated(["x", "y", "z"], keep="first")
    # Derive within-pixel fractions directly from the raster affine transform,
    # so subcell assignment remains correct if a future chip has a different
    # origin or pixel orientation.
    inverse = ~transform
    pixel_col_f, pixel_row_f = inverse * (points.x_img.to_numpy(), points.y_img.to_numpy())
    points["subcell_col"] = np.floor((pixel_col_f - points.col.to_numpy()) * SUBCELLS_PER_AXIS).clip(0, 2).astype(int)
    points["subcell_row"] = np.floor((pixel_row_f - points.row.to_numpy()) * SUBCELLS_PER_AXIS).clip(0, 2).astype(int)

    rows_out = []
    for (row, col), group in points.groupby(["row", "col"], sort=False):
        depth = group.depth_m.to_numpy(dtype=float)
        median = float(np.median(depth))
        q25, q75 = np.quantile(depth, [0.25, 0.75])
        mad = float(np.median(np.abs(depth - median)))
        unique = group.loc[~group["_is_exact_xyz_duplicate"]]
        deduplicated_median = float(unique.depth_m.median())
        sub = group.groupby(["subcell_row", "subcell_col"], sort=False).depth_m.median()
        subcell_medians = sub.to_numpy(dtype=float)
        occupied = int(len(subcell_medians))
        enough = occupied >= MIN_OCCUPIED_SUBCELLS_FOR_SHIFT
        rows_out.append(
            {
                "site": site_id,
                "row": int(row),
                "col": int(col),
                "n_raw_soundings": int(len(group)),
                "n_unique_xyz_soundings": int(len(unique)),
                "occupied_10m_subcells_n": occupied,
                "within_pixel_depth_median_m": median,
                "within_pixel_depth_iqr_m": float(q75 - q25),
                "within_pixel_depth_mad_scaled_m": 1.4826 * mad,
                "within_pixel_depth_p05_m": float(np.quantile(depth, 0.05)),
                "within_pixel_depth_p95_m": float(np.quantile(depth, 0.95)),
                "subcell_median_iqr_m": float(np.quantile(subcell_medians, 0.75) - np.quantile(subcell_medians, 0.25)) if occupied >= 2 else np.nan,
                "equal_subcell_median_m": float(np.median(subcell_medians)),
                "abs_point_vs_equal_subcell_median_shift_m": abs(median - float(np.median(subcell_medians))) if enough else np.nan,
                "abs_raw_vs_deduplicated_median_shift_m": abs(median - deduplicated_median),
                "deduplicated_pixel_median_m": deduplicated_median,
            }
        )

    result = pd.DataFrame(rows_out)
    summary = {
        "site": site_id,
        "survey_reference": site.get("vertical_reference", "site-local vertical reference"),
        "submerged_soundings_inside_hls_chip_n": int(len(points)),
        "exact_duplicate_xyz_rows_n": int(points["_is_exact_xyz_duplicate"].sum()),
        "pixels_n": int(len(result)),
        "median_soundings_per_pixel": float(result.n_raw_soundings.median()),
        "median_within_pixel_iqr_m": float(result.within_pixel_depth_iqr_m.median()),
        "p90_within_pixel_iqr_m": float(result.within_pixel_depth_iqr_m.quantile(0.90)),
        "median_within_pixel_mad_scaled_m": float(result.within_pixel_depth_mad_scaled_m.median()),
        "p90_within_pixel_mad_scaled_m": float(result.within_pixel_depth_mad_scaled_m.quantile(0.90)),
        "pixels_with_at_least_3_occupied_10m_subcells_n": int((result.occupied_10m_subcells_n >= MIN_OCCUPIED_SUBCELLS_FOR_SHIFT).sum()),
        "median_abs_point_vs_equal_subcell_median_shift_m": float(result.abs_point_vs_equal_subcell_median_shift_m.median()),
        "p90_abs_point_vs_equal_subcell_median_shift_m": float(result.abs_point_vs_equal_subcell_median_shift_m.quantile(0.90)),
        "pixels_with_nonzero_duplicate_median_shift_n": int((result.abs_raw_vs_deduplicated_median_shift_m > 0).sum()),
        "max_abs_raw_vs_deduplicated_median_shift_m": float(result.abs_raw_vs_deduplicated_median_shift_m.max()),
        "interpretation": "within-pixel seabed heterogeneity and aggregation sensitivity; not instrument accuracy, datum uncertainty, or a formal U95",
    }
    result["interpretation"] = summary["interpretation"]
    return result, summary


def main() -> None:
    config = json.loads((ROOT / "Config" / "experiment.json").read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / "Data" / "Satellite" / "scene_inventory.json").read_text(encoding="utf-8"))
    pixel_rows = []
    site_rows = []
    for site_id, site in config["sites"].items():
        pixels, summary = aggregate_site(site_id, site, inventory)
        pixel_rows.append(pixels)
        site_rows.append(summary)

    out = ROOT / "Results" / "Additional_Analyses"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(pixel_rows, ignore_index=True).to_csv(
        out / "survey_pixel_label_support.csv.gz", index=False, compression="gzip", float_format="%.6f"
    )
    summary = pd.DataFrame(site_rows)
    summary.to_csv(out / "survey_pixel_label_support_summary.csv", index=False, float_format="%.6f")
    print("Label-support diagnostics complete; these do not estimate survey instrument accuracy.")
    print(summary[[
        "site", "submerged_soundings_inside_hls_chip_n", "pixels_n", "median_soundings_per_pixel",
        "median_within_pixel_iqr_m", "p90_within_pixel_iqr_m",
        "p90_abs_point_vs_equal_subcell_median_shift_m",
        "max_abs_raw_vs_deduplicated_median_shift_m",
    ]].to_string(index=False))
    print(f"Wrote {out / 'survey_pixel_label_support.csv.gz'}")
    print(f"Wrote {out / 'survey_pixel_label_support_summary.csv'}")


if __name__ == "__main__":
    main()
