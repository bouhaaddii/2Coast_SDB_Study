#!/usr/bin/env python3
"""Estimate astronomical tide at HLS overpasses and on survey dates with GOT5.6.

Outputs include model-reference tidal elevations and acquisition anomalies
relative to the modeled mean over the local survey date. The anomaly removes the
unknown constant model-reference offset; it is not an absolute local-ZH level.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyTMD
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "Data" / "Tide_Model"
MODEL_NAME = "GOT5.6"
MODEL_CUTOFF_KM = 10.0
LOCAL_TIME_ZONE = ZoneInfo("Africa/Casablanca")


def representative_position(site_id: str, site: dict) -> tuple[float, float, int]:
    """Return WGS84 median position after removing exact duplicate XYZ exports."""
    survey_path = ROOT / site["survey_points"]
    if site_id == "Anza_Agadir":
        points = pd.read_csv(
            survey_path,
            sep=r"\s+",
            header=None,
            names=["point_id", "x", "y", "z"],
        )
    else:
        points = pd.read_csv(survey_path).rename(columns={"z_source": "z"})

    points = points[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").dropna()
    points = points.drop_duplicates(subset=["x", "y", "z"], keep="first")
    if points.empty:
        raise ValueError(f"No valid survey coordinates found in {survey_path}")

    x_med = float(points["x"].median())
    y_med = float(points["y"].median())
    transformer = Transformer.from_crs(site["survey_crs"], "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(x_med, y_med)
    return float(longitude), float(latitude), int(len(points))


def predict(lon: float, lat: float, utc_times: pd.DatetimeIndex) -> np.ndarray:
    """Run GOT5.6 with a local crop and a conservative coastal wet-node cutoff."""
    times = utc_times.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    result = pyTMD.compute.tide_elevations(
        np.full(len(times), lon, dtype=float),
        np.full(len(times), lat, dtype=float),
        times,
        directory=MODEL_DIR,
        model=MODEL_NAME,
        crs="4326",
        standard="datetime",
        type="trajectory",
        method="linear",
        crop=True,
        bounds=[lon - 0.5, lon + 0.5, lat - 0.5, lat + 0.5],
        buffer=0.25,
        extrapolate=True,
        cutoff=MODEL_CUTOFF_KM,
    )
    return np.asarray(result.values, dtype=float).reshape(-1)


def scene_name(notes: str) -> str:
    match = re.search(r"HLS\.[^;\s]+", notes or "")
    return match.group(0) if match else ""


def main() -> None:
    config_path = ROOT / "Config" / "experiment.json"
    inventory_path = ROOT / "Config" / "tide_levels.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tide_rows = pd.read_csv(inventory_path, keep_default_na=False)
    out_dir = ROOT / "Results" / "Tides"
    out_dir.mkdir(parents=True, exist_ok=True)

    overpass_outputs: list[pd.DataFrame] = []
    survey_range_outputs: list[dict] = []
    summary_outputs: list[dict] = []

    for site_id, site in config["sites"].items():
        lon, lat, n_soundings = representative_position(site_id, site)
        scene_rows = tide_rows.loc[
            (tide_rows["site"] == site_id)
            & (tide_rows["record_type"] == "satellite_overpass")
            & (tide_rows["datetime_utc"].astype(str).str.len() > 0)
        ].copy()
        scene_rows["datetime_utc"] = pd.to_datetime(scene_rows["datetime_utc"], utc=True)
        scene_rows = scene_rows.sort_values("datetime_utc").reset_index(drop=True)

        # The survey plan supplies the date but not a time. Use the modeled mean
        # over that local civil day as the anomaly reference, and retain the
        # full-day range so the assumption is visible in the outputs.
        survey_date = date.fromisoformat(site["survey_date"])
        local_start = datetime.combine(survey_date, time.min, tzinfo=LOCAL_TIME_ZONE)
        local_end = local_start + timedelta(days=1)
        local_times = pd.date_range(local_start, local_end, freq="10min", inclusive="left")
        day_values = predict(lon, lat, local_times)
        finite = np.isfinite(day_values)
        survey_day_mean = float(np.mean(day_values[finite])) if finite.any() else np.nan

        scene_times = pd.DatetimeIndex(scene_rows["datetime_utc"])
        scene_values = predict(lon, lat, scene_times)
        scene_rows["scene_id"] = scene_rows["notes"].map(scene_name)
        scene_rows["representative_longitude"] = lon
        scene_rows["representative_latitude"] = lat
        scene_rows["representative_soundings_n"] = n_soundings
        scene_rows["tide_model"] = MODEL_NAME
        scene_rows["model_tidal_elevation_m"] = scene_values
        scene_rows["survey_day_mean_m"] = survey_day_mean
        scene_rows["tide_anomaly_m_relative_to_survey_day_mean"] = scene_values - survey_day_mean
        scene_rows["model_reference"] = "GOT5.6 harmonic reference; acquisition value minus local survey-day mean"
        scene_rows["spatial_method"] = (
            "linear interpolation; wet-node extrapolation allowed within 10 km"
        )
        scene_rows["local_ZH_level_m"] = np.nan
        scene_rows["status"] = np.where(
            np.isfinite(scene_values - survey_day_mean), "estimated model anomaly; no absolute local-ZH tie asserted", "no model value"
        )
        overpass_outputs.append(scene_rows)

        # The survey plans identify a date, not a measurement time. Calculate a
        # 10-minute modelled envelope over the local civil day instead of inventing
        # an exact survey-time tide.
        if finite.any():
            valid_idx = np.flatnonzero(finite)
            min_idx = valid_idx[np.argmin(day_values[finite])]
            max_idx = valid_idx[np.argmax(day_values[finite])]
            low, high = float(day_values[min_idx]), float(day_values[max_idx])
            low_time = local_times[min_idx].isoformat()
            high_time = local_times[max_idx].isoformat()
        else:
            low = high = np.nan
            low_time = high_time = ""

        survey_range_outputs.append(
            {
                "site": site_id,
                "survey_date_local": survey_date.isoformat(),
                "survey_time_status": "not supplied; daily envelope reported",
                "representative_longitude": lon,
                "representative_latitude": lat,
                "representative_soundings_n": n_soundings,
                "tide_model": MODEL_NAME,
                "modelled_daily_mean_m": survey_day_mean,
                "modelled_daily_min_m": low,
                "modelled_daily_min_time_local": low_time,
                "modelled_daily_max_m": high,
                "modelled_daily_max_time_local": high_time,
                "modelled_daily_range_m": high - low if np.isfinite(low) else np.nan,
                "model_reference": "GOT5.6 harmonic reference; daily mean is the relative anomaly baseline",
                "spatial_method": "linear interpolation; wet-node extrapolation allowed within 10 km",
                "status": "estimated model-reference envelope; survey time unknown" if np.isfinite(low) else "no model value",
            }
        )

        finite_scene_values = scene_values[np.isfinite(scene_values)]
        summary_outputs.append(
            {
                "site": site_id,
                "site_name": site["display_name"],
                "satellite_scenes_n": len(scene_values),
                "estimated_scenes_n": int(np.isfinite(scene_values).sum()),
                "modelled_min_overpass_m": float(np.min(finite_scene_values)) if finite_scene_values.size else np.nan,
                "modelled_median_overpass_m": float(np.median(finite_scene_values)) if finite_scene_values.size else np.nan,
                "modelled_max_overpass_m": float(np.max(finite_scene_values)) if finite_scene_values.size else np.nan,
                "survey_day_min_m": low,
                "survey_day_mean_m": survey_day_mean,
                "survey_day_max_m": high,
                "survey_day_range_m": high - low if np.isfinite(low) else np.nan,
                "model_reference": "GOT5.6 harmonic reference; acquisition anomalies are relative to survey-day mean",
            }
        )

        # Fill the pipeline's explicit relative-anomaly field. Keep the legacy
        # local-ZH level blank: GOT5.6 has not been tied to either port datum.
        if "tide_anomaly_m_relative_to_survey_day_mean" not in tide_rows.columns:
            tide_rows["tide_anomaly_m_relative_to_survey_day_mean"] = ""
        lookup = {
            (str(row.site), str(row.scene_id)): row.tide_anomaly_m_relative_to_survey_day_mean
            for row in scene_rows.itertuples(index=False)
        }
        for row_index, config_row in tide_rows.iterrows():
            if config_row["site"] == site_id and config_row["record_type"] == "satellite_overpass":
                scene_id = scene_name(str(config_row.get("notes", "")))
                value = lookup.get((site_id, scene_id), np.nan)
                if pd.notna(value) and np.isfinite(float(value)):
                    tide_rows.at[row_index, "tide_anomaly_m_relative_to_survey_day_mean"] = f"{float(value):.6f}"

    pd.concat(overpass_outputs, ignore_index=True).to_csv(
        out_dir / "modelled_tide_overpasses_GOT5.6.csv", index=False, float_format="%.5f"
    )
    pd.DataFrame(survey_range_outputs).to_csv(
        out_dir / "modelled_survey_day_tide_range_GOT5.6.csv", index=False, float_format="%.5f"
    )
    pd.DataFrame(summary_outputs).to_csv(
        out_dir / "modelled_tide_summary_GOT5.6.csv", index=False, float_format="%.5f"
    )
    tide_rows.to_csv(inventory_path, index=False)

    note = """# GOT5.6 tide estimates for Anza and Ras El Ma

These files contain predicted astronomical tidal elevations from the NASA GSFC
GOT5.6 model, calculated with pyTMD. Raw heights use the model harmonic
reference. The pipeline uses each acquisition's anomaly relative to the modeled
mean over that site's local survey date. This removes the unknown constant
model-reference offset; it does not establish an absolute tie to either port's
Hydrographic Zero.

## Coordinates and timing

The model point for each site is the WGS84 position obtained by transforming the
median projected X/Y coordinates of that site's bathymetric sounding dataset.
The table `modelled_tide_overpasses_GOT5.6.csv` contains the estimate at each HLS
acquisition time in UTC. Survey plans supply dates but not survey times, so
`modelled_survey_day_tide_range_GOT5.6.csv` gives a 10-minute-sampled range across
the local civil survey date in Africa/Casablanca time.

## Coastal, timing, and vertical-datum limits

GOT5 is a global 1/8-degree harmonic atlas. For these coastal points, pyTMD was
allowed to extrapolate from the nearest valid wet model node only when it was
within 10 km. This is a regional astronomical estimate, not a gauge observation;
it does not include local wind setup, atmospheric pressure residuals, wave setup,
or run-up. The survey plans provide dates but not survey times. The daily mean is
therefore an explicit reference assumption; the full-day min/max range is
provided to show the possible within-day variation. The `local_ZH_level_m` field
remains empty because no absolute tie to the port zeros is asserted. The new
`tide_anomaly_m_relative_to_survey_day_mean` field in `Config/tide_levels.csv`
contains only relative anomalies for the model's intended depth-plus-tide
adjustment.

## Model files and references

GOT5.6 combines the four additional degree-3 constituents in NASA's GOT5.6
archive with the base GOT5.5 constituents. GOT5.5 files used here are from the
pyTMD developer model-data mirror on Zenodo (record 10.5281/zenodo.22970362;
the record states that the models remain the property of their developers and
may be used for scientific purposes). The GOT5.6 supplement was fetched from
NASA GSFC. Cite the original GOT model documentation and software, not this
calculation note, in any manuscript.

- NASA GSFC, Ocean Tide Models: https://earth.gsfc.nasa.gov/geo/data/ocean-tide-models
- Ray, R. D. (2025), GOT5 technical memorandum, NASA TM-20250002085:
  https://sedvme.gsfc.nasa.gov/sed/content/uploadFiles/publication_files/GOT5-TechMemo.pdf
- pyTMD documentation: https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html
- pyTMD Zenodo model-data record: https://doi.org/10.5281/zenodo.22970362
"""
    (out_dir / "README_GOT5.6_estimates.md").write_text(note, encoding="utf-8")

    print(f"Wrote tide estimates to {out_dir}")
    print(pd.DataFrame(summary_outputs).to_string(index=False))


if __name__ == "__main__":
    main()
