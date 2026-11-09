#!/usr/bin/env python3
"""Prepare a site-specific vertical-survey uncertainty budget.

This script does not infer instrument accuracy from XYZ scatter. It reports
GOT5.6 tidal variability over the reported collection window as a timing
sensitivity diagnostic, audits possible repeated XY locations, and combines
only documented standard-uncertainty components supplied in
Config/vertical_uncertainty_components.csv. A complete TVU is withheld while
required evidence is missing.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
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
REQUIRED_COMPONENTS = [
    "measurement_system",
    "vertical_control",
    "survey_water_level_reduction",
    "local_vertical_reference_realization",
    "measurement_corrections",
]


def read_points(site_id: str, site: dict) -> pd.DataFrame:
    path = ROOT / site["survey_points"]
    if site_id == "Anza_Agadir":
        points = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["point_id", "x", "y", "z"],
        )
    else:
        points = pd.read_csv(path).rename(columns={"z_source": "z"})
    for col in ("x", "y", "z"):
        points[col] = pd.to_numeric(points[col], errors="coerce")
    points = points.dropna(subset=["x", "y", "z"]).copy()
    if points.empty:
        raise ValueError(f"No usable XYZ points in {path}")
    return points


def representative_position(site: dict, points: pd.DataFrame) -> tuple[float, float]:
    # Exact XYZ copies in CAD exports must not receive extra weight when the
    # representative tide-model location is calculated.
    points = points.drop_duplicates(subset=["x", "y", "z"], keep="first")
    transformer = Transformer.from_crs(site["survey_crs"], "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(float(points.x.median()), float(points.y.median()))
    return float(lon), float(lat)


def predict_tide(lon: float, lat: float, times_local: pd.DatetimeIndex) -> np.ndarray:
    times_utc = times_local.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    result = pyTMD.compute.tide_elevations(
        np.full(len(times_utc), lon, dtype=float),
        np.full(len(times_utc), lat, dtype=float),
        times_utc,
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


def ensure_component_template(path: Path, sites: dict) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)

    component_notes = {
        "measurement_system": (
            "Required: standard uncertainty from instrument specification/calibration or a documented repeatability estimate."
        ),
        "vertical_control": (
            "Required: GNSS/levelling control uncertainty used to establish point elevations."
        ),
        "survey_water_level_reduction": (
            "Required: uncertainty of the water-level/tide reduction used for the field survey; do not use the satellite-acquisition GOT5.6 anomaly as a substitute."
        ),
        "local_vertical_reference_realization": (
            "Required: uncertainty of the realization/transfer of this site's own port hydrographic zero. Keep the two sites separate."
        ),
        "measurement_corrections": (
            "Required: combined standard uncertainty of applicable corrections (e.g. offsets, draft, sound speed, heave, attitude); document any non-applicable terms."
        ),
    }
    rows = []
    for site_id in sites:
        for component in REQUIRED_COMPONENTS:
            rows.append(
                {
                    "site": site_id,
                    "component": component,
                    "standard_uncertainty_m": "",
                    "evidence_or_method": "",
                    "notes": component_notes[component],
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def summarize_components(site_id: str, components: pd.DataFrame) -> dict:
    rows = components.loc[components["site"].astype(str) == site_id].copy()
    if rows.empty:
        return {
            "documented_components_n": 0,
            "required_components_n": len(REQUIRED_COMPONENTS),
            "missing_components": ";".join(REQUIRED_COMPONENTS),
            "u_c_complete_m": np.nan,
            "U95_complete_m": np.nan,
            "budget_status": "INCOMPLETE: no documented components entered",
        }

    rows["standard_uncertainty_m"] = pd.to_numeric(rows["standard_uncertainty_m"], errors="coerce")
    by_component = rows.set_index("component")
    missing = []
    used = []
    for component in REQUIRED_COMPONENTS:
        if component not in by_component.index:
            missing.append(component)
            continue
        item = by_component.loc[component]
        if isinstance(item, pd.DataFrame):
            item = item.iloc[0]
        value = item["standard_uncertainty_m"]
        evidence = str(item.get("evidence_or_method", "")).strip()
        if pd.isna(value) or value < 0 or not evidence:
            missing.append(component)
        else:
            used.append(float(value))

    if missing:
        return {
            "documented_components_n": len(used),
            "required_components_n": len(REQUIRED_COMPONENTS),
            "missing_components": ";".join(missing),
            "u_c_complete_m": np.nan,
            "U95_complete_m": np.nan,
            "budget_status": "INCOMPLETE: full survey TVU not calculated",
        }

    uc = float(np.sqrt(np.sum(np.square(used))))
    return {
        "documented_components_n": len(used),
        "required_components_n": len(REQUIRED_COMPONENTS),
        "missing_components": "",
        "u_c_complete_m": uc,
        "U95_complete_m": 1.96 * uc,
        "budget_status": "COMPLETE under independent-component and normal-coverage assumptions",
    }


def duplicate_xy_diagnostic(points: pd.DataFrame, site_id: str) -> dict:
    # Exclude repeated exports of the same XYZ record before assessing whether
    # distinct Z values exist at an exactly repeated XY coordinate. Rounded
    # coordinate bins can merge nearby, non-identical survey points and must not
    # be interpreted as repeat observations.
    duplicate_xyz_rows = int(points.duplicated(["x", "y", "z"]).sum())
    unique_xyz = points.drop_duplicates(["x", "y", "z"])
    groups = unique_xyz.groupby(["x", "y"], sort=False).z.agg(["count", "min", "max"])
    repeated = groups.loc[groups["count"] > 1].copy()
    ranges = repeated["max"] - repeated["min"]
    return {
        "site": site_id,
        "survey_points_n": len(points),
        "exact_duplicate_xyz_rows_removed": duplicate_xyz_rows,
        "unique_xyz_points_n": len(unique_xyz),
        "repeated_exact_xy_groups_n": len(repeated),
        "points_in_repeated_exact_xy_groups_n": int(repeated["count"].sum()) if len(repeated) else 0,
        "median_within_group_z_range_m": float(ranges.median()) if len(ranges) else np.nan,
        "p95_within_group_z_range_m": float(ranges.quantile(0.95)) if len(ranges) else np.nan,
        "max_within_group_z_range_m": float(ranges.max()) if len(ranges) else np.nan,
        "interpretation": "exact-coordinate consistency diagnostic only; not a repeatability uncertainty unless independent repeat observations are verified from survey provenance",
    }


def main() -> None:
    config = json.loads((ROOT / "Config" / "experiment.json").read_text(encoding="utf-8"))
    out = ROOT / "Results" / "Additional_Analyses"
    out.mkdir(parents=True, exist_ok=True)
    component_path = ROOT / "Config" / "vertical_uncertainty_components.csv"
    component_inputs = ensure_component_template(component_path, config["sites"])

    tide_rows = []
    duplicate_rows = []
    budget_rows = []
    for site_id, site in config["sites"].items():
        window = site.get("survey_collection_window_local")
        if not window or not window.get("start") or not window.get("end"):
            raise ValueError(f"No reported collection window configured for {site_id}")
        tz = ZoneInfo(window.get("timezone", "Africa/Casablanca"))
        survey_date = date.fromisoformat(site["survey_date"])
        start_local = datetime.combine(survey_date, time.fromisoformat(window["start"]), tzinfo=tz)
        end_local = datetime.combine(survey_date, time.fromisoformat(window["end"]), tzinfo=tz)
        if end_local <= start_local:
            raise ValueError(f"Collection-window end must follow start for {site_id}")

        points = read_points(site_id, site)
        lon, lat = representative_position(site, points)
        times = pd.date_range(start_local, end_local, freq="1min", inclusive="both")
        tide = predict_tide(lon, lat, times)
        finite = np.isfinite(tide)
        values = tide[finite]
        if not len(values):
            raise ValueError(f"GOT5.6 returned no valid predictions for {site_id}")
        mean = float(np.mean(values))
        anomalies = values - mean
        tide_rows.append(
            {
                "site": site_id,
                "survey_date_local": survey_date.isoformat(),
                "collection_start_local": start_local.isoformat(),
                "collection_end_local": end_local.isoformat(),
                "timezone": window.get("timezone", "Africa/Casablanca"),
                "representative_longitude": lon,
                "representative_latitude": lat,
                "survey_points_n": len(points),
                "model": MODEL_NAME,
                "valid_minute_predictions_n": len(values),
                "modelled_window_mean_model_reference_m": mean,
                "tide_anomaly_sd_about_window_mean_m": float(np.std(anomalies, ddof=0)),
                "tide_anomaly_p2_5_m": float(np.quantile(anomalies, 0.025)),
                "tide_anomaly_p97_5_m": float(np.quantile(anomalies, 0.975)),
                "tide_anomaly_min_m": float(np.min(anomalies)),
                "tide_anomaly_max_m": float(np.max(anomalies)),
                "tide_anomaly_range_m": float(np.max(anomalies) - np.min(anomalies)),
                "interpretation": "timing-ambiguity sensitivity under uniform time sampling; not GOT5.6 model error, not an absolute port-datum tie, and not full survey TVU",
            }
        )
        duplicate_rows.append(duplicate_xy_diagnostic(points, site_id))
        budget_rows.append(
            {
                "site": site_id,
                "vertical_reference": site.get("vertical_reference", "site-local vertical reference"),
                **summarize_components(site_id, component_inputs),
            }
        )

    pd.DataFrame(tide_rows).to_csv(out / "survey_window_tide_timing_diagnostic.csv", index=False, float_format="%.6f")
    pd.DataFrame(duplicate_rows).to_csv(out / "survey_duplicate_xy_diagnostic.csv", index=False, float_format="%.6f")
    pd.DataFrame(budget_rows).to_csv(out / "survey_vertical_uncertainty_budget.csv", index=False, float_format="%.6f")

    print("Vertical uncertainty assessment prepared.")
    print(f"  Tide-window sensitivity: {out / 'survey_window_tide_timing_diagnostic.csv'}")
    print(f"  Repeated-coordinate diagnostic: {out / 'survey_duplicate_xy_diagnostic.csv'}")
    print(f"  Survey uncertainty budget: {out / 'survey_vertical_uncertainty_budget.csv'}")
    print(f"  Fill documented standard uncertainties and evidence in: {component_path}")
    print("A full U95 is reported only after every required component has a documented value.")


if __name__ == "__main__":
    main()
