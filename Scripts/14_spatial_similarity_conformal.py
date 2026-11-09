"""Exploratory spatial-similarity calibration of SDB prediction intervals.

This script adds an uncertainty-calibration layer to the existing HLS ±30-day
models. It does not change the bathymetric estimators or overwrite prior outputs.
The local weighting idea is adapted from recent geospatial conformal work; the
spatially dependent blocks here do NOT justify claiming an exact finite-sample
coverage guarantee. Results must be described as empirical validation.

Method references:
  GeoSIMCP (2026), DOI: 10.1080/24694452.2026.2639721
  Depth-conditional conformal SDB (2026), DOI: 10.1109/LGRS.2026.3743086
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
spec = importlib.util.spec_from_file_location(
    "validation_stage", ROOT / "Scripts" / "02_run_validation.py"
)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)

# Prespecified primary analysis settings. Change only before inspecting the
# resulting test-fold coverage, and report any sensitivity analysis separately.
PRIMARY_WINDOW_DAYS = int(V.CFG.get("primary_temporal_window_days", 30))
CALIBRATION_FRACTION = 0.40
CONFIDENCE_LEVELS = (0.80, 0.90)
SPATIAL_BANDWIDTH_M = 1000.0
FEATURE_BANDWIDTH_RMS = 1.0
MIN_EFFECTIVE_CALIBRATION_BLOCKS = 3.0
MODELS = ("StumpfLogRatio", "RandomForest", "HistGradientBoosting", "SVR_RBF")


def group_ids(coords: np.ndarray) -> np.ndarray:
    block = np.floor(coords / V.BLOCK_M).astype(int)
    return np.asarray([f"{x}_{y}" for x, y in block], dtype=object)


def conservative_order_statistic(values: np.ndarray, coverage: float) -> float:
    """Finite-sample split-conformal order statistic over block scores.

    Returns +inf if the number of calibration blocks cannot support the
    requested nominal coverage with a finite order statistic.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("inf")
    rank = int(math.ceil((n + 1) * coverage))
    if rank > n:
        return float("inf")
    return float(np.sort(values)[rank - 1])


def weighted_empirical_quantile(values: np.ndarray, weights: np.ndarray,
                                coverage: float) -> float:
    """Weighted local quantile; empirical only under spatial dependence."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return float("inf")
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[min(np.searchsorted(cumulative, coverage, side="left"),
                            len(values) - 1)])


def depth_band(predicted_depth: np.ndarray) -> np.ndarray:
    return pd.cut(
        np.asarray(predicted_depth, dtype=float),
        bins=[-np.inf, 5.0, 10.0, np.inf],
        labels=["<=5 m", ">5–10 m", ">10 m"],
        right=True,
    ).astype(str)


def block_summaries(cal_xy: np.ndarray, cal_z: np.ndarray,
                    cal_abs_error: np.ndarray, cal_band: np.ndarray,
                    cal_groups: np.ndarray):
    """Use each calibration block as one spatial unit, not each pixel."""
    records = []
    for block in pd.unique(cal_groups):
        take = cal_groups == block
        if not take.any():
            continue
        record = {
            "block_id": str(block),
            "x_m": float(np.mean(cal_xy[take, 0])),
            "y_m": float(np.mean(cal_xy[take, 1])),
            "feature": np.mean(cal_z[take], axis=0),
            # Maximum absolute calibration error gives a conservative
            # block-level nonconformity score.
            "score": float(np.max(cal_abs_error[take])),
        }
        for band in ("<=5 m", ">5–10 m", ">10 m"):
            btake = take & (cal_band == band)
            record[f"score::{band}"] = (
                float(np.max(cal_abs_error[btake])) if btake.any() else np.nan
            )
        records.append(record)
    return records


def main():
    tides, tide_available = V.load_tide_anomalies(False)
    if not tide_available:
        raise SystemExit(
            "GOT5.6 anomalies are incomplete. Run Scripts/05_estimate_tides.py first."
        )

    all_predictions = []
    skipped = []
    seed = int(V.SEED)
    alpha_levels = tuple(CONFIDENCE_LEVELS)
    print(
        "Spatial-similarity interval study: HLS Mixed, quality-weighted composite, "
        f"±{PRIMARY_WINDOW_DAYS} days; GOT5.6 anomaly adjustment."
    )
    print(
        "Three interval methods: global block calibration, predicted-depth "
        "conditional block calibration, and local geography+spectral weighting."
    )
    print(
        "The local weighted method is exploratory; no exact coverage guarantee "
        "is claimed for these spatially dependent observations."
    )

    for site_key, site in V.CFG["sites"].items():
        df = V.SAMPLES[V.SAMPLES.site == site_key].reset_index(drop=True)
        quality = V.QUALITY[V.QUALITY.site == site_key].copy()
        quality = quality[quality.date_offset_days.abs() <= PRIMARY_WINDOW_DAYS]
        quality = quality.sort_values("datetime_utc")
        if quality.empty:
            skipped.append({"site": site_key, "reason": "No scenes in primary time window"})
            continue

        X, h = V.read_composite(
            df, quality.item_id.tolist(), quality, tides, "quality_weighted_mean"
        )
        y = df.depth_m.to_numpy(dtype=float)
        xy = df[["x_m", "y_m"]].to_numpy(dtype=float)
        groups = group_ids(xy)
        valid = np.isfinite(y) & np.isfinite(h) & np.isfinite(X).all(axis=1)
        original = np.flatnonzero(valid)
        X, y, h, xy, groups = X[valid], y[valid], h[valid], xy[valid], groups[valid]
        if len(np.unique(groups)) < V.N_SPLITS + 5:
            skipped.append({"site": site_key,
                            "reason": "Too few 500-m spatial blocks for nested calibration"})
            continue

        all_blocks = np.unique(groups)
        outer = GroupKFold(n_splits=V.N_SPLITS)
        splits = list(outer.split(np.zeros(len(all_blocks)), groups=all_blocks))
        site_block_count = len(all_blocks)
        print(f"  {site_key}: {site_block_count} spatial blocks, {len(original)} usable pixels")

        for model_name in MODELS:
            model_X = V.stumpf_ratio(X) if model_name == "StumpfLogRatio" else X
            model_valid = np.isfinite(model_X).all(axis=1)
            if not model_valid.all():
                # The Stumpf ratio may be undefined where blue/green is invalid.
                take = np.flatnonzero(model_valid)
                model_data = {
                    "X": model_X[take], "raw_X": X[take], "y": y[take],
                    "h": h[take], "xy": xy[take], "groups": groups[take],
                    "original": original[take],
                }
            else:
                model_data = {"X": model_X, "raw_X": X, "y": y, "h": h,
                              "xy": xy, "groups": groups, "original": original}

            mx, raw_x = model_data["X"], model_data["raw_X"]
            my, mh, mxy, mg = (model_data[k] for k in ("y", "h", "xy", "groups"))
            original_ix = model_data["original"]
            block_values = np.unique(mg)

            for fold, (train_group_ix, test_group_ix) in enumerate(splits, start=1):
                train_groups = all_blocks[train_group_ix]
                test_groups = all_blocks[test_group_ix]
                dev_groups = np.asarray([g for g in train_groups if g in set(block_values)], dtype=object)
                if len(dev_groups) < 6:
                    skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                    "reason": "Fewer than six development blocks"})
                    continue

                rng_seed = seed + sum(map(ord, site_key + model_name)) + fold * 1009
                rng = np.random.default_rng(rng_seed)
                n_cal = max(5, int(math.ceil(CALIBRATION_FRACTION * len(dev_groups))))
                n_cal = min(n_cal, len(dev_groups) - 3)
                if n_cal < 3:
                    skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                    "reason": "Insufficient calibration blocks"})
                    continue
                cal_groups = np.sort(rng.choice(dev_groups, size=n_cal, replace=False))
                fit_groups = np.asarray([g for g in dev_groups if g not in set(cal_groups)], dtype=object)

                test0 = np.flatnonzero(np.isin(mg, test_groups))
                cal0 = np.flatnonzero(np.isin(mg, cal_groups))
                fit0 = np.flatnonzero(np.isin(mg, fit_groups))
                if min(len(test0), len(cal0), len(fit0)) == 0:
                    continue

                # Keep calibration and fit samples outside the test exclusion buffer.
                test_tree = cKDTree(mxy[test0])
                cal_distance, _ = test_tree.query(mxy[cal0], k=1)
                cal_ix = cal0[cal_distance > V.BUFFER_M]
                if len(cal_ix) == 0:
                    skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                    "reason": "No calibration pixels remain after spatial buffer"})
                    continue
                excluded_xy = np.vstack([mxy[test0], mxy[cal_ix]])
                excluded_tree = cKDTree(excluded_xy)
                fit_distance, _ = excluded_tree.query(mxy[fit0], k=1)
                fit_ix = fit0[fit_distance > V.BUFFER_M]
                if len(fit_ix) < 30:
                    skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                    "reason": "Fewer than 30 fit pixels after spatial buffers"})
                    continue

                # Fit the prediction model only on the fit blocks.
                model = V.new_model(model_name)
                model.fit(mx[fit_ix], my[fit_ix] + mh[fit_ix])
                cal_prediction = model.predict(mx[cal_ix]) - mh[cal_ix]
                test_prediction = model.predict(mx[test0]) - mh[test0]
                cal_abs_error = np.abs(cal_prediction - my[cal_ix])
                cal_band = depth_band(cal_prediction)
                test_band = depth_band(test_prediction)

                # Local feature distances are standardized from the fit set only.
                scaler = StandardScaler().fit(raw_x[fit_ix])
                cal_z = scaler.transform(raw_x[cal_ix])
                test_z = scaler.transform(raw_x[test0])
                cal_group_ids = mg[cal_ix]
                summaries = block_summaries(
                    mxy[cal_ix], cal_z, cal_abs_error, cal_band, cal_group_ids
                )
                if len(summaries) < 3:
                    skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                    "reason": "Fewer than three buffered calibration blocks remain"})
                    continue

                score_all = np.asarray([r["score"] for r in summaries], dtype=float)
                centers = np.asarray([[r["x_m"], r["y_m"]] for r in summaries])
                features = np.vstack([r["feature"] for r in summaries])

                # Block-score sets for depth-conditional calibration.
                band_scores = {}
                for band in ("<=5 m", ">5–10 m", ">10 m"):
                    vals = np.asarray([r[f"score::{band}"] for r in summaries], dtype=float)
                    band_scores[band] = vals[np.isfinite(vals)]

                for level in alpha_levels:
                    q_global = conservative_order_statistic(score_all, level)
                    q_by_band = {
                        band: conservative_order_statistic(vals, level)
                        for band, vals in band_scores.items()
                    }

                    for j, sample in enumerate(test0):
                        geo_distance = np.sqrt(np.sum((centers - mxy[sample]) ** 2, axis=1))
                        feature_distance = np.sqrt(np.mean((features - test_z[j]) ** 2, axis=1))
                        log_weight = -0.5 * (
                            (geo_distance / SPATIAL_BANDWIDTH_M) ** 2
                            + (feature_distance / FEATURE_BANDWIDTH_RMS) ** 2
                        )
                        log_weight -= np.max(log_weight)
                        weights = np.exp(log_weight)
                        effective_blocks = float(weights.sum() ** 2 / np.sum(weights ** 2))
                        q_local = weighted_empirical_quantile(score_all, weights, level)
                        local_fallback = effective_blocks < MIN_EFFECTIVE_CALIBRATION_BLOCKS
                        if local_fallback:
                            q_local = q_global
                        q_depth = q_by_band.get(str(test_band[j]), q_global)

                        for method, radius, fallback in (
                            ("Global block", q_global, False),
                            ("Predicted-depth conditional", q_depth,
                             not np.isfinite(q_depth) and np.isfinite(q_global)),
                            ("Local geography + spectrum", q_local, local_fallback),
                        ):
                            all_predictions.append({
                                "site": site_key,
                                "model": model_name,
                                "fold": fold,
                                "block_id": str(mg[sample]),
                                "sample_index": int(original_ix[sample]),
                                "x_m": float(mxy[sample, 0]),
                                "y_m": float(mxy[sample, 1]),
                                "observed_depth_m": float(my[sample]),
                                "predicted_depth_m": float(test_prediction[j]),
                                "absolute_error_m": float(abs(test_prediction[j] - my[sample])),
                                "predicted_depth_band": str(test_band[j]),
                                "nominal_coverage": float(level),
                                "interval_method": method,
                                "radius_m": float(radius),
                                "lower_m": float(test_prediction[j] - radius),
                                "upper_m": float(test_prediction[j] + radius),
                                "covered": bool(abs(test_prediction[j] - my[sample]) <= radius),
                                "interval_width_m": float(2 * radius),
                                "calibration_blocks_n": int(len(summaries)),
                                "local_effective_blocks_n": effective_blocks,
                                "fallback_used": bool(fallback),
                                "tide_case": "GOT5.6_anomaly",
                                "scene_pool": "HLS_Mixed",
                                "temporal_window_days": PRIMARY_WINDOW_DAYS,
                                "spatial_block_size_m": float(V.BLOCK_M),
                                "spatial_buffer_m": float(V.BUFFER_M),
                            })
                print(
                    f"    {model_name}, fold {fold}: fit={len(fit_ix):,}; "
                    f"calibration blocks={len(summaries)}; test={len(test0):,}",
                    flush=True,
                )

    if not all_predictions:
        raise SystemExit("No interval predictions were created; inspect the skip diagnostics.")

    out = ROOT / "Results" / "Additional_Analyses"
    out.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(all_predictions)
    pred_path = out / "spatial_similarity_conformal_oof.csv.gz"
    predictions.to_csv(pred_path, index=False, compression="gzip")

    summary = (predictions.groupby(
        ["site", "model", "nominal_coverage", "interval_method", "predicted_depth_band"],
        dropna=False, as_index=False
    ).agg(
        test_pixels_n=("covered", "size"),
        test_blocks_n=("block_id", "nunique"),
        empirical_pixel_coverage=("covered", "mean"),
        mean_interval_width_m=("interval_width_m", "mean"),
        median_interval_width_m=("interval_width_m", "median"),
        mean_absolute_error_m=("absolute_error_m", "mean"),
        calibration_blocks_mean=("calibration_blocks_n", "mean"),
        local_effective_blocks_median=("local_effective_blocks_n", "median"),
        fallback_fraction=("fallback_used", "mean"),
    ))
    summary["coverage_error"] = summary["empirical_pixel_coverage"] - summary["nominal_coverage"]
    summary.to_csv(out / "spatial_similarity_conformal_summary.csv", index=False)

    block_summary = (predictions.groupby(
        ["site", "model", "nominal_coverage", "interval_method", "block_id"],
        as_index=False
    ).agg(
        block_pixels_n=("covered", "size"),
        block_pixel_coverage=("covered", "mean"),
        block_all_pixels_covered=("covered", "all"),
        mean_interval_width_m=("interval_width_m", "mean"),
        calibration_blocks_n=("calibration_blocks_n", "first"),
        local_effective_blocks_median=("local_effective_blocks_n", "median"),
        fallback_fraction=("fallback_used", "mean"),
    ))
    block_summary.to_csv(out / "spatial_similarity_conformal_block_summary.csv", index=False)

    settings = {
        "method": "Exploratory localized weighted residual calibration by geographic and spectral similarity",
        "inspiration": [
            "GeoSIMCP, DOI 10.1080/24694452.2026.2639721",
            "Depth-conditional conformal SDB, DOI 10.1109/LGRS.2026.3743086",
        ],
        "input": "Existing prepared pixel samples and HLS ±30-day quality-weighted composite",
        "models": list(MODELS),
        "tide_case": "GOT5.6 anomaly relative to local survey-day model mean",
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "outer_validation": f"{V.N_SPLITS}-fold GroupKFold over {V.BLOCK_M:g}-m spatial blocks",
        "calibration_block_fraction_of_outer_training": CALIBRATION_FRACTION,
        "spatial_buffer_m": float(V.BUFFER_M),
        "local_spatial_bandwidth_m": SPATIAL_BANDWIDTH_M,
        "local_feature_bandwidth_standardized_rms": FEATURE_BANDWIDTH_RMS,
        "minimum_effective_calibration_blocks_before_global_fallback": MIN_EFFECTIVE_CALIBRATION_BLOCKS,
        "score_unit": "maximum absolute error per buffered calibration block",
        "limitations": [
            "Spatial block scores may remain dependent; exact finite-sample coverage is not claimed.",
            "Local weighted intervals are empirical and exploratory under spatial dependence.",
            "The two sites retain separate local hydrographic zeros; this script does not tie their vertical datums.",
            "Anza has few 500-m calibration blocks; inspect calibration-block support and infinite intervals before interpretation.",
            "This estimates prediction error relative to the supplied survey points; it does not estimate missing survey TVU/U95.",
        ],
        "outputs": [pred_path.name,
                    "spatial_similarity_conformal_summary.csv",
                    "spatial_similarity_conformal_block_summary.csv"],
        "skipped_folds": skipped,
    }
    (out / "spatial_similarity_conformal_settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWrote interval predictions: {pred_path}")
    print(f"Wrote summaries and settings: {out}")
    print("Inspect interval width, block coverage, effective calibration blocks, and fallbacks before citing any result.")
    if skipped:
        print(f"Skipped split diagnostics recorded: {len(skipped)}")


if __name__ == "__main__":
    main()
