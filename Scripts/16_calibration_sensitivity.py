"""Repeated spatial-calibration and bandwidth sensitivity for SDB intervals.

Each repeat uses the same outer held-out 500-m test blocks, but independently
draws calibration blocks from the outer training area. Training points are
buffered from both calibration and test points. Bandwidth settings are applied
to identical fitted/calibration/test samples within each repeat.

Outputs are empirical diagnostics under spatial dependence; no exact
finite-sample conformal coverage guarantee is claimed.
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

N_REPEATS = 10
CALIBRATION_FRACTION = 0.40
CONFIDENCE_LEVELS = (0.80, 0.90)
MIN_EFFECTIVE_BLOCKS = 3.0
# One-at-a-time sensitivity around the v3 primary setting (1000 m, 1.0 RMS).
BANDWIDTHS = ((500.0, 1.0), (1000.0, 0.5), (1000.0, 1.0),
              (1000.0, 2.0), (2000.0, 1.0))
MODELS = ("HistGradientBoosting", "SVR_RBF")


def make_groups(coords: np.ndarray) -> np.ndarray:
    blocks = np.floor(coords / V.BLOCK_M).astype(int)
    return np.asarray([f"{x}_{y}" for x, y in blocks], dtype=object)


def depth_band(values: np.ndarray) -> np.ndarray:
    return pd.cut(np.asarray(values, dtype=float),
                  bins=[-np.inf, 5.0, 10.0, np.inf],
                  labels=["<=5 m", ">5–10 m", ">10 m"], right=True).astype(str)


def balanced_weights(blocks: np.ndarray) -> np.ndarray:
    counts = pd.Series(np.asarray(blocks, dtype=object)).value_counts().to_dict()
    return np.asarray([1.0 / counts[b] for b in blocks], dtype=float)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return float("nan")
    vals, wts = values[keep], weights[keep]
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cdf = np.cumsum(wts) / wts.sum()
    return float(vals[min(np.searchsorted(cdf, level, side="left"), len(vals) - 1)])


def effective_blocks(weights: np.ndarray, blocks: np.ndarray) -> float:
    mass = pd.Series(np.asarray(weights, dtype=float)).groupby(
        np.asarray(blocks, dtype=object), sort=False
    ).sum().to_numpy(dtype=float)
    if not len(mass) or np.sum(mass ** 2) == 0:
        return 0.0
    return float(mass.sum() ** 2 / np.sum(mass ** 2))


def block_bootstrap(values: np.ndarray, seed: int, draws: int = 3000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[picks].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main():
    tides, tide_available = V.load_tide_anomalies(False)
    if not tide_available:
        raise SystemExit("GOT5.6 anomalies are incomplete. Run Scripts/05_estimate_tides.py first.")

    block_rows = []
    skipped = []
    print("Repeated blocked calibration sensitivity: same outer spatial test folds; no test labels used for calibration.")
    print(f"Repeats={N_REPEATS}; calibration share={CALIBRATION_FRACTION:.0%}; bandwidth settings={len(BANDWIDTHS)}.")
    print("Models: HistGradientBoosting and SVR_RBF; HLS Mixed ±30 days; GOT5.6 adjustment.")

    for site_key, site in V.CFG["sites"].items():
        df = V.SAMPLES[V.SAMPLES.site == site_key].reset_index(drop=True)
        quality = V.QUALITY[V.QUALITY.site == site_key].copy()
        window = int(V.CFG.get("primary_temporal_window_days", 30))
        quality = quality[quality.date_offset_days.abs() <= window].sort_values("datetime_utc")
        if quality.empty:
            skipped.append({"site": site_key, "reason": "No scenes in primary temporal window"})
            continue
        X, h = V.read_composite(df, quality.item_id.tolist(), quality, tides,
                                "quality_weighted_mean")
        y = df.depth_m.to_numpy(dtype=float)
        xy = df[["x_m", "y_m"]].to_numpy(dtype=float)
        groups = make_groups(xy)
        valid = np.isfinite(y) & np.isfinite(h) & np.isfinite(X).all(axis=1)
        original = np.flatnonzero(valid)
        X, y, h, xy, groups = X[valid], y[valid], h[valid], xy[valid], groups[valid]
        all_groups = np.unique(groups)
        if len(all_groups) < V.N_SPLITS + 5:
            skipped.append({"site": site_key, "reason": "Insufficient spatial blocks"})
            continue
        splitter = GroupKFold(n_splits=V.N_SPLITS)
        folds = list(splitter.split(np.zeros(len(all_groups)), groups=all_groups))
        print(f"  {site_key}: {len(all_groups)} spatial blocks")

        for model_name in MODELS:
            model_X = X
            model_valid = np.isfinite(model_X).all(axis=1)
            take = np.flatnonzero(model_valid)
            mx, raw_x = model_X[take], X[take]
            my, mh, mxy, mg, original_ix = y[take], h[take], xy[take], groups[take], original[take]

            for fold, (train_group_ix, test_group_ix) in enumerate(folds, start=1):
                fold_train_groups = all_groups[train_group_ix]
                test_groups = all_groups[test_group_ix]
                dev_groups = np.asarray([g for g in fold_train_groups if g in set(np.unique(mg))], dtype=object)
                if len(dev_groups) < 7:
                    skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                    "reason": "Fewer than seven non-test blocks"})
                    continue
                test_ix = np.flatnonzero(np.isin(mg, test_groups))
                if len(test_ix) < 10:
                    continue
                for repeat in range(1, N_REPEATS + 1):
                    # Use the same calibration-block draw for both estimators at a site.
                    seed = int(V.SEED + fold * 1009 + repeat * 10007 +
                               sum(ord(ch) for ch in site_key))
                    rng = np.random.default_rng(seed)
                    n_cal = max(5, int(math.ceil(CALIBRATION_FRACTION * len(dev_groups))))
                    n_cal = min(n_cal, len(dev_groups) - 3)
                    cal_groups = np.sort(rng.choice(dev_groups, size=n_cal, replace=False))
                    fit_groups = np.asarray([g for g in dev_groups if g not in set(cal_groups)], dtype=object)
                    cal0 = np.flatnonzero(np.isin(mg, cal_groups))
                    fit0 = np.flatnonzero(np.isin(mg, fit_groups))
                    if not len(cal0) or not len(fit0):
                        continue

                    test_tree = cKDTree(mxy[test_ix])
                    cal_dist, _ = test_tree.query(mxy[cal0], k=1)
                    cal_ix = cal0[cal_dist > V.BUFFER_M]
                    if not len(cal_ix) or len(np.unique(mg[cal_ix])) < 3:
                        skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                        "repeat": repeat, "reason": "Insufficient buffered calibration blocks"})
                        continue
                    excluded = np.vstack([mxy[test_ix], mxy[cal_ix]])
                    fit_dist, _ = cKDTree(excluded).query(mxy[fit0], k=1)
                    fit_ix = fit0[fit_dist > V.BUFFER_M]
                    if len(fit_ix) < 30:
                        skipped.append({"site": site_key, "model": model_name, "fold": fold,
                                        "repeat": repeat, "reason": "Too few fit pixels after buffers"})
                        continue

                    model = V.new_model(model_name)
                    model.fit(mx[fit_ix], my[fit_ix] + mh[fit_ix])
                    cal_pred = model.predict(mx[cal_ix]) - mh[cal_ix]
                    test_pred = model.predict(mx[test_ix]) - mh[test_ix]
                    cal_error = np.abs(cal_pred - my[cal_ix])
                    cal_band = depth_band(cal_pred)
                    test_band = depth_band(test_pred)
                    cal_blocks = mg[cal_ix]
                    n_cal_blocks = len(np.unique(cal_blocks))
                    global_w = balanced_weights(cal_blocks)
                    scaler = StandardScaler().fit(raw_x[fit_ix])
                    cal_z = scaler.transform(raw_x[cal_ix])
                    test_z = scaler.transform(raw_x[test_ix])

                    # Radii for global and depth-conditional baselines.
                    q_global = {}
                    q_depth = {}
                    depth_support = {}
                    for level in CONFIDENCE_LEVELS:
                        q_global[level] = weighted_quantile(cal_error, global_w, level)
                        for band in ("<=5 m", ">5–10 m", ">10 m"):
                            mask = cal_band == band
                            depth_support[(level, band)] = len(np.unique(cal_blocks[mask]))
                            if depth_support[(level, band)] >= 3:
                                w_depth = balanced_weights(cal_blocks[mask])
                                q_depth[(level, band)] = weighted_quantile(cal_error[mask], w_depth, level)
                            else:
                                q_depth[(level, band)] = q_global[level]

                    # Store per-pixel interval outcomes by setting; reduce to test-block
                    # metrics immediately so the output remains compact and spatially clustered.
                    for level in CONFIDENCE_LEVELS:
                        metrics_by_setting = {}
                        metrics_by_setting[("Global equal-block", None, None)] = (
                            np.full(len(test_ix), q_global[level]),
                            np.full(len(test_ix), False),
                            np.full(len(test_ix), float(n_cal_blocks)),
                        )
                        depth_radii = np.asarray([q_depth[(level, b)] for b in test_band])
                        metrics_by_setting[("Predicted-depth conditional", None, None)] = (
                            depth_radii,
                            np.asarray([depth_support[(level, b)] < 3 for b in test_band]),
                            np.full(len(test_ix), float(n_cal_blocks)),
                        )

                        for spatial_bw, feature_bw in BANDWIDTHS:
                            local_r = np.empty(len(test_ix), dtype=float)
                            local_fb = np.zeros(len(test_ix), dtype=bool)
                            local_neff = np.empty(len(test_ix), dtype=float)
                            depth_local_r = np.empty(len(test_ix), dtype=float)
                            depth_local_fb = np.zeros(len(test_ix), dtype=bool)
                            depth_local_neff = np.zeros(len(test_ix), dtype=float)
                            for j, sample in enumerate(test_ix):
                                d_space = np.sqrt(np.sum((mxy[cal_ix] - mxy[sample]) ** 2, axis=1))
                                d_feat = np.sqrt(np.mean((cal_z - test_z[j]) ** 2, axis=1))
                                log_w = -0.5 * ((d_space / spatial_bw) ** 2 + (d_feat / feature_bw) ** 2)
                                log_w -= np.max(log_w)
                                weights = np.exp(log_w) * global_w
                                neff = effective_blocks(weights, cal_blocks)
                                local_neff[j] = neff
                                local_fb[j] = neff < MIN_EFFECTIVE_BLOCKS
                                local_r[j] = (q_global[level] if local_fb[j]
                                              else weighted_quantile(cal_error, weights, level))

                                band_mask = cal_band == test_band[j]
                                block_n = len(np.unique(cal_blocks[band_mask]))
                                qd = q_depth[(level, test_band[j])]
                                if block_n < 3:
                                    depth_local_fb[j] = True
                                    depth_local_r[j] = qd
                                    continue
                                w_band = balanced_weights(cal_blocks[band_mask])
                                lw_band = np.exp(log_w[band_mask] - np.max(log_w[band_mask])) * w_band
                                neff_band = effective_blocks(lw_band, cal_blocks[band_mask])
                                depth_local_neff[j] = neff_band
                                depth_local_fb[j] = neff_band < MIN_EFFECTIVE_BLOCKS
                                depth_local_r[j] = (qd if depth_local_fb[j]
                                                    else weighted_quantile(cal_error[band_mask], lw_band, level))
                            metrics_by_setting[("Local geography + spectrum", spatial_bw, feature_bw)] = (
                                local_r, local_fb, local_neff)
                            metrics_by_setting[("Local geography + spectrum within depth band", spatial_bw, feature_bw)] = (
                                depth_local_r, depth_local_fb, depth_local_neff)

                        test_abs = np.abs(test_pred - my[test_ix])
                        test_blocks = mg[test_ix]
                        for (method, spatial_bw, feature_bw), (radius, fallback, neff) in metrics_by_setting.items():
                            interval_width = 2 * radius
                            covered = test_abs <= radius
                            for block in np.unique(test_blocks):
                                bmask = test_blocks == block
                                block_rows.append({
                                    "site": site_key,
                                    "model": model_name,
                                    "fold": fold,
                                    "repeat": repeat,
                                    "test_block_id": str(block),
                                    "nominal_coverage": level,
                                    "interval_method": method,
                                    "spatial_bandwidth_m": spatial_bw,
                                    "feature_bandwidth_standardized_rms": feature_bw,
                                    "test_pixels_n": int(bmask.sum()),
                                    "empirical_coverage": float(np.mean(covered[bmask])),
                                    "mean_interval_width_m": float(np.mean(interval_width[bmask])),
                                    "calibration_blocks_n": int(n_cal_blocks),
                                    "effective_local_blocks_mean": float(np.mean(neff[bmask])),
                                    "fallback_fraction": float(np.mean(fallback[bmask])),
                                })
                    print(f"    {model_name} fold {fold}: {repeat}/{N_REPEATS} calibration draws",
                          end="\r", flush=True)
                print(f"    {model_name} fold {fold}: {N_REPEATS}/{N_REPEATS} draws complete.")

    if not block_rows:
        raise SystemExit("No sensitivity results were created; inspect input coverage and spatial block settings.")

    out = ROOT / "Results" / "Additional_Analyses"
    out.mkdir(parents=True, exist_ok=True)
    blocks = pd.DataFrame(block_rows)
    block_path = out / "spatial_similarity_calibration_sensitivity_blocks.csv.gz"
    blocks.to_csv(block_path, index=False, compression="gzip")

    group_cols = ["site", "model", "nominal_coverage", "interval_method",
                  "spatial_bandwidth_m", "feature_bandwidth_standardized_rms"]
    summary_rows = []
    for keys, group in blocks.groupby(group_cols, dropna=False, sort=True):
        site_key, model_name, level, method, spatial_bw, feature_bw = keys
        # First average repeated calibration draws within each held-out spatial block,
        # then bootstrap whole test blocks to avoid treating pixels as independent.
        by_block = group.groupby("test_block_id", as_index=False).agg(
            coverage=("empirical_coverage", "mean"),
            width=("mean_interval_width_m", "mean"),
            split_coverage_sd=("empirical_coverage", "std"),
            split_width_sd=("mean_interval_width_m", "std"),
        )
        seed = int(V.SEED + sum(ord(c) for c in str(keys)))
        cov_mean, cov_low, cov_high = block_bootstrap(by_block.coverage.to_numpy(), seed)
        width_mean, width_low, width_high = block_bootstrap(by_block.width.to_numpy(), seed + 1)
        summary_rows.append({
            "site": site_key,
            "model": model_name,
            "nominal_coverage": float(level),
            "interval_method": method,
            "spatial_bandwidth_m": spatial_bw,
            "feature_bandwidth_standardized_rms": feature_bw,
            "heldout_spatial_blocks_n": int(len(by_block)),
            "repeats_n": int(group.repeat.nunique()),
            "block_equal_coverage": cov_mean,
            "block_equal_coverage_ci95_low": cov_low,
            "block_equal_coverage_ci95_high": cov_high,
            "block_equal_mean_width_m": width_mean,
            "block_equal_width_ci95_low_m": width_low,
            "block_equal_width_ci95_high_m": width_high,
            "calibration_split_coverage_sd_mean": float(by_block.split_coverage_sd.mean()),
            "calibration_split_width_sd_mean_m": float(by_block.split_width_sd.mean()),
            "fallback_fraction": float(group.fallback_fraction.mean()),
            "effective_local_blocks_mean": float(group.effective_local_blocks_mean.mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = out / "spatial_similarity_calibration_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    settings = {
        "outer_validation": f"{V.N_SPLITS}-fold GroupKFold over {V.BLOCK_M:g}-m blocks",
        "training_exclusion_buffer_m": float(V.BUFFER_M),
        "calibration_fraction_of_outer_training_blocks": CALIBRATION_FRACTION,
        "calibration_repeats_per_fold": N_REPEATS,
        "models": list(MODELS),
        "temporal_window_days": int(V.CFG.get("primary_temporal_window_days", 30)),
        "tide_case": "GOT5.6 anomaly relative to local survey-day model mean",
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "bandwidths_spatial_m_and_feature_standardized_rms": [list(v) for v in BANDWIDTHS],
        "reference_bandwidth": [1000.0, 1.0],
        "interpretation": "Empirical sensitivity assessment; spatial dependence remains and exact finite-sample coverage is not claimed.",
        "outputs": [block_path.name, summary_path.name],
        "skipped_splits": skipped,
    }
    settings_path = out / "spatial_similarity_calibration_sensitivity_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")

    print(f"\nWrote block-level repeated results: {block_path}")
    print(f"Wrote block-bootstrap sensitivity summary: {summary_path}")
    print(f"Wrote settings and skipped-split diagnostics: {settings_path}")
    print("Compare coverage, interval width, calibration-split variability, and fallback rates across settings.")


if __name__ == "__main__":
    main()
