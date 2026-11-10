"""Visualize completed spatial calibration-sensitivity results.

This script reads the saved summary from Script 16. It does not refit models or
rerun the calibration experiment. The figure displays the local + depth method
across the five predeclared bandwidth settings, with block-bootstrap intervals.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "Results" / "Additional_Analyses" / "spatial_similarity_calibration_sensitivity_summary.csv"
OUT = ROOT / "Results" / "Figures"
TABLES = ROOT / "Results" / "Additional_Analyses"

LOCAL_METHOD = "Local geography + spectrum within depth band"
GLOBAL_METHOD = "Global equal-block"
DEPTH_METHOD = "Predicted-depth conditional"
SITE_LABELS = {
    "Anza_Agadir": "Anza / Agadir",
    "Ras_El_Ma_Ras_Kebdana": "Ras El Ma / Ras Kebdana",
}
MODELS = ["HistGradientBoosting", "SVR_RBF"]
MODEL_LABELS = {
    "HistGradientBoosting": "HistGradientBoosting",
    "SVR_RBF": "RBF-SVR",
}
LEVELS = [0.80, 0.90]
LEVEL_COLORS = {0.80: "#007C91", 0.90: "#D98E04"}
MODEL_STYLES = {"HistGradientBoosting": "-", "SVR_RBF": "--"}
SETTINGS = [(500.0, 1.0), (1000.0, 0.5), (1000.0, 1.0),
            (1000.0, 2.0), (2000.0, 1.0)]
SETTING_LABELS = ["500 m\nfeature 1.0", "1,000 m\nfeature 0.5",
                  "1,000 m\nfeature 1.0", "1,000 m\nfeature 2.0",
                  "2,000 m\nfeature 1.0"]


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#98A2B3")
    ax.spines["bottom"].set_color("#98A2B3")
    ax.grid(axis="y", color="#DCE2E8", linewidth=.7, alpha=.85)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#344054", labelsize=8)


def validate_summary(data: pd.DataFrame) -> None:
    needed = {
        "site", "model", "nominal_coverage", "interval_method",
        "spatial_bandwidth_m", "feature_bandwidth_standardized_rms",
        "block_equal_coverage", "block_equal_coverage_ci95_low",
        "block_equal_coverage_ci95_high", "block_equal_mean_width_m",
        "block_equal_width_ci95_low_m", "block_equal_width_ci95_high_m",
        "calibration_split_coverage_sd_mean", "fallback_fraction",
    }
    missing = sorted(needed - set(data.columns))
    if missing:
        raise SystemExit(f"Sensitivity summary is missing fields: {', '.join(missing)}")
    if data.empty:
        raise SystemExit("Sensitivity summary is empty.")


def ordered_local_rows(data: pd.DataFrame, site: str, model: str,
                       nominal: float) -> pd.DataFrame:
    rows = data[(data.site == site) & (data.model == model) &
                (data.nominal_coverage == nominal) &
                (data.interval_method == LOCAL_METHOD)].copy()
    rows["setting_order"] = rows.apply(
        lambda row: SETTINGS.index((float(row.spatial_bandwidth_m),
                                    float(row.feature_bandwidth_standardized_rms))),
        axis=1,
    )
    return rows.sort_values("setting_order")


def make_reference_table(data: pd.DataFrame) -> pd.DataFrame:
    """Export reference-setting global/depth/local comparisons and deltas."""
    reference = data[(data.interval_method.isin(
        [GLOBAL_METHOD, DEPTH_METHOD, LOCAL_METHOD])) &
        ((data.interval_method != LOCAL_METHOD) |
         ((data.spatial_bandwidth_m == 1000.0) &
          (data.feature_bandwidth_standardized_rms == 1.0)))].copy()

    reference = reference.rename(columns={
        "block_equal_coverage": "block_equal_coverage_mean",
        "block_equal_mean_width_m": "block_equal_interval_width_mean_m",
    })
    reference["bandwidth_setting"] = np.where(
        reference.interval_method == LOCAL_METHOD, "1000 m / feature 1.0", "not applicable"
    )
    reference["coverage_delta_vs_global"] = np.nan
    reference["width_delta_vs_global_m"] = np.nan
    for _, group in reference.groupby(["site", "model", "nominal_coverage"]):
        baseline = group[group.interval_method == GLOBAL_METHOD]
        if baseline.empty:
            continue
        baseline_coverage = float(baseline.iloc[0].block_equal_coverage_mean)
        baseline_width = float(baseline.iloc[0].block_equal_interval_width_mean_m)
        mask = ((reference.site == group.iloc[0].site) &
                (reference.model == group.iloc[0].model) &
                (reference.nominal_coverage == group.iloc[0].nominal_coverage))
        reference.loc[mask, "coverage_delta_vs_global"] = (
            reference.loc[mask, "block_equal_coverage_mean"] - baseline_coverage
        )
        reference.loc[mask, "width_delta_vs_global_m"] = (
            reference.loc[mask, "block_equal_interval_width_mean_m"] - baseline_width
        )

    columns = [
        "site", "model", "nominal_coverage", "interval_method", "bandwidth_setting",
        "heldout_spatial_blocks_n", "repeats_n", "block_equal_coverage_mean",
        "block_equal_coverage_ci95_low", "block_equal_coverage_ci95_high",
        "block_equal_interval_width_mean_m", "block_equal_width_ci95_low_m",
        "block_equal_width_ci95_high_m", "coverage_delta_vs_global",
        "width_delta_vs_global_m", "calibration_split_coverage_sd_mean",
        "calibration_split_width_sd_mean_m", "fallback_fraction",
        "effective_local_blocks_mean",
    ]
    return reference[columns].sort_values(
        ["site", "model", "nominal_coverage", "interval_method"]
    )


def make_figure(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), sharex=True)
    panel_specs = [
        (axes[0, 0], "Anza_Agadir", "coverage"),
        (axes[0, 1], "Ras_El_Ma_Ras_Kebdana", "coverage"),
        (axes[1, 0], "Anza_Agadir", "width"),
        (axes[1, 1], "Ras_El_Ma_Ras_Kebdana", "width"),
    ]
    marker_by_level = {0.80: "o", 0.90: "s"}

    for panel_index, (ax, site, metric) in enumerate(panel_specs):
        for model in MODELS:
            x_offset = -0.055 if model == "HistGradientBoosting" else 0.055
            for nominal in LEVELS:
                rows = ordered_local_rows(data, site, model, nominal)
                if len(rows) != len(SETTINGS):
                    raise SystemExit(
                        f"Expected {len(SETTINGS)} local-depth settings for "
                        f"{site}/{model}/{nominal:.0%}; found {len(rows)}."
                    )
                x = rows.setting_order.to_numpy(dtype=float) + x_offset
                if metric == "coverage":
                    y = rows.block_equal_coverage.to_numpy(dtype=float)
                    low = y - rows.block_equal_coverage_ci95_low.to_numpy(dtype=float)
                    high = rows.block_equal_coverage_ci95_high.to_numpy(dtype=float) - y
                else:
                    y = rows.block_equal_mean_width_m.to_numpy(dtype=float)
                    low = y - rows.block_equal_width_ci95_low_m.to_numpy(dtype=float)
                    high = rows.block_equal_width_ci95_high_m.to_numpy(dtype=float) - y
                label = f"{MODEL_LABELS[model]}, {nominal:.0%} target"
                ax.errorbar(
                    x, y, yerr=np.vstack([low, high]), fmt=marker_by_level[nominal],
                    color=LEVEL_COLORS[nominal], linestyle=MODEL_STYLES[model],
                    marker=marker_by_level[nominal], markersize=4.4,
                    markerfacecolor="white", markeredgewidth=1.2,
                    linewidth=1.35, capsize=2.5, label=label,
                )
        if metric == "coverage":
            ax.axhline(.80, color=LEVEL_COLORS[.80], linestyle=":", linewidth=1.0)
            ax.axhline(.90, color=LEVEL_COLORS[.90], linestyle=":", linewidth=1.0)
            ax.set_ylim(.60, 1.0)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
            ax.set_ylabel("Empirical block-equal coverage")
        else:
            ax.set_ylabel("Mean prediction-interval width (m)")
            ax.set_ylim(bottom=0)
        ax.set_title(SITE_LABELS[site], loc="left", fontsize=10,
                     fontweight="bold", color="#101828")
        ax.text(.01, .98, f"({chr(97 + panel_index)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=10, fontweight="bold", color="#101828")
        ax.set_xticks(range(len(SETTINGS)), SETTING_LABELS)
        style_axes(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8.2, bbox_to_anchor=(.5, .015))
    fig.text(.5, .075,
             "Local geography + spectrum within predicted-depth bands; whiskers: 95% block-bootstrap intervals. "
             "Feature bandwidth is standardized RMS.",
             ha="center", va="center", fontsize=8, color="#475467")
    fig.tight_layout(rect=(0, .115, 1, .98), h_pad=2.2, w_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "Figure_20_Calibration_Bandwidth_Sensitivity.png",
                dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Figure_20_Calibration_Bandwidth_Sensitivity.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(
            "Run Scripts/16_calibration_sensitivity.py first; its completed summary is required."
        )
    data = pd.read_csv(SUMMARY)
    validate_summary(data)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    TABLES.mkdir(parents=True, exist_ok=True)
    comparison = make_reference_table(data)
    comparison.to_csv(TABLES / "calibration_reference_method_comparison.csv", index=False)
    make_figure(data)
    caption = (
        "Figure 20. Sensitivity of empirical coverage and prediction-interval width to the "
        "spatial and spectral bandwidth settings for the local geography-and-spectrum method "
        "conditioned on predicted-depth bands. The upper panels show block-equal empirical "
        "coverage; the lower panels show mean interval width. Columns correspond to Anza / "
        "Agadir and Ras El Ma / Ras Kebdana. Colors identify nominal 80% and 90% targets; "
        "line styles identify HistGradientBoosting (solid) and RBF-SVR (dashed). The five "
        "settings vary one bandwidth at a time around the reference setting (1,000 m spatial, "
        "standardized RMS feature bandwidth 1.0). Whiskers are 95% block-bootstrap intervals "
        "over held-out spatial blocks. Results are empirical under spatial dependence and do "
        "not provide an exact finite-sample coverage guarantee."
    )
    (OUT / "Calibration_Sensitivity_Figure_Caption_EN.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    print("Created Figure 20 as 350-dpi PNG and vector PDF in Results/Figures.")
    print("Wrote the reference-method comparison table to Results/Additional_Analyses.")
    print("No models were fitted; only the saved sensitivity summary was read.")


if __name__ == "__main__":
    main()
