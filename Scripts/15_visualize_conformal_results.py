"""Create publication-style figures and block-bootstrap tables for v3 intervals."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "Results" / "Additional_Analyses" / "spatial_similarity_conformal_v3_oof.csv.gz"
SAMPLES = ROOT / "Results" / "Prepared" / "pixel_samples.csv.gz"
OUT = ROOT / "Results" / "Figures"
TABLES = ROOT / "Results" / "Additional_Analyses"
BOOTSTRAPS = 5000
SEED = 20261109

METHODS = [
    "Global equal-block",
    "Predicted-depth conditional",
    "Local geography + spectrum",
    "Local geography + spectrum within depth band",
]
METHOD_LABELS = {
    "Global equal-block": "Global",
    "Predicted-depth conditional": "Depth conditional",
    "Local geography + spectrum": "Geo-spectral local",
    "Local geography + spectrum within depth band": "Local + depth",
}
COLORS = {
    "Global equal-block": "#3B528B",
    "Predicted-depth conditional": "#E69F00",
    "Local geography + spectrum": "#009E73",
    "Local geography + spectrum within depth band": "#CC79A7",
}
SELECTED_MODELS = {
    "Anza_Agadir": "HistGradientBoosting",
    "Ras_El_Ma_Ras_Kebdana": "SVR_RBF",
}
SITE_LABELS = {
    "Anza_Agadir": "Anza / Agadir",
    "Ras_El_Ma_Ras_Kebdana": "Ras El Ma / Ras Kebdana",
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        value = float(values[0])
        return value, value, value
    draw_ix = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    draws = values[draw_ix].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def block_metrics(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    by_block = frame.groupby("block_id", sort=False).agg(
        coverage=("covered", "mean"),
        width=("interval_width_m", "mean"),
    )
    cov_mean, cov_low, cov_high = bootstrap_ci(by_block.coverage.to_numpy(), rng)
    width_mean, width_low, width_high = bootstrap_ci(by_block.width.to_numpy(), rng)
    return {
        "test_pixels_n": int(len(frame)),
        "test_blocks_n": int(len(by_block)),
        "pixel_weighted_coverage": float(frame.covered.mean()),
        "block_equal_coverage": cov_mean,
        "block_equal_coverage_ci95_low": cov_low,
        "block_equal_coverage_ci95_high": cov_high,
        "pixel_weighted_mean_width_m": float(frame.interval_width_m.mean()),
        "block_equal_mean_width_m": width_mean,
        "block_equal_width_ci95_low_m": width_low,
        "block_equal_width_ci95_high_m": width_high,
        "fallback_fraction": float(frame.fallback_used.mean()),
        "effective_calibration_blocks_median": float(frame.local_effective_blocks_n.median()),
    }


def make_bootstrap_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    group_cols = ["site", "model", "nominal_coverage", "interval_method"]
    for keys, part in data.groupby(group_cols, sort=True):
        site, model, nominal, method = keys
        rows.append({"site": site, "model": model, "nominal_coverage": nominal,
                     "interval_method": method, **block_metrics(part, rng)})
    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "spatial_similarity_conformal_v3_block_bootstrap.csv", index=False)
    return table


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DCE2E8", linewidth=.7, alpha=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#344054", labelsize=9)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#98A2B3")


def save_figure(fig, stem: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_compass_rose(ax):
    """Draw a compact eight-point compass rose indicating UTM grid north."""
    cx, cy, outer, inner = .91, .84, .055, .019
    for index in range(8):
        angle = np.deg2rad(90 + index * 45)
        side_a = angle + np.deg2rad(22.5)
        side_b = angle - np.deg2rad(22.5)
        tip = (cx + outer * np.cos(angle), cy + outer * np.sin(angle))
        a = (cx + inner * np.cos(side_a), cy + inner * np.sin(side_a))
        b = (cx + inner * np.cos(side_b), cy + inner * np.sin(side_b))
        ax.add_patch(Polygon([(cx, cy), a, tip, b], closed=True,
                             facecolor="#101828" if index % 2 == 0 else "white",
                             edgecolor="#101828", linewidth=.45,
                             transform=ax.transAxes, zorder=8))
    ax.add_patch(Circle((cx, cy), outer * 1.08, fill=False,
                        edgecolor="#101828", linewidth=.55,
                        transform=ax.transAxes, zorder=8))
    ax.text(cx, cy + outer + .018, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7, fontweight="bold",
            color="#101828", zorder=9)


def figure_17(table: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), constrained_layout=True)
    offsets = np.linspace(-.24, .24, len(METHODS))
    panel_specs = [
        (0, "Anza_Agadir", "HistGradientBoosting"),
        (1, "Anza_Agadir", "HistGradientBoosting"),
        (2, "Ras_El_Ma_Ras_Kebdana", "SVR_RBF"),
        (3, "Ras_El_Ma_Ras_Kebdana", "SVR_RBF"),
    ]
    for panel, site, model in panel_specs:
        ax = axes.flat[panel]
        is_coverage = panel % 2 == 0
        nominal_levels = [0.8, 0.9]
        for m_index, method in enumerate(METHODS):
            subset = table[(table.site == site) & (table.model == model) &
                           (table.interval_method == method)]
            xs, ys, lows, highs = [], [], [], []
            for x_index, nominal in enumerate(nominal_levels):
                match = subset[subset.nominal_coverage == nominal]
                if match.empty:
                    continue
                row = match.iloc[0]
                xs.append(x_index + offsets[m_index])
                if is_coverage:
                    ys.append(row.block_equal_coverage)
                    lows.append(row.block_equal_coverage - row.block_equal_coverage_ci95_low)
                    highs.append(row.block_equal_coverage_ci95_high - row.block_equal_coverage)
                else:
                    ys.append(row.block_equal_mean_width_m)
                    lows.append(row.block_equal_mean_width_m - row.block_equal_width_ci95_low_m)
                    highs.append(row.block_equal_width_ci95_high_m - row.block_equal_mean_width_m)
            ax.errorbar(xs, ys, yerr=[lows, highs], fmt="o-", capsize=3,
                        markersize=5, linewidth=1.5, color=COLORS[method],
                        label=METHOD_LABELS[method])
        if is_coverage:
            for x_index, nominal in enumerate(nominal_levels):
                ax.hlines(nominal, x_index - .34, x_index + .34,
                          colors="#D1495B", linestyles="--", linewidth=1.2)
            ax.set_ylabel("Empirical block-equal coverage")
            ax.set_ylim(.45, 1.02)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        else:
            ax.set_ylabel("Mean prediction-interval width (m)")
            ax.set_ylim(bottom=0)
        ax.set_xticks([0, 1], ["80% target", "90% target"])
        style_axes(ax)
        ax.text(.01, .98, f"({chr(97 + panel)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold", color="#101828")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, -.02))
    save_figure(fig, "Figure_17_Conformal_Coverage_and_Width")


def figure_18(data: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.8), constrained_layout=True)
    method = "Local geography + spectrum within depth band"
    nominal = .9
    panels = [
        (0, "Anza_Agadir", "HistGradientBoosting", "width"),
        (1, "Anza_Agadir", "HistGradientBoosting", "miss"),
        (2, "Ras_El_Ma_Ras_Kebdana", "SVR_RBF", "width"),
        (3, "Ras_El_Ma_Ras_Kebdana", "SVR_RBF", "miss"),
    ]
    width_points = []
    miss_points = []
    for panel, site, model, kind in panels:
        ax = axes.flat[panel]
        frame = data[(data.site == site) & (data.model == model) &
                     (data.interval_method == method) &
                     (data.nominal_coverage == nominal)]
        if kind == "width":
            sc = ax.scatter(frame.x_m / 1000, frame.y_m / 1000,
                            c=frame.interval_width_m, s=6, marker="s",
                            cmap="viridis", linewidths=0, rasterized=True)
            width_points.append((ax, sc))
        else:
            covered = frame.covered.to_numpy(dtype=bool)
            ax.scatter(frame.loc[covered, "x_m"] / 1000,
                       frame.loc[covered, "y_m"] / 1000,
                       c="#8AB17D", s=5, marker="s", linewidths=0,
                       rasterized=True, label="Within interval")
            ax.scatter(frame.loc[~covered, "x_m"] / 1000,
                       frame.loc[~covered, "y_m"] / 1000,
                       c="#D1495B", s=8, marker="s", linewidths=0,
                       rasterized=True, label="Outside interval")
            miss_points.append(ax)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#98A2B3")
        ax.spines["bottom"].set_color("#98A2B3")
        ax.tick_params(colors="#344054", labelsize=8)
        add_compass_rose(ax)
        ax.text(.02, .98, f"({chr(97 + panel)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold", color="#101828")
    cbar = fig.colorbar(width_points[0][1], ax=[axes[0, 0], axes[1, 0]],
                        shrink=.84, pad=.02)
    cbar.set_label("Interval width (m)")
    handles, labels = miss_points[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, -.015))
    save_figure(fig, "Figure_18_Spatial_Interval_Width_and_Misses")


def figure_19(data: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    nominal = .9
    plotted_methods = [
        "Global equal-block",
        "Predicted-depth conditional",
        "Local geography + spectrum within depth band",
    ]
    offsets = np.linspace(-.18, .18, len(plotted_methods))
    for panel, (site, model) in enumerate(SELECTED_MODELS.items()):
        ax = axes[panel]
        sub = data[(data.site == site) & (data.model == model) &
                   (data.nominal_coverage == nominal)]
        bands = ["<=5 m", ">5–10 m", ">10 m"]
        rng = np.random.default_rng(SEED + panel)
        for m_index, method in enumerate(plotted_methods):
            method_data = sub[sub.interval_method == method]
            xs, ys, ylo, yhi = [], [], [], []
            for x_index, band in enumerate(bands):
                part = method_data[method_data.predicted_depth_band == band]
                if part.empty:
                    continue
                block_cov = part.groupby("block_id").covered.mean().to_numpy(dtype=float)
                mean, low, high = bootstrap_ci(block_cov, rng)
                xs.append(x_index + offsets[m_index])
                ys.append(mean)
                ylo.append(mean - low)
                yhi.append(high - mean)
            ax.errorbar(xs, ys, yerr=[ylo, yhi], fmt="o-", capsize=3,
                        markersize=5, linewidth=1.5, color=COLORS[method],
                        label=METHOD_LABELS[method])
        ax.axhline(nominal, color="#D1495B", linestyle="--", linewidth=1.2)
        ax.set_xticks(range(len(bands)), [
            f"≤5 m\n(n={len(sub[(sub.interval_method == plotted_methods[0]) & (sub.predicted_depth_band == bands[0])]):,})",
            f"5–10 m\n(n={len(sub[(sub.interval_method == plotted_methods[0]) & (sub.predicted_depth_band == bands[1])]):,})",
            f">10 m\n(n={len(sub[(sub.interval_method == plotted_methods[0]) & (sub.predicted_depth_band == bands[2])]):,})",
        ])
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.set_ylabel("Empirical block-equal coverage")
        style_axes(ax)
        ax.text(.01, .98, f"({chr(97 + panel)})", transform=ax.transAxes,
                ha="left", va="top", fontsize=11, fontweight="bold", color="#101828")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(.5, -.04))
    save_figure(fig, "Figure_19_Depth_Strata_Coverage")


def main():
    if not PREDICTIONS.exists():
        raise SystemExit("Run Scripts/14_spatial_similarity_conformal.py first to create v3 outputs.")
    data = pd.read_csv(PREDICTIONS)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    table = make_bootstrap_table(data)
    figure_17(table)
    figure_18(data)
    figure_19(data)
    captions = (
        "Figure 17. Block-bootstrap empirical coverage and prediction-interval width for the "
        "selected site-level models (Anza: HistGradientBoosting; Ras El Ma: SVR_RBF). "
        "Points show equal-weight means across held-out 500-m spatial blocks; error bars are "
        "95% block-bootstrap intervals. Dashed red segments mark the nominal coverage targets. "
        "The upper row is Anza and the lower row is Ras El Ma.\n\n"
        "Figure 18. Spatial distribution of nominal 90% interval width and held-out misses for "
        "the local geography-and-spectrum method conditioned on predicted depth. Panels (a,b) "
        "show Anza with HistGradientBoosting; panels (c,d) show Ras El Ma with SVR_RBF. Each "
        "square represents one 30-m survey pixel. Compass roses indicate UTM grid north. The two sites use distinct local hydrographic "
        "zeros; map depths and intervals are not on a common vertical datum.\n\n"
        "Figure 19. Block-equal empirical coverage of nominal 90% intervals by predicted-depth "
        "band for the selected site-level models. Error bars are 95% block-bootstrap intervals. "
        "Parenthetical values give the number of held-out pixels in each band; sparse depth "
        "bands should be interpreted cautiously. The dashed line denotes nominal coverage.\n"
    )
    (OUT / "Conformal_Figure_Captions_EN.txt").write_text(captions, encoding="utf-8")
    print("Wrote Figures 17–19 as 350-dpi PNG and vector PDF to Results/Figures.")
    print("Wrote block-bootstrap summary table to Results/Additional_Analyses.")
    print("Figure captions are in Results/Figures/Conformal_Figure_Captions_EN.txt.")


if __name__ == "__main__":
    main()
