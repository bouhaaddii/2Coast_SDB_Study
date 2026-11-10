"""Create publication figures for the actual two-coast SDB workflow.

The map is reconstructed only at sampled 30 m cells from the saved
quality-weighted HLS composite; no spatial interpolation or new model fit is
performed.  The workflow graphic documents the implemented study pipeline.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Figures"
SAMPLES = pd.read_csv(ROOT / "Results" / "Prepared" / "pixel_samples.csv.gz")
QUALITY = pd.read_csv(ROOT / "Results" / "Prepared" / "scene_quality.csv")

INK = "#19324A"
NAVY = "#123B5D"
TEAL = "#148C8C"
SEA = "#DDF2F1"
PALE = "#F1F6F8"
GOLD = "#E39B38"
LILAC = "#EDE8F8"
BLUE = "#5E86C1"
MUTED = "#536675"
GRID = "#D8E1E7"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.2,
    "axes.labelsize": 7.0, "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
    "axes.edgecolor": "#758795", "axes.linewidth": 0.55,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def arrow(ax, start, end, color=BLUE, lw=0.95, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7.5,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def trajectory(ax, points, color=BLUE, lw=1.05):
    """Draw a routed cubic arrow that stays clear of the destination labels."""
    path = MplPath(points, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=8,
                                 linewidth=lw, color=color, zorder=2))


def quality_weighted_composite(site: str) -> pd.DataFrame:
    frame = SAMPLES.loc[SAMPLES.site.eq(site)].copy().reset_index(drop=True)
    scene_quality = QUALITY.loc[QUALITY.site.eq(site)]
    bands = ["blue", "green", "red", "nir"]
    sums = np.zeros((len(frame), len(bands)), dtype=np.float64)
    denom = np.zeros(len(frame), dtype=np.float64)
    for item in scene_quality.itertuples(index=False):
        cols = [f"{item.item_id}__{band}" for band in bands]
        values = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(values).all(axis=1)
        weight = valid.astype(float) * max(float(item.quality_score), 1e-6)
        sums += np.nan_to_num(values, nan=0.0) * weight[:, None]
        denom += weight
    composite = np.divide(sums, denom[:, None], out=np.full_like(sums, np.nan),
                          where=denom[:, None] > 0)
    for j, band in enumerate(bands):
        frame[f"rho_{band}"] = composite[:, j]
    return frame.loc[denom > 0].copy()


def rgb_at_grid(frame: pd.DataFrame):
    # B04/B03/B02 are displayed as RGB. Each site gets an independent 2-98%
    # contrast stretch for visual legibility; the model consumes all four bands.
    channels = []
    for band in ("red", "green", "blue"):
        values = frame[f"rho_{band}"].to_numpy(float)
        lo, hi = np.nanpercentile(values, [2, 98])
        stretched = np.clip((values - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
        # Lift dark water tones slightly while preserving the RGB ordering.
        channels.append(np.power(stretched, 0.66))
    rgb = np.column_stack(channels)
    xvals = np.sort(frame.x_m.unique())
    yvals = np.sort(frame.y_m.unique())
    ix = np.searchsorted(xvals, frame.x_m.to_numpy())
    iy = np.searchsorted(yvals, frame.y_m.to_numpy())
    image = np.zeros((len(yvals), len(xvals), 4), dtype=float)
    image[iy, ix, :3] = rgb
    # Keep unsampled grid locations transparent instead of drawing them black.
    image[iy, ix, 3] = 1.0
    return image, xvals, yvals


def add_scalebar(ax, length_m=1000.0):
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x0 = xlim[0] + 0.07 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.08 * (ylim[1] - ylim[0])
    length = length_m / 1000.0
    ax.plot([x0, x0 + length], [y0, y0], color="white", linewidth=2.8, zorder=6,
            solid_capstyle="butt")
    ax.plot([x0, x0 + length], [y0, y0], color=INK, linewidth=0.85, zorder=7,
            solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.045 * (ylim[1] - ylim[0]), "1 km",
            ha="center", va="bottom", color=INK, fontsize=5.7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
            zorder=8)


def draw_site_input_map(ax, frame: pd.DataFrame, letter: str, site: str,
                        depth_norm: Normalize, point_size: float = 1.7):
    image, xv, yv = rgb_at_grid(frame)
    half_x = float(np.median(np.diff(xv))) / 2 if len(xv) > 1 else 15.0
    half_y = float(np.median(np.diff(yv))) / 2 if len(yv) > 1 else 15.0
    x_origin = np.floor(float(xv.min()) / 1000.0) * 1000.0
    y_origin = np.floor(float(yv.min()) / 1000.0) * 1000.0
    extent = ((xv.min() - half_x - x_origin) / 1000,
              (xv.max() + half_x - x_origin) / 1000,
              (yv.min() - half_y - y_origin) / 1000,
              (yv.max() + half_y - y_origin) / 1000)
    ax.set_facecolor("#EAF1F4")
    ax.imshow(image, origin="lower", extent=extent, interpolation="nearest",
              aspect="equal", zorder=1)
    x = (frame.x_m.to_numpy(float) - x_origin) / 1000
    y = (frame.y_m.to_numpy(float) - y_origin) / 1000
    ax.scatter(x, y, c=frame.depth_m.to_numpy(float), cmap="cividis",
               norm=depth_norm, s=point_size, linewidths=0.10,
               edgecolors="white", alpha=0.95, zorder=4, rasterized=True)
    ax.set_title(f"({letter})  {site}", loc="left", pad=5,
                 fontsize=8.2, weight="bold", color=INK)
    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
    ax.set_xlabel(f"Easting offset from {x_origin/1000:.0f} km (km)")
    ax.set_ylabel(f"Northing offset from {y_origin/1000:.0f} km (km)")
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(labelsize=6.1, length=2, pad=1.5)
    ax.grid(color="white", linewidth=0.36, alpha=0.60, zorder=3)
    ax.set_axisbelow(False)
    add_scalebar(ax)


def draw_input_overlay() -> dict[str, pd.DataFrame]:
    sites = ["Anza_Agadir", "Ras_El_Ma_Ras_Kebdana"]
    frames = {site: quality_weighted_composite(site) for site in sites}
    vmax = 25.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    fig = plt.figure(figsize=(7.25, 3.45))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045],
                            left=0.09, right=0.94, bottom=0.25, top=0.86, wspace=0.20)
    draw_site_input_map(fig.add_subplot(grid[0, 0]), frames[sites[0]], "A",
                        "Anza / Agadir", norm, point_size=1.25)
    draw_site_input_map(fig.add_subplot(grid[0, 1]), frames[sites[1]], "B",
                        "Ras El Ma / Ras Kebdana", norm, point_size=0.95)
    cax = fig.add_subplot(grid[0, 2])
    sm = ScalarMappable(norm=norm, cmap="cividis"); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Survey depth below local port zero (m; positive down)", fontsize=6.2, labelpad=6)
    cb.ax.tick_params(labelsize=6.0, length=2)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#6B8175", linewidth=6, solid_capstyle="butt",
               label="HLS RGB reflectance composite (B04/B03/B02)"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#667894",
               markeredgecolor="white", markeredgewidth=0.45, markersize=4.5,
               label="Measured 30 m cell-depth median"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.49, 0.105),
               ncol=2, frameon=False, fontsize=6.1, handlelength=1.8,
               columnspacing=1.7, handletextpad=0.65)
    fig.text(0.49, 0.038,
             "Reflectance forms the map background; depth-coloured points are overlaid at the matching sample-cell locations.",
             ha="center", va="center", fontsize=5.9, color=MUTED)
    save(fig, "Figure_Sampled_HLS_RGB_Bathymetry_Overlay")
    return frames


def draw_map_based_workflow(frames: dict[str, pd.DataFrame]) -> None:

    fig, ax = plt.subplots(figsize=(7.25, 4.65))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # The blue database header and grouped input lanes follow the supplied
    # example: sources at the top, vertical preparation steps, then outputs.
    ax.add_patch(FancyBboxPatch((41, 94), 18, 4.2,
        boxstyle="round,pad=0.25,rounding_size=0.8", facecolor="#8CCFE5",
        edgecolor=NAVY, linewidth=0.75))
    ax.text(50, 96.1, "DATABASE", ha="center", va="center",
            fontsize=8.2, weight="bold", color=NAVY)
    ax.add_patch(FancyBboxPatch((1.2, 37.5), 97.6, 54.5,
        boxstyle="round,pad=0.3,rounding_size=0.6", facecolor="#FCFDFC",
        edgecolor="#77A66D", linewidth=0.85, linestyle=(0, (3, 2))))
    ax.add_patch(FancyBboxPatch((2.5, 89.0), 31.5, 2.4,
        boxstyle="round,pad=0.18,rounding_size=0.35", facecolor="#EDF5E9",
        edgecolor="#77A66D", linewidth=0.55))
    ax.text(18.25, 90.2, "INPUT DATA AND PRE-PROCESSING", ha="center", va="center",
            fontsize=6.4, weight="bold", color="#54834A")

    xs = [3.0, 27.0, 51.0, 75.0]
    w = 22.0
    lane_titles = ["Satellite reflectance", "Bathymetric survey", "Tide-model covariate",
                   "Open bathymetry"]
    source_text = ["HLS v2.0\nS30 + L30", "XYZ soundings\nlocal port zeros",
                   "GOT5.6 + pyTMD\nharmonic tide", "EMODnet + GEBCO"]
    lane_edges = [TEAL, GOLD, "#8972B4", "#8997A0"]
    for x, title, source, edge in zip(xs, lane_titles, source_text, lane_edges):
        ax.add_patch(FancyBboxPatch((x, 82.8), w, 6.0,
            boxstyle="round,pad=0.25,rounding_size=0.45", facecolor="#EDF5E9",
            edgecolor="#75A56B", linewidth=0.65))
        ax.text(x + w / 2, 85.8, title, ha="center", va="center",
                fontsize=7.0, weight="bold", color="#416D39")
        ax.add_patch(FancyBboxPatch((x, 75.5), w, 5.3,
            boxstyle="round,pad=0.22,rounding_size=0.4", facecolor="white",
            edgecolor=edge, linewidth=0.65))
        ax.text(x + w / 2, 78.15, source, ha="center", va="center",
                fontsize=5.6, weight="bold", color=INK, linespacing=1.05)
        arrow(ax, (x + w / 2, 75.2), (x + w / 2, 73.4), color=edge, lw=0.8)

    def stage_box(x, y, text, edge, face="white", fs=6.0):
        ax.add_patch(FancyBboxPatch((x + 0.7, y), w - 1.4, 6.0,
            boxstyle="round,pad=0.2,rounding_size=0.35", facecolor=face,
            edgecolor=edge, linewidth=0.6))
        ax.text(x + w / 2, y + 3.0, text, ha="center", va="center",
                fontsize=fs, color=INK, linespacing=1.08)

    lane_stages = [
        ["Cloud / invalid-pixel screening", "Harmonize B02/B03/B04 + NIR", "Quality-weighted composite · ±30 d"],
        ["Transform XYZ to local UTM", "Review exact XY duplicates", "Median depth per 30 m cell"],
        ["Predict tide at 10 min steps", "Sample each scene overpass", "Subtract survey-day mean"],
    ]
    for lane, x in enumerate(xs[:3]):
        ys = [67.2, 58.0, 48.8]
        for j, (y, text) in enumerate(zip(ys, lane_stages[lane])):
            stage_box(x, y, text, lane_edges[lane], fs=5.9)
            if j < 2:
                arrow(ax, (x + w / 2, y - 0.35), (x + w / 2, y - 2.3),
                      color=lane_edges[lane], lw=0.75)
        out_text = ["FOUR-BAND COMPOSITE", "30 m DEPTH CELLS", "RELATIVE TIDE ANOMALY"][lane]
        ax.add_patch(FancyBboxPatch((x, 40.0), w, 6.0,
            boxstyle="round,pad=0.23,rounding_size=0.35", facecolor="#DCEFD7",
            edgecolor="#6D9B61", linewidth=0.7))
        ax.text(x + w / 2, 43.0, out_text, ha="center", va="center",
                fontsize=6.0, weight="bold", color="#315D2C")
        arrow(ax, (x + w / 2, 48.4), (x + w / 2, 46.4), color=lane_edges[lane], lw=0.75)

    # Open grids are shown as context only and have no connection to training,
    # validation or vertical-datum adjustment.
    ax.add_patch(FancyBboxPatch((xs[3] + 0.7, 49.0), w - 1.4, 24.0,
        boxstyle="round,pad=0.25,rounding_size=0.4", facecolor="#F4F5F5",
        edgecolor="#8997A0", linewidth=0.65, linestyle=(0, (2, 1))))
    ax.text(xs[3] + w / 2, 67.5, "Reproject / inspect\nsource coverage and\nproduct metadata",
            ha="center", va="center", fontsize=6.1, color=INK, linespacing=1.25)
    ax.text(xs[3] + w / 2, 55.5, "CONTEXT MAPS ONLY\nNot training truth\nNo datum substitution",
            ha="center", va="center", fontsize=6.0, weight="bold", color="#687780",
            linespacing=1.2)
    ax.add_patch(FancyBboxPatch((xs[3], 40.0), w, 6.0,
        boxstyle="round,pad=0.23,rounding_size=0.35", facecolor="#EEF0F1",
        edgecolor="#8997A0", linewidth=0.65))
    ax.text(xs[3] + w / 2, 43.0, "SOURCE CONTEXT", ha="center", va="center",
            fontsize=6.0, weight="bold", color="#687780")

    # Three actual study inputs converge on a common pixel table; open grids
    # deliberately do not feed this overlay.
    ax.add_patch(FancyBboxPatch((8.0, 28.8), 66.0, 7.2,
        boxstyle="round,pad=0.28,rounding_size=0.45", facecolor="#E6F3F2",
        edgecolor=TEAL, linewidth=0.8))
    ax.text(41.0, 33.7, "COMMON 30 m PIXEL OVERLAY", ha="center", va="center",
            fontsize=7.2, weight="bold", color=NAVY)
    ax.text(41.0, 30.8,
            "HLS reflectance + surveyed depth + weighted tide anomaly; retain each local port zero",
            ha="center", va="center", fontsize=6.1, color=INK)
    for x, color in zip(xs[:3], lane_edges[:3]):
        arrow(ax, (x + w / 2, 39.6), (x + w / 2, 36.3), color=color, lw=0.8)
    arrow(ax, (41, 28.4), (41, 26.6), color=BLUE, lw=0.9)

    ax.add_patch(FancyBboxPatch((13.0, 19.4), 56.0, 6.5,
        boxstyle="round,pad=0.28,rounding_size=0.45", facecolor="#EAF1F8",
        edgecolor=BLUE, linewidth=0.75))
    ax.text(41.0, 23.6, "BATHYMETRY ESTIMATORS", ha="center", va="center",
            fontsize=7.0, weight="bold", color=NAVY)
    ax.text(41.0, 21.0, "Stumpf log-ratio  ·  Random Forest  ·  HistGradientBoosting  ·  RBF-SVR",
            ha="center", va="center", fontsize=6.0, color=INK)

    # Split model use into held-out evaluation and a supported full-data map.
    left_card = FancyBboxPatch((2.5, 3.7), 46.0, 12.8,
        boxstyle="round,pad=0.28,rounding_size=0.45", facecolor="#FFF4E5",
        edgecolor=GOLD, linewidth=0.8)
    right_card = FancyBboxPatch((51.5, 3.7), 46.0, 12.8,
        boxstyle="round,pad=0.28,rounding_size=0.45", facecolor="#F1F6F8",
        edgecolor=BLUE, linewidth=0.8)
    ax.add_patch(left_card); ax.add_patch(right_card)
    ax.text(25.5, 14.3, "SPATIAL HOLD-OUT EVALUATION", ha="center", va="center",
            fontsize=6.8, weight="bold", color="#9A6423")
    ax.text(25.5, 8.6,
            "5 grouped folds · 500 m blocks · 90 m buffer\nOut-of-fold RMSE, MAE and bias\nDepth-stratified errors\nBlock-bootstrap intervals and sensitivity analyses",
            ha="center", va="center", fontsize=5.55, color=INK, linespacing=1.22)
    ax.text(74.5, 14.3, "SUPPORTED PREDICTION PRODUCTS", ha="center", va="center",
            fontsize=6.4, weight="bold", color=NAVY)
    ax.text(74.5, 8.6,
            "Refit selected estimators on matched samples\nMap within 90 m of valid survey cells\nRetain each site's local port zero\nMaps shown separately in Figure 13",
            ha="center", va="center", fontsize=5.55, color=INK, linespacing=1.22)
    # Routed curves leave the estimator box and meet the top edge of each
    # destination card. Their control points keep both trajectories above all
    # card text and prevent the previous sweeping arrow from crossing labels.
    trajectory(ax, [(34, 19.15), (32.0, 18.7), (25.5, 18.6), (25.5, 16.65)], color=GOLD)
    trajectory(ax, [(48, 19.15), (53.0, 18.7), (74.5, 18.6), (74.5, 16.65)], color=BLUE)
    save(fig, "Figure3_MapBased_Methods_Workflow_TwoCoast")


def write_captions_and_attribution() -> None:
    captions = """Figure 2. Overlay of HLS reflectance and measured bathymetric samples on the same 30 m grid. Panels A and B show Anza / Agadir and Ras El Ma / Ras Kebdana, respectively. The RGB map background displays the quality-weighted B04/B03/B02 surface-reflectance composite; depth-coloured points mark the measured median depth in each matching 30 m cell. RGB contrast is stretched independently for each site. Depth is positive downward from each site's local port hydrographic zero; no common vertical datum or interpolated seabed surface is implied. Axes show local UTM offsets.

Figure 3. Source-to-output workflow for the two-coast satellite-derived bathymetry study, arranged as parallel data-source lanes. HLS S30/L30 reflectance, local XYZ soundings and GOT5.6/pyTMD tide anomalies pass through their respective screening and preparation steps before being overlaid on matched 30 m cells. Stumpf log-ratio, Random Forest, HistGradientBoosting and RBF-SVR are evaluated with five grouped spatial folds, 500 m blocks and a 90 m buffer. The two final branches distinguish out-of-fold performance assessment from full-data prediction products, which are mapped only within 90 m of valid survey cells. Each site's depths retain its local port zero. EMODnet and GEBCO are shown as source context only and do not enter model fitting or validation.
"""
    (OUT / "Materials_Methods_Additional_Captions_EN.txt").write_text(captions, encoding="utf-8")
    credits = """Third-party logo source notes for the methods workflow figure

ESA Sentinel-2 mission logo: https://www.esa.int/eologos/sentinel-2.html
Direct official asset: https://download.esa.int/multimedia/mission_logos/EO/sentinel-2_logo/sentinel-2.jpg

Python Powered logo: Python Software Foundation, official logo downloads and usage guidance:
https://www.python.org/community/logos/

The NASA and USGS identifiers are shown as plain text only. NASA's brand guidance states its insignia/logotype are protected and subject to restrictions; USGS reserves its visual identifier for official use and authorized partners. No NASA or USGS logo is included. Logos in the figure identify source mission/software only and do not imply endorsement.
"""
    (ROOT / "Assets" / "Logos" / "ATTRIBUTION.txt").write_text(credits, encoding="utf-8")


def main() -> None:
    frames = draw_input_overlay()
    draw_map_based_workflow(frames)
    write_captions_and_attribution()
    print(f"Created the sampled HLS/depth overlay and map-based workflow figures in {OUT}")
    print("Only saved 30 m sample cells are mapped; no model was fitted and no spatial interpolation was applied.")


if __name__ == "__main__":
    main()
