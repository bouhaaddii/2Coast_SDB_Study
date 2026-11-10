"""Create publication figures for the actual two-coast SDB workflow.

The map is reconstructed only at sampled 30 m cells from the saved
quality-weighted HLS composite; no spatial interpolation or new model fit is
performed.  The workflow graphic documents the implemented study pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Figures"
ASSETS = ROOT / "Assets" / "Logos"
CFG = json.loads((ROOT / "Config" / "experiment.json").read_text(encoding="utf-8"))
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


def rounded_box(ax, x, y, w, h, title, body, face=PALE, edge=GRID,
                title_color=INK, lw=0.8, fs=6.8, title_fs=7.4, radius=0.8):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.32,rounding_size={radius}",
                         facecolor=face, edgecolor=edge, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + 0.65, y + h - 1.1, title, ha="left", va="top",
            fontsize=title_fs, weight="bold", color=title_color)
    ax.text(x + 0.65, y + h - 3.5, body, ha="left", va="top",
            fontsize=fs, color=INK, linespacing=1.25)
    return box


def arrow(ax, start, end, color=BLUE, lw=0.95, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7.5,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 5.65))
    ax.set_xlim(0, 100); ax.set_ylim(-7, 100); ax.axis("off")

    # Source cards: the small ESA logo identifies the Sentinel-2 mission only.
    card_y, card_h = 78, 16
    widths = [24.2, 24.2, 24.2, 24.2]
    xs = [0.8, 25.5, 50.2, 74.9]
    titles = ["Satellite reflectance", "Bathymetric surveys", "Tide-model input", "Context grids"]
    bodies = [
        "HLS v2.0 · 30 m\nS30 Sentinel-2 MSI\nL30 Landsat 8/9 OLI",
        "XYZ soundings\nAnza / Agadir · 10 Nov 2022\nRas El Ma · 8 Aug 2024",
        "GOT5.6 harmonic model\nAcquisition overpass times\nNo local-gauge validation",
        "EMODnet + GEBCO\nCoverage/source context only\nNot validation truth",
    ]
    for x, w, title, body in zip(xs, widths, titles, bodies):
        rounded_box(ax, x, card_y, w, card_h, title, body, face="white", edge=GRID,
                    title_fs=7.3, fs=6.3)
    esa_path = ASSETS / "ESA_Sentinel-2.jpg"
    if esa_path.exists():
        logo = mpimg.imread(esa_path)
        ax.imshow(logo, extent=(18.8, 23.3, 82.0, 84.2), zorder=3, aspect="auto")

    # Three independent preprocessing branches.
    pre_y, pre_h = 55, 17.5
    rounded_box(ax, 2.0, pre_y, 30.0, pre_h, "HLS quality control",
                "Scale factor 0.0001; Fmask/validity\nCommon blue, green, red and NIR bands\nScene quality from fixed water mask",
                face=SEA, edge=TEAL, title_color=NAVY, fs=6.5)
    rounded_box(ax, 35.0, pre_y, 30.0, pre_h, "Sounding preparation",
                "Remove exact duplicate XYZ rows\nProject to HLS UTM grid; 30 m cells\nMedian depth; retain local port zero",
                face="#FFF4E5", edge=GOLD, title_color=NAVY, fs=6.5)
    rounded_box(ax, 68.0, pre_y, 30.0, pre_h, "Relative tide anomaly",
                "pyTMD with GOT5.6; 10 min daily reference\nOverpass height minus local survey-day mean\nRelative adjustment; not an absolute datum tie",
                face=LILAC, edge="#8972B4", title_color=NAVY, fs=6.35)
    for sx, ex in [(13.0, 17.0), (37.5, 50.0), (62.5, 83.0)]:
        arrow(ax, (sx, card_y - 0.7), (ex, pre_y + pre_h + 0.5))
    # Open bathymetry grids are retained for source context and do not feed the
    # sample generation or validation branch.

    # Paired feature/label table.
    rounded_box(ax, 8.0, 34.5, 84.0, 14.5, "Paired 30 m training samples",
                "±30-day window · 33 Anza scenes / 20 Ras El Ma scenes\n"
                "Quality-weighted four-band mean paired with the 30 m median survey depth\n"
                "The same scene weights composite the tide anomaly; fit depth + anomaly, then subtract it from predictions",
                face="#EAF2F8", edge=BLUE, title_color=NAVY, fs=6.55, title_fs=7.5)
    # Three simple, non-crossing connectors show the data streams joining at
    # the paired sample table. Keep the arrows outside the box interiors.
    for start_x, end_x in [(17.0, 25.0), (50.0, 50.0), (83.0, 75.0)]:
        arrow(ax, (start_x, pre_y - 0.5), (end_x, 49.1), color=BLUE, lw=0.85)

    # Estimation and independent spatial assessment.
    rounded_box(ax, 2.0, 14.2, 46.0, 14.8, "Depth estimators",
                "Stumpf blue/green log-ratio · Random Forest\nHistGradientBoosting · RBF-SVR\nFixed analysis configurations; no global optimum claimed",
                face="white", edge=TEAL, title_color=NAVY, fs=6.55)
    rounded_box(ax, 52.0, 14.2, 46.0, 14.8, "Spatially held-out evaluation",
                "Five folds of 500 m spatial blocks\n90 m exclusion buffer around test cells\nOut-of-fold metrics and block-bootstrap intervals",
                face="white", edge=GOLD, title_color=NAVY, fs=6.55)
    arrow(ax, (26.0, 34.3), (26.0, 29.2), color=TEAL)
    arrow(ax, (48.0, 21.5), (51.8, 21.5), color=BLUE)

    rounded_box(ax, 2.0, 2.0, 96.0, 8.0, "Reported outputs",
                "Out-of-fold metrics and depth errors · paired tide ablation · scene/sensor sensitivity\n"
                "Target-calibrated cross-site transfer · exploratory spatial intervals · supported maps (≤90 m)",
                face="#E8F3F5", edge=TEAL, title_color=NAVY, fs=5.95, title_fs=7.0, radius=0.6)
    arrow(ax, (75.0, 14.0), (75.0, 10.3), color=BLUE)

    # Compact software strip: Python mark plus names from the actual environment.
    logo_path = ASSETS / "Python_Powered_200x80.png"
    if logo_path.exists():
        logo = mpimg.imread(logo_path)
        ax.imshow(logo, extent=(82.0, 96.0, -6.2, -3.2), zorder=4, aspect="auto")
    ax.text(2.0, -4.7, "Python · NumPy/pandas · rasterio/pyproj · pyTMD · scikit-learn · Matplotlib",
            fontsize=5.3, color=MUTED, va="center")
    save(fig, "Figure_Methods_Workflow_TwoCoast")


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
        channels.append(np.power(stretched, 0.82))
    rgb = np.column_stack(channels)
    xvals = np.sort(frame.x_m.unique())
    yvals = np.sort(frame.y_m.unique())
    ix = np.searchsorted(xvals, frame.x_m.to_numpy())
    iy = np.searchsorted(yvals, frame.y_m.to_numpy())
    image = np.zeros((len(yvals), len(xvals), 4), dtype=float)
    image[iy, ix, :3] = rgb
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


def draw_input_overlay() -> None:
    sites = ["Anza_Agadir", "Ras_El_Ma_Ras_Kebdana"]
    frames = {site: quality_weighted_composite(site) for site in sites}
    all_depth = np.concatenate([f.depth_m.to_numpy(float) for f in frames.values()])
    vmax = float(np.ceil(np.nanpercentile(all_depth, 99.5)))
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1.0))
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(7.25, 5.65))
    grid = fig.add_gridspec(2, 2, left=0.10, right=0.88, bottom=0.10, top=0.96,
                            hspace=0.26, wspace=0.19)
    panel_id = iter(["A", "B", "C", "D"])
    for row, site in enumerate(sites):
        frame = frames[site]
        image, xv, yv = rgb_at_grid(frame)
        half_x = float(np.median(np.diff(xv))) / 2 if len(xv) > 1 else 15.0
        half_y = float(np.median(np.diff(yv))) / 2 if len(yv) > 1 else 15.0
        x_origin = np.floor(float(xv.min()) / 1000.0) * 1000.0
        y_origin = np.floor(float(yv.min()) / 1000.0) * 1000.0
        extent = ((xv.min() - half_x - x_origin) / 1000,
                  (xv.max() + half_x - x_origin) / 1000,
                  (yv.min() - half_y - y_origin) / 1000,
                  (yv.max() + half_y - y_origin) / 1000)
        depth = frame.depth_m.to_numpy(float)
        for col in range(2):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(image, origin="lower", extent=extent, interpolation="nearest", aspect="equal")
            if col == 1:
                x = (frame.x_m.to_numpy(float) - x_origin) / 1000
                y = (frame.y_m.to_numpy(float) - y_origin) / 1000
                ax.scatter(x, y, c=depth, cmap=cmap, norm=norm, s=4.2,
                           linewidths=0.16, edgecolors="white", alpha=0.96, zorder=4,
                           rasterized=True)
            letter = next(panel_id)
            ax.text(0.025, 0.96, f"({letter})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.1, weight="bold", color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.1})
            ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
            ax.set_xlabel(f"Easting offset from {x_origin/1000:.0f} km (km)")
            ax.set_ylabel(f"Northing offset from {y_origin/1000:.0f} km (km)")
            ax.xaxis.set_major_locator(plt.MaxNLocator(4))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.grid(color="white", linewidth=0.38, alpha=0.48)
            ax.set_axisbelow(True)
            add_scalebar(ax)
    cax = fig.add_axes([0.91, 0.19, 0.022, 0.63])
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Survey depth below each local port zero (m; positive down)", fontsize=6.8, labelpad=7)
    cb.ax.tick_params(labelsize=6.0, length=2)
    save(fig, "Figure_Sampled_HLS_RGB_Bathymetry_Overlay")


def write_captions_and_attribution() -> None:
    captions = """Figure. Methods workflow for the two-coast HLS satellite-derived bathymetry study. The diagram summarizes the implemented data sources, screening and harmonization, 30 m sample pairing, relative GOT5.6 anomaly adjustment, estimators, spatial holdout design and reported outputs. EMODnet and GEBCO are used as source-context grids only. The NASA/USGS/ESA names identify the HLS data lineage; the Sentinel-2 and Python marks identify the mission and software and do not imply agency or foundation endorsement.

Figure. Sampled HLS surface-reflectance composite and bathymetric observations at the retained 30 m cells. Panels A-B show Anza / Agadir and panels C-D show Ras El Ma / Ras Kebdana; A and C show the stretched HLS RGB input (B04/B03/B02), and B and D overlay the measured cell-depth medians. The RGB stretch is calculated separately for each site for display. Depth colours refer to each survey's own local port hydrographic zero, positive downward; no datum transformation or interpolated seabed surface is shown. Coordinate axes are local offsets in the respective UTM system.
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
    draw_workflow()
    draw_input_overlay()
    write_captions_and_attribution()
    print(f"Created workflow and sampled HLS/depth overlay figures in {OUT}")
    print("Only saved 30 m sample cells are mapped; no model was fitted and no spatial interpolation was applied.")


if __name__ == "__main__":
    main()
