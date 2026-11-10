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
import rasterio
from PIL import Image
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.transforms import Affine2D

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
    ax.text(0.50, 0.975, f"({letter})  {site}", transform=ax.transAxes,
            ha="center", va="top", fontsize=6.0, weight="bold", color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 2.0})
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
    fig = plt.figure(figsize=(7.25, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045],
                            left=0.09, right=0.94, bottom=0.22, top=0.91, wspace=0.20)
    draw_site_input_map(fig.add_subplot(grid[0, 0]), frames[sites[0]], "A",
                        "Anza / Agadir", norm, point_size=1.65)
    draw_site_input_map(fig.add_subplot(grid[0, 1]), frames[sites[1]], "B",
                        "Ras El Ma / Ras Kebdana", norm, point_size=1.30)
    cax = fig.add_subplot(grid[0, 2])
    sm = ScalarMappable(norm=norm, cmap="cividis"); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Survey depth below local port zero (m; positive down)", fontsize=6.4, labelpad=6)
    cb.ax.tick_params(labelsize=6.0, length=2)
    fig.text(0.50, 0.065,
             "HLS RGB composite at retained 30 m cells; small coloured points show the corresponding surveyed cell-depth medians.",
             ha="center", va="center", fontsize=5.9, color=MUTED)
    save(fig, "Figure_Sampled_HLS_RGB_Bathymetry_Overlay")
    return frames


def draw_map_based_workflow(frames: dict[str, pd.DataFrame]) -> None:
    sites = ["Anza_Agadir", "Ras_El_Ma_Ras_Kebdana"]
    depth_norm = Normalize(vmin=0.0, vmax=25.0)
    cmap = plt.get_cmap("cividis")

    def depth_rgba(frame: pd.DataFrame):
        _, xv, yv = rgb_at_grid(frame)
        ix = np.searchsorted(xv, frame.x_m.to_numpy(float))
        iy = np.searchsorted(yv, frame.y_m.to_numpy(float))
        image = np.zeros((len(yv), len(xv), 4), dtype=float)
        rgba = cmap(depth_norm(frame.depth_m.to_numpy(float)))
        image[iy, ix] = rgba
        return image

    def pair_texture(site_arrays, aspect_values, height_px=240):
        gap = 24
        widths = [max(1, int(round(height_px * aspect))) for aspect in aspect_values]
        canvas = Image.new("RGBA", (sum(widths) + gap, height_px), (255, 255, 255, 0))
        xoff = 0
        for array, width in zip(site_arrays, widths):
            rgba8 = np.clip(array * 255, 0, 255).astype(np.uint8)
            tile = Image.fromarray(rgba8, mode="RGBA").resize(
                (width, height_px), resample=Image.Resampling.NEAREST)
            canvas.alpha_composite(tile, (xoff, 0))
            xoff += width + gap
        return np.asarray(canvas), canvas.width / canvas.height

    def site_texture(frame: pd.DataFrame, kind: str):
        rgb, xv, yv = rgb_at_grid(frame)
        d = depth_rgba(frame)
        if kind == "rgb":
            texture = rgb
        elif kind == "depth":
            texture = d
        elif kind == "paired":
            texture = np.zeros_like(rgb)
            valid = (rgb[..., 3] > 0) & (d[..., 3] > 0)
            texture[valid, :3] = 0.48 * rgb[valid, :3] + 0.52 * d[valid, :3]
            texture[valid, 3] = 1.0
        else:
            raise ValueError(kind)
        aspect = (float(xv.max() - xv.min() + 30) /
                  float(yv.max() - yv.min() + 30))
        return texture, aspect

    def prediction_texture(site: str):
        path = ROOT / "Results" / "Predictions" / "Maps" / f"{site}_RandomForest_supported_depth.tif"
        with rasterio.open(path) as src:
            values = src.read(1, masked=True).filled(np.nan).astype(float)
            aspect = (src.bounds.right - src.bounds.left) / (src.bounds.top - src.bounds.bottom)
        rgba = cmap(depth_norm(np.nan_to_num(values, nan=0.0)))
        rgba[..., 3] = np.isfinite(values).astype(float)
        return np.flipud(rgba), aspect

    def sheet(kind: str):
        textures, aspects = zip(*(site_texture(frames[s], kind) for s in sites))
        return pair_texture(textures, aspects)

    rgb_sheet, rgb_aspect = sheet("rgb")
    depth_sheet, depth_aspect = sheet("depth")
    paired_sheet, paired_aspect = sheet("paired")
    pred_arrays, pred_aspects = zip(*(prediction_texture(s) for s in sites))
    pred_sheet, pred_aspect = pair_texture(pred_arrays, pred_aspects, height_px=240)

    fig, ax = plt.subplots(figsize=(7.25, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    x0, sheet_w, sheet_h = 4.3, 43.5, 15.7
    shear, yscale = 0.30, 0.72

    def draw_plane(image, y0, label, edge, zorder, site_aspects=None):
        corners = [(x0, y0), (x0 + sheet_w, y0),
                   (x0 + sheet_w + shear * sheet_h, y0 + yscale * sheet_h),
                   (x0 + shear * sheet_h, y0 + yscale * sheet_h)]
        ax.add_patch(Polygon(corners, closed=True, facecolor="white", edgecolor=edge,
                             linewidth=0.8, joinstyle="round", zorder=zorder))
        local_transform = Affine2D.from_values(1, 0, shear, yscale, x0, y0)
        transform = local_transform + ax.transData
        ax.imshow(image, extent=(0, sheet_w, 0, sheet_h), origin="lower",
                  interpolation="nearest", transform=transform, zorder=zorder + 1,
                  aspect="auto")
        ax.add_patch(Polygon(corners, closed=True, facecolor="none", edgecolor=edge,
                             linewidth=0.65, joinstyle="round", zorder=zorder + 2))
        ax.text(3.8, y0 + yscale * sheet_h * 0.48, label, ha="right", va="center",
                fontsize=6.1, weight="bold", color=edge,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.94, "pad": 1.1},
                zorder=zorder + 4)
        if site_aspects:
            hpx = 240
            gap_px = 24
            widths = [hpx * a for a in site_aspects]
            total_px = sum(widths) + gap_px
            centers = [widths[0] / 2,
                       widths[0] + gap_px + widths[1] / 2]
            for center, name in zip(centers, ("A", "B")):
                tx = sheet_w * center / total_px
                ty = sheet_h * 0.78
                px, py = local_transform.transform((tx, ty))
                ax.text(px, py, name, ha="center", va="center", fontsize=5.8,
                        color=INK, bbox={"facecolor": "white", "edgecolor": "none",
                                        "alpha": 0.90, "pad": 0.9}, zorder=zorder + 5)

    # The exploded map stack contains the actual site-specific HLS composite,
    # survey-cell depths, and their co-registered overlay.
    draw_plane(paired_sheet, 42.5, "CO-REGISTERED\nINPUTS", BLUE, 10)
    draw_plane(depth_sheet, 56.0, "SURVEY\nDEPTH", GOLD, 20)
    draw_plane(rgb_sheet, 69.5, "HLS RGB\nCOMPOSITE", TEAL, 30,
               [site_texture(frames[s], "rgb")[1] for s in sites])

    # Data-source labels and arrows mirror the supplied exploded-layer style.
    rounded_box(ax, 64.0, 82.0, 34.0, 12.0, "Satellite reflectance",
                "HLS v2.0 · S30 Sentinel-2 + L30 Landsat 8/9\n30 m · B02/B03/B04 + harmonized NIR",
                face="white", edge=TEAL, title_color=TEAL, fs=6.0, title_fs=7.0, radius=0.8)
    rounded_box(ax, 64.0, 66.5, 34.0, 12.0, "Bathymetric observations",
                "XYZ soundings · 30 m cell medians\nAnza 10 Nov 2022 · Ras El Ma 8 Aug 2024",
                face="white", edge=GOLD, title_color=GOLD, fs=6.0, title_fs=7.0, radius=0.8)
    rounded_box(ax, 64.0, 51.0, 34.0, 12.0, "Tide-model covariate",
                "pyTMD + GOT5.6 overpass anomaly\nRelative to survey-day mean; no local gauge tie",
                face="white", edge="#8972B4", title_color="#72579B", fs=5.9, title_fs=7.0, radius=0.8)
    arrow(ax, (63.3, 87.5), (54.5, 82.0), color=TEAL, lw=0.85)
    arrow(ax, (63.3, 72.0), (54.5, 69.0), color=GOLD, lw=0.85)
    arrow(ax, (63.3, 56.0), (54.5, 53.0), color="#8972B4", lw=0.85)

    # Processing and evaluation stages continue beneath the stacked inputs.
    rounded_box(ax, 2.0, 25.0, 29.5, 13.0, "Pre-processing",
                "HLS Fmask + common-band harmonization\nXYZ projection + duplicate removal\n30 m cell medians; retain local zeros",
                face=SEA, edge=TEAL, title_color=NAVY, fs=6.15, title_fs=7.2, radius=0.8)
    rounded_box(ax, 35.2, 25.0, 29.5, 13.0, "Sample construction",
                "Quality-weighted four-band composite · ±30 d\nMatch cell depth + relative tide anomaly\nKeep the two local vertical references",
                face="#EAF2F8", edge=BLUE, title_color=NAVY, fs=6.15, title_fs=7.2, radius=0.8)
    rounded_box(ax, 68.4, 25.0, 29.5, 13.0, "Models and validation",
                "Stumpf · RF · HGB · RBF-SVR\nFive spatial folds · 500 m blocks\n90 m buffer · out-of-fold scores",
                face="#FFF4E5", edge=GOLD, title_color=NAVY, fs=6.05, title_fs=7.2, radius=0.8)
    arrow(ax, (32.0, 31.5), (34.7, 31.5), color=BLUE, lw=0.85)
    arrow(ax, (65.2, 31.5), (67.9, 31.5), color=BLUE, lw=0.85)
    arrow(ax, (48.0, 42.0), (17.0, 38.2), color=TEAL, lw=0.75, rad=0.12)
    arrow(ax, (81.0, 50.7), (50.0, 38.2), color="#8972B4", lw=0.75, rad=-0.10)

    # Show actual supported Random Forest outputs saved by the completed run.
    output_extent = (2.5, 45.0, 4.0, 21.0)
    ax.imshow(pred_sheet, extent=output_extent, origin="lower", interpolation="nearest",
              aspect="auto", zorder=2)
    ax.add_patch(Polygon([(output_extent[0], output_extent[2]),
                          (output_extent[1], output_extent[2]),
                          (output_extent[1], output_extent[3]),
                          (output_extent[0], output_extent[3])], closed=True,
                         facecolor="none", edgecolor=GRID, linewidth=0.7, zorder=3))
    ax.text(output_extent[0], 22.4, "SUPPORTED BATHYMETRY OUTPUT · RANDOM FOREST",
            fontsize=6.1, weight="bold", color=NAVY, ha="left", va="bottom")
    # Compact depth key in the gap beside the pair of actual prediction maps.
    gradient = np.linspace(0.0, 1.0, 128).reshape(-1, 1)
    ax.imshow(gradient, extent=(47.0, 48.0, 4.5, 18.0), origin="lower",
              aspect="auto", cmap=cmap, zorder=2)
    ax.add_patch(Polygon([(47.0, 4.5), (48.0, 4.5), (48.0, 18.0), (47.0, 18.0)],
                         closed=True, facecolor="none", edgecolor=GRID,
                         linewidth=0.5, zorder=3))
    ax.text(47.5, 19.0, "m", fontsize=4.8, color=MUTED, ha="center", va="bottom")
    for value in (0, 10, 20, 25):
        ytick = 4.5 + 13.5 * value / 25.0
        ax.plot([48.0, 48.35], [ytick, ytick], color=MUTED, lw=0.45, zorder=4)
        ax.text(48.5, ytick, str(value), fontsize=4.4, color=MUTED,
                ha="left", va="center")
    ax.text(53.0, 20.0,
            "Reported results\nSpatially held-out RMSE, MAE and bias\nDepth-stratified errors + block-bootstrap intervals\nTide, scene/sensor and cross-site sensitivity\n\nFull-data maps are limited to 90 m from survey support;\naccuracy claims use out-of-fold predictions.\n\nLeft: Anza / Agadir · Right: Ras El Ma / Ras Kebdana\nEach map keeps its local port zero.\nEMODnet / GEBCO: source context only.",
            fontsize=5.8, color=INK, ha="left", va="top", linespacing=1.16,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": PALE,
                  "edgecolor": GRID, "linewidth": 0.65})
    save(fig, "Figure3_MapBased_Methods_Workflow_TwoCoast")


def write_captions_and_attribution() -> None:
    captions = """Figure 2. HLS surface-reflectance composite and bathymetric observations at retained 30 m cells. Panels A and B show Anza / Agadir and Ras El Ma / Ras Kebdana, respectively. The RGB background is the quality-weighted composite from B04/B03/B02 at sampled cells; small points show the corresponding median survey depth in each 30 m cell. RGB contrast is stretched independently by site. Depth is positive downward and referenced to each site's local port hydrographic zero; no common vertical datum or interpolated seabed surface is implied. Axes show local UTM offsets.

Figure 3. Map-overlay workflow for the two-coast satellite-derived bathymetry study. The exploded input layers show quality-weighted HLS RGB composites, surveyed depth medians on the 30 m grid, and the co-registered paired inputs for Anza / Agadir and Ras El Ma / Ras Kebdana. Source arrows identify HLS v2.0 S30/L30 reflectance, the local XYZ surveys and the relative GOT5.6 anomaly calculated with pyTMD. The workflow summarizes band screening and harmonization, survey preparation, sample pairing, depth estimators and buffered spatial validation. The lower maps show saved full-data Random Forest predictions within the 90 m support limit; accuracy is assessed from out-of-fold predictions. Depths retain their separate local port zeros. EMODnet and GEBCO are context grids only.
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
