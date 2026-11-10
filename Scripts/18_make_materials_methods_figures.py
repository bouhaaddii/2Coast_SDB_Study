"""Create Materials & Methods figures from the saved study inputs.

No bathymetry model is fitted. Reflectances are recomposited from the saved
30 m sample table using the exact quality-weighted HLS recipe used in Stage 2.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results" / "Figures"
CFG = json.loads((ROOT / "Config" / "experiment.json").read_text(encoding="utf-8"))
SAMPLES = pd.read_csv(ROOT / "Results" / "Prepared" / "pixel_samples.csv.gz")
QUALITY = pd.read_csv(ROOT / "Results" / "Prepared" / "scene_quality.csv")
TIDE = pd.read_csv(ROOT / "Results" / "Tides" / "modelled_tide_overpasses_GOT5.6.csv")
WINDOW = pd.read_csv(ROOT / "Results" / "Additional_Analyses" / "survey_window_tide_timing_diagnostic.csv")

INK = "#203447"
MUTED = "#718096"
GRID = "#E4EAF0"
TEAL = "#168C8C"
ORANGE = "#E58B2A"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": .7,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

BANDS = [("blue", "B02 · blue"), ("green", "B03 · green"),
         ("red", "B04 · red"), ("nir", "NIR · B08 / B05")]


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def quality_weighted_sample_composite(site_key: str) -> pd.DataFrame:
    """Replicate read_composite(..., quality_weighted_mean) on the saved samples."""
    df = SAMPLES.loc[SAMPLES.site.eq(site_key)].copy().reset_index(drop=True)
    qsite = QUALITY.loc[QUALITY.site.eq(site_key)].copy()
    sums = np.zeros((len(df), 4), dtype=np.float64)
    denom = np.zeros(len(df), dtype=np.float64)
    for row in qsite.itertuples(index=False):
        cols = [f"{row.item_id}__{b}" for b, _ in BANDS]
        x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x).all(axis=1)
        w = valid.astype(float) * max(float(row.quality_score), 1e-6)
        sums += np.nan_to_num(x, nan=0.0) * w[:, None]
        denom += w
    composite = np.divide(sums, denom[:, None], out=np.full_like(sums, np.nan), where=denom[:, None] > 0)
    for j, (band, _) in enumerate(BANDS):
        df[f"composite_{band}"] = composite[:, j]
    df["composite_clear"] = denom > 0
    return df.loc[df.composite_clear].copy()


def style_3d(ax, x_origin_km: int, y_origin_km: int):
    ax.view_init(elev=25, azim=-56)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="both", which="major", pad=0, labelsize=6.3, colors=INK)
    ax.tick_params(axis="z", which="major", pad=0, labelsize=6.3, colors=INK)
    ax.set_xlabel("Easting X offset (km)", labelpad=1, fontsize=6.5, color=INK)
    ax.set_ylabel("Northing Y offset (km)", labelpad=1, fontsize=6.5, color=INK)
    ax.set_zlabel("Depth Z (m, positive down)", labelpad=1, fontsize=6.5, color=INK)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.975, 0.985, 0.99, 1))
        axis.pane.set_edgecolor("#D9E1E8")
    ax.grid(True, color=GRID, linewidth=.45)


VERTICAL_EXAGGERATION = {}


def plot_band_depth_3d(site_key: str, site: dict, letter: str):
    df = quality_weighted_sample_composite(site_key)
    if df.empty:
        raise SystemExit(f"No valid quality-weighted HLS sample composite for {site_key}.")
    # Local coordinate offsets retain UTM references while preventing
    # scientific-notation offsets and overlapping tick labels in the panels.
    x_origin_km = int(np.floor(df.x_m.min() / 1000.0))
    y_origin_km = int(np.floor(df.y_m.min() / 1000.0))
    x = df.x_m.to_numpy(dtype=float) / 1000.0 - x_origin_km
    y = df.y_m.to_numpy(dtype=float) / 1000.0 - y_origin_km
    z = df.depth_m.to_numpy(dtype=float)
    xr = max(np.ptp(x), 1e-4)
    yr = max(np.ptp(y), 1e-4)
    zr = max(np.nanpercentile(z, 99.5), 1.0)
    vertical_exaggeration = 0.65 * max(xr, yr) / (zr / 1000.0)
    VERTICAL_EXAGGERATION[site_key] = int(round(vertical_exaggeration))
    fig = plt.figure(figsize=(11.2, 8.2), layout="constrained")
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]
    for ax, (band, label), panel in zip(axes, BANDS, "ABCD"):
        values = pd.to_numeric(df[f"composite_{band}"], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(values) & np.isfinite(z)
        vmin, vmax = np.nanpercentile(values[good], [2, 98])
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
        points = ax.scatter(x[good], y[good], z[good], c=values[good], cmap="cividis", norm=norm,
                            s=4.2, alpha=.82, linewidths=0, depthshade=False, rasterized=True)
        style_3d(ax, x_origin_km, y_origin_km)
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
        ax.set_zlim(float(np.nanpercentile(z, 99.8)), 0)
        ax.set_box_aspect((xr, yr, zr / 1000.0 * vertical_exaggeration))
        ax.text2D(.02, .98, panel, transform=ax.transAxes, ha="left", va="top",
                  fontsize=10, fontweight="bold", color=INK)
        ax.set_title(label, loc="left", pad=2, color=INK, fontweight="bold")
        cb = fig.colorbar(points, ax=ax, shrink=.68, pad=.055, fraction=.05)
        cb.set_label("HLS reflectance ρ", fontsize=6.5, color=INK, labelpad=2)
        cb.ax.tick_params(labelsize=6, length=2, pad=1, colors=INK)
    stem = "Figure_2_HLS_Band_Depth_Overlay_Anza" if site_key == "Anza_Agadir" else "Figure_3_HLS_Band_Depth_Overlay_Ras_El_Ma"
    save(fig, stem)
    print(f"Wrote {stem}: n={len(df):,} cells; quality-weighted HLS reflectance, four bands.")


def plot_tide_inputs():
    q = QUALITY[["site", "item_id", "product", "quality_score"]]
    t = TIDE.loc[TIDE.record_type.eq("satellite_overpass")].merge(q, left_on=["site", "scene_id"],
                                                                    right_on=["site", "item_id"], how="left")
    t["datetime_utc"] = pd.to_datetime(t.datetime_utc, utc=True)
    t["anomaly_m"] = pd.to_numeric(t.tide_anomaly_m_relative_to_survey_day_mean, errors="coerce")
    t["quality_score"] = pd.to_numeric(t.quality_score, errors="coerce")
    fig = plt.figure(figsize=(10.8, 5.8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, .88], hspace=.16)
    top = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0, 1)
    for ax, (site_key, site) in zip(top, CFG["sites"].items()):
        sub = t.loc[t.site.eq(site_key)].copy()
        survey_day = pd.Timestamp(site["survey_date"], tz="UTC")
        sub["days_from_survey"] = (sub.datetime_utc - survey_day).dt.total_seconds() / 86400.0
        for product, marker in [("S30", "o"), ("L30", "s")]:
            s = sub.loc[sub["product"].eq(product)]
            ax.scatter(s.days_from_survey, s.anomaly_m, c=s.quality_score, cmap=cmap, norm=norm,
                       marker=marker, s=26, edgecolor="white", linewidth=.38, alpha=.94,
                       label=f"HLS {product}", zorder=3)
        ax.axvline(0, color=INK, lw=.85, ls=(0, (3, 2)))
        ax.axhline(0, color=MUTED, lw=.65, ls=(0, (2, 2)))
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_xlim(-34, 34)
        ax.set_xlabel("Days from local survey date", color=INK)
        ax.set_ylabel("GOT5.6 anomaly Δη (m)", color=INK)
        ax.set_title("Anza / Agadir" if site_key == "Anza_Agadir" else "Ras El Ma / Ras Kebdana",
                     loc="left", fontsize=9, fontweight="bold", color=INK)
        ax.text(.98, .96, f"n = {len(sub)} scenes", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=INK)
        ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=7, handletextpad=.35, columnspacing=.8)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=top, location="right", shrink=.83, pad=.015)
    cbar.set_label("Scene quality score", fontsize=7.3, color=INK)
    cbar.ax.tick_params(labelsize=6.5, colors=INK)
    axw = fig.add_subplot(gs[1, :])
    w = WINDOW.copy()
    labels = []
    lows = pd.to_numeric(w.tide_anomaly_p2_5_m, errors="coerce").to_numpy()
    highs = pd.to_numeric(w.tide_anomaly_p97_5_m, errors="coerce").to_numpy()
    for row in w.itertuples(index=False):
        site_name = "Anza / Agadir" if row.site == "Anza_Agadir" else "Ras El Ma / Ras Kebdana"
        labels.append(f"{site_name}\n{row.collection_start_local[11:16]}–{row.collection_end_local[11:16]} local")
    ypos = np.arange(len(w))[::-1]
    axw.axvline(0, color=INK, lw=.8, ls=(0, (3, 2)))
    # Plot only the documented 2.5th–97.5th percentile span. The saved
    # reference-model mean is not the centre of these survey-day anomalies.
    axw.hlines(ypos, lows, highs, color=TEAL, linewidth=2.2, zorder=3)
    axw.vlines(lows, ypos - .10, ypos + .10, color=TEAL, linewidth=1.0, zorder=3)
    axw.vlines(highs, ypos - .10, ypos + .10, color=TEAL, linewidth=1.0, zorder=3)
    axw.set_yticks(ypos, labels)
    axw.set_xlabel("Modelled anomaly relative to survey-day mean (m)", color=INK)
    axw.grid(axis="x", color=GRID, lw=.6)
    axw.tick_params(axis="y", labelsize=7, length=0, colors=INK)
    axw.tick_params(axis="x", labelsize=7, colors=INK)
    axw.text(1.0, -.36,
             "Uniform-time diagnostic only; not observed gauge levels, model-error bounds, or full survey vertical uncertainty.",
             transform=axw.transAxes, ha="right", va="top", fontsize=6.7, color=MUTED)
    save(fig, "Figure_4_GOT56_Acquisition_and_Survey_Window_Anomalies")
    print("Wrote Figure_4_GOT56_Acquisition_and_Survey_Window_Anomalies from saved HLS and tide diagnostics.")


def main():
    for site_key, site in CFG["sites"].items():
        plot_band_depth_3d(site_key, site, "")
    plot_tide_inputs()
    captions = f"""Figure 2. Quality-weighted HLS surface-reflectance inputs plotted at measured 30 m bathymetric cell coordinates for Anza / Agadir. Panels show B02 (blue), B03 (green), B04 (red), and the harmonized near-infrared feature (B08 for S30; B05 for L30). X and Y show UTM offsets in kilometres from E = 434 km and N = 3368 km; Z is measured depth below the Port of Agadir local hydrographic zero, positive downward. Point colour encodes composite reflectance rho (unitless). Only observed cell locations are shown; no seabed surface is interpolated. Vertical exaggeration is {VERTICAL_EXAGGERATION['Anza_Agadir']}x for legibility.

Figure 3. As Figure 2, for Ras El Ma / Ras Kebdana. X and Y show UTM offsets in kilometres from E = 551 km and N = 3888 km. Depths are relative to the local hydrographic zero of the Port of Ras Kebdana and are positive downward; this reference is independent of the Agadir zero. Vertical exaggeration is {VERTICAL_EXAGGERATION['Ras_El_Ma_Ras_Kebdana']}x for legibility.

Figure 4. GOT5.6 acquisition anomalies and survey-window timing sensitivity. Upper panels show one modelled anomaly per HLS overpass relative to the mean on the site's survey date; marker shape identifies S30 / L30 and colour indicates the image-only scene-quality score. The lower panel shows modelled anomaly percentiles over each reported local field-collection window under uniform-time sampling. These are model diagnostics, not tide-gauge observations, absolute port-datum corrections, or uncertainty bounds for the full bathymetric survey."""
    (OUT / "Materials_Methods_Figure_Captions_EN.txt").write_text(captions, encoding="utf-8")


if __name__ == "__main__":
    main()
