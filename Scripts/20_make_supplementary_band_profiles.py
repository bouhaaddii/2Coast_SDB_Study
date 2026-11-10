"""Create legible 3D supplementary reflectance-by-depth profiles from saved inputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "Config" / "experiment.json").read_text(encoding="utf-8"))
SAMPLES = pd.read_csv(ROOT / "Results" / "Prepared" / "pixel_samples.csv.gz")
QUALITY = pd.read_csv(ROOT / "Results" / "Prepared" / "scene_quality.csv")
OUT_RESULTS = ROOT / "Results" / "Figures"
OUT_SUPP = ROOT / "Submission_Package" / "Supplementary" / "Figures"

BANDS = [
    ("blue", "B02 · Blue", "#2878B5"),
    ("green", "B03 · Green", "#27966F"),
    ("red", "B04 · Red", "#D65A4A"),
    ("nir", "NIR · B08/B05", "#7056A8"),
]
INK = "#203447"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 11,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def quality_weighted_composite(site_key: str) -> pd.DataFrame:
    """Recreate the saved HLS quality-weighted composite at sounding pixels."""
    df = SAMPLES.loc[SAMPLES.site.eq(site_key)].copy().reset_index(drop=True)
    qsite = QUALITY.loc[QUALITY.site.eq(site_key)]
    sums = np.zeros((len(df), len(BANDS)), dtype=np.float64)
    denom = np.zeros(len(df), dtype=np.float64)
    for row in qsite.itertuples(index=False):
        columns = [f"{row.item_id}__{band}" for band, _, _ in BANDS]
        values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        weight = valid.astype(float) * max(float(row.quality_score), 1e-6)
        sums += np.nan_to_num(values, nan=0.0) * weight[:, None]
        denom += weight
    composite = np.divide(
        sums, denom[:, None], out=np.full_like(sums, np.nan), where=denom[:, None] > 0
    )
    for index, (band, _, _) in enumerate(BANDS):
        df[f"rho_{band}"] = composite[:, index]
    return df.loc[denom > 0].copy()


def depth_band_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize cell composites by 1 m observed-depth intervals."""
    df = df.copy()
    df["depth_bin"] = np.floor(pd.to_numeric(df.depth_m, errors="coerce"))
    result = {}
    for band, _, _ in BANDS:
        grouped = df.groupby("depth_bin", observed=True)[f"rho_{band}"]
        stats = grouped.agg(median="median", q25=lambda x: x.quantile(.25),
                            q75=lambda x: x.quantile(.75), n="count").reset_index()
        stats = stats.loc[stats.n >= 3].copy()
        stats["depth_center"] = stats.depth_bin + .5
        result[band] = stats
    return result


def render(site_key: str, display_name: str, stem: str, shared_zlim: tuple[float, float]) -> None:
    cells = quality_weighted_composite(site_key)
    summaries = depth_band_summaries(cells)
    fig = plt.figure(figsize=(9.2, 6.6), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")

    for layer, (band, label, color) in enumerate(BANDS):
        stats = summaries[band]
        if stats.empty:
            continue
        x = stats.depth_center.to_numpy(dtype=float)
        y = np.full(len(stats), layer, dtype=float)
        med = stats["median"].to_numpy(dtype=float)
        lo = stats.q25.to_numpy(dtype=float)
        hi = stats.q75.to_numpy(dtype=float)
        # Thin stems encode the 25th–75th percentile range; circles show medians.
        for xd, yd, low, high in zip(x, y, lo, hi):
            ax.plot([xd, xd], [yd, yd], [low, high], color=color, alpha=.62, lw=1.05)
            cap = .075
            ax.plot([xd, xd], [yd-cap, yd+cap], [low, low], color=color, alpha=.62, lw=.7)
            ax.plot([xd, xd], [yd-cap, yd+cap], [high, high], color=color, alpha=.62, lw=.7)
        ax.plot(x, y, med, color=color, lw=1.8, alpha=.96)
        ax.scatter(x, y, med, color=color, s=25, edgecolor="white", linewidth=.5,
                   depthshade=False, zorder=4)

    ax.set_title(display_name, loc="left", color=INK, fontweight="bold", pad=10)
    ax.set_xlabel("Survey depth below local hydrographic zero (m; positive downward)", labelpad=11)
    ax.set_ylabel("")
    ax.set_zlabel("Surface reflectance ρ (unitless)", labelpad=8)
    ax.set_yticks(np.arange(len(BANDS)))
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    all_depth = pd.to_numeric(cells.depth_m, errors="coerce").to_numpy(dtype=float)
    ax.set_xlim(0, float(np.nanpercentile(all_depth, 99.5)))
    ax.set_ylim(-.45, len(BANDS)-.55)
    ax.set_zlim(*shared_zlim)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.zaxis.set_major_locator(MaxNLocator(5))
    ax.view_init(elev=23, azim=-58)
    ax.set_box_aspect((1.65, 1.05, .78))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((.97, .98, .99, 1))
        axis.pane.set_edgecolor("#D9E1E8")
    ax.grid(True, color="#DCE5EA", linewidth=.45)
    stat_handles=[
        Line2D([0], [0], marker="o", color=INK, markerfacecolor=INK, lw=0,
               markersize=4, label="Median"),
        Line2D([0], [0], color=INK, lw=1.1, label="IQR (25th–75th percentile)"),
    ]
    band_handles=[Line2D([0], [0], color=color, lw=2.1, label=label) for _, label, color in BANDS]
    fig.legend(handles=band_handles, loc="upper center", bbox_to_anchor=(.53, .955),
               ncol=4, frameon=False, fontsize=7.5, handlelength=1.6,
               columnspacing=1.4, handletextpad=.4)
    fig.legend(handles=stat_handles, loc="upper center", bbox_to_anchor=(.53, .91),
               ncol=2, frameon=False, fontsize=7.2, handlelength=1.5,
               columnspacing=1.5, handletextpad=.45)
    note = f"Quality-weighted HLS composite · {len(cells):,} sampled 30 m cells · depth-bin summaries require n ≥ 3"
    fig.text(.5, .015, note, ha="center", va="bottom", fontsize=7.5, color="#526573")

    for folder in (OUT_RESULTS, OUT_SUPP):
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
        fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}: {len(cells):,} quality-composited cells; 1 m depth bins.")


def main() -> None:
    caption_path = OUT_SUPP.parent / "Supplementary_Figure_Captions_EN.txt"
    prior_captions = caption_path.read_text(encoding="utf-8").split("\n\n") if caption_path.exists() else []
    remaining_captions = [text for text in prior_captions
                          if text.strip() and not text.lstrip().startswith(("Figure S1.", "Figure S2."))]
    cells = {site: quality_weighted_composite(site) for site in CFG["sites"]}
    reflectance = np.concatenate([
        cells[site][f"rho_{band}"].to_numpy(dtype=float)
        for site in cells for band, _, _ in BANDS
    ])
    zlim = tuple(float(v) for v in np.nanpercentile(reflectance[np.isfinite(reflectance)], [1, 99]))
    captions = []
    for site_key, site in CFG["sites"].items():
        if site_key == "Anza_Agadir":
            stem = "Supplementary_Figure_S1_HLS_bands_Anza"
            site_name = "Anza / Agadir"
            zero = "Port of Agadir local hydrographic zero"
            number = "S1"
        else:
            stem = "Supplementary_Figure_S2_HLS_bands_Ras_El_Ma"
            site_name = "Ras El Ma / Ras Kebdana"
            zero = "Port of Ras Kebdana local hydrographic zero"
            number = "S2"
        render(site_key, site_name, stem, zlim)
        captions.append(
            f"Figure {number}. Three-dimensional, band-layered profile of quality-weighted HLS surface reflectance "
            f"against measured bathymetric depth at {site_name}. The x-axis is survey depth below the {zero} "
            f"(positive downward); the y-axis separates B02 (blue), B03 (green), B04 (red), and the harmonized "
            f"near-infrared feature (B08 for Sentinel-2 S30; B05 for Landsat L30); the z-axis is unitless "
            f"surface reflectance. Reflectance is first composited at each sampled 30 m cell using the saved "
            f"scene-quality weights. Within each 1 m depth interval, the marker is the median across cells and "
            f"the vertical whisker is the 25th–75th percentile range (only bins with at least three cells are shown). "
            f"The band axis is categorical; the 3D separation is for visual comparison and does not represent a "
            f"physical elevation or spatial map. The two sites retain their independent local vertical references."
        )
    caption_path.write_text("\n\n".join(captions + remaining_captions) + "\n", encoding="utf-8")
    print("Updated supplementary S1/S2 captions.")


if __name__ == "__main__":
    main()
