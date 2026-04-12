#!/usr/bin/env -S uv run --with geopandas --with matplotlib --with pandas --with numpy --with shapely --with pyogrio
"""
Admin-1 choropleth maps of flood BN IBF v1 risk outcomes.

For a date range, joins each daily BN result CSV to the ICPAC admin-1 GeoJSON
and plots risk_level with a fixed discrete colormap. Produces:
  - A combined 2×5 panel figure (output/flood_bn_v1_risk_maps_panel.png)
  - One PNG per day (output/maps/flood_bn_v1_risk_<date>.png)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

RISK_ORDER = ["Minimal", "Low", "Moderate", "High", "Extreme"]
RISK_RANK = {k: i for i, k in enumerate(RISK_ORDER)}
# Colorbrewer-ish 5-step sequential (green → yellow → orange → red → dark purple)
RISK_COLORS = {
    "Minimal":  "#1a9850",
    "Low":      "#d9ef8b",
    "Moderate": "#fee08b",
    "High":     "#f46d43",
    "Extreme":  "#a50026",
}
CMAP = ListedColormap([RISK_COLORS[k] for k in RISK_ORDER])
NORM = BoundaryNorm(range(len(RISK_ORDER) + 1), CMAP.N)


def load_day(adm1: gpd.GeoDataFrame, result_csv: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(result_csv)
    joined = adm1.merge(df, left_on="GID_1", right_on="boundary_id", how="left")
    joined["risk_rank"] = joined["risk_level"].map(RISK_RANK)
    return joined


def plot_day(ax, gdf: gpd.GeoDataFrame, title: str) -> None:
    gdf.plot(
        ax=ax,
        column="risk_rank",
        cmap=CMAP,
        norm=NORM,
        edgecolor="#555555",
        linewidth=0.25,
        missing_kwds={"color": "lightgrey", "edgecolor": "#999", "linewidth": 0.2},
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlim(21.0, 52.0)
    ax.set_ylim(-12.0, 23.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def add_risk_legend(fig) -> None:
    patches = [mpatches.Patch(color=RISK_COLORS[k], label=k) for k in RISK_ORDER]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.01),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adm1", default="icpac_adm1v3.geojson")
    ap.add_argument("--input-dir", default="output")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-03-10")
    ap.add_argument("--panel-out", default="output/flood_bn_v1_risk_maps_panel.png")
    ap.add_argument("--per-day-dir", default="output/maps")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    adm1 = gpd.read_file(args.adm1)
    dates = pd.date_range(args.start, args.end, freq="D")
    Path(args.per_day_dir).mkdir(parents=True, exist_ok=True)

    # Combined 2×5 panel
    ncols = 5
    nrows = (len(dates) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.3))
    fig.suptitle("Flood BN IBF v1 — admin-1 daily risk (2-yr RP, ECMWF 7-day forecast)",
                 fontsize=14, y=0.985)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, d in enumerate(dates):
        ax = axes_flat[i]
        csv = Path(args.input_dir) / f"flood_bn_v1_{d.date()}.csv"
        if not csv.exists():
            ax.text(0.5, 0.5, f"missing\n{d.date()}", ha="center", va="center")
            ax.set_axis_off()
            continue
        day_gdf = load_day(adm1, csv)
        plot_day(ax, day_gdf, title=str(d.date()))

        # per-day standalone PNG
        fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 6.0))
        plot_day(ax1, day_gdf, title=f"Flood risk — {d.date()}")
        patches = [mpatches.Patch(color=RISK_COLORS[k], label=k) for k in RISK_ORDER]
        ax1.legend(handles=patches, loc="lower left", fontsize=8, frameon=True, title="Risk level")
        out1 = Path(args.per_day_dir) / f"flood_bn_v1_risk_{d.date()}.png"
        fig1.tight_layout()
        fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig1)
        print(f"[map] wrote {out1}")

    for j in range(len(dates), len(axes_flat)):
        axes_flat[j].set_axis_off()

    add_risk_legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    Path(args.panel_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.panel_out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[map] wrote panel → {args.panel_out}")


if __name__ == "__main__":
    main()
