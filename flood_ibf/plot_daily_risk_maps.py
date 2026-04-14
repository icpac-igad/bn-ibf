#!/usr/bin/env -S uv run --with geopandas --with matplotlib --with pandas --with numpy --with shapely --with pyogrio
"""
Admin-1 choropleth maps of flood BN IBF v1 CRMA state.

For a date range, joins each daily BN result CSV to the ICPAC admin-1 GeoJSON
and plots the 4-state CRMA output (Monitor / Evaluate / Assess / Actionable_Risk)
with a traffic-light colormap (Green / Yellow / Orange / Red). Produces:
  - A combined 2×5 panel figure (output/flood_bn_v1_crma_maps_panel.png)
  - One PNG per day (output/maps/flood_bn_v1_crma_<date>.png)

Use --mode risk_level to plot the old 5-state risk_level output instead.
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
RISK_COLORS = {
    "Minimal":  "#1a9850",
    "Low":      "#d9ef8b",
    "Moderate": "#fee08b",
    "High":     "#f46d43",
    "Extreme":  "#a50026",
}

# CRMA 4-state traffic-light palette (WMO / FbF convention)
CRMA_ORDER = ["Monitor", "Evaluate", "Assess", "Actionable_Risk"]
CRMA_RANK = {k: i for i, k in enumerate(CRMA_ORDER)}
CRMA_COLORS = {
    "Monitor":         "#1a9850",  # Green
    "Evaluate":        "#fee08b",  # Yellow
    "Assess":          "#f46d43",  # Orange
    "Actionable_Risk": "#a50026",  # Red
}


def load_day(adm1: gpd.GeoDataFrame, result_csv: Path, mode: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(result_csv, keep_default_na=False, na_values=[""])
    joined = adm1.merge(df, left_on="GID_1", right_on="boundary_id", how="left")
    if mode == "crma_state":
        joined["rank"] = joined["crma_state"].map(CRMA_RANK)
    else:
        joined["rank"] = joined["risk_level"].map(RISK_RANK)
    return joined


def plot_day(ax, gdf: gpd.GeoDataFrame, title: str, cmap, norm) -> None:
    gdf.plot(
        ax=ax,
        column="rank",
        cmap=cmap,
        norm=norm,
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adm1", default="icpac_adm1v3.geojson")
    ap.add_argument("--input-dir", default="output")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-03-10")
    ap.add_argument("--mode", choices=["crma_state", "risk_level"], default="crma_state")
    ap.add_argument("--panel-out", default=None)
    ap.add_argument("--per-day-dir", default="output/maps")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    if args.mode == "crma_state":
        order, colors = CRMA_ORDER, CRMA_COLORS
        panel_default = "output/flood_bn_v1_crma_maps_panel.png"
        per_day_tag = "crma"
        title_label = "CRMA state (2-yr RP, C/L=0.20)"
    else:
        order, colors = RISK_ORDER, RISK_COLORS
        panel_default = "output/flood_bn_v1_risk_maps_panel.png"
        per_day_tag = "risk"
        title_label = "risk_level (2-yr RP, C/L=0.20)"
    cmap = ListedColormap([colors[k] for k in order])
    norm = BoundaryNorm(range(len(order) + 1), cmap.N)
    panel_out = args.panel_out or panel_default

    adm1 = gpd.read_file(args.adm1)
    dates = pd.date_range(args.start, args.end, freq="D")
    Path(args.per_day_dir).mkdir(parents=True, exist_ok=True)

    ncols = 5
    nrows = (len(dates) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.3))
    fig.suptitle(f"Flood BN IBF v1 — admin-1 daily {title_label}",
                 fontsize=14, y=0.985)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, d in enumerate(dates):
        ax = axes_flat[i]
        csv = Path(args.input_dir) / f"flood_bn_v1_{d.date()}.csv"
        if not csv.exists():
            ax.text(0.5, 0.5, f"missing\n{d.date()}", ha="center", va="center")
            ax.set_axis_off()
            continue
        day_gdf = load_day(adm1, csv, args.mode)
        plot_day(ax, day_gdf, title=str(d.date()), cmap=cmap, norm=norm)

        fig1, ax1 = plt.subplots(1, 1, figsize=(6.5, 6.0))
        plot_day(ax1, day_gdf, title=f"Flood {args.mode} — {d.date()}", cmap=cmap, norm=norm)
        patches = [mpatches.Patch(color=colors[k], label=k.replace("_", " ")) for k in order]
        ax1.legend(handles=patches, loc="lower left", fontsize=8, frameon=True, title=args.mode)
        out1 = Path(args.per_day_dir) / f"flood_bn_v1_{per_day_tag}_{d.date()}.png"
        fig1.tight_layout()
        fig1.savefig(out1, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig1)
        print(f"[map] wrote {out1}")

    for j in range(len(dates), len(axes_flat)):
        axes_flat[j].set_axis_off()

    patches = [mpatches.Patch(color=colors[k], label=k.replace("_", " ")) for k in order]
    fig.legend(handles=patches, loc="lower center", ncol=len(order),
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    Path(panel_out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[map] wrote panel → {panel_out}")


if __name__ == "__main__":
    main()
