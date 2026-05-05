#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "geopandas",
#   "pandas",
#   "matplotlib",
# ]
# ///
"""
Choropleth grid for drought BN IBF results.

Reads the BN result CSVs written by drought_bn_ibf_v1.jl
(`/tmp/drought_bn_v1_YYYY-MM.csv` — 227 admin-1 boundaries × CRMA state +
risk posterior) and the ICPAC adm1 GeoJSON, joins by `id`/`GID_1`, and
plots one panel per init-month coloured by `traffic_light`
(Green / Yellow / Orange / Red), with a per-panel CRMA-state count.

Usage:
    uv run plot_drought_bn_choropleth.py \\
        --pattern '/tmp/drought_bn_v1_*.csv' \\
        --adm1 /scratch/notebook/bn-ibf/drought_crma/icpac_adm1v3.geojson \\
        --out /tmp/drought_bn_choropleth.png
"""
import argparse
import glob
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

TRAFFIC_COLORS = {
    "Green":  "#4caf50",
    "Yellow": "#ffeb3b",
    "Orange": "#ff9800",
    "Red":    "#f44336",
}
CRMA_ORDER = ["Monitor", "Evaluate", "Assess", "Actionable_Risk"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="/tmp/drought_bn_v1_*.csv")
    ap.add_argument("--adm1", required=True)
    ap.add_argument("--out", default="/tmp/drought_bn_choropleth.png")
    ap.add_argument("--cols", type=int, default=3, help="grid columns (default 3)")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    # Sort CSVs by date encoded in filename (drought_bn_v1_2025-04.csv etc.)
    files = sorted(
        glob.glob(args.pattern),
        key=lambda p: re.search(r"(\d{4}-\d{2})", p).group(1),
    )
    if not files:
        raise SystemExit(f"No files matched {args.pattern}")
    print(f"[plot] {len(files)} CSVs:")
    for f in files:
        print(f"  {f}")

    gdf = gpd.read_file(args.adm1)
    print(f"[plot] adm1: {len(gdf)} polygons")

    n = len(files)
    cols = args.cols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, f in enumerate(files):
        ax = axes[i]
        date = re.search(r"(\d{4}-\d{2})", f).group(1)
        df = pd.read_csv(f)
        # join: BN result column 'boundary_id' ↔ GeoJSON 'GID_1'
        merged = gdf.merge(df, left_on="GID_1", right_on="boundary_id", how="left")
        # Fall back to a neutral grey for any unmatched polygon (ocean / extra island)
        merged["_color"] = merged["traffic_light"].map(TRAFFIC_COLORS).fillna("#cccccc")
        merged.plot(color=merged["_color"], edgecolor="black",
                    linewidth=0.3, ax=ax)
        counts = df["crma_state"].value_counts().reindex(CRMA_ORDER, fill_value=0)
        title = (
            f"init {date}\n"
            f"Mon={counts['Monitor']}  Eva={counts['Evaluate']}  "
            f"Ass={counts['Assess']}  Act={counts['Actionable_Risk']}"
        )
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Single legend
    handles = [mpatches.Patch(color=TRAFFIC_COLORS[c], label=lab)
               for c, lab in zip(["Green", "Yellow", "Orange", "Red"],
                                  CRMA_ORDER)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle("Drought BN IBF — CRMA traffic light per admin-1, by init month",
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.985])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot] saved {out}  ({rows}×{cols} grid, {n} panels)")


if __name__ == "__main__":
    main()
