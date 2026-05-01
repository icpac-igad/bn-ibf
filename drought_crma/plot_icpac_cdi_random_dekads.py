#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=0.2.0",
#   "xarray",
#   "zarr>=3",
#   "numpy",
#   "pandas",
#   "matplotlib",
#   "geopandas",
#   "pyogrio",
#   "shapely",
# ]
# ///
"""
Pick N random dekads from the icpac_cdi_dekadal_icechunk store on source.coop,
plot each as a CDI map with the ICPAC admin-1 polygon outlines overlaid as
thin black lines.

Reads the store anonymously — works as soon as the relevant icechunk
commits land on source.coop (no creds needed). Ignores any AWS_* in the
caller's environment so a stale .env does not interfere.

Usage:
    ./plot_icpac_cdi_random_dekads.py [-n 5] [--seed 0]
        [--adm1 ../drought_ibf/icpac_adm1v3.geojson]
        [--out icpac_cdi_random_dekads.png]
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

import icechunk

DEFAULT_BUCKET = "us-west-2.opendata.source.coop"
DEFAULT_PREFIX = "e4drr-project/observations/icpac_cdi_dekadal_icechunk"
DEFAULT_REGION = "us-west-2"


# JRC EADW CDI 14-class palette (Watch / Warning / Alert / Recovery / etc.)
# Index 0 reserved for "no data"; 1..14 mirror cdi-method.md.
CDI_COLORS = {
    0:  "#cccccc",  # No data / unclassified
    1:  "#ffff00",  # Watch — SPI1<-2 only
    2:  "#ffff00",  # Watch
    3:  "#ffff00",  # Watch
    4:  "#ffa600",  # Warning — + soil moisture deficit
    5:  "#ffa600",  # Warning
    6:  "#ffa600",  # Warning
    7:  "#fe0000",  # Alert — + vegetation anomaly
    8:  "#fe0000",  # Alert
    9:  "#fe0000",  # Alert
    10: "#fe0000",  # Alert
    11: "#9f8001",  # Partial recovery
    12: "#9f8001",  # Partial recovery
    13: "#9ec75f",  # Full recovery
    14: "#f5f5f5",  # No drought
}


def open_store_anonymous(bucket: str, prefix: str, region: str) -> xr.Dataset:
    # Hide any STS creds the user may have in .env / shell — anonymous read
    # bypasses the expired-token issue.
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN", "AWS_DEFAULT_REGION"):
        os.environ.pop(key, None)
    storage = icechunk.s3_storage(
        bucket=bucket, prefix=prefix, region=region, anonymous=True,
    )
    repo = icechunk.Repository.open(storage=storage)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5, help="Number of random dekads (default 5)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--adm1", default="../drought_ibf/icpac_adm1v3.geojson",
                    help="Admin-1 GeoJSON to overlay")
    ap.add_argument("--out", default="icpac_cdi_random_dekads.png",
                    help="Output panel PNG (default cwd)")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    print(f"[plot] opening icechunk store anonymously: s3://{args.bucket}/{args.prefix}")
    ds = open_store_anonymous(args.bucket, args.prefix, args.region)
    times = pd.to_datetime(ds.time.values)
    print(f"[plot] store: {len(times)} dekads "
          f"({times[0].date()} .. {times[-1].date()})")

    rng = random.Random(args.seed)
    picks = sorted(rng.sample(range(len(times)), k=min(args.n, len(times))))
    pick_dates = [times[i] for i in picks]
    print(f"[plot] random picks: {[d.date() for d in pick_dates]}")

    print(f"[plot] reading admin-1 boundaries: {args.adm1}")
    adm1 = gpd.read_file(args.adm1)
    print(f"[plot] adm1: {len(adm1)} polygons, crs={adm1.crs}")

    # Discrete colormap covering 0..14
    classes = list(range(15))
    colors = [CDI_COLORS[c] for c in classes]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 15.5, 1.0), cmap.N)

    n = len(picks)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 5.5, nrows * 5.0), squeeze=False,
    )
    fig.suptitle(
        "ICPAC EADW dekadal CDI — random samples from "
        f"icpac_cdi_dekadal_icechunk ({len(times)} dekads available)",
        fontsize=12, y=0.995,
    )

    lat = ds.lat.values
    lon = ds.lon.values
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    print(f"[plot] grid extent: lon[{lon.min():.2f}..{lon.max():.2f}] "
          f"lat[{lat.min():.2f}..{lat.max():.2f}] "
          f"({len(lat)}×{len(lon)} pixels)")

    for k, (t_idx, t) in enumerate(zip(picks, pick_dates)):
        r, c = divmod(k, ncols)
        ax = axes[r][c]
        print(f"[plot]   loading CDI t={t.date()} (idx {t_idx}) ...", flush=True)
        arr = ds.cdi.isel(time=t_idx).load().values
        # Some EADW CDI files have lat in descending order — that flips the
        # array under imshow with origin='lower'. Normalise.
        if lat[0] > lat[-1]:
            arr_disp = arr[::-1, :]
        else:
            arr_disp = arr
        ax.imshow(arr_disp, extent=extent, origin="lower",
                  cmap=cmap, norm=norm, interpolation="nearest")
        adm1.boundary.plot(ax=ax, color="black", linewidth=0.4)
        ax.set_title(f"{t.date()}  (n_pixels={arr.size})", fontsize=10)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    for k in range(n, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_axis_off()

    # One shared discrete legend across the panel
    legend_classes = [
        (1, "Watch (1-3)"),
        (4, "Warning (4-6)"),
        (7, "Alert (7-10)"),
        (11, "Partial recovery (11-12)"),
        (13, "Full recovery (13)"),
        (14, "No drought (14)"),
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CDI_COLORS[c], label=lbl)
        for c, lbl in legend_classes
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot] saved → {args.out}")


if __name__ == "__main__":
    main()
