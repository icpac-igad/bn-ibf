#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk==2.0.3",
#   "xarray",
#   "zarr>=3",
#   "numpy",
#   "pandas",
#   "geopandas",
#   "regionmask",
#   "fsspec",
#   "s3fs",
#   "scipy",
# ]
# ///
"""
CDI data prep — recompute the JRC-style Combined Drought Indicator from the
source.coop component stores and aggregate per ICPAC admin-1 boundary.

Reads (all anonymous on source.coop, us-west-2):
  - chirps_spi_icechunk        spi1, spi3, spi9 (or spi12)            monthly
  - gdo_sma_icechunk           smang (soil-moisture anomaly)          dekadal
  - gdo_fpar_icechunk          fpanv (vegetation anomaly)             dekadal

Implements `calculate_cdi(...)` per drought_crma/cdi-method.md, returning a
14-class integer CDI index plus a 6-level classification.

Output (CSV): one row per admin-1 boundary with columns
  id, name, country, target_date,
  cdi_class           int   1..14   (worst class touching the boundary)
  cdi_level           str   No_drought / Watch / Warning / Alert / Partial_recovery / Full_recovery
  cdi_level_idx       int   1..6   (1=No_drought, 6=Alert)
  cdi_level_fraction  float fraction of pixels at this level
  cdi_modal_class     int   1..14   (most common class — diagnostic)
  cdi_source          str   "recomputed"

Usage:
    uv run cdi_data_prep.py \\
        --date 2026-04 --adm1 icpac_adm1v3.geojson \\
        --out /tmp/cdi_inputs_2026-04.csv

The output CSV joins on `id` with the CSV from `drought_data_prep.py`.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import geopandas as gpd
import icechunk as ic
import numpy as np
import pandas as pd
import regionmask
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

S3_BUCKET            = "us-west-2.opendata.source.coop"
S3_REGION            = "us-west-2"
CHIRPS_SPI_PREFIX    = "e4drr-project/observations/chirps_spi_icechunk"
GDO_SMA_PREFIX       = "e4drr-project/observations/gdo_sma_icechunk"
GDO_FPAR_PREFIX      = "e4drr-project/observations/gdo_fpar_icechunk"

ISO_TO_COUNTRY = {
    "BDI": "Burundi", "DJI": "Djibouti", "ERI": "Eritrea", "ETH": "Ethiopia",
    "KEN": "Kenya", "RWA": "Rwanda", "SOM": "Somalia", "SSD": "South Sudan",
    "SDN": "Sudan", "TZA": "Tanzania", "UGA": "Uganda",
}

# CDI level numbering (1=least, 6=most stressed). Order matches the BN's
# CDI_STATES = ["No_drought", "Full_recovery", "Partial_recovery",
#               "Watch", "Warning", "Alert"]
LEVEL_TO_IDX = {
    "No_drought":       1,
    "Full_recovery":    2,
    "Partial_recovery": 3,
    "Watch":            4,
    "Warning":          5,
    "Alert":            6,
}
IDX_TO_LEVEL = {v: k for k, v in LEVEL_TO_IDX.items()}


# ─── store openers ───────────────────────────────────────────────────────────


def open_icechunk_anon(prefix: str) -> xr.Dataset:
    storage = ic.s3_storage(
        bucket=S3_BUCKET, prefix=prefix, region=S3_REGION, anonymous=True,
    )
    repo = ic.Repository.open(storage, config=ic.RepositoryConfig.default())
    return xr.open_zarr(
        repo.readonly_session("main").store, consolidated=False,
    )


# ─── calculate_cdi (port of cdi-method.md::calculate_cdi) ───────────────────


def calculate_cdi(
    spi9_12_lt_m1: bool, spi3_lt_m1: bool, spi1_lt_m2: bool,
    spi3_prev_lt_m1: bool, spi1_prev_lt_m2: bool,
    sma_lt_m1: bool, fapar_lt_m1: bool,
) -> tuple[int, str]:
    """Return (cdi_class 1..14, level_string) per drought_crma/cdi-method.md."""

    # Alert: precipitation shortage + vegetation anomaly
    if fapar_lt_m1 and sma_lt_m1 and spi9_12_lt_m1 and spi3_lt_m1:
        return 10, "Alert"
    if fapar_lt_m1 and sma_lt_m1 and spi3_lt_m1:
        return 9, "Alert"
    if fapar_lt_m1 and spi3_lt_m1:
        return 8, "Alert"
    if fapar_lt_m1 and spi1_lt_m2:
        return 7, "Alert"

    # Warning: precipitation shortage + soil moisture anomaly
    if sma_lt_m1 and spi9_12_lt_m1 and spi3_lt_m1:
        return 6, "Warning"
    if sma_lt_m1 and spi3_lt_m1:
        return 5, "Warning"
    if sma_lt_m1 and spi1_lt_m2:
        return 4, "Warning"

    # Watch: precipitation shortage only
    if spi9_12_lt_m1 and spi3_lt_m1:
        return 3, "Watch"
    if spi3_lt_m1:
        return 2, "Watch"
    if spi1_lt_m2:
        return 1, "Watch"

    # Partial recovery: previous precipitation deficit + vegetation anomaly
    if fapar_lt_m1 and spi3_prev_lt_m1:
        return 12, "Partial_recovery"
    if fapar_lt_m1 and spi1_prev_lt_m2:
        return 11, "Partial_recovery"

    # Full recovery: previous precipitation deficit only
    if spi3_prev_lt_m1:
        return 14, "Full_recovery"
    if spi1_prev_lt_m2:
        return 13, "Full_recovery"

    return 0, "No_drought"


def calculate_cdi_grid(
    spi9_12: np.ndarray, spi3: np.ndarray, spi1: np.ndarray,
    spi3_prev: np.ndarray, spi1_prev: np.ndarray,
    sma: np.ndarray, fapar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised pixel-wise CDI. All inputs are 2-D arrays on the same grid.

    Returns (cdi_class[H, W] uint8, cdi_level_idx[H, W] uint8). Pixels with any
    NaN among the inputs become class 0 / level 1 (No_drought) — the rule
    table says "no condition met" and we conservatively treat missing data
    as "no information" rather than bias toward severe.
    """
    # Boolean condition grids
    c_spi9_12 = np.isfinite(spi9_12) & (spi9_12 < -1.0)
    c_spi3    = np.isfinite(spi3)    & (spi3    < -1.0)
    c_spi1    = np.isfinite(spi1)    & (spi1    < -2.0)
    c_spi3_p  = np.isfinite(spi3_prev) & (spi3_prev < -1.0)
    c_spi1_p  = np.isfinite(spi1_prev) & (spi1_prev < -2.0)
    c_sma     = np.isfinite(sma)     & (sma     < -1.0)
    c_fapar   = np.isfinite(fapar)   & (fapar   < -1.0)

    h, w = spi3.shape
    cdi_class = np.zeros((h, w), dtype=np.uint8)
    cdi_level_idx = np.full((h, w), LEVEL_TO_IDX["No_drought"], dtype=np.uint8)

    # Apply rules in priority order — same as calculate_cdi() above.
    # Each rule writes into pixels that haven't matched a higher-priority rule.
    matched = np.zeros((h, w), dtype=bool)

    def assign(mask, cls: int, level: str):
        nonlocal matched
        sel = mask & ~matched
        cdi_class[sel] = cls
        cdi_level_idx[sel] = LEVEL_TO_IDX[level]
        matched |= sel

    # Alert
    assign(c_fapar & c_sma & c_spi9_12 & c_spi3, 10, "Alert")
    assign(c_fapar & c_sma & c_spi3,              9, "Alert")
    assign(c_fapar & c_spi3,                      8, "Alert")
    assign(c_fapar & c_spi1,                      7, "Alert")
    # Warning
    assign(c_sma & c_spi9_12 & c_spi3,            6, "Warning")
    assign(c_sma & c_spi3,                        5, "Warning")
    assign(c_sma & c_spi1,                        4, "Warning")
    # Watch
    assign(c_spi9_12 & c_spi3,                    3, "Watch")
    assign(c_spi3,                                2, "Watch")
    assign(c_spi1,                                1, "Watch")
    # Partial recovery
    assign(c_fapar & c_spi3_p,                    12, "Partial_recovery")
    assign(c_fapar & c_spi1_p,                    11, "Partial_recovery")
    # Full recovery
    assign(c_spi3_p,                              14, "Full_recovery")
    assign(c_spi1_p,                              13, "Full_recovery")

    return cdi_class, cdi_level_idx


# ─── slice helpers ───────────────────────────────────────────────────────────


def latest_le_idx(times: np.ndarray, target: pd.Timestamp) -> int:
    """Index of the most recent timestamp ≤ target."""
    ts = pd.to_datetime(times)
    upper_mask = ts <= target
    if not upper_mask.any():
        raise SystemExit(f"No time slice ≤ {target.date()} in dataset")
    return int(np.where(upper_mask)[0].max())


def previous_month_idx(times: np.ndarray, latest_idx: int) -> int:
    """Index roughly 1 month before times[latest_idx]."""
    ts = pd.to_datetime(times)
    target = ts[latest_idx] - pd.DateOffset(months=1)
    upper_mask = ts <= target
    if not upper_mask.any():
        return latest_idx
    return int(np.where(upper_mask)[0].max())


def regrid_to(
    da_src: xr.DataArray, lat_target: xr.DataArray, lon_target: xr.DataArray,
) -> xr.DataArray:
    lat_asc = np.sort(lat_target.values)
    lon_asc = np.sort(lon_target.values)
    interp = da_src.interp(lat=lat_asc, lon=lon_asc, method="nearest")
    return interp.reindex(lat=lat_target.values, lon=lon_target.values)


# ─── adm1 zonal aggregation ──────────────────────────────────────────────────


def build_mask(
    gdf: gpd.GeoDataFrame, lat: xr.DataArray, lon: xr.DataArray,
) -> xr.DataArray:
    regions = regionmask.Regions(
        outlines=list(gdf.geometry), numbers=list(range(len(gdf))),
        names=list(gdf["NAME_1"]), abbrevs=list(gdf["GID_1"]),
        name="adm1",
    )
    return regions.mask(lon, lat)


def aggregate_per_boundary(
    cdi_class: np.ndarray, cdi_level_idx: np.ndarray,
    mask_arr: np.ndarray, n_regions: int,
    gdf: gpd.GeoDataFrame, lat_vals: np.ndarray, lon_vals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each boundary return (max_class, max_level_idx, modal_class, level_fraction)."""
    max_class    = np.zeros(n_regions, dtype=np.int16)
    max_level    = np.full(n_regions, LEVEL_TO_IDX["No_drought"], dtype=np.int16)
    modal_class  = np.zeros(n_regions, dtype=np.int16)
    level_frac   = np.zeros(n_regions, dtype=np.float64)

    for r in range(n_regions):
        sel = mask_arr == r
        if not sel.any():
            # Centroid fallback
            pt = gdf.iloc[r].geometry.centroid
            i = int(np.argmin(np.abs(lat_vals - pt.y)))
            j = int(np.argmin(np.abs(lon_vals - pt.x)))
            max_class[r]   = int(cdi_class[i, j])
            max_level[r]   = int(cdi_level_idx[i, j])
            modal_class[r] = max_class[r]
            level_frac[r]  = 1.0
            continue

        cls_pixels = cdi_class[sel]
        lvl_pixels = cdi_level_idx[sel]

        max_level[r] = int(lvl_pixels.max())
        # Worst class within the worst level
        in_max = lvl_pixels == max_level[r]
        max_class[r] = int(cls_pixels[in_max].max()) if in_max.any() else 0
        # Modal class (most frequent)
        vals, counts = np.unique(cls_pixels, return_counts=True)
        modal_class[r] = int(vals[np.argmax(counts)])
        # Fraction of pixels at the max level
        level_frac[r] = float(in_max.sum()) / float(sel.sum())

    return max_class, max_level, modal_class, level_frac


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target month (YYYY-MM or YYYY-MM-DD)")
    ap.add_argument("--adm1", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--spi-long", default="spi9", choices=["spi9", "spi12"],
                    help="Long-window SPI variable name (default spi9)")
    args = ap.parse_args()

    D = pd.Timestamp(args.date).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    print(f"[cdi-prep] target month: {D.date()}  long-SPI: {args.spi_long}", flush=True)

    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    n_adm = len(adm1)
    print(f"[cdi-prep] adm1: {n_adm} boundaries", flush=True)

    # ── Open the three component stores ─────────────────────────────────────
    print("[cdi-prep] opening CHIRPS SPI icechunk ...", flush=True)
    spi_ds = open_icechunk_anon(CHIRPS_SPI_PREFIX)
    print("[cdi-prep] opening GDO SMA icechunk ...", flush=True)
    sma_ds = open_icechunk_anon(GDO_SMA_PREFIX)
    print("[cdi-prep] opening GDO fAPAR icechunk ...", flush=True)
    fp_ds  = open_icechunk_anon(GDO_FPAR_PREFIX)

    # ── Pick the right time-step indices (isel to avoid fuzzy .sel matches) ─
    spi_idx      = latest_le_idx(spi_ds.time.values, D)
    spi_prev_idx = previous_month_idx(spi_ds.time.values, spi_idx)
    sma_idx      = latest_le_idx(sma_ds.time.values, D)
    fp_idx       = latest_le_idx(fp_ds.time.values, D)
    spi_t      = pd.Timestamp(spi_ds.time.values[spi_idx])
    spi_t_prev = pd.Timestamp(spi_ds.time.values[spi_prev_idx])
    sma_t      = pd.Timestamp(sma_ds.time.values[sma_idx])
    fp_t       = pd.Timestamp(fp_ds.time.values[fp_idx])
    print(f"[cdi-prep] time slices: SPI={spi_t.date()} prev={spi_t_prev.date()}  "
          f"SMA={sma_t.date()}  fAPAR={fp_t.date()}", flush=True)

    # ── Load slices via isel for unique selection ──────────────────────────
    spi3      = spi_ds.spi3.isel(time=spi_idx).load()
    spi1      = spi_ds.spi1.isel(time=spi_idx).load()
    spi9_12   = spi_ds[args.spi_long].isel(time=spi_idx).load()
    spi3_prev = spi_ds.spi3.isel(time=spi_prev_idx).load()
    spi1_prev = spi_ds.spi1.isel(time=spi_prev_idx).load()
    sma       = sma_ds.smang.isel(time=sma_idx).load()
    fapar     = fp_ds.fpanv.isel(time=fp_idx).load()

    # ── Regrid SMA + fAPAR to CHIRPS SPI grid ─────────────────────────────
    print("[cdi-prep] regridding SMA + fAPAR to CHIRPS grid ...", flush=True)
    sma_rg   = regrid_to(sma,   spi3.lat, spi3.lon)
    fapar_rg = regrid_to(fapar, spi3.lat, spi3.lon)

    # ── Compute CDI ─────────────────────────────────────────────────────────
    print("[cdi-prep] computing CDI grid ...", flush=True)
    cdi_class, cdi_level_idx = calculate_cdi_grid(
        spi9_12.values, spi3.values, spi1.values,
        spi3_prev.values, spi1_prev.values,
        sma_rg.values, fapar_rg.values,
    )
    print(f"[cdi-prep] CDI class distribution (pixels): "
          f"{dict(zip(*np.unique(cdi_class, return_counts=True)))}", flush=True)
    print(f"[cdi-prep] CDI level distribution (pixels): "
          f"{ {IDX_TO_LEVEL[i]: int(c) for i, c in zip(*np.unique(cdi_level_idx, return_counts=True))} }",
          flush=True)

    # ── Build mask + aggregate per boundary ─────────────────────────────────
    print("[cdi-prep] building admin-1 mask ...", flush=True)
    mask = build_mask(adm1, spi3.lat, spi3.lon)
    mask_arr = mask.values

    print("[cdi-prep] aggregating per boundary ...", flush=True)
    max_class, max_level, modal_class, level_frac = aggregate_per_boundary(
        cdi_class, cdi_level_idx, mask_arr, n_adm, adm1,
        spi3.lat.values, spi3.lon.values,
    )

    country = (adm1["GID_1"].str.split(".").str[0]
               .map(ISO_TO_COUNTRY).fillna("Unknown"))
    df = pd.DataFrame({
        "id":                adm1["GID_1"],
        "name":              adm1["NAME_1"],
        "country":           country,
        "target_date":       str(D.date()),
        "cdi_class":         max_class,
        "cdi_level_idx":     max_level,
        "cdi_level":         [IDX_TO_LEVEL[int(i)] for i in max_level],
        "cdi_level_fraction": np.round(level_frac, 4),
        "cdi_modal_class":   modal_class,
        "cdi_source":        "recomputed",
        "spi_time":          str(spi_t.date()),
        "sma_time":          str(sma_t.date()),
        "fapar_time":        str(fp_t.date()),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    counts = df["cdi_level"].value_counts()
    print(f"[cdi-prep] wrote {out}  rows={len(df)}", flush=True)
    print(f"[cdi-prep] per-boundary CDI level: {counts.to_dict()}", flush=True)


if __name__ == "__main__":
    main()
