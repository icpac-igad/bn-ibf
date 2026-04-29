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
#   "pyarrow",
#   "scipy",
#   "fsspec",
#   "s3fs",
# ]
# ///
"""
Drought BN IBF v1 — per-month admin-1 input generator.

Drought analogue of flood_data_prep.py. Reads:
  - ERA5 SPI pencil zarr (observations; SPI1..SPI48 monthly, 1940-now)
  - SEAS5 SPI-3 pencil zarr (51-member forecast; 6 lead months)
  - ERA5 SPI return-period icechunk (fitted normal SPI thresholds, 5 RPs)
  - ICPAC admin-1 GeoJSON

Writes a CSV with one row per admin-1 boundary holding the evidence vector
consumed by drought_bn_ibf_v1.py / .jl. Schema is parallel to the flood
prep with drought semantics:

    id, name, country,
    current_spi3, current_spi3_category,            <- antecedent analogue
    spi3_trend, trend_slope_spi_per_month,
    forecast_deficit_prob,                          <- exceedance analogue
    deficit_prob_lead1,
    spatial_coverage,
    spatial_cov_mean_p, hotspot_fraction,
    forecast_agreement,
    ens_min_spi,                                    <- p5 of ens-min SPI per pixel
    ens_min_spi_mean, ens_min_spi_peak,
    ens_mean_lead1_spi, ens_min_lead1_spi, ens_max_lead1_spi,
    target_date

Bin / threshold conventions follow ECMWF SPI return-period semantics:
  • Drought bins use SPI cutoffs at -1.5/-1.0/-0.5/+0.5
  • A "deficit" event is forecast SPI ≤ -1.0 (matches McKee moderate drought)
  • Trend is slope of monthly SPI over the last `--trend-months` months,
    band ±0.3 SPI/month

Differences from flood_data_prep.py:
  • Time unit: month (not day)
  • Threshold direction: SPI ≤ RP (drought) vs TP ≥ RP (flood)
  • Forecast forms 6 lead months × 51 members; ensemble-min is the
    "worst-case" (analogue of ensemble-max for floods)
  • RP is per-pixel SPI (already standardized) — no mm conversion.
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

# ─── source.coop layout ──────────────────────────────────────────────────────
S3_BUCKET                = "us-west-2.opendata.source.coop"
S3_REGION                = "us-west-2"
SPI_OBS_PREFIX           = "e4drr-project/observations/era5_ecmwf_pencil"           # plain zarr
SPI_RP_PREFIX            = "e4drr-project/observations/era5_ecmwf_rp_icechunk"       # icechunk
SEAS5_SPI3_PREFIX        = "e4drr-project/forecasts/seas51_spi3_10km_icechunk_v2"    # slab icechunk
                                                                                     # (NOT the pencil zarr; pencil is full-time per pixel,
                                                                                     #  bad for 'single-init across full grid' queries)

ISO_TO_COUNTRY = {
    "BDI": "Burundi", "DJI": "Djibouti", "ERI": "Eritrea", "ETH": "Ethiopia",
    "KEN": "Kenya", "RWA": "Rwanda", "SOM": "Somalia", "SSD": "South Sudan",
    "SDN": "Sudan", "TZA": "Tanzania", "UGA": "Uganda",
}

# ─── Seasonal mapping (v2) ───────────────────────────────────────────────────
# East-Africa SPI-3 seasons. SPI-3 at month X is the 3-month mean ending at
# X, so the season's "anchor" month is the *last* month of the season.
# Source: drought_crma/itt-seasonal-docs.rst (lines 72-112).

SEASON_ANCHOR_MONTH = {
    "MAM": 5,   # Mar-Apr-May, anchor = May
    "JJA": 8,   # Jun-Jul-Aug, anchor = Aug
    "OND": 11,  # Sep-Oct-Nov, anchor = Nov
    "DJF": 2,   # Dec-Jan-Feb, anchor = Feb (next year)
}

# Default valid (init_month, target_season) pairs. SEAS5 has 6 leads (1..6),
# so a season is reachable iff the anchor is 1..6 months ahead. The doc's
# operational table uses leads 2/3/4 only (drops lead 5 and lead 6 as too
# uncertain); we accept any lead in 1..6 with a warning when outside 2..4.
SEASON_INIT_LEAD = {
    "MAM": {12: 5, 1: 4, 2: 3},          # doc table
    "JJA": {3: 5, 4: 4, 5: 3},
    "OND": {6: 5, 7: 4, 8: 3},
    "DJF": {9: 5, 10: 4, 11: 3},
}

# Drought-specific SPI threshold ("forecast deficit" = forecast SPI ≤ this).
# -1.0 = McKee moderate drought; tunable via --deficit-spi.
DEFAULT_DEFICIT_SPI = -1.0

# Soft-evidence binning for the BN (mirrors the flood _NODE_EDGES dict
# structure but with drought-side cutoffs).  Bin order is increasing on
# the underlying axis. For SPI categories, lower SPI = more severe drought.
_NODE_EDGES = {
    # current_spi3 categories (5): Severe / Moderate / Mild / Normal / Above
    "cur":  [-np.inf, -1.5, -1.0, -0.5, 0.5, np.inf],
    # deficit-prob categories (5): Very_Low … Very_High
    "def":  [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
    # spatial-coverage (3): Localized / Moderate / Widespread
    "spa":  [-np.inf, 0.3, 0.6, np.inf],
    # SPI trend (3): Deteriorating / Stable / Improving
    # bins on slope_spi_per_month; -0.3..+0.3 → Stable
    "trn":  [-np.inf, -0.3, 0.3, np.inf],
    # tail (4): worst-case ens-min SPI per boundary.
    # bins ordered low→high SPI = severe→nil drought.
    # state mapping: <-1.5: High, -1.5..-1.0: Moderate, -1.0..-0.5: Low, >-0.5: Nil
    "tail": [-np.inf, -1.5, -1.0, -0.5, np.inf],
}
_NODE_SIGMA_DEFAULT = {"cur": 0.3, "def": 0.05, "spa": 0.05, "trn": 0.1, "tail": 0.2}


# ─── store openers ───────────────────────────────────────────────────────────


def open_zarr_anon(prefix: str) -> xr.Dataset:
    """Open a public zarr store on source.coop anonymously."""
    return xr.open_zarr(
        f"s3://{S3_BUCKET}/{prefix}",
        storage_options={"anon": True},
        consolidated=True,
        decode_timedelta=True,
    )


def open_icechunk_anon(prefix: str) -> xr.Dataset:
    storage = ic.s3_storage(
        bucket=S3_BUCKET, prefix=prefix, region=S3_REGION, anonymous=True,
    )
    repo = ic.Repository.open(storage, config=ic.RepositoryConfig.default())
    return xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False, decode_timedelta=True,
    )


# ─── helpers ─────────────────────────────────────────────────────────────────


# Nodes whose soft-evidence column order must be REVERSED to match the
# drought-side Julia STATES (which list low-stress at idx 1, high-stress at
# idx K). _NODE_EDGES uses physical-axis order (increasing SPI for cur/tail,
# increasing slope for trn) — for SPI-style axes that means low-stress is
# at the *high* end, so the natural CDF-bin order needs to flip.
_REVERSE_NODES = {"cur", "tail", "trn"}


def soft_bin(x: float, node: str, sigma: float | None = None) -> np.ndarray:
    from scipy import stats as _st
    edges = _NODE_EDGES[node]
    k = len(edges) - 1
    if not np.isfinite(x):
        return np.full(k, 1.0 / k)
    s = _NODE_SIGMA_DEFAULT[node] if sigma is None else sigma
    probs = np.diff(_st.norm.cdf(edges, loc=x, scale=s))
    if node in _REVERSE_NODES:
        probs = probs[::-1]
    tot = probs.sum()
    return probs / tot if tot > 0 else np.full(k, 1.0 / k)


def add_soft_columns(df: pd.DataFrame, cur_spi: np.ndarray, def_p: np.ndarray,
                     spa: np.ndarray, trn_slope: np.ndarray,
                     tail_spi: np.ndarray) -> None:
    """In-place: add 5+5+3+3+4=20 soft-evidence columns."""
    blocks = [("cur", cur_spi, 5), ("def", def_p, 5), ("spa", spa, 3),
              ("trn", trn_slope, 3), ("tail", tail_spi, 4)]
    for node, vals, k in blocks:
        probs = np.vstack([soft_bin(float(v), node) for v in vals])
        for i in range(k):
            df[f"{node}_p{i+1}"] = np.round(probs[:, i], 4)


def lead_for_season(init_month: int, target_season: str) -> int:
    """Return the 1-based SEAS5 lead index pointing at the season's anchor
    month from the given init month, or raise if not reachable in 1..6 leads.
    """
    if target_season not in SEASON_ANCHOR_MONTH:
        raise ValueError(f"Unknown season: {target_season} "
                         f"(expected {list(SEASON_ANCHOR_MONTH)})")
    anchor = SEASON_ANCHOR_MONTH[target_season]
    lead = ((anchor - init_month) % 12) or 12
    if not 1 <= lead <= 6:
        raise SystemExit(
            f"[prep] {target_season} from init month {init_month} requires "
            f"lead {lead}, but SEAS5 only has leads 1..6. Pick another init.")
    if lead == 6:
        print(f"[prep] WARNING: {target_season} from init {init_month} uses "
              f"lead 6 (max horizon, large forecast uncertainty)")
    return lead


def categorize_current_spi(spi: float) -> str:
    if not np.isfinite(spi):
        return "Unknown"
    if spi < -1.5: return "Severe_Drought"
    if spi < -1.0: return "Moderate_Drought"
    if spi < -0.5: return "Mild_Drought"
    if spi <  0.5: return "Normal"
    return "Above_Normal"


def classify_trend(slope: float, band: float) -> str:
    """For SPI: positive slope = improving (less drought), negative = worsening."""
    if not np.isfinite(slope):
        return "Stable"
    if slope > band:
        return "Improving"
    if slope < -band:
        return "Deteriorating"
    return "Stable"


# ─── obs loading (ERA5 SPI3) ─────────────────────────────────────────────────


def load_obs_spi3_window(obs: xr.Dataset, target: pd.Timestamp,
                         lookback_months: int) -> xr.DataArray:
    """Last `lookback_months` of monthly SPI3 ending at `target` (inclusive
    of the latest available month ≤ target).
    """
    spi3 = obs.SPI3
    times = pd.to_datetime(spi3.time.values)
    upper = times[times <= target]
    if len(upper) == 0:
        raise SystemExit(f"[prep] no ERA5 SPI3 obs at or before {target.date()}")
    end = upper.max()
    start = end - pd.DateOffset(months=lookback_months - 1)
    print(f"[prep] obs SPI3 window: {start.date()} .. {end.date()} "
          f"({lookback_months} months)")
    return spi3.sel(time=slice(start, end)).load()


# ─── forecast loading (SEAS5 SPI3) ───────────────────────────────────────────


def load_forecast_spi3(forecast: xr.Dataset, target: pd.Timestamp) -> xr.DataArray:
    """Forecast SPI3 (lead × member × lat × lon) for the init at `target`,
    or the latest init <= target if target is not present.

    Loads lead-by-lead and stitches into a numpy-backed DataArray. This
    avoids the chunk-decompression memory spike that crashes 8 GB VMs
    when the full (6, 51, 351, 321) slab is requested in one .load().
    """
    inits = pd.to_datetime(forecast.init.values)
    upper = inits[inits <= target]
    if len(upper) == 0:
        raise SystemExit(f"[prep] no SEAS5 init at or before {target.date()} "
                         f"(forecast inits {inits.min().date()}..{inits.max().date()})")
    chosen = upper.max()
    if chosen != target:
        print(f"[prep] note: target={target.date()}, using nearest "
              f"SEAS5 init={chosen.date()}")
    da = forecast.spi3.sel(init=chosen)  # (lead, member, lat, lon), still lazy
    leads = da.lead.values
    parts = []
    for lv in leads:
        chunk = da.sel(lead=lv).load()  # ~22 MB each (51 × 351 × 321 × 4)
        parts.append(chunk)
        print(f"  loaded lead={int(lv)}: shape={chunk.shape}, "
              f"mem={chunk.nbytes/1024**2:.0f} MB", flush=True)
    return xr.concat(parts, dim="lead")


# ─── RP loading ──────────────────────────────────────────────────────────────


def load_rp_thresholds(rp: xr.Dataset, rp_year: int,
                       spi_period: str = "SPI3",
                       prefer: str = "fitted") -> xr.DataArray:
    """Per-pixel SPI return-period threshold (negative). prefer='fitted' uses
    the per-pixel fitted-normal threshold; 'standard' uses the theoretical
    Φ⁻¹(1/T) (constant across pixels)."""
    if prefer == "fitted":
        da = rp.fitted_threshold.sel(spi_period=spi_period, return_period=rp_year)
    elif prefer == "empirical":
        da = rp.empirical_threshold.sel(spi_period=spi_period, return_period=rp_year)
    else:
        std = float(rp.standard_threshold.sel(return_period=rp_year).values)
        da = xr.full_like(rp.fitted_threshold.sel(spi_period=spi_period,
                                                  return_period=rp_year), std)
    # Ensure ascending lat/lon for downstream interp
    if float(da.lat[0]) > float(da.lat[-1]):
        da = da.isel(lat=slice(None, None, -1))
    if float(da.lon[0]) > float(da.lon[-1]):
        da = da.isel(lon=slice(None, None, -1))
    return da.load()


# ─── grid helpers ────────────────────────────────────────────────────────────


def regrid_to(da_src: xr.DataArray, lat_target: xr.DataArray,
              lon_target: xr.DataArray) -> xr.DataArray:
    lat_asc = np.sort(lat_target.values)
    lon_asc = np.sort(lon_target.values)
    interp = da_src.interp(lat=lat_asc, lon=lon_asc, method="nearest")
    return interp.reindex(lat=lat_target.values, lon=lon_target.values)


def build_mask(gdf: gpd.GeoDataFrame, lat: xr.DataArray, lon: xr.DataArray) -> xr.DataArray:
    regions = regionmask.Regions(
        outlines=list(gdf.geometry),
        numbers=list(range(len(gdf))),
        names=list(gdf["NAME_1"]),
        abbrevs=list(gdf["GID_1"]),
        name="adm1",
    )
    return regions.mask(lon, lat)


def zonal_reduce(da: xr.DataArray, mask: xr.DataArray, lat: xr.DataArray,
                 n_regions: int, thresh: float | None = None,
                 below: bool = False) -> np.ndarray:
    """Area-weighted mean per region; if `thresh` given, fraction of pixels
    where da ≤ thresh (`below=True`) or da ≥ thresh (`below=False`)."""
    weights = np.cos(np.deg2rad(lat))
    w2d = weights.broadcast_like(da)
    if thresh is not None:
        src = ((da <= thresh) if below else (da >= thresh)).astype("float32")
    else:
        src = da
    valid = (~da.isnull()).astype("float32")
    mask_vals = mask.values
    src_vals = src.values
    w_vals = w2d.values
    v_vals = valid.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        w = w_vals[sel] * v_vals[sel]
        den = w.sum()
        if den <= 0:
            continue
        num = float((src_vals[sel] * w).sum())
        out[r] = num / float(den)
    return out


def zonal_quantile(da: xr.DataArray, mask: xr.DataArray, n_regions: int,
                   q: float = 0.05) -> np.ndarray:
    """Per-region q-th quantile (unweighted). Default q=0.05 = drought-side."""
    mask_vals = mask.values
    vals = da.values
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        v = vals[sel]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[r] = float(np.quantile(v, q))
    return out


def zonal_extreme(da: xr.DataArray, mask: xr.DataArray, n_regions: int,
                  agg: str = "min") -> np.ndarray:
    """Per-region min/max of pixel values."""
    mask_vals = mask.values
    vals = da.values
    fn = np.min if agg == "min" else np.max
    out = np.full(n_regions, np.nan, dtype=np.float64)
    for r in range(n_regions):
        sel = mask_vals == r
        if not sel.any():
            continue
        v = vals[sel]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        out[r] = float(fn(v))
    return out


def fill_small_boundaries(values: np.ndarray, da: xr.DataArray,
                          gdf: gpd.GeoDataFrame, thresh: float | None = None,
                          below: bool = False) -> np.ndarray:
    """For boundaries with no pixel hit, sample nearest pixel at centroid."""
    out = values.copy()
    missing = np.where(np.isnan(out))[0]
    if len(missing) == 0:
        return out
    cent = gdf.iloc[missing].geometry.centroid
    if thresh is not None:
        src = ((da <= thresh) if below else (da >= thresh)).astype("float32")
    else:
        src = da
    for pos, (i, pt) in enumerate(zip(missing, cent)):
        try:
            val = float(src.sel(lat=pt.y, lon=pt.x, method="nearest").values)
        except Exception:
            val = np.nan
        out[i] = val
    return out


# ─── per-member sidecar ──────────────────────────────────────────────────────


def compute_per_member_evidence(
    fcst: xr.DataArray,            # (lead, member, lat, lon) on RP grid
    rp_thresh: xr.DataArray,       # (lat, lon)
    mask: xr.DataArray,
    adm1: gpd.GeoDataFrame,
    n_regions: int,
    cur_spi: np.ndarray,
    slopes: np.ndarray,
    trend_band: float,
    deficit_spi: float,
    target_date: str,
    soft: bool = False,
) -> pd.DataFrame:
    """Per-member storyline evidence for storyline BN runs."""
    members = fcst.member.values
    n_mem   = len(members)
    n_lat   = fcst.sizes["lat"]
    n_lon   = fcst.sizes["lon"]
    mask_vals = mask.values

    # Per-member, per-pixel: min-over-leads SPI (worst-case across leads)
    per_member_min_spi = fcst.min(dim="lead").values            # (member, lat, lon)
    # Per-member, per-pixel: any lead crosses RP (binary deficit indicator)
    per_member_deficit = (fcst <= rp_thresh).any(dim="lead").astype("float32").values

    rows = []
    for r_idx in range(n_regions):
        sel = mask_vals == r_idx
        gid = adm1.iloc[r_idx]["GID_1"]
        nm  = adm1.iloc[r_idx]["NAME_1"]
        cc  = ISO_TO_COUNTRY.get(gid.split(".")[0], "Unknown")
        cur_val = float(cur_spi[r_idx])
        slope_val = float(slopes[r_idx])
        trend_str = classify_trend(slope_val, trend_band)

        for m_idx, m in enumerate(members):
            if sel.any():
                pix_min = per_member_min_spi[m_idx, sel]
                pix_def = per_member_deficit[m_idx, sel]
                m_min = float(np.quantile(pix_min[np.isfinite(pix_min)], 0.05)) \
                    if np.isfinite(pix_min).any() else np.nan
                m_def = float(np.mean(pix_def))
                m_spa = float(np.mean(pix_def >= 0.5)) if pix_def.size else 0.0
            else:
                pt = adm1.iloc[r_idx].geometry.centroid
                lat_v = fcst.lat.values; lon_v = fcst.lon.values
                i = int(np.argmin(np.abs(lat_v - pt.y)))
                j = int(np.argmin(np.abs(lon_v - pt.x)))
                m_min = float(per_member_min_spi[m_idx, i, j])
                m_def = float(per_member_deficit[m_idx, i, j])
                m_spa = m_def

            row = {
                "boundary_id": gid, "boundary_name": nm, "country": cc,
                "member": str(int(m) if np.issubdtype(type(m), np.integer) else m),
                "target_date": target_date,
                "current_spi3": round(cur_val, 4),
                "spi3_trend": trend_str,
                "trend_slope_spi_per_month": round(slope_val, 4),
                "member_def_frac": round(m_def, 4),
                "member_spa_cov":  round(m_spa, 4),
                "member_min_spi":  round(m_min, 4),
            }
            if soft:
                for node, val, k in [("cur", cur_val, 5), ("def", m_def, 5),
                                      ("spa", m_spa, 3), ("trn", slope_val, 3),
                                      ("tail", m_min, 4)]:
                    probs = soft_bin(val, node)
                    for ki in range(k):
                        row[f"{node}_p{ki+1}"] = round(float(probs[ki]), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=False, default=None,
                    help="Target month D (YYYY-MM-01 or YYYY-MM) — v1 single-month "
                         "snapshot mode. Mutually exclusive with --init-month.")
    ap.add_argument("--init-month", required=False, default=None,
                    help="v2 seasonal mode: init month for the SEAS5 forecast "
                         "(YYYY-MM). Use with --target-season.")
    ap.add_argument("--target-season", choices=["MAM", "JJA", "OND", "DJF"],
                    default=None,
                    help="v2 seasonal mode: SPI-3 target season anchored at the "
                         "season's last month (May/Aug/Nov/Feb). Use with --init-month.")
    ap.add_argument("--ensemble-size", type=int, default=51,
                    help="Number of ensemble members to use for P2 (deficit_prob) "
                         "and P5 (tail_risk). Default 51 (all). v2 uses 25 "
                         "(first 25 = members with full 1981-now hindcast).")
    ap.add_argument("--rp-years", type=int, default=5,
                    help="Return period years (default 5; choices: 3,5,10,20,50)")
    ap.add_argument("--spi-period", default="SPI3",
                    help="SPI accumulation period (default SPI3)")
    ap.add_argument("--rp-prefer", default="fitted",
                    choices=["fitted", "empirical", "standard"])
    ap.add_argument("--deficit-spi", type=float, default=DEFAULT_DEFICIT_SPI,
                    help="Forecast SPI threshold below which a lead counts as "
                         "a deficit (default -1.0 = McKee moderate drought)")
    ap.add_argument("--trend-months", type=int, default=6,
                    help="Months of obs SPI to fit slope over (default 6)")
    ap.add_argument("--trend-band", type=float, default=0.1,
                    help="Slope ±band (SPI/month) bracketing 'Stable' (default 0.1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adm1", default="icpac_adm1v3.geojson")
    ap.add_argument("--clip-spi", type=float, default=4.0,
                    help="Clip SPI obs and forecast values to ±this absolute "
                         "value (default 4.0). Standard SPI analysis convention "
                         "— guards against ill-conditioned gamma-fit tails at "
                         "arid pixels. Pass 0 to disable.")
    ap.add_argument("--soft-evidence", action="store_true",
                    help="Emit Gaussian-soft-binned probability columns")
    ap.add_argument("--member-evidence-sidecar", default=None,
                    help="Per-member sidecar CSV path (one row per boundary × member)")
    args = ap.parse_args()

    # ── v1 vs v2 mode dispatch ──────────────────────────────────────────────
    seasonal_mode = args.init_month is not None or args.target_season is not None
    if seasonal_mode:
        if args.init_month is None or args.target_season is None:
            raise SystemExit("[prep] v2 seasonal mode requires both "
                             "--init-month YYYY-MM and --target-season MAM|JJA|OND|DJF")
        if args.date is not None:
            raise SystemExit("[prep] --date is v1-only; use --init-month + --target-season for v2")
        I = pd.Timestamp(args.init_month).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        target_season = args.target_season
        target_lead = lead_for_season(I.month, target_season)
        anchor_month = SEASON_ANCHOR_MONTH[target_season]
        # The season's anchor calendar month relative to the init year.
        anchor_year = I.year + (1 if anchor_month <= I.month else 0) if target_season == "DJF" else I.year + (1 if anchor_month < I.month else 0)
        D = pd.Timestamp(year=anchor_year, month=anchor_month, day=1)
        print(f"[prep] v2 seasonal: init={I.date()}  target_season={target_season}  "
              f"anchor_month={D.date()}  lead={target_lead}  "
              f"members={args.ensemble_size}  RP={args.rp_years}yr  "
              f"trend_band=±{args.trend_band}/month")
    else:
        if args.date is None:
            raise SystemExit("[prep] either --date (v1) or --init-month+--target-season (v2)")
        D = pd.Timestamp(args.date).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        I = D
        target_season = None
        target_lead = None
        print(f"[prep] v1 single-month: D={D.date()}  RP={args.rp_years}yr  "
              f"spi_period={args.spi_period}  deficit_spi={args.deficit_spi}  "
              f"trend_band=±{args.trend_band}/month")

    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    n_adm = len(adm1)
    print(f"[prep] adm1 boundaries: {n_adm}")

    # ── Open stores ──────────────────────────────────────────────────────────
    print("[prep] opening ERA5 SPI obs (zarr) ...")
    obs = open_zarr_anon(SPI_OBS_PREFIX)

    print("[prep] opening ERA5 SPI return-period (icechunk) ...")
    rp = open_icechunk_anon(SPI_RP_PREFIX)

    print("[prep] opening SEAS5 SPI3 forecast (icechunk slab) ...")
    forecast = open_icechunk_anon(SEAS5_SPI3_PREFIX)

    # ── Observation window (current SPI + trend) ────────────────────────────
    obs_spi3 = load_obs_spi3_window(obs, D, args.trend_months)
    if args.clip_spi > 0:
        obs_spi3 = obs_spi3.clip(-args.clip_spi, args.clip_spi)
        print(f"[prep] clipped obs SPI to ±{args.clip_spi}", flush=True)
    cur_spi3_grid = obs_spi3.isel(time=-1)  # most recent month

    print("[prep] building obs mask ...", flush=True)
    obs_mask = build_mask(adm1, obs_spi3.lat, obs_spi3.lon)
    print(f"[prep] obs mask built: {obs_mask.shape}, {int((obs_mask >= 0).sum())} masked pixels", flush=True)
    cur_spi_adm = zonal_reduce(cur_spi3_grid, obs_mask, obs_spi3.lat, n_adm)
    cur_spi_adm = fill_small_boundaries(cur_spi_adm, cur_spi3_grid, adm1)
    print("[prep] cur_spi_adm computed", flush=True)

    # Slope of monthly SPI3 over the trend window — fast numpy path that
    # avoids re-broadcasting xarray weights inside a per-timestep zonal loop
    # (which OOMs on small VMs because xarray re-allocates the weight grid
    # each call).
    n_t = obs_spi3.sizes["time"]
    print(f"[prep] computing per-region SPI3 slope over {n_t} months ...", flush=True)
    obs_vals = np.asarray(obs_spi3.values, dtype=np.float64)         # (t, lat, lon)
    mask_arr = np.asarray(obs_mask.values)                            # (lat, lon)
    weights_1d = np.cos(np.deg2rad(np.asarray(obs_spi3.lat.values))) # (lat,)
    w2d = weights_1d[:, None] * np.ones(obs_vals.shape[1:])           # (lat, lon)
    obs_adm = np.full((n_t, n_adm), np.nan, dtype=np.float64)
    for r in range(n_adm):
        sel = mask_arr == r
        if not sel.any():
            continue
        w_sel = w2d[sel]
        denom = float(w_sel.sum())
        if denom <= 0:
            continue
        # Vectorised over time: (t, n_pixels) * (n_pixels,)
        vals_sel = obs_vals[:, sel]                                    # (t, n_pix)
        valid = np.isfinite(vals_sel)
        # weighted mean per timestep, ignoring NaN pixels
        ws = (w_sel * valid).sum(axis=1)
        num = np.where(valid, vals_sel * w_sel, 0.0).sum(axis=1)
        obs_adm[:, r] = np.where(ws > 0, num / np.where(ws > 0, ws, 1.0), np.nan)
    print(f"[prep] obs_adm slope-input matrix built: {obs_adm.shape}", flush=True)

    x = np.arange(n_t, dtype=np.float64)
    slopes = np.full(n_adm, np.nan)
    for i in range(n_adm):
        y = obs_adm[:, i]
        if np.isfinite(y).all() and n_t >= 2:
            slopes[i] = float(np.polyfit(x, y, 1)[0])
    print(f"[prep] slopes computed", flush=True)
    trend_cls = np.array([classify_trend(s, args.trend_band) for s in slopes])
    cur_cat = np.array([categorize_current_spi(s) for s in cur_spi_adm])

    # ── Free obs arrays before opening the forecast — small VMs can't
    # hold both the obs reader, the regionmask machinery, and a 135 MB
    # forecast load + intermediate chunk decompression buffers.
    import gc
    del obs_vals, obs_adm, w2d, weights_1d, mask_arr, obs_spi3, obs
    gc.collect()

    # ── Forecast: lead × member × lat × lon for the chosen init ─────────────
    # v1: init = target month D
    # v2: init = I (--init-month); slice will then pick the season's anchor lead.
    print(f"[prep] loading SEAS5 forecast for init {I.date()} ...", flush=True)
    fcst = load_forecast_spi3(forecast, I)
    print(f"[prep] loaded SEAS5 forecast {dict(fcst.sizes)}, "
          f"dtype={fcst.dtype}, mem≈{fcst.nbytes/1024**2:.0f} MB", flush=True)
    if args.clip_spi > 0:
        fcst = fcst.clip(-args.clip_spi, args.clip_spi)
    print(f"[prep] forecast dims: {dict(fcst.sizes)} "
          f"{'(clipped to ±' + str(args.clip_spi) + ')' if args.clip_spi > 0 else ''}")

    # v2: restrict to first N ensemble members (default 25 = full 1981-now
    # hindcast block; members 25-50 only exist from 2017 onwards).
    if args.ensemble_size < fcst.sizes["member"]:
        fcst = fcst.isel(member=slice(0, args.ensemble_size))
        print(f"[prep] forecast restricted to first {args.ensemble_size} members "
              f"(calibration coverage: 1981-now)")

    # Regrid forecast (10 km, 351×321) to obs/RP grid (~25 km, 161×133)
    rp_thresh = load_rp_thresholds(rp, rp_year=args.rp_years,
                                   spi_period=args.spi_period,
                                   prefer=args.rp_prefer)
    fcst_rg = regrid_to(fcst, rp_thresh.lat, rp_thresh.lon)
    print(f"[prep] forecast regridded to RP grid: {dict(fcst_rg.sizes)}")

    # ── Forecast deficit metrics ─────────────────────────────────────────────
    # v2: deficit threshold = per-pixel SPI return-period threshold from
    # era5_ecmwf_rp_icechunk (negative SPI value, e.g. ~-0.84 for 5-yr).
    # v1: scalar -1.0 (--deficit-spi).
    if seasonal_mode:
        # Slice to the single anchor lead for the target season.
        fc_target = fcst_rg.sel(lead=target_lead)               # (member, lat, lon)
        deficit_thresh = rp_thresh                              # (lat, lon)
        deficit_lead = (fc_target <= deficit_thresh)            # (member, lat, lon)
        p_def_l1 = deficit_lead.mean(dim="member")              # (lat, lon)
        p_deficit = p_def_l1                                    # single lead → same value
        ens_mean_l1 = fc_target.mean(dim="member")
        ens_min_l1  = fc_target.min(dim="member")
        ens_max_l1  = fc_target.max(dim="member")
        ens_min_anylead = ens_min_l1                            # single lead → same
        deficit_threshold_label = (f"era5_ecmwf_rp_icechunk:{args.rp_years}yr "
                                   f"{args.rp_prefer}")
        crosses_rp = (fc_target <= rp_thresh).astype("float32") # (member, lat, lon)
        crosses_rp_any = crosses_rp.any(dim="member").astype("float32")
    else:
        # v1: scalar deficit_spi, max over leads.
        deficit_lead = (fcst_rg <= args.deficit_spi)            # (lead, member, lat, lon)
        deficit_prob_per_lead = deficit_lead.mean(dim="member") # (lead, lat, lon)
        p_deficit = deficit_prob_per_lead.max(dim="lead")
        if 1 in fcst_rg.lead.values:
            p_def_l1 = deficit_prob_per_lead.sel(lead=1)
            l1 = fcst_rg.sel(lead=1)
        else:
            p_def_l1 = deficit_prob_per_lead.isel(lead=0)
            l1 = fcst_rg.isel(lead=0)
        ens_mean_l1 = l1.mean(dim="member")
        ens_min_l1  = l1.min(dim="member")
        ens_max_l1  = l1.max(dim="member")
        ens_min_anylead = fcst_rg.min(dim="member").min(dim="lead")
        deficit_threshold_label = f"scalar:{args.deficit_spi}"
        crosses_rp_any = (fcst_rg <= rp_thresh).any(dim=["member", "lead"]).astype("float32")

    # ── Zonal aggregation ────────────────────────────────────────────────────
    fc_mask = build_mask(adm1, rp_thresh.lat, rp_thresh.lon)
    eprob_def_adm = zonal_reduce(p_deficit, fc_mask, rp_thresh.lat, n_adm)
    eprob_l1_adm  = zonal_reduce(p_def_l1, fc_mask, rp_thresh.lat, n_adm)
    spatial_cov_adm = zonal_reduce(p_deficit, fc_mask, rp_thresh.lat, n_adm, thresh=0.5)

    # Hotspot fraction: pixels where any forecast member crosses RP
    hotspot_frac_adm = zonal_reduce(crosses_rp_any, fc_mask, rp_thresh.lat, n_adm, thresh=0.5)

    # Tail-risk: ens-min SPI per boundary (p5 = drought-side; mean; peak)
    ens_min_p5_adm   = zonal_quantile(ens_min_anylead, fc_mask, n_adm, q=0.05)
    ens_min_mean_adm = zonal_reduce(ens_min_anylead, fc_mask, rp_thresh.lat, n_adm)
    ens_min_peak_adm = zonal_extreme(ens_min_anylead, fc_mask, n_adm, agg="min")

    ens_mean_l1_adm = zonal_reduce(ens_mean_l1, fc_mask, rp_thresh.lat, n_adm)
    ens_min_l1_adm  = zonal_reduce(ens_min_l1,  fc_mask, rp_thresh.lat, n_adm)
    ens_max_l1_adm  = zonal_reduce(ens_max_l1,  fc_mask, rp_thresh.lat, n_adm)

    # Centroid fallbacks for tiny/missing boundaries
    eprob_def_adm    = fill_small_boundaries(eprob_def_adm,    p_deficit, adm1)
    eprob_l1_adm     = fill_small_boundaries(eprob_l1_adm,     p_def_l1, adm1)
    spatial_cov_adm  = fill_small_boundaries(spatial_cov_adm,  p_deficit, adm1, thresh=0.5)
    hotspot_frac_adm = fill_small_boundaries(hotspot_frac_adm, crosses_rp_any, adm1, thresh=0.5)
    ens_min_p5_adm   = fill_small_boundaries(ens_min_p5_adm,   ens_min_anylead, adm1)
    ens_min_mean_adm = fill_small_boundaries(ens_min_mean_adm, ens_min_anylead, adm1)
    ens_min_peak_adm = fill_small_boundaries(ens_min_peak_adm, ens_min_anylead, adm1)
    ens_mean_l1_adm  = fill_small_boundaries(ens_mean_l1_adm,  ens_mean_l1, adm1)
    ens_min_l1_adm   = fill_small_boundaries(ens_min_l1_adm,   ens_min_l1, adm1)
    ens_max_l1_adm   = fill_small_boundaries(ens_max_l1_adm,   ens_max_l1, adm1)

    # Spatial coverage: max of (P_deficit ≥ 0.5 mask) and (hotspot fraction)
    spatial_cov_final = np.fmax(spatial_cov_adm, hotspot_frac_adm)

    # ── Assemble ──────────────────────────────────────────────────────────────
    country = (adm1["GID_1"].str.split(".").str[0]
               .map(ISO_TO_COUNTRY).fillna("Unknown"))

    df = pd.DataFrame({
        "id": adm1["GID_1"],
        "name": adm1["NAME_1"],
        "country": country,
        "current_spi3": np.round(cur_spi_adm, 4),
        "current_spi3_category": cur_cat,
        "spi3_trend": trend_cls,
        "trend_slope_spi_per_month": np.round(slopes, 4),
        "forecast_deficit_prob": np.round(eprob_def_adm, 4),
        "deficit_prob_lead1": np.round(eprob_l1_adm, 4),
        "spatial_coverage": np.round(spatial_cov_final, 4),
        "spatial_cov_mean_p": np.round(spatial_cov_adm, 4),
        "hotspot_fraction": np.round(hotspot_frac_adm, 4),
        "forecast_agreement": "Medium",
        "ens_min_spi": np.round(ens_min_p5_adm, 4),       # tail-risk evidence
        "ens_min_spi_mean": np.round(ens_min_mean_adm, 4),
        "ens_min_spi_peak": np.round(ens_min_peak_adm, 4),
        "ens_mean_lead1_spi": np.round(ens_mean_l1_adm, 4),
        "ens_min_lead1_spi":  np.round(ens_min_l1_adm,  4),
        "ens_max_lead1_spi":  np.round(ens_max_l1_adm,  4),
        "target_date": str(D.date()),
    })
    if seasonal_mode:
        df.insert(3, "init_month", str(I.date()))
        df.insert(4, "target_season", target_season)
        df.insert(5, "lead_index_used", target_lead)
        df.insert(6, "deficit_threshold_source", deficit_threshold_label)
        df.insert(7, "ensemble_size", args.ensemble_size)

    if args.soft_evidence:
        add_soft_columns(df,
                         cur_spi   = cur_spi_adm,
                         def_p     = eprob_def_adm,
                         spa       = spatial_cov_final,
                         trn_slope = slopes,
                         tail_spi  = ens_min_p5_adm)
        print("[prep] soft-evidence columns added (20 cols)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    mode_label = ("v2 seasonal" if seasonal_mode else "v1 single-month")
    print(f"[prep] wrote {out}  ({mode_label})  rows={len(df)}  cols={len(df.columns)}  "
          f"cur_spi_mean={np.nanmean(cur_spi_adm):.2f}  "
          f"def_prob_mean={np.nanmean(eprob_def_adm):.3f}")

    if args.member_evidence_sidecar:
        me_df = compute_per_member_evidence(
            fcst_rg, rp_thresh, fc_mask, adm1, n_adm,
            cur_spi_adm, slopes, args.trend_band, args.deficit_spi,
            target_date=str(D.date()),
            soft=args.soft_evidence,
        )
        me_path = Path(args.member_evidence_sidecar)
        me_path.parent.mkdir(parents=True, exist_ok=True)
        me_df.to_csv(me_path, index=False)
        n_severe = (me_df["member_min_spi"] <= -1.5).sum()
        print(f"[prep] wrote member-evidence sidecar {me_path}  rows={len(me_df)}  "
              f"severe-drought members ({len(me_df)} total)={n_severe} "
              f"({n_severe/len(me_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
