#!/usr/bin/env -S uv run --with icechunk --with xarray --with "zarr>=3" --with numpy --with pandas --with geopandas --with regionmask --with netcdf4 --with pyarrow --with scipy
"""
Flood BN IBF v1 — per-day admin-1 input generator.

Reads:
  - IMERG half-hourly icechunk store (observations)
  - ECMWF TP icechunk store (forecasts)
  - CMORPH return-period NetCDF (pixel-wise thresholds)
  - ICPAC admin-1 GeoJSON

Writes a CSV with one row per admin-1 boundary holding the evidence vector
consumed by flood_bn_ibf_v1.jl:
    id, name, country,
    antecedent_rainfall_mm, antecedent_category,
    rainfall_trend, trend_slope_mm_per_day,
    gefs_eprob_heavy, eprob_24h, spatial_coverage,
    forecast_agreement, target_date
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

DURATIONS = ["3hr", "6hr", "12hr", "24hr", "48hr", "72hr", "7day"]
DURATION_HOURS = {"3hr": 3, "6hr": 6, "12hr": 12, "24hr": 24,
                  "48hr": 48, "72hr": 72, "7day": 168}

ISO_TO_COUNTRY = {
    "BDI": "Burundi", "DJI": "Djibouti", "ERI": "Eritrea", "ETH": "Ethiopia",
    "KEN": "Kenya", "RWA": "Rwanda", "SOM": "Somalia", "SSD": "South Sudan",
    "SDN": "Sudan", "TZA": "Tanzania", "UGA": "Uganda",
}


def open_icechunk(prefix: str) -> xr.Dataset:
    storage = ic.s3_storage(
        bucket="e4drr-project",
        prefix=prefix,
        endpoint_url="https://data.source.coop",
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    repo = ic.Repository.open(storage)
    return xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
        decode_timedelta=True,
    )


def imerg_daily_totals(imerg: xr.Dataset, date_utc: pd.Timestamp) -> xr.DataArray:
    start = pd.Timestamp(date_utc) - pd.Timedelta(days=7)
    end = pd.Timestamp(date_utc) - pd.Timedelta(seconds=1)
    hh = imerg.precipitation.sel(time=slice(start, end))  # mm/hr
    mm = hh * 0.5  # half-hour → mm
    daily = mm.resample(time="1D").sum()
    return daily.astype("float32")


def ecmwf_window_accums(ecmwf: xr.Dataset, init_date: pd.Timestamp) -> dict[str, xr.DataArray]:
    tp = ecmwf.tp.sel(init_date=init_date)  # (member, lead_time, lat, lon) in metres
    lt = tp.lead_time.values
    out: dict[str, xr.DataArray] = {}
    for dur, h in DURATION_HOURS.items():
        td = np.timedelta64(h, "h")
        idx_arr = np.where(lt == td)[0]
        if idx_arr.size == 0:
            idx = int(np.argmin(np.abs(lt - td)))
        else:
            idx = int(idx_arr[0])
        out[dur] = (tp.isel(lead_time=idx) * 1000.0).astype("float32")  # → mm
    return out


def load_cmorph_thresholds(path: str, rp_year: int) -> dict[str, xr.DataArray]:
    ds = xr.open_dataset(path)
    rp = ds.return_period_precip.sel(return_period=rp_year)
    if float(rp.lat[0]) > float(rp.lat[-1]):
        rp = rp.isel(lat=slice(None, None, -1))
    if float(rp.lon[0]) > float(rp.lon[-1]):
        rp = rp.isel(lon=slice(None, None, -1))
    return {dur: rp.sel(duration=dur).drop_vars("duration") for dur in DURATIONS}


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
                 n_regions: int, thresh: float | None = None) -> np.ndarray:
    """Area-weighted mean (or fraction ≥ thresh) per region. NaN where empty."""
    weights = np.cos(np.deg2rad(lat))
    w2d = weights.broadcast_like(da)
    src = (da >= thresh).astype("float32") if thresh is not None else da
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


def fill_small_boundaries(values: np.ndarray, da: xr.DataArray,
                          gdf: gpd.GeoDataFrame, thresh: float | None = None) -> np.ndarray:
    """For boundaries with no pixel hit, sample nearest pixel at centroid."""
    out = values.copy()
    missing = np.where(np.isnan(out))[0]
    if len(missing) == 0:
        return out
    cent = gdf.iloc[missing].geometry.centroid
    src = (da >= thresh).astype("float32") if thresh is not None else da
    for pos, (i, pt) in enumerate(zip(missing, cent)):
        try:
            val = float(src.sel(lat=pt.y, lon=pt.x, method="nearest").values)
        except Exception:
            val = np.nan
        out[i] = val
    return out


def classify_trend(slope: float, band: float) -> str:
    if not np.isfinite(slope):
        return "Stable"
    if slope > band:
        return "Increasing"
    if slope < -band:
        return "Decreasing"
    return "Stable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Target date D (YYYY-MM-DD)")
    ap.add_argument("--rp-years", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adm1", default="icpac_adm1v3.geojson")
    ap.add_argument("--cmorph-rp", default="cmorph_ea_return_periods.nc")
    ap.add_argument("--trend-band", type=float, default=2.0)
    args = ap.parse_args()

    D = pd.Timestamp(args.date)
    print(f"[prep] D={D.date()}  RP={args.rp_years}yr  band=±{args.trend_band} mm/day")

    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    n_adm = len(adm1)
    print(f"[prep] adm1 boundaries: {n_adm}")

    # ---------------- IMERG antecedent ----------------
    print("[prep] opening IMERG icechunk...")
    imerg = open_icechunk("observations/imerg_hh_icechunk")
    daily = imerg_daily_totals(imerg, D).load()
    t0 = pd.to_datetime(daily.time.values[0]).date()
    t1 = pd.to_datetime(daily.time.values[-1]).date()
    print(f"[prep] IMERG 7-day totals {t0}..{t1}  shape={daily.shape}")

    imerg_mask = build_mask(adm1, daily.lat, daily.lon)

    daily_adm = np.full((daily.sizes["time"], n_adm), np.nan, dtype=np.float64)
    for di in range(daily.sizes["time"]):
        daily_adm[di] = zonal_reduce(daily.isel(time=di), imerg_mask, daily.lat, n_adm)

    antecedent_mm = np.nansum(daily_adm, axis=0)
    antecedent_mm[np.isnan(daily_adm).all(axis=0)] = np.nan

    x = np.arange(daily.sizes["time"], dtype=np.float64)
    slopes = np.full(n_adm, np.nan)
    for i in range(n_adm):
        y = daily_adm[:, i]
        if np.isfinite(y).all():
            slopes[i] = float(np.polyfit(x, y, 1)[0])
    trend_cls = np.array([classify_trend(s, args.trend_band) for s in slopes])

    # ---------------- ECMWF exceedance ----------------
    print("[prep] opening ECMWF icechunk...")
    ecmwf = open_icechunk("forecasts/ecmwf_ea_tp_icechunk")
    init_dates = pd.to_datetime(ecmwf.init_date.values)
    if D not in init_dates:
        raise SystemExit(f"[prep] init_date {D.date()} not in ECMWF store "
                         f"(range {init_dates.min().date()}..{init_dates.max().date()})")

    accums = ecmwf_window_accums(ecmwf, D)
    for k in list(accums):
        accums[k] = accums[k].load()
    print(f"[prep] ECMWF accums loaded for {list(accums)}")

    thresh = load_cmorph_thresholds(args.cmorph_rp, args.rp_years)
    ref = accums["24hr"].isel(member=0)
    thresh_ec = {dur: regrid_to(thresh[dur], ref.lat, ref.lon) for dur in DURATIONS}

    eprob = {}
    for dur in DURATIONS:
        eprob[dur] = ((accums[dur] >= thresh_ec[dur])
                      .astype("float32").mean(dim="member"))
    eprob_24 = eprob["24hr"]
    p_heavy = xr.concat([eprob[d] for d in DURATIONS], dim="duration").max("duration")

    ec_mask = build_mask(adm1, ref.lat, ref.lon)
    eprob_heavy_adm = zonal_reduce(p_heavy, ec_mask, ref.lat, n_adm)
    eprob_24h_adm = zonal_reduce(eprob_24, ec_mask, ref.lat, n_adm)
    spatial_cov_adm = zonal_reduce(p_heavy, ec_mask, ref.lat, n_adm, thresh=0.5)

    eprob_heavy_adm = fill_small_boundaries(eprob_heavy_adm, p_heavy, adm1)
    eprob_24h_adm = fill_small_boundaries(eprob_24h_adm, eprob_24, adm1)
    spatial_cov_adm = fill_small_boundaries(spatial_cov_adm, p_heavy, adm1, thresh=0.5)

    # ---------------- Assemble output ----------------
    country = (adm1["GID_1"].str.split(".").str[0]
               .map(ISO_TO_COUNTRY).fillna("Unknown"))

    df = pd.DataFrame({
        "id": adm1["GID_1"],
        "name": adm1["NAME_1"],
        "country": country,
        "antecedent_rainfall_mm": np.round(antecedent_mm, 3),
        "antecedent_category": "",
        "rainfall_trend": trend_cls,
        "trend_slope_mm_per_day": np.round(slopes, 3),
        "gefs_eprob_heavy": np.round(eprob_heavy_adm, 4),
        "eprob_24h": np.round(eprob_24h_adm, 4),
        "spatial_coverage": np.round(spatial_cov_adm, 4),
        "forecast_agreement": "Medium",
        "target_date": str(D.date()),
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[prep] wrote {out}  rows={len(df)}  "
          f"ant_mean={np.nanmean(antecedent_mm):.1f}mm  "
          f"heavy_mean={np.nanmean(eprob_heavy_adm):.3f}")


if __name__ == "__main__":
    main()
