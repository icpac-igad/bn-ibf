#!/usr/bin/env -S uv run --with requests --with rasterio --with numpy --with xarray --with zarr --with icechunk --with geopandas --with pyproj --with tqdm
"""
collect_worldpop_arco.py
========================
Download WorldPop 2020 UN-adjusted population count GeoTIFFs for the 11
ICPAC East-Africa countries, mosaic them onto the same regular 0.1° grid
used by the ECMWF/IMERG icechunk stores, and write the result into an
icechunk zarr store on disk (or S3).

Spatial extent (matching source.coop CHIRPS SPI store):
    lat : -12 … +23 N   (step 0.1°)
    lon :  21 … +53 E   (step 0.1°)

Store layout
------------
  worldpop_ea_icechunk/
    population (year, lat, lon)  float32  — person-count per pixel
    attrs: source, resolution, crs, year_range

Usage
-----
    uv run collect_worldpop_arco.py --years 2020 --out ./worldpop_ea_icechunk
    uv run collect_worldpop_arco.py --years 2015 2020 --out ./worldpop_ea_icechunk
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import icechunk as ic
import numpy as np
import requests
import xarray as xr
import zarr
from tqdm import tqdm

try:
    import rasterio
    from rasterio.merge import merge as rasterio_merge
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
except ImportError:
    raise SystemExit("rasterio is required: uv add rasterio")

# ── East-Africa domain (matches ECMWF/IMERG icechunk extent) ──────────────────
EA_LAT_MIN, EA_LAT_MAX = -12.0, 23.0
EA_LON_MIN, EA_LON_MAX =  21.0, 53.0
GRID_STEP = 0.1  # degrees

# ── ICPAC member-state ISO3 codes ─────────────────────────────────────────────
EA_ISO3 = ["BDI", "DJI", "ERI", "ETH", "KEN", "RWA", "SOM", "SSD", "SDN", "TZA", "UGA"]

# ── WorldPop unconstrained UN-adjusted 100 m resolution URL template ──────────
# https://hub.worldpop.org/geodata/summary?id=24777  (dataset catalogue)
WP_URL_TMPL = (
    "https://data.worldpop.org/GIS/Population/"
    "Global_2000_2020/{year}/{iso3}/{iso3_lower}_ppp_{year}_UNadj.tif"
)

# Some countries have slightly different naming on the WorldPop server.
# Override here if needed.
WP_ISO_OVERRIDES: dict[str, str] = {
    "SSD": "SSD",   # South Sudan uses SSD
}


def worldpop_url(iso3: str, year: int) -> str:
    iso3 = WP_ISO_OVERRIDES.get(iso3, iso3)
    return WP_URL_TMPL.format(
        year=year,
        iso3=iso3,
        iso3_lower=iso3.lower(),
    )


def download_file(url: str, dest: Path, chunk_bytes: int = 1 << 20) -> bool:
    """Stream-download *url* to *dest*. Returns True on success."""
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_bytes):
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as exc:
        print(f"  [warn] failed to download {url}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


def make_target_grid() -> tuple[np.ndarray, np.ndarray]:
    """Build the target lat/lon 1-D arrays for the EA domain."""
    lats = np.arange(EA_LAT_MIN, EA_LAT_MAX + GRID_STEP / 2, GRID_STEP)
    lons = np.arange(EA_LON_MIN, EA_LON_MAX + GRID_STEP / 2, GRID_STEP)
    return lats.astype("float32"), lons.astype("float32")


def mosaic_and_resample(tif_paths: list[Path], out_lat: np.ndarray,
                        out_lon: np.ndarray) -> np.ndarray:
    """
    Merge multiple GeoTIFFs (same CRS, WGS84) and resample to the target grid.
    Returns a 2-D float32 array (lat, lon) with NaN for no-data.
    """
    if not tif_paths:
        nlat, nlon = len(out_lat), len(out_lon)
        return np.full((nlat, nlon), np.nan, dtype="float32")

    # Open all rasters and merge
    srcs = [rasterio.open(p) for p in tif_paths]
    mosaic_data, mosaic_transform = rasterio_merge(srcs)
    nodata_val = srcs[0].nodata
    for s in srcs:
        s.close()

    # mosaic_data shape: (bands, height, width) – band 0 is population
    pop = mosaic_data[0].astype("float32")
    if nodata_val is not None:
        pop[pop == nodata_val] = np.nan

    # Build target transform from the output grid
    nlat, nlon = len(out_lat), len(out_lon)
    dst_transform = from_bounds(
        left=float(out_lon[0]) - GRID_STEP / 2,
        bottom=float(out_lat[0]) - GRID_STEP / 2,
        right=float(out_lon[-1]) + GRID_STEP / 2,
        top=float(out_lat[-1]) + GRID_STEP / 2,
        width=nlon,
        height=nlat,
    )

    # Reproject mosaic → target grid (sum resampling to conserve population)
    dst = np.full((nlat, nlon), np.nan, dtype="float32")
    reproject(
        source=pop,
        destination=dst,
        src_transform=mosaic_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.sum,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    # Flip so lat is ascending (south → north)
    dst = dst[::-1, :]
    return dst


def build_store(out_dir: Path, years: list[int], cache_dir: Path) -> None:
    lats, lons = make_target_grid()
    nlat, nlon = len(lats), len(lons)
    nyear = len(years)

    # ── icechunk store on local filesystem ────────────────────────────────────
    storage = ic.local_filesystem_storage(str(out_dir))
    try:
        repo = ic.Repository.open(storage)
        print(f"[wp] opened existing store at {out_dir}")
    except Exception:
        repo = ic.Repository.create(storage)
        print(f"[wp] created new store at {out_dir}")

    session = repo.writable_session("main")
    store = session.store

    # Initialise zarr arrays
    root = zarr.open_group(store, mode="a")

    if "population" not in root:
        root.create_array(
            "population",
            shape=(nyear, nlat, nlon),
            chunks=(1, 256, 256),
            dtype="float32",
            fill_value=np.nan,
        )
        root.create_array("year", shape=(nyear,), dtype="int32")
        root.create_array("lat",  shape=(nlat,),  dtype="float32")
        root.create_array("lon",  shape=(nlon,),  dtype="float32")

        root["lat"][:] = lats
        root["lon"][:] = lons
        root["year"][:] = np.array(years, dtype="int32")
        root.attrs.update({
            "source": "WorldPop Global 2000-2020, UN-adjusted, 100 m",
            "citation": "WorldPop (www.worldpop.org) - School of Geography and Environmental Science, University of Southampton",
            "license": "CC-BY 4.0",
            "resolution_deg": GRID_STEP,
            "crs": "EPSG:4326",
            "resampling": "sum (population-conserving)",
            "lat_min": EA_LAT_MIN,
            "lat_max": EA_LAT_MAX,
            "lon_min": EA_LON_MIN,
            "lon_max": EA_LON_MAX,
        })
        pop_arr = root["population"]
    else:
        # Extend year axis if new years requested
        existing_years = list(root["year"][:])
        pop_arr = root["population"]
        for yi, yr in enumerate(years):
            if yr in existing_years:
                print(f"[wp] year {yr} already in store, skipping download")

    # ── Download and ingest each year ─────────────────────────────────────────
    for yi, year in enumerate(years):
        existing_years_arr = list(root["year"][:])
        if year in existing_years_arr and pop_arr[yi].max() > 0:
            print(f"[wp] year {year} already ingested")
            continue

        print(f"\n[wp] === year {year} ===")
        tif_paths: list[Path] = []
        year_cache = cache_dir / str(year)
        year_cache.mkdir(parents=True, exist_ok=True)

        for iso3 in EA_ISO3:
            url = worldpop_url(iso3, year)
            dest = year_cache / f"{iso3.lower()}_ppp_{year}_UNadj.tif"
            if dest.exists():
                print(f"  [wp] {iso3} already cached at {dest}")
                tif_paths.append(dest)
            else:
                print(f"  [wp] downloading {iso3} from {url}")
                ok = download_file(url, dest)
                if ok:
                    tif_paths.append(dest)

        print(f"[wp] mosaicking {len(tif_paths)} tiles for year {year}...")
        grid = mosaic_and_resample(tif_paths, lats, lons)
        print(f"[wp]   pop total = {np.nansum(grid):,.0f}  NaN frac = {np.isnan(grid).mean():.2%}")

        pop_arr[yi, :, :] = grid

    session.commit(f"WorldPop EA mosaic years={years}")
    print(f"\n[wp] committed to {out_dir}")
    print(f"[wp] store shape: population{root['population'].shape}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=[2020],
                    help="WorldPop years to download (default: 2020)")
    ap.add_argument("--out", default="./worldpop_ea_icechunk",
                    help="Output icechunk store directory")
    ap.add_argument("--cache-dir", default="./worldpop_cache",
                    help="Local cache directory for raw GeoTIFFs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    build_store(out_dir, sorted(args.years), cache_dir)


if __name__ == "__main__":
    main()
