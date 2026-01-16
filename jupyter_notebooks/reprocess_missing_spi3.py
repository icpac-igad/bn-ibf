#!/usr/bin/env python3
"""
Script to reprocess missing SPI3 months from intermediate file,
regrid to 10km, and upload to IceChunk.

Usage:
    python reprocess_missing_spi3.py
"""

import os
import sys
import logging
import time
import xarray as xr
import numpy as np
import xesmf as xe
import icechunk
import gcsfs

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
INTERMEDIATE_FILE = "/srv/202512-itt/seas51_spi3_full.nc"
SERVICE_ACCOUNT_FILE = "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "seas51_spi3_10km"

# East Africa bounds (consistent with other scripts)
EA_BOUNDS = {
    'lat_min': -12,
    'lat_max': 23,
    'lon_min': 21,
    'lon_max': 53
}
RESOLUTION_DEG = 0.1


def get_uploaded_inits():
    """Get list of already uploaded initialization months."""
    fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_FILE)
    try:
        paths = fs.ls(f"{GCS_BUCKET}/{GCS_PATH}")
        uploaded = set(p.split('/')[-1] for p in paths if p.split('/')[-1].isdigit())
        return uploaded
    except Exception as e:
        logger.error(f"Error listing uploaded inits: {e}")
        return set()


def create_target_grid(bounds, resolution_deg):
    """Create target grid for regridding to 10km."""
    lats = np.arange(bounds['lat_min'], bounds['lat_max'] + resolution_deg, resolution_deg)
    lons = np.arange(bounds['lon_min'], bounds['lon_max'] + resolution_deg, resolution_deg)

    ds_out = xr.Dataset({
        "lat": (["lat"], lats, {"units": "degrees_north"}),
        "lon": (["lon"], lons, {"units": "degrees_east"}),
    })

    return ds_out


def regrid_to_10km(ds, regridder):
    """Regrid dataset to 10km resolution."""
    if isinstance(ds, xr.Dataset):
        regridded_vars = {}
        for var in ds.data_vars:
            regridded_vars[var] = regridder(ds[var], keep_attrs=True)
        regridded = xr.Dataset(regridded_vars)
        regridded.attrs = ds.attrs
    else:
        regridded = regridder(ds, keep_attrs=True)

    return regridded


def upload_to_icechunk(ds, init_str):
    """Upload dataset to IceChunk."""
    try:
        storage = icechunk.gcs_storage(
            bucket=GCS_BUCKET,
            prefix=f"{GCS_PATH}/{init_str}",
            service_account_file=SERVICE_ACCOUNT_FILE
        )

        try:
            repo = icechunk.Repository.open(storage=storage)
            logger.info(f"Opened existing IceChunk repository for {init_str}")
        except Exception:
            repo = icechunk.Repository.create(storage=storage)
            logger.info(f"Created new IceChunk repository for {init_str}")

        session = repo.writable_session("main")
        store = session.store

        ds.to_zarr(store, mode='w', consolidated=False)

        commit_id = session.commit(f"Add SPI3 10km data for init {init_str}")
        logger.info(f"Committed {init_str} with ID: {commit_id}")

        return True

    except Exception as e:
        logger.error(f"Failed to upload {init_str}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Reprocess Missing SPI3 Months")
    logger.info("=" * 60)

    # Get already uploaded inits
    uploaded = get_uploaded_inits()
    logger.info(f"Already uploaded: {len(uploaded)} months")

    # Generate expected months (1981-01 to 2025-12)
    expected = []
    for year in range(1981, 2026):
        for month in range(1, 13):
            expected.append(f'{year}{month:02d}')

    # Find missing
    missing = [m for m in expected if m not in uploaded]
    logger.info(f"Missing months: {len(missing)}")

    if not missing:
        logger.info("No missing months. All done!")
        return

    # Load intermediate file
    logger.info(f"Loading intermediate file: {INTERMEDIATE_FILE}")
    ds_full = xr.open_dataset(INTERMEDIATE_FILE)
    logger.info(f"Loaded. Dimensions: {dict(ds_full.sizes)}")

    # Create target grid
    ds_out = create_target_grid(EA_BOUNDS, RESOLUTION_DEG)
    logger.info(f"Target grid: {len(ds_out.lat)} x {len(ds_out.lon)} cells")

    # Create regridder once (reuse for all months)
    logger.info("Creating regridder...")
    ds_template = ds_full['spi3'].isel(init=0, member=0, lead=0)
    if 'latitude' in ds_template.coords:
        ds_template = ds_template.rename({'latitude': 'lat', 'longitude': 'lon'})
    regridder = xe.Regridder(ds_template, ds_out, 'bilinear', periodic=False)
    logger.info("Regridder created.")

    # Process each missing month
    success = 0
    failed = 0
    start_time = time.time()

    for i, init_str in enumerate(missing):
        logger.info(f"\n[{i+1}/{len(missing)}] Processing {init_str}...")

        try:
            # Parse init string to datetime
            year = int(init_str[:4])
            month = int(init_str[4:])
            init_time = np.datetime64(f'{year}-{month:02d}-01')

            # Select this init from the dataset
            ds_init = ds_full.sel(init=init_time)

            # Rename coordinates for regridding
            if 'latitude' in ds_init.coords:
                ds_init = ds_init.rename({'latitude': 'lat', 'longitude': 'lon'})

            # Regrid
            regridded = regrid_to_10km(ds_init, regridder)

            # Add metadata
            regridded.attrs['title'] = 'SEAS51 SPI3 Forecast - 10km resolution'
            regridded.attrs['resolution'] = '10km (~0.1 degrees)'
            regridded.attrs['region'] = 'East Africa'
            regridded.attrs['init_time'] = str(init_time)

            # Upload
            if upload_to_icechunk(regridded, init_str):
                success += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Error processing {init_str}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

        # Small delay
        time.sleep(0.5)

    elapsed = time.time() - start_time
    ds_full.close()

    logger.info("\n" + "=" * 60)
    logger.info("REPROCESSING COMPLETE")
    logger.info(f"Successful: {success}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
