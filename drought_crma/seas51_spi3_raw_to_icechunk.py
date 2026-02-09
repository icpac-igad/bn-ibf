#!/usr/bin/env python3
"""
Script to upload raw ~100km SEAS51 SPI-3 data to IceChunk store in GCS bucket.

This script:
1. Loads the intermediate SPI3 file (already calculated, ~100km resolution)
2. Uploads each initialization month to IceChunk store (no regridding)

Usage:
    python seas51_spi3_raw_to_icechunk.py
"""

import sys
import os
import logging
import time
import xarray as xr
import numpy as np
import gcsfs

# Try to import icechunk
try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False
    print("Warning: icechunk not available")

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
INTERMEDIATE_FILE = "/srv/202512-itt/seas51_spi3_full.nc"
OUTPUT_DIR = "/srv/202512-itt/"
SERVICE_ACCOUNT_FILE = "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "seas51_spi3_raw"  # Different path for raw data


def get_uploaded_inits():
    """Get list of already uploaded initialization months."""
    fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_FILE)
    try:
        paths = fs.ls(f"{GCS_BUCKET}/{GCS_PATH}")
        uploaded = set(p.split('/')[-1] for p in paths if p.split('/')[-1].isdigit())
        return uploaded
    except Exception as e:
        # Path might not exist yet
        logger.info(f"No existing uploads found (or path doesn't exist yet)")
        return set()


def upload_to_icechunk(ds, init_time):
    """
    Upload raw SPI3 data to IceChunk store.
    """
    if not HAS_ICECHUNK:
        logger.warning("IceChunk not available. Saving locally instead.")
        return save_locally(ds, init_time)

    try:
        # Format init time for storage path
        init_str = str(init_time)[:7].replace('-', '')  # e.g., "198101"

        # Configure GCS storage for IceChunk
        storage = icechunk.gcs_storage(
            bucket=GCS_BUCKET,
            prefix=f"{GCS_PATH}/{init_str}",
            service_account_file=SERVICE_ACCOUNT_FILE
        )

        # Try to open existing repository, or create new one
        try:
            repo = icechunk.Repository.open(storage=storage)
            logger.info(f"Opened existing IceChunk repository for {init_str}")
        except Exception:
            repo = icechunk.Repository.create(storage=storage)
            logger.info(f"Created new IceChunk repository for {init_str}")

        session = repo.writable_session("main")
        store = session.store

        # Write data to IceChunk
        ds.to_zarr(store, mode='w', consolidated=False)

        # Commit the changes
        commit_id = session.commit(f"Add SPI3 raw (~100km) data for init {init_str}")
        logger.info(f"Committed with ID: {commit_id}")

        return True

    except Exception as e:
        logger.error(f"IceChunk upload failed: {e}")
        import traceback
        traceback.print_exc()
        return save_locally(ds, init_time)


def save_locally(ds, init_time):
    """Save dataset locally as fallback."""
    init_str = str(init_time)[:7].replace('-', '')
    output_file = os.path.join(OUTPUT_DIR, f'spi3_raw_{init_str}.nc')
    ds.to_netcdf(output_file)
    logger.info(f"Saved locally: {output_file}")
    return True


def process_all_inits(intermediate_file):
    """
    Load intermediate SPI3 file and upload each init to IceChunk.
    """
    logger.info("=" * 60)
    logger.info("SEAS51 SPI3 Raw (~100km) to IceChunk Pipeline")
    logger.info("=" * 60)

    # Load the intermediate file
    logger.info(f"Loading intermediate file: {intermediate_file}")
    try:
        ds = xr.open_dataset(intermediate_file)
        logger.info(f"Dataset loaded. Dimensions: {dict(ds.sizes)}")
        logger.info(f"Init range: {ds.init.values[0]} to {ds.init.values[-1]}")
        logger.info(f"Lead times: {ds.sizes.get('lead', 'N/A')}")
        logger.info(f"Ensemble members: {ds.sizes.get('member', 'N/A')}")
        logger.info(f"Lat x Lon: {ds.sizes.get('lat', 'N/A')} x {ds.sizes.get('lon', 'N/A')}")
    except Exception as e:
        logger.error(f"Error loading intermediate file: {e}")
        raise

    # Get already uploaded inits
    uploaded = get_uploaded_inits()
    logger.info(f"Already uploaded: {len(uploaded)} months")

    # Get all init times
    init_times = ds.init.values
    logger.info(f"Total init times in file: {len(init_times)}")

    # Filter to only upload missing inits
    to_upload = []
    for init_time in init_times:
        init_str = str(init_time)[:7].replace('-', '')
        if init_str not in uploaded:
            to_upload.append(init_time)

    logger.info(f"Inits to upload: {len(to_upload)}")

    if not to_upload:
        logger.info("All inits already uploaded. Nothing to do!")
        ds.close()
        return

    # Process each init time
    total_success = 0
    total_failed = 0
    start_time = time.time()

    for i, init_time in enumerate(to_upload):
        init_str = str(init_time)[:7].replace('-', '')
        logger.info(f"\n[{i+1}/{len(to_upload)}] Processing init: {init_str}")

        try:
            # Select this init
            ds_init = ds.sel(init=init_time)

            # Ensure it's a dataset with proper structure
            if isinstance(ds_init, xr.DataArray):
                ds_init = ds_init.to_dataset(name='spi3')

            # Add metadata
            ds_init.attrs['title'] = 'SEAS51 SPI3 Forecast - Raw (~100km) resolution'
            ds_init.attrs['source'] = intermediate_file
            ds_init.attrs['resolution'] = '~100km (native SEAS51 resolution)'
            ds_init.attrs['region'] = 'East Africa'
            ds_init.attrs['init_time'] = str(init_time)

            # Upload to IceChunk
            success = upload_to_icechunk(ds_init, init_time)

            if success:
                total_success += 1
            else:
                total_failed += 1

        except Exception as e:
            logger.error(f"Error processing init {init_time}: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1

        # Small delay between uploads
        time.sleep(0.5)

    elapsed = time.time() - start_time
    ds.close()

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_failed}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info("=" * 60)


def main():
    """Main function."""
    logger.info("Starting SEAS51 SPI3 Raw to IceChunk pipeline...")

    # Check if intermediate file exists
    if not os.path.exists(INTERMEDIATE_FILE):
        logger.error(f"Intermediate file not found: {INTERMEDIATE_FILE}")
        sys.exit(1)

    # Check if service account file exists
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        start_time = time.time()
        process_all_inits(INTERMEDIATE_FILE)
        elapsed = time.time() - start_time

        logger.info(f"\nTotal processing time: {elapsed/60:.2f} minutes")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
