#!/usr/bin/env python3
"""
Script to upload missing SPI3 10km NetCDF files to IceChunk store.
Removes local NC file after successful upload.

Usage:
    python upload_missing_spi3_to_icechunk.py
"""

import os
import sys
import glob
import logging
import time
import xarray as xr
import icechunk

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
LOCAL_DIR = "/srv/202512-itt/"
SERVICE_ACCOUNT_FILE = "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "seas51_spi3_10km"


def get_uploaded_inits():
    """Get list of already uploaded initialization months."""
    import gcsfs

    fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_FILE)
    try:
        paths = fs.ls(f"{GCS_BUCKET}/{GCS_PATH}")
        uploaded = set(p.split('/')[-1] for p in paths if p.split('/')[-1].isdigit())
        return uploaded
    except Exception as e:
        logger.error(f"Error listing uploaded inits: {e}")
        return set()


def get_local_files():
    """Get list of local NC files to upload."""
    pattern = os.path.join(LOCAL_DIR, "spi3_10km_*.nc")
    files = glob.glob(pattern)
    return sorted(files)


def upload_to_icechunk(nc_file):
    """
    Upload a single NC file to IceChunk.
    Returns True on success, False on failure.
    """
    # Extract init string from filename (e.g., spi3_10km_201102.nc -> 201102)
    basename = os.path.basename(nc_file)
    init_str = basename.replace("spi3_10km_", "").replace(".nc", "")

    logger.info(f"Uploading {init_str} from {nc_file}...")

    try:
        # Load the NetCDF file
        ds = xr.open_dataset(nc_file)

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
        commit_id = session.commit(f"Add SPI3 10km data for init {init_str}")
        logger.info(f"Committed {init_str} with ID: {commit_id}")

        ds.close()
        return True

    except Exception as e:
        logger.error(f"Failed to upload {init_str}: {e}")
        import traceback
        traceback.print_exc()
        return False


def remove_local_file(nc_file):
    """Remove local NC file after successful upload."""
    try:
        os.remove(nc_file)
        logger.info(f"Removed local file: {nc_file}")
        return True
    except Exception as e:
        logger.warning(f"Failed to remove {nc_file}: {e}")
        return False


def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Upload Missing SPI3 10km Files to IceChunk")
    logger.info("=" * 60)

    # Get already uploaded inits
    uploaded = get_uploaded_inits()
    logger.info(f"Already uploaded: {len(uploaded)} months")

    # Get local files
    local_files = get_local_files()
    logger.info(f"Local files available: {len(local_files)}")

    # Filter to only files not yet uploaded
    to_upload = []
    for f in local_files:
        basename = os.path.basename(f)
        init_str = basename.replace("spi3_10km_", "").replace(".nc", "")
        if init_str not in uploaded:
            to_upload.append(f)

    logger.info(f"Files to upload: {len(to_upload)}")

    if not to_upload:
        logger.info("No files to upload. All done!")
        return

    # Upload each file
    success = 0
    failed = 0
    removed = 0
    start_time = time.time()

    for i, nc_file in enumerate(to_upload):
        logger.info(f"\n[{i+1}/{len(to_upload)}] Processing...")

        if upload_to_icechunk(nc_file):
            success += 1
            # Remove local file after successful upload
            if remove_local_file(nc_file):
                removed += 1
        else:
            failed += 1

        # Small delay between uploads
        time.sleep(0.5)

    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("UPLOAD COMPLETE")
    logger.info(f"Successful uploads: {success}")
    logger.info(f"Failed uploads: {failed}")
    logger.info(f"Local files removed: {removed}")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
