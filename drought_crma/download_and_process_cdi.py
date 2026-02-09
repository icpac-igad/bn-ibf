#!/usr/bin/env python3
"""
Script to download CDI NetCDF files from ICPAC FTP, convert to IceChunk/Zarr,
regrid from 1km to 10km, and store in GCS bucket.
"""

import os
import requests
from bs4 import BeautifulSoup
import xarray as xr
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account
import zarr
import tempfile
import re
from pathlib import Path

# Try to import icechunk - if not available, we'll use regular zarr
try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False
    print("Warning: icechunk not available, will use regular zarr storage")

# Configuration
SERVICE_ACCOUNT_FILE = "/scratch/notebook/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "bn_icpac_cdi_store"
BASE_FTP_URL = "https://droughtwatch.icpac.net/ftp/monthly/netcdf"
YEARS = range(2011, 2026)  # 2011 to 2025 inclusive
DOWNLOAD_DIR = "/scratch/notebook/nc_downloads"

# Create download directory
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_gcs_credentials():
    """Get GCS credentials from service account file."""
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return credentials


def list_nc_files(year):
    """List all NC files for a given year from the FTP server."""
    url = f"{BASE_FTP_URL}/{year}/"
    print(f"Fetching file list from: {url}")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse HTML to find .nc files
        soup = BeautifulSoup(response.text, 'html.parser')
        nc_files = []

        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('.nc'):
                nc_files.append(href)

        return nc_files
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list for year {year}: {e}")
        return []


def download_nc_file(year, filename):
    """Download a single NC file."""
    url = f"{BASE_FTP_URL}/{year}/{filename}"
    local_path = os.path.join(DOWNLOAD_DIR, f"{year}_{filename}")

    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return local_path

    print(f"Downloading: {url}")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded: {local_path}")
        return local_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None


def regrid_to_10km(ds):
    """
    Regrid dataset from 1km to 10km resolution using xarray coarsen
    with proper coordinate handling to preserve extent.
    Memory-efficient approach for large datasets.
    """
    # Get the dimension names
    lat_dim = None
    lon_dim = None
    lat_coord = None
    lon_coord = None

    for dim in ds.dims:
        dim_lower = dim.lower()
        if 'lat' in dim_lower or dim_lower == 'y':
            lat_dim = dim
        elif 'lon' in dim_lower or dim_lower == 'x':
            lon_dim = dim

    # Also check coordinates
    for coord in ds.coords:
        coord_lower = coord.lower()
        if 'lat' in coord_lower or coord_lower == 'y':
            lat_coord = coord
        elif 'lon' in coord_lower or coord_lower == 'x':
            lon_coord = coord

    if lat_coord is None:
        lat_coord = lat_dim
    if lon_coord is None:
        lon_coord = lon_dim

    if lat_dim is None or lon_dim is None:
        print(f"Warning: Could not identify lat/lon dimensions. Dims: {list(ds.dims)}")
        return ds

    print(f"Regridding from 1km to 10km using xarray coarsen (memory-efficient)")
    print(f"Using dimensions: lat={lat_dim}, lon={lon_dim}")

    # Get original extent
    if lat_coord in ds.coords:
        lat_vals = ds[lat_coord].values
        lon_vals = ds[lon_coord].values
        lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
        lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
        print(f"Original extent: lat=[{lat_min:.4f}, {lat_max:.4f}], lon=[{lon_min:.4f}, {lon_max:.4f}]")

    lat_size = ds.sizes[lat_dim]
    lon_size = ds.sizes[lon_dim]
    print(f"Original shape: lat={lat_size}, lon={lon_size}")

    # Coarsening factor (1km to 10km = factor of 10)
    coarsen_factor = 10

    if lat_size < coarsen_factor or lon_size < coarsen_factor:
        print(f"Warning: Dimensions too small to coarsen by {coarsen_factor}. Skipping regrid.")
        return ds

    try:
        # Load to memory if dask-backed, then coarsen
        if hasattr(ds, 'chunks') and ds.chunks:
            print("Loading data to memory for coarsening...")
            ds = ds.compute()

        print(f"Applying coarsening with factor {coarsen_factor}...")

        # Use coarsen with mean aggregation
        ds_coarse = ds.coarsen(
            {lat_dim: coarsen_factor, lon_dim: coarsen_factor},
            boundary='trim'
        ).mean()

        new_lat_size = ds_coarse.sizes.get(lat_dim, ds_coarse.sizes.get('lat', 0))
        new_lon_size = ds_coarse.sizes.get(lon_dim, ds_coarse.sizes.get('lon', 0))
        print(f"Regridded shape: lat={new_lat_size}, lon={new_lon_size}")

        # Verify extent is preserved (approximately)
        if lat_coord in ds_coarse.coords:
            new_lat_vals = ds_coarse[lat_coord].values if lat_coord in ds_coarse.coords else ds_coarse[lat_dim].values
            new_lat_min, new_lat_max = float(new_lat_vals.min()), float(new_lat_vals.max())
            print(f"Regridded extent: lat=[{new_lat_min:.4f}, {new_lat_max:.4f}]")

        return ds_coarse

    except Exception as e:
        print(f"Coarsening failed: {e}")
        import traceback
        traceback.print_exc()
        return ds


def process_and_upload_to_gcs(nc_file_path, credentials, year, filename):
    """
    Process NC file: convert to Zarr and upload to GCS.
    """
    print(f"\nProcessing: {nc_file_path}")

    # Open the NetCDF file with chunking for memory efficiency
    try:
        ds = xr.open_dataset(nc_file_path, chunks={'y': 500, 'x': 500})
        print(f"Dataset loaded. Variables: {list(ds.data_vars)}")
        print(f"Dimensions: {dict(ds.sizes)}")
    except Exception as e:
        print(f"Error opening NetCDF file: {e}")
        return False

    # Create base filename without extension
    base_name = os.path.splitext(filename)[0]

    # Store original 1km resolution data
    zarr_path_1km = f"{GCS_PATH}/1km/{year}/{base_name}.zarr"
    gcs_path_1km = f"gs://{GCS_BUCKET}/{zarr_path_1km}"

    print(f"Uploading 1km data to: {gcs_path_1km}")

    try:
        # Use gcsfs for zarr storage with service account file path
        import gcsfs
        fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_FILE)

        # Create zarr store
        store_1km = fs.get_mapper(f"{GCS_BUCKET}/{zarr_path_1km}")

        # Write to zarr
        ds.to_zarr(store_1km, mode='w', consolidated=True)
        print(f"Successfully uploaded 1km data")

    except Exception as e:
        print(f"Error uploading 1km data: {e}")
        # Try local zarr storage as fallback
        local_zarr_path = os.path.join(DOWNLOAD_DIR, f"{year}_{base_name}_1km.zarr")
        print(f"Saving locally to: {local_zarr_path}")
        ds.to_zarr(local_zarr_path, mode='w', consolidated=True)

    # Regrid to 10km
    ds_10km = regrid_to_10km(ds)

    # Store 10km resolution data
    zarr_path_10km = f"{GCS_PATH}/10km/{year}/{base_name}.zarr"
    gcs_path_10km = f"gs://{GCS_BUCKET}/{zarr_path_10km}"

    print(f"Uploading 10km data to: {gcs_path_10km}")

    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_FILE)

        store_10km = fs.get_mapper(f"{GCS_BUCKET}/{zarr_path_10km}")
        ds_10km.to_zarr(store_10km, mode='w', consolidated=True)
        print(f"Successfully uploaded 10km data")

    except Exception as e:
        print(f"Error uploading 10km data: {e}")
        local_zarr_path = os.path.join(DOWNLOAD_DIR, f"{year}_{base_name}_10km.zarr")
        print(f"Saving locally to: {local_zarr_path}")
        ds_10km.to_zarr(local_zarr_path, mode='w', consolidated=True)

    # Close datasets
    ds.close()
    ds_10km.close()

    return True


def process_with_icechunk(nc_file_path, credentials, year, filename):
    """
    Process NC file using IceChunk store (if available).
    """
    if not HAS_ICECHUNK:
        return process_and_upload_to_gcs(nc_file_path, credentials, year, filename)

    print(f"\nProcessing with IceChunk: {nc_file_path}")

    # Open the NetCDF file with chunking for memory efficiency
    try:
        ds = xr.open_dataset(nc_file_path, chunks={'y': 500, 'x': 500})
        print(f"Dataset loaded. Variables: {list(ds.data_vars)}")
    except Exception as e:
        print(f"Error opening NetCDF file: {e}")
        return False

    base_name = os.path.splitext(filename)[0]

    try:
        # Configure GCS storage for IceChunk using service account file
        storage = icechunk.gcs_storage(
            bucket=GCS_BUCKET,
            prefix=f"{GCS_PATH}/icechunk/{year}/{base_name}",
            service_account_file=SERVICE_ACCOUNT_FILE
        )

        # Try to open existing repository, or create new one
        try:
            repo = icechunk.Repository.open(storage=storage)
            print("Opened existing IceChunk repository")
        except Exception:
            repo = icechunk.Repository.create(storage=storage)
            print("Created new IceChunk repository")

        session = repo.writable_session("main")
        store = session.store

        # Write 1km data
        print("Writing 1km data to IceChunk...")
        ds.to_zarr(store, mode='w', consolidated=False)

        # Commit the changes
        commit_id = session.commit(f"Add 1km CDI data for {year}/{base_name}")
        print(f"Committed 1km data with ID: {commit_id}")

        # Regrid and write 10km data
        ds_10km = regrid_to_10km(ds)

        # Create separate store for 10km data
        storage_10km = icechunk.gcs_storage(
            bucket=GCS_BUCKET,
            prefix=f"{GCS_PATH}/icechunk_10km/{year}/{base_name}",
            service_account_file=SERVICE_ACCOUNT_FILE
        )

        # Try to open existing repository, or create new one
        try:
            repo_10km = icechunk.Repository.open(storage=storage_10km)
            print("Opened existing IceChunk repository for 10km data")
        except Exception:
            repo_10km = icechunk.Repository.create(storage=storage_10km)
            print("Created new IceChunk repository for 10km data")

        session_10km = repo_10km.writable_session("main")
        store_10km = session_10km.store

        print("Writing 10km data to IceChunk...")
        ds_10km.to_zarr(store_10km, mode='w', consolidated=False)
        commit_id_10km = session_10km.commit(f"Add 10km regridded CDI data for {year}/{base_name}")
        print(f"Committed 10km data with ID: {commit_id_10km}")

        ds.close()
        ds_10km.close()

        return True

    except Exception as e:
        print(f"IceChunk processing failed: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to regular zarr storage...")
        ds.close()
        return process_and_upload_to_gcs(nc_file_path, credentials, year, filename)


def main():
    """Main function to orchestrate the download and processing."""
    print("=" * 60)
    print("CDI NetCDF to IceChunk/Zarr Processing Pipeline")
    print("=" * 60)

    # Get credentials
    print("\nLoading GCS credentials...")
    try:
        credentials = get_gcs_credentials()
        print(f"Credentials loaded for project: {credentials.project_id}")
    except Exception as e:
        print(f"Error loading credentials: {e}")
        return

    # Process each year
    total_processed = 0
    total_errors = 0

    for year in YEARS:
        print(f"\n{'=' * 40}")
        print(f"Processing year: {year}")
        print(f"{'=' * 40}")

        # List files for this year
        nc_files = list_nc_files(year)

        if not nc_files:
            print(f"No NC files found for year {year}")
            continue

        print(f"Found {len(nc_files)} NC files for year {year}")

        for filename in nc_files:
            # Download the file
            local_path = download_nc_file(year, filename)

            if local_path is None:
                total_errors += 1
                continue

            # Process and upload
            if HAS_ICECHUNK:
                success = process_with_icechunk(local_path, credentials, year, filename)
            else:
                success = process_and_upload_to_gcs(local_path, credentials, year, filename)

            if success:
                total_processed += 1
            else:
                total_errors += 1

            # Clean up downloaded file to save space
            # os.remove(local_path)

    print("\n" + "=" * 60)
    print("Processing Complete")
    print(f"Total files processed: {total_processed}")
    print(f"Total errors: {total_errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()
