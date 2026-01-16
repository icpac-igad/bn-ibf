#!/usr/bin/env python3
"""
Script to verify and access the SEAS51 SPI3 IceChunk repository.

Usage:
    python verify_icechunk_spi3.py [--init YYYYMM]
"""

import argparse
import icechunk
import xarray as xr
import numpy as np

# Configuration
SERVICE_ACCOUNT_FILE = "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "seas51_spi3_10km"


def list_available_inits():
    """List all available initialization months in the bucket."""
    import gcsfs

    fs = gcsfs.GCSFileSystem(token=SERVICE_ACCOUNT_FILE)

    try:
        # List directories under the SPI3 path
        paths = fs.ls(f"{GCS_BUCKET}/{GCS_PATH}")
        inits = [p.split('/')[-1] for p in paths if p.split('/')[-1].isdigit()]
        inits.sort()
        return inits
    except Exception as e:
        print(f"Error listing inits: {e}")
        return []


def open_icechunk_store(init_str):
    """
    Open an IceChunk store for a specific initialization month.

    Args:
        init_str: Initialization month in YYYYMM format (e.g., "198101")

    Returns:
        xarray.Dataset: The SPI3 dataset
    """
    print(f"Opening IceChunk store for init: {init_str}")

    # Configure GCS storage
    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=f"{GCS_PATH}/{init_str}",
        service_account_file=SERVICE_ACCOUNT_FILE
    )

    # Open repository
    repo = icechunk.Repository.open(storage=storage)
    print(f"Repository opened. Checking branch...")

    # Get the store from main branch
    session = repo.readonly_session("main")
    store = session.store

    # Open with xarray
    ds = xr.open_zarr(store, consolidated=False)

    return ds, repo


def verify_dataset(ds, init_str):
    """
    Verify the dataset structure and content.

    Args:
        ds: xarray.Dataset
        init_str: Initialization month string
    """
    print("\n" + "=" * 60)
    print(f"DATASET VERIFICATION: {init_str}")
    print("=" * 60)

    # Check dimensions
    print("\n📊 Dimensions:")
    for dim, size in ds.sizes.items():
        print(f"  {dim}: {size}")

    # Expected dimensions for 10km East Africa
    expected = {
        'lat': 351,
        'lon': 321,
        'member': 51,
        'lead': 6
    }

    print("\n✅ Dimension Check:")
    all_ok = True
    for dim, expected_size in expected.items():
        actual_size = ds.sizes.get(dim, 0)
        status = "✓" if actual_size == expected_size else "✗"
        print(f"  {dim}: {actual_size} (expected {expected_size}) {status}")
        if actual_size != expected_size:
            all_ok = False

    # Check coordinates
    print("\n🌍 Coordinate Ranges:")
    if 'lat' in ds.coords:
        lat_vals = ds.lat.values
        print(f"  lat: {lat_vals.min():.2f} to {lat_vals.max():.2f}")
    if 'lon' in ds.coords:
        lon_vals = ds.lon.values
        print(f"  lon: {lon_vals.min():.2f} to {lon_vals.max():.2f}")

    # Check data variables
    print("\n📈 Data Variables:")
    for var in ds.data_vars:
        da = ds[var]
        print(f"  {var}:")
        print(f"    shape: {da.shape}")
        print(f"    dtype: {da.dtype}")

        # Sample statistics
        sample = da.isel(member=0, lead=0).values
        valid = ~np.isnan(sample)
        if valid.sum() > 0:
            print(f"    min: {np.nanmin(sample):.3f}")
            print(f"    max: {np.nanmax(sample):.3f}")
            print(f"    mean: {np.nanmean(sample):.3f}")
            print(f"    NaN%: {(~valid).sum() / sample.size * 100:.1f}%")

    # Check attributes
    print("\n📝 Attributes:")
    for key, val in ds.attrs.items():
        print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify SEAS51 SPI3 IceChunk repository"
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Initialization month in YYYYMM format (e.g., 198101). If not provided, lists available inits."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available initialization months"
    )

    args = parser.parse_args()

    if args.list or args.init is None:
        print("Listing available initialization months...")
        inits = list_available_inits()
        if inits:
            print(f"\nFound {len(inits)} initialization months:")
            # Group by year
            years = {}
            for init in inits:
                year = init[:4]
                if year not in years:
                    years[year] = []
                years[year].append(init)

            for year in sorted(years.keys()):
                months = [i[4:] for i in years[year]]
                print(f"  {year}: {', '.join(months)}")
        else:
            print("No initialization months found.")

        if args.init is None and inits:
            # Use first available init for demo
            args.init = inits[0]
            print(f"\nUsing first available init for verification: {args.init}")

    if args.init:
        try:
            ds, repo = open_icechunk_store(args.init)
            verify_dataset(ds, args.init)

            # Show how to access data
            print("\n💡 Example Usage:")
            print("```python")
            print("import icechunk")
            print("import xarray as xr")
            print("")
            print(f"storage = icechunk.gcs_storage(")
            print(f"    bucket='{GCS_BUCKET}',")
            print(f"    prefix='{GCS_PATH}/{args.init}',")
            print(f"    service_account_file='<your_service_account.json>'")
            print(f")")
            print("repo = icechunk.Repository.open(storage=storage)")
            print("session = repo.readonly_session('main')")
            print("ds = xr.open_zarr(session.store, consolidated=False)")
            print("")
            print("# Get SPI3 for first ensemble member, lead time 1")
            print("spi3 = ds['spi3'].isel(member=0, lead=0)")
            print("spi3.plot()")
            print("```")

            ds.close()

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
