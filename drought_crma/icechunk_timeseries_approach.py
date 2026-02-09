#!/usr/bin/env python3
"""
Demonstration of two approaches for IceChunk storage:
1. Appending to a single store with time dimension (RECOMMENDED)
2. Concatenating separate stores for analysis (current approach workaround)
"""

import xarray as xr
import numpy as np
import pandas as pd
import icechunk
from datetime import datetime

# Configuration
SERVICE_ACCOUNT_FILE = "/scratch/notebook/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "bn_icpac_cdi_store"


# =============================================================================
# APPROACH 1: Single IceChunk Store with Time Dimension (RECOMMENDED)
# =============================================================================

def create_timeseries_store(resolution="10km"):
    """
    Create a single IceChunk store that will hold all time steps.
    Each month is appended as a new time slice and committed.

    This is the RECOMMENDED approach for time-series data.
    """
    prefix = f"{GCS_PATH}/icechunk_timeseries_{resolution}"

    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=prefix,
        service_account_file=SERVICE_ACCOUNT_FILE
    )

    # Create new repository
    repo = icechunk.Repository.create(storage=storage)

    return repo, prefix


def append_month_to_timeseries(repo, nc_file_path, year, month):
    """
    Append a single month's data to the timeseries store.

    The key insight: each month gets added along the time dimension,
    and we commit after each addition for version control.
    """
    # Open the NC file
    ds = xr.open_dataset(nc_file_path)

    # Add time coordinate
    time_value = pd.Timestamp(f"{year}-{month:02d}-01")
    ds = ds.expand_dims(time=[time_value])

    # Get writable session
    session = repo.writable_session("main")
    store = session.store

    # Check if this is the first write or an append
    try:
        existing = xr.open_zarr(store, consolidated=False)
        # Append along time dimension using region
        # Calculate the time index for this new data
        time_idx = len(existing.time)

        # Resize the time dimension and write
        ds.to_zarr(store, append_dim='time')
        existing.close()
    except:
        # First write - initialize the store
        # Set up chunking optimized for time-series analysis
        encoding = {
            'cdi': {
                'chunks': (1, 348, 295) if '10km' in str(repo) else (1, 500, 500)
            }
        }
        ds.to_zarr(store, mode='w', encoding=encoding)

    # Commit this month's data
    commit_id = session.commit(f"Add {year}-{month:02d} data")
    print(f"Committed {year}-{month:02d} with ID: {commit_id}")

    ds.close()
    return commit_id


def query_timeseries(repo, start_date, end_date):
    """
    Query a time range from the timeseries store.

    This is where the single-store approach shines:
    - Native time dimension for slicing
    - Lazy loading - only loads requested data
    - Efficient for temporal analysis
    """
    session = repo.readonly_session(branch='main')
    store = session.store

    ds = xr.open_zarr(store, consolidated=False)

    # Select time range - this is lazy and efficient!
    ds_subset = ds.sel(time=slice(start_date, end_date))

    return ds_subset


# =============================================================================
# APPROACH 2: Concatenate Separate Stores (Workaround for current structure)
# =============================================================================

def open_multiple_months_concat(years, months, resolution="10km"):
    """
    Open multiple separate IceChunk stores and concatenate them.

    This is a WORKAROUND for the current separate-prefix structure.
    It works but is less efficient than a single timeseries store.
    """
    datasets = []

    for year in years:
        for month in months:
            month_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec'][month - 1]

            try:
                filename = f"eadw-cdi-data-{year}-{month_name}"
                if resolution == "1km":
                    prefix = f"{GCS_PATH}/icechunk/{year}/{filename}"
                else:
                    prefix = f"{GCS_PATH}/icechunk_10km/{year}/{filename}"

                storage = icechunk.gcs_storage(
                    bucket=GCS_BUCKET,
                    prefix=prefix,
                    service_account_file=SERVICE_ACCOUNT_FILE
                )

                repo = icechunk.Repository.open(storage=storage)
                session = repo.readonly_session(branch='main')
                store = session.store

                ds = xr.open_zarr(store, consolidated=False)

                # Add time coordinate
                time_value = pd.Timestamp(f"{year}-{month:02d}-01")
                ds = ds.expand_dims(time=[time_value])

                datasets.append(ds)
                print(f"Loaded {year}-{month_name}")

            except Exception as e:
                print(f"Could not load {year}-{month_name}: {e}")

    if not datasets:
        return None

    # Concatenate along time dimension
    ds_combined = xr.concat(datasets, dim='time')
    ds_combined = ds_combined.sortby('time')

    return ds_combined


def demonstrate_pancake_analysis(ds_timeseries):
    """
    Demonstrate 'pancake' style analysis across multiple time steps.

    Pancake analysis = analyzing stacked 2D layers across time
    """
    print("\n" + "=" * 60)
    print("PANCAKE ANALYSIS EXAMPLES")
    print("=" * 60)

    # 1. Temporal mean (collapse time dimension)
    print("\n1. Temporal Mean (average across all months):")
    temporal_mean = ds_timeseries['cdi'].mean(dim='time')
    print(f"   Shape: {temporal_mean.shape}")
    print(f"   Mean CDI: {float(temporal_mean.mean()):.4f}")

    # 2. Temporal standard deviation (variability)
    print("\n2. Temporal Variability (std across time):")
    temporal_std = ds_timeseries['cdi'].std(dim='time')
    print(f"   Shape: {temporal_std.shape}")
    print(f"   Mean Std: {float(temporal_std.mean()):.4f}")

    # 3. Monthly climatology
    print("\n3. Monthly Climatology (group by month):")
    monthly_clim = ds_timeseries['cdi'].groupby('time.month').mean()
    print(f"   Shape: {monthly_clim.shape}")

    # 4. Anomalies
    print("\n4. Anomaly Calculation:")
    climatology = ds_timeseries['cdi'].groupby('time.month').mean()
    anomalies = ds_timeseries['cdi'].groupby('time.month') - climatology
    print(f"   Shape: {anomalies.shape}")

    # 5. Trend analysis (linear trend at each pixel)
    print("\n5. Trend Analysis:")
    # Simple approach: correlation with time
    time_numeric = np.arange(len(ds_timeseries.time))
    # This would compute trend at each pixel
    print("   (Would compute linear trend at each pixel)")

    # 6. Drought frequency
    print("\n6. Drought Frequency (CDI > threshold):")
    drought_threshold = 3.0  # Example threshold
    drought_frequency = (ds_timeseries['cdi'] > drought_threshold).sum(dim='time')
    print(f"   Shape: {drought_frequency.shape}")

    return {
        'temporal_mean': temporal_mean,
        'temporal_std': temporal_std,
        'monthly_climatology': monthly_clim,
        'anomalies': anomalies,
        'drought_frequency': drought_frequency
    }


# =============================================================================
# COMPARISON: When to use which approach
# =============================================================================

def print_comparison():
    """Print comparison of the two approaches."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ICECHUNK STORAGE ARCHITECTURE COMPARISON                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  APPROACH 1: Single Store with Time Dimension (RECOMMENDED)                  ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  Structure:                                                                  ║
║    gs://bucket/store/                                                        ║
║    └── zarr array with dims: (time, y, x)                                   ║
║                                                                              ║
║  Pros:                                                                       ║
║    ✓ Native time dimension for efficient temporal queries                   ║
║    ✓ Single repository to manage                                            ║
║    ✓ Lazy loading works seamlessly                                          ║
║    ✓ Optimal chunking along time dimension                                  ║
║    ✓ Each month = new commit (full version history)                         ║
║    ✓ Can query any time range efficiently: ds.sel(time=slice('2011','2015'))║
║                                                                              ║
║  Cons:                                                                       ║
║    ✗ More complex initial setup                                             ║
║    ✗ Appending requires careful handling                                    ║
║                                                                              ║
║  Best for:                                                                   ║
║    • Time-series analysis                                                    ║
║    • Climatology calculations                                               ║
║    • Trend analysis                                                          ║
║    • Multi-year queries                                                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  APPROACH 2: Separate Prefix per Month (Current Structure)                   ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  Structure:                                                                  ║
║    gs://bucket/store/2011/jan/                                              ║
║    gs://bucket/store/2011/feb/                                              ║
║    gs://bucket/store/2012/jan/                                              ║
║    ...                                                                       ║
║                                                                              ║
║  Pros:                                                                       ║
║    ✓ Simple file-by-file processing                                         ║
║    ✓ Easy to update individual months                                       ║
║    ✓ Isolated failures                                                       ║
║    ✓ Good for single-month analysis                                         ║
║                                                                              ║
║  Cons:                                                                       ║
║    ✗ No native time dimension                                               ║
║    ✗ Must concatenate manually for multi-month analysis                     ║
║    ✗ Concatenation loads metadata from many stores (slow)                   ║
║    ✗ Many separate repositories to manage                                   ║
║    ✗ Harder to maintain consistency                                         ║
║                                                                              ║
║  Best for:                                                                   ║
║    • Single-month snapshots                                                  ║
║    • Independent processing pipelines                                        ║
║    • When each file has different schema                                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  WHEN TO USE CONCAT:                                                         ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║                                                                              ║
║  xr.concat() is useful when:                                                 ║
║    1. You have existing separate stores and need to analyze together        ║
║    2. Data comes from different sources with same schema                    ║
║    3. You want to create a virtual combined dataset                         ║
║                                                                              ║
║  But it's a WORKAROUND, not the optimal solution:                           ║
║    • Each store's metadata must be read (slow with many files)              ║
║    • Memory overhead for tracking multiple stores                            ║
║    • No unified version control                                              ║
║                                                                              ║
║  RECOMMENDATION: Migrate to single timeseries store                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# MIGRATION SCRIPT: Convert separate stores to single timeseries
# =============================================================================

def migrate_to_timeseries_store(years, resolution="10km"):
    """
    Migrate from separate monthly stores to a single timeseries store.

    This creates a new IceChunk repository with proper time dimension
    and commits each month as a version.
    """
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_nums = list(range(1, 13))

    # Create the timeseries store
    prefix = f"{GCS_PATH}/icechunk_timeseries_{resolution}"

    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=prefix,
        service_account_file=SERVICE_ACCOUNT_FILE
    )

    # Try to open existing or create new
    try:
        repo = icechunk.Repository.open(storage=storage)
        print(f"Opened existing timeseries store at: {prefix}")
    except:
        repo = icechunk.Repository.create(storage=storage)
        print(f"Created new timeseries store at: {prefix}")

    all_datasets = []

    # Collect all monthly data
    for year in years:
        for month_name, month_num in zip(months, month_nums):
            try:
                filename = f"eadw-cdi-data-{year}-{month_name}"
                if resolution == "1km":
                    src_prefix = f"{GCS_PATH}/icechunk/{year}/{filename}"
                else:
                    src_prefix = f"{GCS_PATH}/icechunk_10km/{year}/{filename}"

                src_storage = icechunk.gcs_storage(
                    bucket=GCS_BUCKET,
                    prefix=src_prefix,
                    service_account_file=SERVICE_ACCOUNT_FILE
                )

                src_repo = icechunk.Repository.open(storage=src_storage)
                src_session = src_repo.readonly_session(branch='main')
                src_store = src_session.store

                ds = xr.open_zarr(src_store, consolidated=False)

                # Add time coordinate
                time_value = pd.Timestamp(f"{year}-{month_num:02d}-01")
                ds = ds.expand_dims(time=[time_value])

                # Load into memory for concatenation
                ds = ds.compute()

                all_datasets.append(ds)
                print(f"Loaded {year}-{month_name}")

            except Exception as e:
                print(f"Skipping {year}-{month_name}: {e}")

    if not all_datasets:
        print("No datasets to migrate!")
        return None

    # Concatenate all data
    print("\nConcatenating all datasets...")
    ds_combined = xr.concat(all_datasets, dim='time')
    ds_combined = ds_combined.sortby('time')

    print(f"Combined dataset shape: {dict(ds_combined.sizes)}")
    print(f"Time range: {ds_combined.time.min().values} to {ds_combined.time.max().values}")

    # Write to the timeseries store
    session = repo.writable_session("main")
    store = session.store

    # Set up optimal chunking for time-series analysis
    # Chunk size of 12 months = 1 year chunks
    encoding = {
        'cdi': {
            'chunks': (12, ds_combined.sizes['y'], ds_combined.sizes['x'])
        }
    }

    print("\nWriting to timeseries store...")
    ds_combined.to_zarr(store, mode='w')

    # Commit
    commit_id = session.commit(f"Migrated {len(all_datasets)} months to timeseries store")
    print(f"\nMigration complete! Commit ID: {commit_id}")

    return repo, ds_combined


if __name__ == "__main__":
    print_comparison()

    print("\n" + "=" * 70)
    print("DEMONSTRATION: Concatenating Current Separate Stores")
    print("=" * 70)

    # Demo with current structure
    print("\nLoading multiple months using concat approach...")
    ds = open_multiple_months_concat(
        years=[2011],
        months=[1, 2, 3, 4],  # Jan-Apr
        resolution="10km"
    )

    if ds is not None:
        print(f"\nCombined Dataset:")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Time range: {ds.time.values}")

        # Run pancake analysis
        results = demonstrate_pancake_analysis(ds)

        ds.close()
