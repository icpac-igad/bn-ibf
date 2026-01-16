#!/usr/bin/env python3
"""
Script to calculate SPI-3 from SEAS51 GRIB file, regrid to 10km,
and upload to IceChunk store in GCS bucket.

This script:
1. Loads a SEAS51 GRIB file with precipitation forecasts
2. Calculates SPI-3 for each initialization month
3. Regrids from ~100km to 10km resolution for East Africa
4. Uploads to IceChunk store in GCS bucket

Usage:
    python seas51_spi3_to_icechunk.py
"""

import sys
import os
import logging
import time
import xarray as xr
import numpy as np
import xesmf as xe
from xclim.indices import standardized_precipitation_index

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
GRIB_FILE = "/srv/202512-itt/938caf39cc807133341ce734391b2f6b.grib"
OUTPUT_DIR = "/srv/202512-itt/"
SERVICE_ACCOUNT_FILE = "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "seas51_spi3_10km"

# East Africa bounds (consistent with download_and_process_cdi.py)
EA_BOUNDS = {
    'lat_min': -12,
    'lat_max': 23,
    'lon_min': 21,
    'lon_max': 53
}

# 10km resolution (~0.1 degrees)
RESOLUTION_DEG = 0.1


def create_target_grid(bounds, resolution_deg):
    """Create target grid for regridding to 10km."""
    lats = np.arange(bounds['lat_min'], bounds['lat_max'] + resolution_deg, resolution_deg)
    lons = np.arange(bounds['lon_min'], bounds['lon_max'] + resolution_deg, resolution_deg)

    ds_out = xr.Dataset({
        "lat": (["lat"], lats, {"units": "degrees_north"}),
        "lon": (["lon"], lons, {"units": "degrees_east"}),
    })

    return ds_out


def calculate_spi3_for_member(precip_data, member_num, cal_start, cal_end):
    """
    Calculate SPI-3 for a single ensemble member.
    """
    try:
        # Set units
        precip_data.attrs['units'] = 'mm/month'

        # Check data validity
        nan_count = np.isnan(precip_data.values).sum()
        if nan_count > 0:
            nan_percent = (nan_count / precip_data.size) * 100
            if nan_percent > 90:
                logger.warning(f"Skipping member {member_num} due to excessive NaNs ({nan_percent:.1f}%)")
                return None

        # Calculate SPI-3
        spi_3 = standardized_precipitation_index(
            precip_data,
            freq="MS",
            window=3,
            dist="gamma",
            method="APP",
            cal_start=cal_start,
            cal_end=cal_end,
            fitkwargs={"floc": 0}
        )

        # Compute the result
        spi_computed = spi_3.compute()

        # Check output validity
        spi_nan_count = np.isnan(spi_computed.values).sum()
        if spi_nan_count > 0:
            spi_nan_percent = (spi_nan_count / spi_computed.size) * 100
            if spi_nan_percent > 95:
                logger.warning(f"Skipping member {member_num} due to excessive NaNs in output")
                return None

        return spi_computed

    except Exception as e:
        logger.error(f"Error processing member {member_num}: {e}")
        return None


def process_forecast_month(dataset, forecast_month, cal_start='1991-01-01', cal_end='2018-01-01'):
    """
    Process a single forecast month (lead time) for all ensemble members.
    """
    logger.info(f"Processing forecast month {forecast_month}...")

    # Select data for this forecast month
    fm_data = dataset.sel(forecastMonth=forecast_month)

    # List to store SPI-3 for each ensemble member
    member_spi_list = []

    # Get number of ensemble members
    n_members = len(fm_data.number.values)

    # Process each ensemble member
    for member_num in fm_data.number.values:
        # Select data for this member
        member_data = fm_data.sel(number=member_num)
        precip = member_data.tprate

        # Use different calibration periods based on member number
        if member_num < 25:
            member_cal_start = cal_start
            member_cal_end = cal_end
        else:
            member_cal_start = '2017-01-01'
            member_cal_end = '2024-01-01'

        # Calculate SPI-3
        spi_result = calculate_spi3_for_member(
            precip, member_num, member_cal_start, member_cal_end
        )

        if spi_result is not None:
            member_spi_list.append(spi_result)

    if not member_spi_list:
        logger.error(f"No valid members for forecast month {forecast_month}")
        return None

    # Concatenate all members
    try:
        combined_spi = xr.concat(member_spi_list, dim='member')
        logger.info(f"Forecast month {forecast_month}: {len(member_spi_list)}/{n_members} members processed")
        return combined_spi
    except Exception as e:
        logger.error(f"Error combining members for forecast month {forecast_month}: {e}")
        return None


def regrid_to_10km(ds, ds_out, method='bilinear'):
    """
    Regrid dataset to 10km resolution for East Africa.
    Consistent with the regridding in download_and_process_cdi.py.
    """
    logger.info(f"Regridding to 10km using {method} method...")

    # Rename coordinates if needed for xesmf
    if 'longitude' in ds.coords:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

    # Get a template for creating regridder (need 2D lat/lon)
    # Handle different dimension structures
    if 'member' in ds.dims and 'lead' in ds.dims:
        ds_template = ds.isel(member=0, lead=0)
    elif 'member' in ds.dims:
        ds_template = ds.isel(member=0)
    elif 'lead' in ds.dims:
        ds_template = ds.isel(lead=0)
    else:
        ds_template = ds.isel(time=0) if 'time' in ds.dims else ds

    # Create regridder
    regridder = xe.Regridder(ds_template, ds_out, method, periodic=False)

    # Regrid each data variable - xesmf returns data on the new grid
    if isinstance(ds, xr.Dataset):
        regridded_vars = {}
        for var in ds.data_vars:
            regridded_vars[var] = regridder(ds[var], keep_attrs=True)
        regridded = xr.Dataset(regridded_vars)
        regridded.attrs = ds.attrs
    else:
        regridded = regridder(ds, keep_attrs=True)

    logger.info(f"Regridded shape: {dict(regridded.sizes) if hasattr(regridded, 'sizes') else regridded.shape}")

    return regridded


def upload_to_icechunk(ds, init_time, storage_config):
    """
    Upload regridded SPI3 data to IceChunk store.
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
        commit_id = session.commit(f"Add SPI3 10km data for init {init_str}")
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
    output_file = os.path.join(OUTPUT_DIR, f'spi3_10km_{init_str}.nc')
    ds.to_netcdf(output_file)
    logger.info(f"Saved locally: {output_file}")
    return True


def process_single_init(ds, init_time, ds_out, cal_start='1991-01-01', cal_end='2018-01-01'):
    """
    Process a single initialization time:
    1. Select data for this init
    2. Calculate SPI3 for all lead times and members
    3. Regrid to 10km
    4. Upload to IceChunk
    """
    logger.info(f"Processing initialization: {init_time}")
    start_time = time.time()

    # Select data for this init
    ds_init = ds.sel(time=init_time)

    # Calculate SPI3 for all forecast months
    all_leads = []

    for fm in ds.forecastMonth.values:
        fm_result = process_forecast_month(ds_init.expand_dims('time'), int(fm), cal_start, cal_end)
        if fm_result is not None:
            all_leads.append(fm_result)

    if not all_leads:
        logger.warning(f"No valid data for init {init_time}")
        return False

    # Combine all leads
    combined = xr.concat(all_leads, dim='lead')
    combined_ds = combined.to_dataset(name='spi3')

    # Add attributes
    combined_ds['spi3'].attrs['long_name'] = 'Standardized Precipitation Index (3-month)'
    combined_ds['spi3'].attrs['calibration_period'] = f"{cal_start} to {cal_end}"
    combined_ds.attrs['init_time'] = str(init_time)

    # Regrid to 10km
    regridded = regrid_to_10km(combined_ds, ds_out)

    # Add metadata
    regridded.attrs['title'] = f'SEAS51 SPI3 Forecast - 10km resolution'
    regridded.attrs['source'] = GRIB_FILE
    regridded.attrs['resolution'] = '10km (~0.1 degrees)'
    regridded.attrs['region'] = 'East Africa'
    regridded.attrs['init_time'] = str(init_time)

    # Upload to IceChunk
    success = upload_to_icechunk(regridded, init_time, None)

    elapsed = time.time() - start_time
    logger.info(f"Init {init_time} completed in {elapsed:.2f}s")

    return success


def process_all_inits_batch(grib_file, output_dir=OUTPUT_DIR):
    """
    Alternative approach: Calculate SPI3 for all data first, then regrid and upload.
    More efficient for memory as we process by month.
    """
    logger.info("=" * 60)
    logger.info("SEAS51 SPI3 to 10km IceChunk Pipeline")
    logger.info("=" * 60)

    # Load the GRIB file
    logger.info(f"Loading GRIB file: {grib_file}")
    try:
        ds = xr.open_dataset(
            grib_file,
            engine='cfgrib',
            backend_kwargs=dict(time_dims=('forecastMonth', 'time'))
        )
        logger.info(f"Dataset loaded. Dimensions: {dict(ds.dims)}")
        logger.info(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        logger.info(f"Forecast months: {ds.forecastMonth.values}")
        logger.info(f"Ensemble members: {len(ds.number.values)}")
    except Exception as e:
        logger.error(f"Error loading GRIB file: {e}")
        raise

    # Create target grid for 10km East Africa
    ds_out = create_target_grid(EA_BOUNDS, RESOLUTION_DEG)
    logger.info(f"Target grid: {len(ds_out.lat)} x {len(ds_out.lon)} cells")

    # Process each forecast month (lead time)
    all_forecast_months = []

    for fm in ds.forecastMonth.values:
        fm_result = process_forecast_month(ds, int(fm))
        if fm_result is not None:
            all_forecast_months.append(fm_result)

    if not all_forecast_months:
        raise ValueError("No forecast months were successfully processed")

    # Combine all forecast months
    logger.info("Combining all forecast months...")
    combined = xr.concat(all_forecast_months, dim='lead')

    if isinstance(combined, xr.DataArray):
        combined_ds = combined.to_dataset(name='spi3')
    else:
        combined_ds = combined

    # Rename coordinates if needed
    if 'longitude' in combined_ds.dims:
        combined_ds = combined_ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    if 'time' in combined_ds.dims or 'time' in combined_ds.coords:
        combined_ds = combined_ds.rename({'time': 'init'})

    logger.info(f"Combined dataset dimensions: {dict(combined_ds.dims)}")

    # Save intermediate SPI3 result (optional)
    intermediate_file = os.path.join(output_dir, 'seas51_spi3_full.nc')
    logger.info(f"Saving intermediate SPI3 to: {intermediate_file}")
    combined_ds.to_netcdf(intermediate_file)

    # Process each init time: regrid and upload
    init_times = combined_ds.init.values
    total_success = 0
    total_failed = 0

    for i, init_time in enumerate(init_times):
        logger.info(f"\n[{i+1}/{len(init_times)}] Processing init: {init_time}")

        try:
            # Select this init
            ds_init = combined_ds.sel(init=init_time)

            # Regrid to 10km
            regridded = regrid_to_10km(ds_init, ds_out)

            # Add metadata
            if isinstance(regridded, xr.DataArray):
                regridded = regridded.to_dataset(name='spi3')

            regridded.attrs['title'] = 'SEAS51 SPI3 Forecast - 10km resolution'
            regridded.attrs['source'] = grib_file
            regridded.attrs['resolution'] = '10km (~0.1 degrees)'
            regridded.attrs['region'] = 'East Africa'
            regridded.attrs['init_time'] = str(init_time)

            # Upload to IceChunk
            success = upload_to_icechunk(regridded, init_time, None)

            if success:
                total_success += 1
            else:
                total_failed += 1

        except Exception as e:
            logger.error(f"Error processing init {init_time}: {e}")
            total_failed += 1

    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Successful: {total_success}")
    logger.info(f"Failed: {total_failed}")
    logger.info("=" * 60)

    return combined_ds


def main():
    """Main function."""
    logger.info("Starting SEAS51 SPI3 to IceChunk pipeline...")

    # Check if GRIB file exists
    if not os.path.exists(GRIB_FILE):
        logger.error(f"GRIB file not found: {GRIB_FILE}")
        sys.exit(1)

    # Check if service account file exists
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        logger.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
        sys.exit(1)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        start_time = time.time()
        result = process_all_inits_batch(GRIB_FILE, OUTPUT_DIR)
        elapsed = time.time() - start_time

        logger.info(f"\nTotal processing time: {elapsed/60:.2f} minutes")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
