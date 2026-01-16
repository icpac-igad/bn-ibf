#!/usr/bin/env python3
"""
GEFS Empirical Probability Computation for Flood BN

This module computes empirical exceedance probabilities from GEFS ensemble
forecasts using arbitrary thresholds. It integrates with the existing
GEFS GIK (Grib-Index-Kerchunk) streaming infrastructure.

The module can:
1. Stream GEFS ensemble data using parquet reference files
2. Compute N-day accumulated precipitation
3. Calculate empirical exceedance probabilities for each threshold
4. Extract boundary-level statistics for the flood BN

Thresholds are arbitrary (not return-period based) and can be easily
modified for different flood risk categories.

Usage:
    python gefs_probability.py --parquet-dir ./output_parquet \
        --date 20250106 --run 00 --accum-days 7 \
        --output ./gefs_eprob_20250106_00z.nc

Author: ICPAC IBF Team
Date: January 2026
"""

import numpy as np
import pandas as pd
import xarray as xr
import json
import os
import sys
import warnings
import time
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up anonymous S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

try:
    import fsspec
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    import gribberish
    GRIBBERISH_AVAILABLE = True
except ImportError:
    GRIBBERISH_AVAILABLE = False

try:
    import geopandas as gpd
    import regionmask
    HAS_GEO = True
except ImportError:
    HAS_GEO = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Precipitation thresholds for exceedance probability (mm/day or mm/period)
# These are arbitrary thresholds - modify as needed
PRECIP_THRESHOLDS = {
    'light': 5,        # Light rainfall
    'moderate': 25,    # Moderate rainfall
    'heavy': 50,       # Heavy rainfall
    'very_heavy': 75,  # Very heavy rainfall
    'extreme': 100,    # Extreme rainfall
    'exceptional': 125 # Exceptional rainfall
}

# GEFS grid specification (0.25 degree global)
GEFS_GRID_SHAPE = (721, 1440)  # lat x lon
GEFS_LATS = np.linspace(90, -90, 721)
GEFS_LONS = np.linspace(0, 359.75, 1440)

# East Africa domain
EA_BOUNDS = {
    'lat_min': -12,
    'lat_max': 23,
    'lon_min': 21,
    'lon_max': 53
}

# Compute East Africa indices once
lat_mask = (GEFS_LATS >= EA_BOUNDS['lat_min']) & (GEFS_LATS <= EA_BOUNDS['lat_max'])
lon_mask = (GEFS_LONS >= EA_BOUNDS['lon_min']) & (GEFS_LONS <= EA_BOUNDS['lon_max'])
LAT_INDICES = np.where(lat_mask)[0]
LON_INDICES = np.where(lon_mask)[0]
EA_LATS = GEFS_LATS[LAT_INDICES[0]:LAT_INDICES[-1]+1]
EA_LONS = GEFS_LONS[LON_INDICES[0]:LON_INDICES[-1]+1]

# GEFS timestep (3 hours)
TIMESTEPS_PER_DAY = 8


# ============================================================================
# PARQUET READING UTILITIES (from run_gefs_data_streaming_v2.py)
# ============================================================================

def read_parquet_refs(parquet_path: str) -> Dict:
    """Read parquet file and extract zstore references."""
    df = pd.read_parquet(parquet_path)

    zstore = {}
    for _, row in df.iterrows():
        key = row['key']
        value = row['value']

        if isinstance(value, bytes):
            value = value.decode('utf-8')

        if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
            try:
                value = json.loads(value)
            except:
                pass

        zstore[key] = value

    return zstore


def discover_precipitation_chunks(zstore: Dict) -> Dict:
    """Discover precipitation (tp) chunks in the zstore."""
    tp_prefix = 'tp/accum/surface/tp'

    chunks = []
    chunk_pattern = re.compile(rf'^{re.escape(tp_prefix)}/(\d+)\.0\.0$')

    for key in zstore.keys():
        match = chunk_pattern.match(key)
        if match:
            step_idx = int(match.group(1))
            chunks.append((step_idx, key))

    chunks.sort(key=lambda x: x[0])

    return {
        'path_prefix': tp_prefix,
        'chunks': chunks
    }


# ============================================================================
# GRIB DECODING (from run_gefs_data_streaming_v2.py)
# ============================================================================

def fetch_grib_bytes(zstore: Dict, chunk_key: str, fs) -> Tuple[bytes, int]:
    """Fetch GRIB bytes from S3 using the reference."""
    ref = zstore[chunk_key]

    if isinstance(ref, list) and len(ref) >= 3:
        url, offset, length = ref[0], ref[1], ref[2]
    else:
        raise ValueError(f"Invalid reference format for {chunk_key}: {ref}")

    with fs.open(url, 'rb') as f:
        f.seek(offset)
        grib_bytes = f.read(length)

    return grib_bytes, length


def decode_grib_hybrid(grib_bytes: bytes, grid_shape=GEFS_GRID_SHAPE) -> Tuple[np.ndarray, str]:
    """Decode GRIB with gribberish, fallback to cfgrib on failure."""
    if GRIBBERISH_AVAILABLE:
        try:
            flat_array = gribberish.parse_grib_array(grib_bytes, 0)
            array_2d = flat_array.reshape(grid_shape)
            return array_2d, 'gribberish'
        except Exception:
            pass

    # Fallback to cfgrib
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.grib2') as tmp:
        tmp.write(grib_bytes)
        tmp_path = tmp.name

    try:
        ds = xr.open_dataset(tmp_path, engine='cfgrib')
        var_name = list(ds.data_vars)[0]
        array_2d = ds[var_name].values.copy()
        ds.close()
    finally:
        os.unlink(tmp_path)

    return array_2d, 'cfgrib'


# ============================================================================
# STREAMING AND PROBABILITY COMPUTATION
# ============================================================================

def stream_member_precipitation(
    parquet_path: str,
    subset_to_ea: bool = True
) -> Optional[np.ndarray]:
    """
    Stream precipitation data for a single ensemble member.

    Args:
        parquet_path: Path to the parquet reference file
        subset_to_ea: Whether to subset to East Africa domain

    Returns:
        3D array (timestep, lat, lon) of precipitation data
    """
    member_name = Path(parquet_path).stem.split('_')[0]
    logger.debug(f"Streaming {member_name}...")

    try:
        # Read parquet references
        zstore = read_parquet_refs(parquet_path)

        # Discover precipitation chunks
        tp_info = discover_precipitation_chunks(zstore)

        if not tp_info['chunks']:
            logger.warning(f"No precipitation data found for {member_name}")
            return None

        n_timesteps = len(tp_info['chunks'])

        # Create S3 filesystem
        fs = fsspec.filesystem('s3', anon=True)

        # Determine output shape
        if subset_to_ea:
            n_lats = len(EA_LATS)
            n_lons = len(EA_LONS)
        else:
            n_lats, n_lons = GEFS_GRID_SHAPE

        # Allocate array
        precip_data = np.full((n_timesteps, n_lats, n_lons), np.nan, dtype=np.float32)

        # Stream and decode all timesteps
        for i, (step_idx, chunk_key) in enumerate(tp_info['chunks']):
            try:
                grib_bytes, _ = fetch_grib_bytes(zstore, chunk_key, fs)
                array_2d, _ = decode_grib_hybrid(grib_bytes)

                if subset_to_ea:
                    data_subset = array_2d[LAT_INDICES[0]:LAT_INDICES[-1]+1,
                                          LON_INDICES[0]:LON_INDICES[-1]+1]
                    precip_data[i] = data_subset.astype(np.float32)
                else:
                    precip_data[i] = array_2d.astype(np.float32)

            except Exception as e:
                logger.warning(f"Failed to decode step {step_idx}: {e}")

        return precip_data

    except Exception as e:
        logger.error(f"Error streaming {member_name}: {e}")
        return None


def compute_accumulated_precipitation(
    precip_3hourly: np.ndarray,
    accum_days: int = 7,
    start_day: int = 0
) -> np.ndarray:
    """
    Compute accumulated precipitation over specified period.

    Args:
        precip_3hourly: 3D array (timestep, lat, lon) of 3-hourly precipitation
        accum_days: Number of days to accumulate
        start_day: Starting day (0 = first forecast day)

    Returns:
        2D array (lat, lon) of accumulated precipitation
    """
    n_timesteps = precip_3hourly.shape[0]

    # Skip first timestep (initial condition) if present
    forecast_data = precip_3hourly[1:] if n_timesteps > 1 else precip_3hourly

    # Calculate timestep indices for accumulation period
    start_idx = start_day * TIMESTEPS_PER_DAY
    end_idx = (start_day + accum_days) * TIMESTEPS_PER_DAY

    if end_idx > forecast_data.shape[0]:
        end_idx = forecast_data.shape[0]
        logger.warning(f"Truncating accumulation to available timesteps: {end_idx}")

    if start_idx >= forecast_data.shape[0]:
        logger.warning(f"Start day {start_day} beyond available data")
        return np.full(forecast_data.shape[1:], np.nan)

    # Sum over accumulation period
    accum = np.nansum(forecast_data[start_idx:end_idx], axis=0)

    return accum


def compute_exceedance_probabilities(
    ensemble_precip: np.ndarray,
    thresholds: Dict[str, float] = PRECIP_THRESHOLDS
) -> Dict[str, np.ndarray]:
    """
    Compute empirical exceedance probabilities for each threshold.

    Args:
        ensemble_precip: 3D array (member, lat, lon) of accumulated precipitation
        thresholds: Dictionary of threshold names to values (mm)

    Returns:
        Dictionary of threshold names to probability arrays (0-1)
    """
    n_members = ensemble_precip.shape[0]

    # Count valid (non-NaN) members per pixel
    valid_counts = np.sum(~np.isnan(ensemble_precip), axis=0)
    valid_counts = np.maximum(valid_counts, 1)  # Avoid division by zero

    eprobs = {}

    for name, threshold in thresholds.items():
        # Count members exceeding threshold
        exceeding = np.nansum(ensemble_precip >= threshold, axis=0)

        # Compute probability
        prob = exceeding / valid_counts

        eprobs[f'eprob_{name}'] = prob.astype(np.float32)

        # Log statistics
        max_prob = np.nanmax(prob)
        mean_prob = np.nanmean(prob)
        logger.debug(f"  {name} (>{threshold}mm): max={max_prob:.2f}, mean={mean_prob:.3f}")

    return eprobs


def process_gefs_ensemble(
    parquet_dir: str,
    target_date: str,
    run_hour: str = '00',
    accum_days: int = 7,
    start_day: int = 0,
    thresholds: Dict[str, float] = PRECIP_THRESHOLDS,
    max_members: int = 30
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Process GEFS ensemble to compute exceedance probabilities.

    Args:
        parquet_dir: Directory containing parquet reference files
        target_date: Forecast date (YYYYMMDD)
        run_hour: Model run hour ('00', '06', '12', '18')
        accum_days: Accumulation period in days
        start_day: Starting day for accumulation
        thresholds: Precipitation thresholds
        max_members: Maximum number of members to process

    Returns:
        Tuple of (exceedance_probs_dict, metadata_dict)
    """
    logger.info("="*60)
    logger.info("GEFS ENSEMBLE PROBABILITY COMPUTATION")
    logger.info(f"Date: {target_date}, Run: {run_hour}Z")
    logger.info(f"Accumulation: {accum_days} days starting day {start_day}")
    logger.info("="*60)

    start_time = time.time()

    parquet_dir = Path(parquet_dir)

    # Find parquet files
    pattern = f"gep*_{target_date}_{run_hour}z.parquet"
    parquet_files = sorted(parquet_dir.glob(pattern))

    if not parquet_files:
        logger.error(f"No parquet files found matching {pattern}")
        return None, None

    logger.info(f"Found {len(parquet_files)} ensemble member files")

    # Limit members if needed
    if len(parquet_files) > max_members:
        parquet_files = parquet_files[:max_members]
        logger.info(f"Processing first {max_members} members")

    n_members = len(parquet_files)

    # Process first member to get dimensions
    logger.info("\nStep 1: Determining data dimensions...")
    first_precip = stream_member_precipitation(str(parquet_files[0]))

    if first_precip is None:
        logger.error("Failed to stream first member")
        return None, None

    first_accum = compute_accumulated_precipitation(first_precip, accum_days, start_day)
    spatial_shape = first_accum.shape
    logger.info(f"  Spatial shape: {spatial_shape}")

    # Allocate ensemble array
    ensemble_accum = np.full((n_members,) + spatial_shape, np.nan, dtype=np.float32)
    ensemble_accum[0] = first_accum

    # Process remaining members
    logger.info(f"\nStep 2: Processing {n_members} ensemble members...")
    successful = 1

    for i, pf in enumerate(parquet_files[1:], start=1):
        try:
            precip_data = stream_member_precipitation(str(pf))

            if precip_data is not None:
                accum = compute_accumulated_precipitation(precip_data, accum_days, start_day)
                ensemble_accum[i] = accum
                successful += 1

            if (i + 1) % 5 == 0:
                logger.info(f"  Processed {i + 1}/{n_members} members")

            # Clean up
            del precip_data
            gc.collect()

        except Exception as e:
            logger.warning(f"Failed to process member {i}: {e}")

    logger.info(f"  Successfully processed: {successful}/{n_members}")

    # Compute exceedance probabilities
    logger.info("\nStep 3: Computing exceedance probabilities...")
    eprobs = compute_exceedance_probabilities(ensemble_accum, thresholds)

    # Log summary statistics
    logger.info("\nExceedance Probability Summary:")
    for name, prob in eprobs.items():
        max_p = np.nanmax(prob) * 100
        mean_p = np.nanmean(prob) * 100
        area_50 = np.sum(prob >= 0.5)
        logger.info(f"  {name}: max={max_p:.1f}%, mean={mean_p:.2f}%, pixels>=50%={area_50}")

    elapsed = time.time() - start_time

    metadata = {
        'target_date': target_date,
        'run_hour': run_hour,
        'accum_days': accum_days,
        'start_day': start_day,
        'n_members': n_members,
        'n_successful': successful,
        'thresholds': thresholds,
        'lat': EA_LATS,
        'lon': EA_LONS,
        'processing_time_s': elapsed
    }

    logger.info(f"\nCompleted in {elapsed:.1f} seconds")

    return eprobs, metadata


def save_probabilities_netcdf(
    eprobs: Dict[str, np.ndarray],
    metadata: Dict,
    output_path: str
):
    """Save exceedance probabilities to NetCDF file."""
    logger.info(f"\nSaving to {output_path}...")

    # Create xarray dataset
    data_vars = {}
    for name, prob in eprobs.items():
        data_vars[name] = (['lat', 'lon'], prob)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'lat': metadata['lat'],
            'lon': metadata['lon']
        },
        attrs={
            'title': 'GEFS Precipitation Exceedance Probabilities',
            'target_date': metadata['target_date'],
            'run_hour': metadata['run_hour'],
            'accum_days': metadata['accum_days'],
            'start_day': metadata['start_day'],
            'n_members': metadata['n_members'],
            'n_successful': metadata['n_successful'],
            'thresholds': str(metadata['thresholds']),
            'created': datetime.now().isoformat()
        }
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds.to_netcdf(output_path)
    logger.info(f"Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    return ds


def extract_boundary_probabilities(
    eprobs: Dict[str, np.ndarray],
    metadata: Dict,
    boundaries_path: str
) -> pd.DataFrame:
    """
    Extract boundary-level exceedance probabilities.

    Args:
        eprobs: Dictionary of exceedance probability arrays
        metadata: Metadata dictionary with lat/lon coordinates
        boundaries_path: Path to boundaries GeoJSON

    Returns:
        DataFrame with boundary-level probabilities
    """
    if not HAS_GEO:
        logger.error("Geopandas/regionmask not available")
        return None

    logger.info(f"\nExtracting boundary-level probabilities...")

    # Load boundaries
    gdf = gpd.read_file(boundaries_path)

    if 'GID_1' in gdf.columns:
        gdf['id'] = gdf['GID_1']
    else:
        gdf['id'] = [f'ADMIN_{i:03d}' for i in range(len(gdf))]

    if 'NAME_1' in gdf.columns:
        gdf['name'] = gdf['NAME_1']
    else:
        gdf['name'] = gdf['id']

    gdf['centroid_lon'] = gdf.geometry.centroid.x
    gdf['centroid_lat'] = gdf.geometry.centroid.y

    # Create region mask
    regions = regionmask.from_geopandas(gdf, names='id', abbrevs='id')
    mask = regions.mask(metadata['lon'], metadata['lat'])

    results = []

    for idx, row in gdf.iterrows():
        boundary_id = row['id']

        try:
            region_num = regions.map_keys(boundary_id)
            boundary_mask = mask.values == region_num
            n_pixels = np.sum(boundary_mask)

            result = {
                'boundary_id': boundary_id,
                'boundary_name': row['name'],
                'n_pixels': n_pixels
            }

            for name, prob in eprobs.items():
                if n_pixels > 0:
                    masked = prob[boundary_mask]
                    result[f'{name}_mean'] = float(np.nanmean(masked))
                    result[f'{name}_max'] = float(np.nanmax(masked))
                    result[f'{name}_pct50'] = float(np.sum(masked >= 0.5) / n_pixels)
                else:
                    # Centroid extraction
                    lat_idx = np.argmin(np.abs(metadata['lat'] - row['centroid_lat']))
                    lon_idx = np.argmin(np.abs(metadata['lon'] - row['centroid_lon']))
                    result[f'{name}_mean'] = float(prob[lat_idx, lon_idx])
                    result[f'{name}_max'] = float(prob[lat_idx, lon_idx])
                    result[f'{name}_pct50'] = 1.0 if prob[lat_idx, lon_idx] >= 0.5 else 0.0

            results.append(result)

        except Exception as e:
            logger.warning(f"Failed for {boundary_id}: {e}")

    df = pd.DataFrame(results)
    logger.info(f"Extracted probabilities for {len(df)} boundaries")

    return df


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main CLI routine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute GEFS exceedance probabilities for flood BN"
    )
    parser.add_argument("--parquet-dir", required=True,
                       help="Directory with parquet reference files")
    parser.add_argument("--date", required=True,
                       help="Target date (YYYYMMDD)")
    parser.add_argument("--run", default="00",
                       help="Model run hour (00, 06, 12, 18)")
    parser.add_argument("--accum-days", type=int, default=7,
                       help="Accumulation period in days")
    parser.add_argument("--start-day", type=int, default=0,
                       help="Start day for accumulation")
    parser.add_argument("--output", required=True,
                       help="Output NetCDF file path")
    parser.add_argument("--boundaries", default=None,
                       help="Optional: boundaries GeoJSON for extraction")
    parser.add_argument("--max-members", type=int, default=30,
                       help="Maximum ensemble members to process")

    args = parser.parse_args()

    # Process ensemble
    eprobs, metadata = process_gefs_ensemble(
        parquet_dir=args.parquet_dir,
        target_date=args.date,
        run_hour=args.run,
        accum_days=args.accum_days,
        start_day=args.start_day,
        max_members=args.max_members
    )

    if eprobs is None:
        logger.error("Failed to compute probabilities")
        sys.exit(1)

    # Save NetCDF
    save_probabilities_netcdf(eprobs, metadata, args.output)

    # Extract boundary probabilities if boundaries provided
    if args.boundaries:
        boundary_df = extract_boundary_probabilities(eprobs, metadata, args.boundaries)
        if boundary_df is not None:
            csv_output = args.output.replace('.nc', '_boundaries.csv')
            boundary_df.to_csv(csv_output, index=False)
            logger.info(f"Saved boundary probabilities: {csv_output}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
