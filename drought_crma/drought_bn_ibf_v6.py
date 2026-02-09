#!/usr/bin/env python3
"""
Drought Impact-Based Forecasting using Bayesian Networks - Version 6

Key improvements over v5:
1. Pixel-wise return period thresholds for empirical probability calculation
2. Uses SPI-3 threshold NetCDF file instead of global scalar thresholds
3. Automatic regridding of thresholds to match target resolution

Data Sources:
- CDI: GCS bucket 'cdi_arco' path 'bn_icpac_cdi_store/10km/{year}/{month}.zarr'
- SPI3: GCS bucket 'cdi_arco' path 'seas51_spi3_raw/{YYYYMM}'
- Thresholds: /srv/spi_3_return_period_thresholds_20250805/spi_3_return_period_thresholds_20250805.nc

Author: Claude Code
Date: January 2025
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import warnings
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Geospatial imports
try:
    import xarray as xr
    import geopandas as gpd
    import regionmask
    import gcsfs
    HAS_GEO_DEPS = True
except ImportError:
    HAS_GEO_DEPS = False

try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False

try:
    import xesmf as xe
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# GCS Configuration
GCS_BUCKET = "cdi_arco"
CDI_GCS_PATH = "bn_icpac_cdi_store"
SPI3_GCS_PATH = "seas51_spi3_raw"

# Pixel-wise return period thresholds file
DEFAULT_THRESHOLD_FILE = "/srv/spi_3_return_period_thresholds_20250805/spi_3_return_period_thresholds_20250805.nc"

# Default service account file paths (can be overridden)
DEFAULT_SERVICE_ACCOUNT_PATHS = [
    "/scratch/notebook/coiled-data-e4drr_202505.json",
    "/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json",
]

MONTH_MAP = {
    1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr',
    5: 'may', 6: 'jun', 7: 'jul', 8: 'aug',
    9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
}

COUNTRY_CODE_MAP = {
    'BDI': 'Burundi', 'DJI': 'Djibouti', 'ERI': 'Eritrea',
    'ETH': 'Ethiopia', 'KEN': 'Kenya', 'RWA': 'Rwanda',
    'SOM': 'Somalia', 'SSD': 'South Sudan', 'SDN': 'Sudan',
    'TZA': 'Tanzania', 'UGA': 'Uganda'
}

CDI_CATEGORIES = {
    0: 'No_Drought',
    1: 'Mild', 2: 'Mild',
    3: 'Moderate', 4: 'Moderate',
    5: 'Severe', 6: 'Severe', 7: 'Severe',
    8: 'Extreme', 9: 'Extreme', 10: 'Extreme'
}

# Season definitions: (months in season, last_month, description)
SEASONS = {
    'MAM': {'months': [3, 4, 5], 'last_month': 5, 'name': 'March-April-May'},
    'JJA': {'months': [6, 7, 8], 'last_month': 8, 'name': 'June-July-August'},
    'SON': {'months': [9, 10, 11], 'last_month': 11, 'name': 'September-October-November'},
    'OND': {'months': [10, 11, 12], 'last_month': 12, 'name': 'October-November-December'},
    'JFM': {'months': [1, 2, 3], 'last_month': 3, 'name': 'January-February-March'},
}

# Return period names (matching threshold file variables)
RETURN_PERIODS = ['3yr', '5yr', '10yr', '20yr', '50yr']

# Fallback scalar thresholds (used when threshold file unavailable)
FALLBACK_THRESHOLDS = {
    '3yr': -0.43,
    '5yr': -0.68,
    '10yr': -0.84,
    '20yr': -1.04,
    '50yr': -1.28,
}

DEFAULT_CDI_MONTHS = 6
TEMPORAL_WEIGHTS = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_service_account_file() -> Optional[str]:
    """Find an available service account file."""
    for path in DEFAULT_SERVICE_ACCOUNT_PATHS:
        if os.path.exists(path):
            return path
    return None


def categorize_cdi(cdi_value: float) -> str:
    """Categorize CDI value into drought severity."""
    if np.isnan(cdi_value):
        return 'Unknown'
    cdi_int = int(round(max(0, min(10, cdi_value))))
    return CDI_CATEGORIES.get(cdi_int, 'Unknown')


def get_season_lead_mapping(season: str, init_year: int, init_month: int) -> Dict:
    """
    Calculate lead time and target information for a season forecast.

    Args:
        season: Season code (MAM, JJA, SON, OND, JFM)
        init_year: Initialization year
        init_month: Initialization month (1-12)

    Returns:
        Dictionary with lead_time, target_year, target_month, season_name
    """
    if season not in SEASONS:
        raise ValueError(f"Unknown season: {season}. Valid: {list(SEASONS.keys())}")

    season_info = SEASONS[season]
    last_month = season_info['last_month']

    # Calculate lead time (months from init to last month of season)
    # Lead 0 = init month is target month
    if last_month >= init_month:
        lead_time = last_month - init_month
        target_year = init_year
    else:
        # Season spans into next year
        lead_time = (12 - init_month) + last_month
        target_year = init_year + 1

    return {
        'lead_time': lead_time,
        'target_year': target_year,
        'target_month': last_month,
        'season': season,
        'season_name': season_info['name'],
        'init_year': init_year,
        'init_month': init_month,
        'init_time': f"{init_year}-{init_month:02d}"
    }


def get_recommended_init_months(season: str) -> List[Tuple[int, int]]:
    """
    Get recommended init months and lead times for a season.

    Returns list of (init_month, lead_time) tuples for leads 3, 4, 5.
    """
    season_info = SEASONS[season]
    last_month = season_info['last_month']

    results = []
    for lead in [5, 4, 3]:
        init_month = last_month - lead
        year_offset = 0
        if init_month <= 0:
            init_month += 12
            year_offset = -1
        results.append((init_month, lead, year_offset))

    return results


def compute_temporal_weights(n_months: int, available_indices: List[int]) -> Dict[int, float]:
    """Compute normalized weights for available months."""
    base_weights = TEMPORAL_WEIGHTS[:n_months]
    available_weights = {i: base_weights[i] for i in available_indices if i < len(base_weights)}
    total = sum(available_weights.values())
    if total > 0:
        return {i: w / total for i, w in available_weights.items()}
    return {}


def compute_trend(cdi_values: List[Tuple[str, float]]) -> Tuple[str, float]:
    """Compute drought trend from time-ordered CDI values."""
    if len(cdi_values) < 2:
        return 'stable', 0.0
    values = [v for _, v in cdi_values if not np.isnan(v)]
    if len(values) < 2:
        return 'stable', 0.0
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]
    trend_magnitude = np.clip(slope / 2.0, -1.0, 1.0)
    if trend_magnitude < -0.15:
        return 'improving', float(trend_magnitude)
    elif trend_magnitude > 0.15:
        return 'worsening', float(trend_magnitude)
    return 'stable', float(trend_magnitude)


def compute_persistence(cdi_values: List[Tuple[str, float]], threshold: float = 1.0) -> int:
    """Count consecutive months in drought state."""
    if not cdi_values:
        return 0
    reversed_values = list(reversed(cdi_values))
    persistence = 0
    for _, cdi in reversed_values:
        if not np.isnan(cdi) and cdi >= threshold:
            persistence += 1
        else:
            break
    return persistence


def compute_data_confidence(n_available: int, n_requested: int, max_lag_months: int = 0) -> float:
    """Compute confidence score based on data availability."""
    lag_penalty = max(0, 1 - max_lag_months * 0.15)
    if n_available == 0:
        return 0.1
    elif n_available == 1:
        return 0.3 * lag_penalty
    elif n_available == 2:
        return 0.5 * lag_penalty
    else:
        return min(1.0, 0.6 + (n_available - 3) * 0.1) * lag_penalty


# ============================================================================
# PIXEL-WISE THRESHOLD LOADER
# ============================================================================

class PixelWiseThresholdLoader:
    """
    Load and manage pixel-wise SPI-3 return period thresholds.
    """

    def __init__(self, threshold_file: str = DEFAULT_THRESHOLD_FILE):
        """
        Initialize the threshold loader.

        Args:
            threshold_file: Path to the NetCDF file with pixel-wise thresholds
        """
        self.threshold_file = threshold_file
        self._thresholds_ds = None
        self._regridded_cache = {}

    def load_thresholds(self) -> xr.Dataset:
        """Load threshold dataset from file."""
        if self._thresholds_ds is not None:
            return self._thresholds_ds

        if not os.path.exists(self.threshold_file):
            raise FileNotFoundError(f"Threshold file not found: {self.threshold_file}")

        logger.info(f"Loading pixel-wise thresholds from {self.threshold_file}")
        self._thresholds_ds = xr.open_dataset(self.threshold_file)

        # Normalize coordinate names if needed
        if 'latitude' in self._thresholds_ds.coords and 'lat' not in self._thresholds_ds.coords:
            self._thresholds_ds = self._thresholds_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

        logger.info(f"Threshold grid: {self._thresholds_ds.lat.size} lat x {self._thresholds_ds.lon.size} lon")

        return self._thresholds_ds

    def get_threshold_for_rp(self, return_period: str) -> xr.DataArray:
        """
        Get threshold DataArray for a specific return period.

        Args:
            return_period: Return period string (e.g., '5yr', '10yr')

        Returns:
            DataArray of thresholds
        """
        ds = self.load_thresholds()
        var_name = f"spi_3_threshold_{return_period}"

        if var_name not in ds.data_vars:
            raise ValueError(f"Threshold variable {var_name} not found. Available: {list(ds.data_vars)}")

        return ds[var_name]

    def regrid_to_target(self, target_lat: np.ndarray, target_lon: np.ndarray,
                         method: str = 'bilinear') -> Dict[str, xr.DataArray]:
        """
        Regrid all threshold variables to target grid.

        Args:
            target_lat: Target latitude coordinates
            target_lon: Target longitude coordinates
            method: Regridding method ('bilinear', 'nearest')

        Returns:
            Dictionary of regridded threshold DataArrays
        """
        cache_key = (tuple(target_lat[:5]), tuple(target_lon[:5]), len(target_lat), len(target_lon))
        if cache_key in self._regridded_cache:
            logger.info("Using cached regridded thresholds")
            return self._regridded_cache[cache_key]

        ds = self.load_thresholds()
        regridded = {}

        logger.info(f"Regridding thresholds from {ds.lat.size}x{ds.lon.size} to {len(target_lat)}x{len(target_lon)}")

        # Prepare source dataset - ensure ascending lat for xesmf
        source_ds = ds.copy()
        if source_ds.lat.values[0] > source_ds.lat.values[-1]:
            source_ds = source_ds.isel(lat=slice(None, None, -1))

        # Prepare target grid - ensure ascending lat for xesmf
        target_lat_sorted = np.sort(target_lat)
        target_ds = xr.Dataset(coords={'lat': target_lat_sorted, 'lon': target_lon})

        if HAS_XESMF:
            try:
                regridder = xe.Regridder(source_ds, target_ds, method, unmapped_to_nan=True)

                for rp in RETURN_PERIODS:
                    var_name = f"spi_3_threshold_{rp}"
                    if var_name in ds.data_vars:
                        # Regrid
                        regridded_var = regridder(source_ds[var_name])

                        # Restore original lat order if needed
                        if target_lat[0] > target_lat[-1]:
                            regridded_var = regridded_var.isel(lat=slice(None, None, -1))
                            regridded_var = regridded_var.assign_coords(lat=target_lat)

                        regridded[rp] = regridded_var
                        logger.info(f"  Regridded {var_name}: {regridded_var.shape}")

                self._regridded_cache[cache_key] = regridded
                return regridded

            except Exception as e:
                logger.warning(f"xesmf regridding failed: {e}, falling back to interpolation")

        # Fallback: nearest neighbor interpolation
        for rp in RETURN_PERIODS:
            var_name = f"spi_3_threshold_{rp}"
            if var_name in ds.data_vars:
                regridded[rp] = ds[var_name].interp(lat=target_lat, lon=target_lon, method='nearest')
                logger.info(f"  Interpolated {var_name}: {regridded[rp].shape}")

        self._regridded_cache[cache_key] = regridded
        return regridded


# ============================================================================
# GCS DATA LOADERS
# ============================================================================

class GCSDataLoader:
    """Base class for GCS data loading."""

    def __init__(self, service_account_file: str = None):
        if service_account_file is None:
            service_account_file = find_service_account_file()
        if service_account_file is None:
            raise ValueError("No service account file found. Please provide one.")

        self.service_account_file = service_account_file
        self.fs = gcsfs.GCSFileSystem(token=service_account_file)
        logger.info(f"GCS filesystem initialized with {service_account_file}")

    def list_available(self, path: str) -> List[str]:
        """List available paths in GCS."""
        try:
            full_path = f"{GCS_BUCKET}/{path}"
            return self.fs.ls(full_path)
        except Exception as e:
            logger.warning(f"Error listing {path}: {e}")
            return []


class GCSCDILoader(GCSDataLoader):
    """Load CDI data from GCS IceChunk stores."""

    def __init__(self, service_account_file: str = None, resolution: str = '10km'):
        super().__init__(service_account_file)
        self.resolution = resolution
        self._cache = {}

    def get_cdi_path(self, year: int, month: int) -> str:
        """Get GCS path for CDI data (IceChunk format)."""
        month_str = MONTH_MAP[month]
        if self.resolution == '10km':
            return f"{CDI_GCS_PATH}/icechunk_10km/{year}/eadw-cdi-data-{year}-{month_str}"
        else:
            return f"{CDI_GCS_PATH}/icechunk/{year}/eadw-cdi-data-{year}-{month_str}"

    def check_availability(self, year: int, month: int) -> bool:
        """Check if CDI data exists for year/month."""
        path = self.get_cdi_path(year, month)
        try:
            return self.fs.exists(f"{GCS_BUCKET}/{path}")
        except:
            return False

    def load_cdi(self, year: int, month: int) -> Optional[xr.Dataset]:
        """Load CDI dataset from GCS IceChunk store."""
        cache_key = (year, month)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.get_cdi_path(year, month)

        try:
            logger.info(f"Loading CDI from gs://{GCS_BUCKET}/{path}")

            if HAS_ICECHUNK:
                try:
                    storage = icechunk.gcs_storage(
                        bucket=GCS_BUCKET,
                        prefix=path,
                        service_account_file=self.service_account_file
                    )
                    repo = icechunk.Repository.open(storage=storage)
                    session = repo.readonly_session(branch="main")
                    store = session.store
                    ds = xr.open_zarr(store, consolidated=False)
                    self._cache[cache_key] = ds
                    return ds
                except Exception as e:
                    logger.warning(f"IceChunk load failed for CDI {year}-{month:02d}: {e}")
                    pass

            # Fallback to regular zarr
            store = self.fs.get_mapper(f"{GCS_BUCKET}/{path}")
            ds = xr.open_zarr(store, consolidated=True)
            self._cache[cache_key] = ds
            return ds

        except Exception as e:
            logger.warning(f"Failed to load CDI {year}-{month:02d}: {e}")
            return None

    def get_month_sequence(self, target_year: int, target_month: int, n_months: int = 6) -> List[Tuple[int, int]]:
        """Generate sequence of (year, month) tuples going backwards."""
        months = []
        year, month = target_year, target_month
        for _ in range(n_months):
            months.append((year, month))
            month -= 1
            if month < 1:
                month = 12
                year -= 1
        return months

    def load_multi_month_cdi(self, target_year: int, target_month: int,
                              n_months: int = DEFAULT_CDI_MONTHS) -> Tuple[Dict, List]:
        """Load multiple months of CDI data."""
        month_sequence = self.get_month_sequence(target_year, target_month, n_months)
        cdi_data = {}
        available = []

        for year, month in month_sequence:
            ds = self.load_cdi(year, month)
            if ds is not None:
                cdi_data[(year, month)] = ds
                available.append((year, month))

        logger.info(f"Loaded {len(available)}/{n_months} CDI months")
        return cdi_data, available


class GCSSPI3LoaderV6(GCSDataLoader):
    """
    Load SPI3 forecast data from GCS IceChunk stores.
    V6: Uses pixel-wise thresholds for empirical probability calculation.
    """

    def __init__(self, service_account_file: str = None,
                 threshold_file: str = DEFAULT_THRESHOLD_FILE):
        super().__init__(service_account_file)
        self._cache = {}
        self.threshold_loader = PixelWiseThresholdLoader(threshold_file)

    def get_spi3_path(self, year: int, month: int) -> str:
        """Get GCS path for SPI3 data."""
        init_str = f"{year}{month:02d}"
        return f"{SPI3_GCS_PATH}/{init_str}"

    def check_availability(self, year: int, month: int) -> bool:
        """Check if SPI3 data exists for init year/month."""
        path = self.get_spi3_path(year, month)
        try:
            return self.fs.exists(f"{GCS_BUCKET}/{path}")
        except:
            return False

    def load_spi3(self, year: int, month: int) -> Optional[xr.Dataset]:
        """Load SPI3 dataset from GCS."""
        cache_key = (year, month)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.get_spi3_path(year, month)

        try:
            logger.info(f"Loading SPI3 from gs://{GCS_BUCKET}/{path}")

            if HAS_ICECHUNK:
                try:
                    storage = icechunk.gcs_storage(
                        bucket=GCS_BUCKET,
                        prefix=path,
                        service_account_file=self.service_account_file
                    )
                    repo = icechunk.Repository.open(storage=storage)
                    session = repo.readonly_session(branch="main")
                    store = session.store
                    ds = xr.open_zarr(store, consolidated=False)
                    self._cache[cache_key] = ds
                    return ds
                except Exception as e:
                    logger.warning(f"IceChunk load failed, trying zarr: {e}")

            # Fallback to regular zarr
            store = self.fs.get_mapper(f"{GCS_BUCKET}/{path}")
            ds = xr.open_zarr(store, consolidated=True)
            self._cache[cache_key] = ds
            return ds

        except Exception as e:
            logger.warning(f"Failed to load SPI3 {year}-{month:02d}: {e}")
            return None

    def get_spi3_for_lead(self, ds: xr.Dataset, lead: int) -> xr.DataArray:
        """
        Extract SPI3 data for a specific lead time.

        Args:
            ds: SPI3 dataset
            lead: Lead time index (0-indexed, or will handle 1-indexed)

        Returns:
            SPI3 DataArray for the specified lead
        """
        if 'spi3' not in ds.data_vars:
            for var in ['spi', 'SPI3', 'SPI']:
                if var in ds.data_vars:
                    spi3 = ds[var]
                    break
            else:
                raise ValueError(f"No SPI3 variable found. Available: {list(ds.data_vars)}")
        else:
            spi3 = ds['spi3']

        # Select lead time
        if 'lead' in spi3.dims:
            lead_vals = spi3.lead.values
            if lead in lead_vals:
                spi3 = spi3.sel(lead=lead)
            elif lead < len(lead_vals):
                spi3 = spi3.isel(lead=lead)
            else:
                raise ValueError(f"Lead {lead} not available. Available: {lead_vals}")
        elif 'forecastMonth' in spi3.dims:
            spi3 = spi3.isel(forecastMonth=lead)

        # Normalize member dimension name
        if 'member' not in spi3.dims and 'number' in spi3.dims:
            spi3 = spi3.rename({'number': 'member'})

        return spi3

    def compute_empirical_probabilities_pixelwise(
            self,
            spi3: xr.DataArray,
            target_lat: np.ndarray = None,
            target_lon: np.ndarray = None,
            regridder: 'xe.Regridder' = None
    ) -> Dict[str, xr.DataArray]:
        """
        Compute empirical exceedance probabilities using pixel-wise thresholds.

        This is the key V6 improvement: instead of using a single global threshold
        for each return period, we use spatially-varying thresholds derived from
        historical SPI-3 data.

        Args:
            spi3: SPI3 DataArray with 'member' dimension (already lead-selected)
            target_lat: Target latitude grid (for regridding to 10km)
            target_lon: Target longitude grid (for regridding to 10km)
            regridder: Pre-built xesmf regridder (optional, for efficiency)

        Returns:
            Dictionary of DataArrays for each return period (at target resolution)
        """
        n_members = spi3.sizes.get('member', 1)
        logger.info(f"Computing PIXEL-WISE empirical probabilities from {n_members} ensemble members")
        logger.info(f"SPI3 original resolution: {spi3.lat.size} x {spi3.lon.size}")

        # Step 1: Regrid SPI3 to target 10km grid if target coordinates provided
        if target_lat is not None and target_lon is not None:
            logger.info(f"Regridding SPI3 to target resolution: {len(target_lat)} x {len(target_lon)}")

            spi3_regridded_members = []

            for member_idx in range(n_members):
                member_data = spi3.isel(member=member_idx)

                if HAS_XESMF and regridder is None:
                    source_ds = member_data.to_dataset(name='spi3')
                    if source_ds.lat.values[0] > source_ds.lat.values[-1]:
                        source_ds = source_ds.isel(lat=slice(None, None, -1))

                    target_lat_sorted = np.sort(target_lat)
                    target_ds = xr.Dataset(coords={'lat': target_lat_sorted, 'lon': target_lon})
                    regridder = xe.Regridder(source_ds, target_ds, 'bilinear', unmapped_to_nan=True)

                if regridder is not None:
                    source_ds = member_data.to_dataset(name='spi3')
                    if source_ds.lat.values[0] > source_ds.lat.values[-1]:
                        source_ds = source_ds.isel(lat=slice(None, None, -1))
                    regridded = regridder(source_ds)['spi3']

                    if target_lat[0] > target_lat[-1]:
                        regridded = regridded.isel(lat=slice(None, None, -1))
                        regridded = regridded.assign_coords(lat=target_lat)
                else:
                    regridded = member_data.interp(lat=target_lat, lon=target_lon, method='nearest')

                spi3_regridded_members.append(regridded)

            spi3 = xr.concat(spi3_regridded_members, dim='member')
            logger.info(f"SPI3 regridded resolution: {spi3.lat.size} x {spi3.lon.size}")

        # Step 2: Load and regrid pixel-wise thresholds to match SPI3 grid
        logger.info("Loading and regridding pixel-wise thresholds...")
        try:
            thresholds = self.threshold_loader.regrid_to_target(
                target_lat=spi3.lat.values,
                target_lon=spi3.lon.values
            )
            use_pixelwise = True
            logger.info(f"Loaded pixel-wise thresholds for {len(thresholds)} return periods")
        except Exception as e:
            logger.warning(f"Failed to load pixel-wise thresholds: {e}")
            logger.warning("Falling back to global scalar thresholds")
            use_pixelwise = False

        # Step 3: Compute empirical probabilities using pixel-wise thresholds
        eprobs = {}

        for rp in RETURN_PERIODS:
            if use_pixelwise and rp in thresholds:
                # PIXEL-WISE: Compare each pixel's SPI3 values against its specific threshold
                threshold_grid = thresholds[rp]

                # Broadcast threshold to member dimension and compare
                # spi3 shape: (member, lat, lon)
                # threshold_grid shape: (lat, lon)
                below = (spi3 <= threshold_grid).sum(dim='member')
                prob = below / n_members

                logger.info(f"  {rp}: pixel-wise thresholds, mean threshold = {float(threshold_grid.mean()):.3f}")
            else:
                # FALLBACK: Use global scalar threshold
                threshold = FALLBACK_THRESHOLDS[rp]
                below = (spi3 <= threshold).sum(dim='member')
                prob = below / n_members
                logger.info(f"  {rp}: scalar threshold = {threshold:.3f}")

            prob.name = f'eprob_{rp}'
            eprobs[f'eprob_{rp}'] = prob

        return eprobs


# ============================================================================
# DATA LOADER V6
# ============================================================================

class DroughtDataLoaderV6:
    """
    Data loader using GCS IceChunk stores for CDI and SPI3.
    V6: Uses pixel-wise thresholds for empirical probability calculation.
    """

    def __init__(self,
                 boundaries_path: str,
                 service_account_file: str = None,
                 n_cdi_months: int = DEFAULT_CDI_MONTHS,
                 cdi_resolution: str = '10km',
                 threshold_file: str = DEFAULT_THRESHOLD_FILE):

        if not HAS_GEO_DEPS:
            raise ImportError("Geospatial dependencies not installed")

        self.boundaries_path = Path(boundaries_path)
        self.n_cdi_months = n_cdi_months

        # Initialize GCS loaders
        self.cdi_loader = GCSCDILoader(service_account_file, resolution=cdi_resolution)
        self.spi3_loader = GCSSPI3LoaderV6(service_account_file, threshold_file=threshold_file)

        # Load boundaries
        self.boundaries = self._load_boundaries()
        self.n_boundaries = len(self.boundaries)

        # Cache for regridded data
        self.target_grid = None
        self._regrid_cache = {}

    def _load_boundaries(self) -> gpd.GeoDataFrame:
        """Load and prepare admin boundaries."""
        gdf = gpd.read_file(self.boundaries_path)

        if 'GID_1' in gdf.columns:
            gdf['id'] = gdf['GID_1']
            gdf['country_code'] = gdf['GID_1'].str.split('.').str[0]
            gdf['country'] = gdf['country_code'].map(COUNTRY_CODE_MAP).fillna('Unknown')
        else:
            gdf['id'] = [f'ADMIN_{i:03d}' for i in range(len(gdf))]
            gdf['country'] = 'Unknown'

        if 'NAME_1' in gdf.columns:
            gdf['name'] = gdf['NAME_1']
        else:
            gdf['name'] = gdf['id']

        gdf['centroid_lon'] = gdf.geometry.centroid.x
        gdf['centroid_lat'] = gdf.geometry.centroid.y

        logger.info(f"Loaded {len(gdf)} boundaries from {self.boundaries_path.name}")
        return gdf

    def _regrid_to_target(self, data: xr.DataArray, target_lat: np.ndarray,
                          target_lon: np.ndarray) -> xr.DataArray:
        """Regrid data to target grid."""
        if not HAS_XESMF:
            logger.warning("xesmf not available, using nearest neighbor interpolation")
            return data.interp(lat=target_lat, lon=target_lon, method='nearest')

        source_ds = data.to_dataset(name='data')

        if source_ds.lat.values[0] > source_ds.lat.values[-1]:
            source_ds = source_ds.isel(lat=slice(None, None, -1))

        target_lat_sorted = np.sort(target_lat)
        target_ds = xr.Dataset(coords={'lat': target_lat_sorted, 'lon': target_lon})

        regridder = xe.Regridder(source_ds, target_ds, 'bilinear', unmapped_to_nan=True)
        regridded = regridder(source_ds)['data']

        if target_lat[0] > target_lat[-1]:
            regridded = regridded.isel(lat=slice(None, None, -1))
            regridded = regridded.assign_coords(lat=target_lat)

        return regridded

    def _extract_boundary_value(self, data: xr.DataArray, boundary_mask: np.ndarray,
                                 centroid_lat: float, centroid_lon: float,
                                 n_pixels: int) -> float:
        """Extract value using area or centroid method."""
        if n_pixels > 0:
            masked = data.values[boundary_mask]
            return float(np.nanmean(masked))
        else:
            try:
                val = float(data.sel(lat=centroid_lat, lon=centroid_lon, method='nearest').values)
                return val if not np.isnan(val) else 0.0
            except:
                return 0.0

    def _extract_boundary_value_numpy(self, data_np: np.ndarray, boundary_mask: np.ndarray,
                                       centroid_lat: float, centroid_lon: float,
                                       n_pixels: int, data_xr: xr.DataArray = None) -> float:
        """Extract value using area or centroid method - OPTIMIZED with pre-computed numpy array."""
        if n_pixels > 0:
            masked = data_np[boundary_mask]
            return float(np.nanmean(masked))
        else:
            if data_xr is not None:
                try:
                    val = float(data_xr.sel(lat=centroid_lat, lon=centroid_lon, method='nearest').values)
                    return val if not np.isnan(val) else 0.0
                except:
                    return 0.0
            return 0.0

    def prepare_analysis_data(self,
                              season: str,
                              init_year: int,
                              init_month: int) -> Tuple[List[Dict], Dict]:
        """
        Prepare data for all boundaries for a specific season forecast.

        Args:
            season: Season code (MAM, JJA, SON, OND, JFM)
            init_year: Forecast initialization year
            init_month: Forecast initialization month

        Processing order:
        1. Load CDI (10km) first to establish target grid
        2. Load SPI3 (~100km) and regrid to 10km
        3. Compute empirical probabilities at 10km using PIXEL-WISE thresholds
        4. Extract boundary statistics
        """
        # Get lead time mapping
        lead_info = get_season_lead_mapping(season, init_year, init_month)
        lead_time = lead_info['lead_time']

        logger.info("="*60)
        logger.info(f"PREPARING ANALYSIS DATA (V6 - Pixel-wise Thresholds)")
        logger.info(f"Season: {season} ({lead_info['season_name']})")
        logger.info(f"Init: {init_year}-{init_month:02d}, Lead: {lead_time}")
        logger.info("="*60)

        # STEP 1: Load CDI data FIRST to get 10km target grid
        logger.info(f"\nStep 1: Loading CDI data (10km) for last {self.n_cdi_months} months...")
        cdi_data_raw, available_cdi = self.cdi_loader.load_multi_month_cdi(
            init_year, init_month, self.n_cdi_months
        )

        if not available_cdi:
            raise ValueError("No CDI data available")

        # Get 10km target grid from first CDI file
        first_cdi_ds = list(cdi_data_raw.values())[0]
        if 'cdi' in first_cdi_ds.data_vars:
            cdi_ref = first_cdi_ds['cdi']
        else:
            cdi_ref = list(first_cdi_ds.data_vars.values())[0]

        # Handle different coordinate naming conventions
        if 'lat' in cdi_ref.coords:
            target_lat = cdi_ref.lat.values
            target_lon = cdi_ref.lon.values
        elif 'y' in cdi_ref.coords:
            target_lat = cdi_ref.y.values
            target_lon = cdi_ref.x.values
        else:
            raise ValueError(f"Cannot determine coordinate names. Coords: {list(cdi_ref.coords)}")

        self.target_grid = xr.Dataset(coords={'lat': target_lat, 'lon': target_lon})
        logger.info(f"Target grid (10km): {len(target_lat)} lat x {len(target_lon)} lon")

        # Pre-compute CDI numpy arrays
        logger.info("Pre-computing CDI arrays to numpy...")
        cdi_data_regridded = {}
        cdi_data_numpy = {}
        for (year, month), cdi_ds in cdi_data_raw.items():
            if 'cdi' in cdi_ds.data_vars:
                cdi_var = cdi_ds['cdi']
            else:
                cdi_var = list(cdi_ds.data_vars.values())[0]

            if 'y' in cdi_var.dims and 'x' in cdi_var.dims:
                cdi_var = cdi_var.rename({'y': 'lat', 'x': 'lon'})

            cdi_data_regridded[(year, month)] = cdi_var
            cdi_data_numpy[(year, month)] = cdi_var.values

        # STEP 2: Load SPI3 forecast data
        logger.info(f"\nStep 2: Loading SPI3 forecast data (~100km)...")
        spi3_ds = self.spi3_loader.load_spi3(init_year, init_month)
        if spi3_ds is None:
            raise ValueError(f"No SPI3 data available for {init_year}-{init_month:02d}")

        spi3_lead = self.spi3_loader.get_spi3_for_lead(spi3_ds, lead_time)
        logger.info(f"SPI3 shape: {spi3_lead.shape}, dims: {spi3_lead.dims}")

        # STEP 3: Compute empirical probabilities with PIXEL-WISE thresholds
        logger.info(f"\nStep 3: Computing PIXEL-WISE empirical probabilities...")
        eprob_data = self.spi3_loader.compute_empirical_probabilities_pixelwise(
            spi3_lead,
            target_lat=target_lat,
            target_lon=target_lon
        )
        logger.info(f"Computed pixel-wise empirical probabilities for {len(eprob_data)} return periods")

        # Pre-compute eprob numpy arrays
        logger.info("Pre-computing eprob arrays to numpy...")
        eprob_data_numpy = {}
        for name, arr in eprob_data.items():
            eprob_data_numpy[name] = arr.values
        logger.info(f"Pre-computed {len(eprob_data_numpy)} eprob arrays")

        # Create regionmask
        regions = regionmask.from_geopandas(self.boundaries, names='id', abbrevs='id')
        mask = regions.mask(target_lon, target_lat)

        # Compute temporal weights
        month_sequence = self.cdi_loader.get_month_sequence(init_year, init_month, self.n_cdi_months)
        available_indices = [month_sequence.index(m) for m in available_cdi]
        weights = compute_temporal_weights(self.n_cdi_months, available_indices)

        # Process all boundaries
        boundaries_data = []
        area_extracted = 0
        centroid_extracted = 0

        logger.info(f"\nExtracting data for {self.n_boundaries} boundaries...")

        for idx, row in self.boundaries.iterrows():
            boundary_id = row['id']
            boundary_name = row['name']
            country = row['country']
            centroid_lat = row['centroid_lat']
            centroid_lon = row['centroid_lon']

            # Get region mask
            try:
                region_num = regions.map_keys(boundary_id)
                boundary_mask = mask.values == region_num
                n_pixels = np.sum(boundary_mask)
            except:
                n_pixels = 0

            if n_pixels > 0:
                extraction_method = 'area'
                area_extracted += 1
            else:
                extraction_method = 'centroid'
                centroid_extracted += 1

            # Extract multi-month CDI
            cdi_time_series = []
            for year, month in reversed(available_cdi):
                cdi_arr = cdi_data_numpy[(year, month)]
                cdi_val = self._extract_boundary_value_numpy(
                    cdi_arr, boundary_mask, centroid_lat, centroid_lon, n_pixels,
                    cdi_data_regridded[(year, month)]
                )
                cdi_time_series.append((f"{year}-{month:02d}", cdi_val if not np.isnan(cdi_val) else 0.0))

            # Compute weighted mean CDI
            weighted_cdi_sum = 0.0
            for (year, month), _ in zip(available_cdi, range(len(available_cdi))):
                idx_in_seq = month_sequence.index((year, month))
                if idx_in_seq in weights:
                    cdi_arr = cdi_data_numpy[(year, month)]
                    cdi_val = self._extract_boundary_value_numpy(
                        cdi_arr, boundary_mask, centroid_lat, centroid_lon, n_pixels,
                        cdi_data_regridded[(year, month)]
                    )
                    weighted_cdi_sum += weights[idx_in_seq] * (cdi_val if not np.isnan(cdi_val) else 0.0)

            # Compute trend and persistence
            trend, trend_magnitude = compute_trend(cdi_time_series)
            persistence = compute_persistence(cdi_time_series)

            # Compute confidence
            max_lag = available_indices[0] if available_indices else self.n_cdi_months
            confidence = compute_data_confidence(len(available_cdi), self.n_cdi_months, max_lag)

            # Extract empirical probabilities
            rp_probs = {}
            for var_name, var_arr in eprob_data_numpy.items():
                rp_probs[var_name] = self._extract_boundary_value_numpy(
                    var_arr, boundary_mask, centroid_lat, centroid_lon, n_pixels,
                    eprob_data.get(var_name)
                )

            # Spatial coverage
            if n_pixels > 0:
                eprob_5yr_arr = eprob_data_numpy.get('eprob_5yr', list(eprob_data_numpy.values())[0])
                eprob_masked = eprob_5yr_arr[boundary_mask]
                spatial_coverage = float(np.sum(eprob_masked > 0.5) / n_pixels)
            else:
                spatial_coverage = 1.0 if rp_probs.get('eprob_5yr', 0) > 0.5 else 0.0

            # Severity index
            severity_index = self._compute_severity_index(rp_probs)

            boundary_data = {
                'id': boundary_id,
                'name': boundary_name,
                'country': country,
                'n_pixels': int(n_pixels),
                'extraction_method': extraction_method,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                # CDI fields
                'cdi_weighted_mean': weighted_cdi_sum,
                'cdi_category': categorize_cdi(weighted_cdi_sum),
                'cdi_trend': trend,
                'cdi_trend_magnitude': trend_magnitude,
                'cdi_persistence_months': persistence,
                'cdi_n_available': len(available_cdi),
                'cdi_n_requested': self.n_cdi_months,
                'cdi_confidence': confidence,
                'cdi_date_range': f"{cdi_time_series[0][0]} to {cdi_time_series[-1][0]}" if cdi_time_series else '',
                'cdi_values': dict(cdi_time_series),
                # Forecast fields
                'spatial_coverage': spatial_coverage,
                'severity_index': severity_index,
                **rp_probs
            }
            boundaries_data.append(boundary_data)

            if (idx + 1) % 50 == 0:
                logger.info(f"  Processed {idx + 1}/{self.n_boundaries}...")

        logger.info(f"\nExtracted {len(boundaries_data)} boundaries:")
        logger.info(f"  Area-based: {area_extracted}")
        logger.info(f"  Centroid-based: {centroid_extracted}")

        metadata = {
            'season': season,
            'season_name': lead_info['season_name'],
            'init_time': lead_info['init_time'],
            'init_year': init_year,
            'init_month': init_month,
            'lead_time': lead_time,
            'target_year': lead_info['target_year'],
            'target_month': lead_info['target_month'],
            'cdi_months_requested': self.n_cdi_months,
            'cdi_months_available': len(available_cdi),
            'cdi_date_range': f"{available_cdi[-1][0]}-{available_cdi[-1][1]:02d} to {available_cdi[0][0]}-{available_cdi[0][1]:02d}",
            'n_boundaries': len(boundaries_data),
            'area_extracted': area_extracted,
            'centroid_extracted': centroid_extracted,
            'data_source': 'GCS IceChunk',
            'threshold_method': 'pixel-wise'  # V6 indicator
        }

        return boundaries_data, metadata

    def _compute_severity_index(self, rp_probs: Dict[str, float]) -> float:
        """Compute severity index from return period probabilities."""
        base_prob = rp_probs.get('eprob_5yr', 0.0)
        boost = 0.0
        if rp_probs.get('eprob_10yr', 0) > 0.3:
            boost += 0.05
        if rp_probs.get('eprob_20yr', 0) > 0.2:
            boost += 0.05
        if rp_probs.get('eprob_50yr', 0) > 0.1:
            boost += 0.05
        return min(1.0, base_prob + boost)


# ============================================================================
# BAYESIAN NETWORK V6 (same as V5)
# ============================================================================

class DroughtBayesianNetworkV6:
    """
    Bayesian Network for drought risk with GCS data integration.
    Same structure as V5, used with pixel-wise threshold data from V6 loader.
    """

    def __init__(self, include_confidence_node: bool = True):
        self.include_confidence_node = include_confidence_node

        edges = [
            ('antecedent_condition', 'risk_level'),
            ('exceedance_prob', 'risk_level'),
            ('spatial_coverage', 'risk_level'),
            ('temporal_trend', 'risk_level'),
            ('risk_level', 'action')
        ]

        if include_confidence_node:
            edges.append(('data_confidence', 'risk_level'))

        self.model = BayesianNetwork(edges)
        self._setup_cpds()

    def _setup_cpds(self):
        """Setup Conditional Probability Distributions."""
        antecedent_cpd = TabularCPD(
            'antecedent_condition', 5,
            [[0.30], [0.25], [0.20], [0.15], [0.10]],
            state_names={'antecedent_condition': ['No_Drought', 'Mild', 'Moderate', 'Severe', 'Extreme']}
        )

        exceed_cpd = TabularCPD(
            'exceedance_prob', 5,
            [[0.30], [0.25], [0.20], [0.15], [0.10]],
            state_names={'exceedance_prob': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']}
        )

        spatial_cpd = TabularCPD(
            'spatial_coverage', 3,
            [[0.40], [0.35], [0.25]],
            state_names={'spatial_coverage': ['Localized', 'Moderate', 'Widespread']}
        )

        trend_cpd = TabularCPD(
            'temporal_trend', 3,
            [[0.30], [0.45], [0.25]],
            state_names={'temporal_trend': ['Improving', 'Stable', 'Worsening']}
        )

        cpds = [antecedent_cpd, exceed_cpd, spatial_cpd, trend_cpd]

        if self.include_confidence_node:
            confidence_cpd = TabularCPD(
                'data_confidence', 3,
                [[0.20], [0.50], [0.30]],
                state_names={'data_confidence': ['Low', 'Medium', 'High']}
            )
            cpds.append(confidence_cpd)

        risk_cpd = self._create_risk_cpt()
        cpds.append(risk_cpd)

        action_cpd = TabularCPD(
            'action', 4,
            np.array([
                [0.95, 0.10, 0.00, 0.00, 0.00],
                [0.05, 0.85, 0.15, 0.00, 0.00],
                [0.00, 0.05, 0.80, 0.20, 0.05],
                [0.00, 0.00, 0.05, 0.80, 0.95],
            ]),
            evidence=['risk_level'],
            evidence_card=[5],
            state_names={
                'action': ['Monitor', 'Be_Aware', 'Be_Prepared', 'Take_Action'],
                'risk_level': ['Minimal', 'Low', 'Moderate', 'High', 'Extreme']
            }
        )
        cpds.append(action_cpd)

        for cpd in cpds:
            self.model.add_cpds(cpd)
        self.model.check_model()

    def _create_risk_cpt(self) -> TabularCPD:
        """Create risk level CPT."""
        if self.include_confidence_node:
            n_combinations = 5 * 5 * 3 * 3 * 3
            evidence = ['antecedent_condition', 'exceedance_prob', 'spatial_coverage',
                       'temporal_trend', 'data_confidence']
            evidence_card = [5, 5, 3, 3, 3]
            state_names = {
                'risk_level': ['Minimal', 'Low', 'Moderate', 'High', 'Extreme'],
                'antecedent_condition': ['No_Drought', 'Mild', 'Moderate', 'Severe', 'Extreme'],
                'exceedance_prob': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
                'spatial_coverage': ['Localized', 'Moderate', 'Widespread'],
                'temporal_trend': ['Improving', 'Stable', 'Worsening'],
                'data_confidence': ['Low', 'Medium', 'High']
            }
        else:
            n_combinations = 5 * 5 * 3 * 3
            evidence = ['antecedent_condition', 'exceedance_prob', 'spatial_coverage', 'temporal_trend']
            evidence_card = [5, 5, 3, 3]
            state_names = {
                'risk_level': ['Minimal', 'Low', 'Moderate', 'High', 'Extreme'],
                'antecedent_condition': ['No_Drought', 'Mild', 'Moderate', 'Severe', 'Extreme'],
                'exceedance_prob': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
                'spatial_coverage': ['Localized', 'Moderate', 'Widespread'],
                'temporal_trend': ['Improving', 'Stable', 'Worsening']
            }

        cpt = np.zeros((5, n_combinations))
        idx = 0

        if self.include_confidence_node:
            for conf in range(3):
                for trend in range(3):
                    for spatial in range(3):
                        for exceed in range(5):
                            for antecedent in range(5):
                                cpt[:, idx] = self._compute_risk_probs(antecedent, exceed, spatial, trend, conf)
                                idx += 1
        else:
            for trend in range(3):
                for spatial in range(3):
                    for exceed in range(5):
                        for antecedent in range(5):
                            cpt[:, idx] = self._compute_risk_probs(antecedent, exceed, spatial, trend, 2)
                            idx += 1

        return TabularCPD('risk_level', 5, cpt, evidence=evidence,
                         evidence_card=evidence_card, state_names=state_names)

    def _compute_risk_probs(self, antecedent: int, exceed: int, spatial: int,
                            trend: int, confidence: int = 2) -> np.ndarray:
        """Compute risk probabilities."""
        base_risk = (antecedent * 0.35 + exceed * 0.50)

        if spatial == 2:
            base_risk += 0.5
        elif spatial == 1:
            base_risk += 0.25

        if trend == 2:
            base_risk += 0.3
        elif trend == 0:
            base_risk -= 0.25

        if antecedent == 4 and exceed >= 3 and trend == 2:
            probs = np.array([0.0, 0.0, 0.05, 0.25, 0.70]) if spatial >= 1 else np.array([0.0, 0.0, 0.10, 0.45, 0.45])
        elif antecedent >= 3 and trend == 2 and exceed >= 2:
            probs = np.array([0.0, 0.0, 0.10, 0.55, 0.35])
        elif trend == 0 and antecedent <= 2 and exceed <= 1:
            probs = np.array([0.60, 0.35, 0.05, 0.0, 0.0])
        elif antecedent <= 1 and exceed >= 3:
            probs = np.array([0.05, 0.20, 0.50, 0.20, 0.05])
        elif base_risk < 1:
            probs = np.array([0.50, 0.40, 0.10, 0.0, 0.0])
        elif base_risk < 2:
            probs = np.array([0.10, 0.35, 0.40, 0.15, 0.0])
        elif base_risk < 3:
            probs = np.array([0.05, 0.15, 0.45, 0.30, 0.05])
        elif base_risk < 4:
            probs = np.array([0.0, 0.05, 0.25, 0.50, 0.20])
        else:
            probs = np.array([0.0, 0.0, 0.10, 0.40, 0.50])

        if confidence == 0:
            uniform = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
            probs = 0.5 * probs + 0.5 * uniform
        elif confidence == 1:
            uniform = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
            probs = 0.8 * probs + 0.2 * uniform

        return probs / probs.sum()

    def _categorize(self, value: float, var: str) -> str:
        """Categorize continuous value to discrete state."""
        if var == 'exceedance':
            if value < 0.2: return 'Very_Low'
            elif value < 0.4: return 'Low'
            elif value < 0.6: return 'Medium'
            elif value < 0.8: return 'High'
            else: return 'Very_High'
        elif var == 'spatial':
            if value < 0.3: return 'Localized'
            elif value < 0.6: return 'Moderate'
            else: return 'Widespread'
        elif var == 'confidence':
            if value < 0.4: return 'Low'
            elif value < 0.7: return 'Medium'
            else: return 'High'
        elif var == 'trend':
            return {'improving': 'Improving', 'stable': 'Stable', 'worsening': 'Worsening'}.get(value.lower(), 'Stable')
        elif var == 'antecedent':
            return {'No_Drought': 'No_Drought', 'Mild': 'Mild', 'Moderate': 'Moderate',
                   'Severe': 'Severe', 'Extreme': 'Extreme', 'Unknown': 'Mild'}.get(value, 'Mild')
        return value

    def process_boundary(self, boundary_data: Dict) -> Dict:
        """Process single boundary through BN."""
        evidence = {
            'antecedent_condition': self._categorize(boundary_data['cdi_category'], 'antecedent'),
            'exceedance_prob': self._categorize(boundary_data['severity_index'], 'exceedance'),
            'spatial_coverage': self._categorize(boundary_data['spatial_coverage'], 'spatial'),
            'temporal_trend': self._categorize(boundary_data.get('cdi_trend', 'stable'), 'trend')
        }

        if self.include_confidence_node:
            evidence['data_confidence'] = self._categorize(boundary_data.get('cdi_confidence', 1.0), 'confidence')

        inference = VariableElimination(self.model)

        risk_query = inference.query(variables=['risk_level'], evidence=evidence)
        risk_probs = risk_query.values
        risk_states = risk_query.state_names['risk_level']
        risk_level = risk_states[np.argmax(risk_probs)]

        action_query = inference.query(variables=['action'], evidence=evidence)
        action_probs = action_query.values
        action_states = action_query.state_names['action']
        recommended_action = action_states[np.argmax(action_probs)]

        return {
            'boundary_id': boundary_data['id'],
            'boundary_name': boundary_data['name'],
            'country': boundary_data['country'],
            'n_pixels': boundary_data['n_pixels'],
            'extraction_method': boundary_data['extraction_method'],
            'cdi_weighted_mean': boundary_data.get('cdi_weighted_mean', 0),
            'cdi_category': boundary_data['cdi_category'],
            'cdi_trend': boundary_data.get('cdi_trend', 'stable'),
            'cdi_trend_magnitude': boundary_data.get('cdi_trend_magnitude', 0),
            'cdi_persistence_months': boundary_data.get('cdi_persistence_months', 0),
            'cdi_n_available': boundary_data.get('cdi_n_available', 1),
            'cdi_confidence': boundary_data.get('cdi_confidence', 1.0),
            'cdi_date_range': boundary_data.get('cdi_date_range', ''),
            'severity_index': boundary_data['severity_index'],
            'spatial_coverage_pct': boundary_data['spatial_coverage'] * 100,
            'eprob_3yr': boundary_data.get('eprob_3yr', np.nan),
            'eprob_5yr': boundary_data.get('eprob_5yr', np.nan),
            'eprob_10yr': boundary_data.get('eprob_10yr', np.nan),
            'eprob_20yr': boundary_data.get('eprob_20yr', np.nan),
            'eprob_50yr': boundary_data.get('eprob_50yr', np.nan),
            'evidence': evidence,
            'risk_level': risk_level,
            'risk_probabilities': dict(zip(risk_states, risk_probs)),
            'action_probabilities': dict(zip(action_states, action_probs)),
            'recommended_action': recommended_action,
            'confidence': float(np.max(action_probs))
        }

    def process_all_boundaries(self, boundaries_data: List[Dict]) -> pd.DataFrame:
        """Process all boundaries."""
        results = []
        for i, data in enumerate(boundaries_data):
            if (i + 1) % 50 == 0:
                logger.info(f"  Processing {i + 1}/{len(boundaries_data)}...")
            try:
                results.append(self.process_boundary(data))
            except Exception as e:
                logger.warning(f"Failed {data.get('id')}: {e}")

        df = pd.DataFrame(results)
        df['high_risk'] = df['recommended_action'].isin(['Be_Prepared', 'Take_Action'])
        return df


# ============================================================================
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def analyze_season(boundaries_path: str,
                   season: str,
                   init_year: int,
                   init_month: int,
                   service_account_file: str = None,
                   threshold_file: str = DEFAULT_THRESHOLD_FILE,
                   n_cdi_months: int = DEFAULT_CDI_MONTHS,
                   output_dir: str = ".",
                   include_confidence_node: bool = True) -> pd.DataFrame:
    """
    Run drought analysis for a specific season using pixel-wise thresholds.

    Args:
        boundaries_path: Path to admin boundaries GeoJSON
        season: Season code (MAM, JJA, SON, OND, JFM)
        init_year: Forecast initialization year
        init_month: Forecast initialization month
        service_account_file: GCS service account JSON file
        threshold_file: Path to pixel-wise threshold NetCDF file
        n_cdi_months: Number of CDI months to use
        output_dir: Directory for output files
        include_confidence_node: Include data confidence in BN
    """
    import time
    start = time.time()

    logger.info("="*70)
    logger.info(f"DROUGHT BN ANALYSIS V6 - {season} Season (Pixel-wise Thresholds)")
    logger.info("="*70)

    # Initialize data loader
    loader = DroughtDataLoaderV6(
        boundaries_path=boundaries_path,
        service_account_file=service_account_file,
        n_cdi_months=n_cdi_months,
        threshold_file=threshold_file
    )

    # Prepare data
    boundaries_data, metadata = loader.prepare_analysis_data(
        season=season,
        init_year=init_year,
        init_month=init_month
    )

    # Initialize BN
    logger.info(f"\nInitializing Bayesian Network (V6)...")
    bn = DroughtBayesianNetworkV6(include_confidence_node=include_confidence_node)

    # Process boundaries
    logger.info("\nProcessing boundaries...")
    results = bn.process_all_boundaries(boundaries_data)

    elapsed = time.time() - start

    # Summary
    logger.info(f"\n" + "="*70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*70)
    logger.info(f"Season: {metadata['season']} ({metadata['season_name']})")
    logger.info(f"Init: {metadata['init_time']}, Lead: {metadata['lead_time']}")
    logger.info(f"CDI: {metadata['cdi_date_range']} ({metadata['cdi_months_available']}/{metadata['cdi_months_requested']} months)")
    logger.info(f"Threshold method: {metadata['threshold_method']}")
    logger.info(f"Boundaries: {len(results)} | Time: {elapsed:.2f}s")

    logger.info(f"\nAction Distribution:")
    for action in ['Monitor', 'Be_Aware', 'Be_Prepared', 'Take_Action']:
        count = (results['recommended_action'] == action).sum()
        pct = count / len(results) * 100
        logger.info(f"  {action:15}: {count:3} ({pct:.1f}%)")

    # Save results
    output_path = Path(output_dir) / f"drought_bn_v6_{season}_{init_year}_{init_month:02d}.csv"
    cols = ['boundary_id', 'boundary_name', 'country', 'cdi_weighted_mean', 'cdi_category',
            'cdi_trend', 'cdi_confidence', 'severity_index', 'spatial_coverage_pct',
            'eprob_3yr', 'eprob_5yr', 'eprob_10yr', 'eprob_20yr', 'eprob_50yr',
            'risk_level', 'recommended_action', 'confidence']
    output_cols = [c for c in cols if c in results.columns]
    results[output_cols].to_csv(output_path, index=False)
    logger.info(f"\nSaved: {output_path}")

    return results


def analyze_year_all_seasons(boundaries_path: str,
                              year: int,
                              service_account_file: str = None,
                              threshold_file: str = DEFAULT_THRESHOLD_FILE,
                              output_dir: str = ".",
                              seasons: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Run analysis for all seasons in a year using pixel-wise thresholds.

    Args:
        boundaries_path: Path to boundaries
        year: Target year for seasonal forecasts
        service_account_file: GCS service account file
        threshold_file: Path to pixel-wise threshold NetCDF file
        output_dir: Output directory
        seasons: List of seasons (default: all)
    """
    if seasons is None:
        seasons = ['JFM', 'MAM', 'JJA', 'SON', 'OND']

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    for season in seasons:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# Processing {season} {year}")
        logger.info(f"{'#'*70}")

        init_configs = get_recommended_init_months(season)
        init_month, lead, year_offset = init_configs[0]
        init_year = year + year_offset

        try:
            df = analyze_season(
                boundaries_path=boundaries_path,
                season=season,
                init_year=init_year,
                init_month=init_month,
                service_account_file=service_account_file,
                threshold_file=threshold_file,
                output_dir=output_dir
            )
            results[season] = df
        except Exception as e:
            logger.error(f"Failed to process {season}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"YEAR {year} SUMMARY (V6 - Pixel-wise Thresholds)")
    logger.info("="*70)
    for season, df in results.items():
        high_risk = df['high_risk'].sum()
        logger.info(f"{season}: {len(df)} boundaries, {high_risk} high-risk")

    return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drought BN IBF V6 - Pixel-wise Thresholds")
    parser.add_argument("--boundaries", required=True, help="Path to boundaries GeoJSON")
    parser.add_argument("--season", default=None, help="Season code (MAM, JJA, SON, OND, JFM)")
    parser.add_argument("--init-year", type=int, required=True, help="Initialization year")
    parser.add_argument("--init-month", type=int, help="Initialization month (auto if not set)")
    parser.add_argument("--year-all-seasons", action="store_true", help="Run all seasons for year")
    parser.add_argument("--service-account", help="GCS service account JSON file")
    parser.add_argument("--threshold-file", default=DEFAULT_THRESHOLD_FILE,
                        help="Path to pixel-wise threshold NetCDF file")
    parser.add_argument("--cdi-months", type=int, default=DEFAULT_CDI_MONTHS)
    parser.add_argument("--output-dir", default=".")

    args = parser.parse_args()

    if args.year_all_seasons:
        analyze_year_all_seasons(
            boundaries_path=args.boundaries,
            year=args.init_year,
            service_account_file=args.service_account,
            threshold_file=args.threshold_file,
            output_dir=args.output_dir
        )
    elif args.season:
        if args.init_month is None:
            init_configs = get_recommended_init_months(args.season)
            args.init_month, _, year_offset = init_configs[0]
            args.init_year += year_offset

        analyze_season(
            boundaries_path=args.boundaries,
            season=args.season,
            init_year=args.init_year,
            init_month=args.init_month,
            service_account_file=args.service_account,
            threshold_file=args.threshold_file,
            n_cdi_months=args.cdi_months,
            output_dir=args.output_dir
        )
    else:
        parser.print_help()
