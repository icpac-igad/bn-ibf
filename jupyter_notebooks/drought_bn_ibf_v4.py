#!/usr/bin/env python3
"""
Drought Impact-Based Forecasting using Bayesian Networks - Version 4

Key improvements over v3:
1. Multi-month CDI support (last 6 months if available)
2. Flexible handling when not all months are available
3. Uncertainty quantification based on CDI data availability
4. Temporal trend detection (improving, stable, worsening)
5. Persistence scoring for drought duration
6. Modified BN structure with data confidence node

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

# Geospatial imports
try:
    import xarray as xr
    import geopandas as gpd
    import regionmask
    HAS_GEO_DEPS = True
except ImportError:
    HAS_GEO_DEPS = False

try:
    import xesmf as xe
    HAS_XESMF = True
except ImportError:
    HAS_XESMF = False


# Constants
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

# Default number of months to look back for CDI
DEFAULT_CDI_MONTHS = 6

# Weights for temporal averaging (most recent month has highest weight)
# Index 0 = most recent, 5 = 6 months ago
TEMPORAL_WEIGHTS = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]


@dataclass
class CDITimeSeries:
    """Container for multi-month CDI data."""
    values: Dict[str, float]  # {YYYY-MM: cdi_value}
    weighted_mean: float
    trend: str  # 'improving', 'stable', 'worsening'
    trend_magnitude: float  # -1 to 1 scale
    persistence_months: int  # consecutive months in drought
    n_available: int  # number of months with data
    n_requested: int  # number of months requested
    confidence: float  # 0-1 based on data availability
    oldest_date: str
    newest_date: str


def categorize_cdi(cdi_value: float) -> str:
    """Categorize CDI value into drought severity."""
    if np.isnan(cdi_value):
        return 'Unknown'
    cdi_int = int(round(max(0, min(10, cdi_value))))
    return CDI_CATEGORIES.get(cdi_int, 'Unknown')


def get_season_from_lead(init_month: int, lead: int) -> str:
    """Get SPI3 season name for given init month and lead."""
    valid_month = ((init_month + lead - 1) % 12) + 1
    month_letters = {
        1: 'J', 2: 'F', 3: 'M', 4: 'A', 5: 'M', 6: 'J',
        7: 'J', 8: 'A', 9: 'S', 10: 'O', 11: 'N', 12: 'D'
    }
    m1 = ((valid_month - 3) % 12) + 1
    m2 = ((valid_month - 2) % 12) + 1
    m3 = valid_month
    return month_letters[m1] + month_letters[m2] + month_letters[m3]


def compute_temporal_weights(n_months: int, available_indices: List[int]) -> Dict[int, float]:
    """
    Compute normalized weights for available months.

    Args:
        n_months: Total months requested
        available_indices: List of available month indices (0 = most recent)

    Returns:
        Dictionary mapping index to normalized weight
    """
    base_weights = TEMPORAL_WEIGHTS[:n_months]
    available_weights = {i: base_weights[i] for i in available_indices if i < len(base_weights)}

    # Normalize
    total = sum(available_weights.values())
    if total > 0:
        return {i: w / total for i, w in available_weights.items()}
    return {}


def compute_trend(cdi_values: List[Tuple[str, float]]) -> Tuple[str, float]:
    """
    Compute drought trend from time-ordered CDI values.

    Args:
        cdi_values: List of (date_str, cdi_value) tuples, oldest to newest

    Returns:
        (trend_category, trend_magnitude)
        - trend_category: 'improving', 'stable', 'worsening'
        - trend_magnitude: -1 (strong improvement) to +1 (strong worsening)
    """
    if len(cdi_values) < 2:
        return 'stable', 0.0

    values = [v for _, v in cdi_values if not np.isnan(v)]
    if len(values) < 2:
        return 'stable', 0.0

    # Simple linear regression slope
    x = np.arange(len(values))
    slope = np.polyfit(x, values, 1)[0]

    # Normalize slope to -1, 1 range (assuming max change of ~2 CDI units/month)
    trend_magnitude = np.clip(slope / 2.0, -1.0, 1.0)

    # Categorize
    if trend_magnitude < -0.15:
        trend = 'improving'
    elif trend_magnitude > 0.15:
        trend = 'worsening'
    else:
        trend = 'stable'

    return trend, float(trend_magnitude)


def compute_persistence(cdi_values: List[Tuple[str, float]], drought_threshold: float = 1.0) -> int:
    """
    Count consecutive months in drought state (CDI >= threshold).

    Args:
        cdi_values: List of (date_str, cdi_value) tuples, oldest to newest
        drought_threshold: CDI value above which is considered drought

    Returns:
        Number of consecutive recent months in drought
    """
    if not cdi_values:
        return 0

    # Start from most recent and count backwards
    reversed_values = list(reversed(cdi_values))
    persistence = 0

    for _, cdi in reversed_values:
        if not np.isnan(cdi) and cdi >= drought_threshold:
            persistence += 1
        else:
            break

    return persistence


def compute_data_confidence(n_available: int, n_requested: int, max_lag_months: int = 0) -> float:
    """
    Compute confidence score based on data availability.

    Args:
        n_available: Number of CDI months actually available
        n_requested: Number of CDI months requested
        max_lag_months: Maximum lag in months for the most recent data

    Returns:
        Confidence score 0-1
    """
    # Base confidence from availability ratio
    availability_ratio = n_available / n_requested if n_requested > 0 else 0

    # Penalize for lag in most recent data
    lag_penalty = max(0, 1 - max_lag_months * 0.15)  # -15% per month lag

    # Minimum confidence thresholds
    if n_available == 0:
        return 0.1  # Very low but not zero (allows BN to work with priors)
    elif n_available == 1:
        return 0.3 * lag_penalty
    elif n_available == 2:
        return 0.5 * lag_penalty
    elif n_available >= 3:
        return min(1.0, 0.6 + (n_available - 3) * 0.1) * lag_penalty

    return availability_ratio * lag_penalty


class MultiMonthCDILoader:
    """Load and process multiple months of CDI data."""

    def __init__(self, cdi_base_path: str = "/srv/icpac_monthly_netcdf"):
        self.cdi_base_path = Path(cdi_base_path)
        self._cache = {}

    def get_cdi_file_path(self, year: int, month: int) -> Path:
        month_str = MONTH_MAP[month]
        filename = f"eadw-cdi-data-{year}-{month_str}.nc"
        return self.cdi_base_path / str(year) / filename

    def check_availability(self, year: int, month: int) -> bool:
        return self.get_cdi_file_path(year, month).exists()

    def load_cdi(self, year: int, month: int) -> Optional[xr.DataArray]:
        """Load single CDI file, return None if not available."""
        cache_key = (year, month)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.get_cdi_file_path(year, month)
        if not path.exists():
            return None

        try:
            ds = xr.open_dataset(path)
            cdi = ds['cdi']
            self._cache[cache_key] = cdi
            return cdi
        except Exception as e:
            print(f"Warning: Failed to load CDI {year}-{month:02d}: {e}")
            return None

    def get_month_sequence(self, target_year: int, target_month: int, n_months: int = 6) -> List[Tuple[int, int]]:
        """
        Generate sequence of (year, month) tuples going backwards from target.

        Returns list ordered from most recent to oldest.
        """
        months = []
        year, month = target_year, target_month

        for _ in range(n_months):
            months.append((year, month))
            month -= 1
            if month < 1:
                month = 12
                year -= 1

        return months

    def load_multi_month_cdi(self,
                             target_year: int,
                             target_month: int,
                             n_months: int = DEFAULT_CDI_MONTHS) -> Tuple[Dict[Tuple[int, int], xr.DataArray], List[Tuple[int, int]]]:
        """
        Load multiple months of CDI data.

        Args:
            target_year: Most recent year to consider
            target_month: Most recent month to consider
            n_months: Number of months to look back

        Returns:
            (cdi_data_dict, available_months)
            - cdi_data_dict: {(year, month): DataArray}
            - available_months: list of (year, month) tuples that were loaded
        """
        month_sequence = self.get_month_sequence(target_year, target_month, n_months)

        cdi_data = {}
        available = []

        for year, month in month_sequence:
            cdi = self.load_cdi(year, month)
            if cdi is not None:
                cdi_data[(year, month)] = cdi
                available.append((year, month))

        return cdi_data, available


class DroughtDataLoaderV4:
    """
    Enhanced data loader with multi-month CDI support and uncertainty quantification.
    """

    def __init__(self,
                 boundaries_path: str,
                 eprob_path: str = "/srv/empirical_probability_output/empirical_probabilities.nc",
                 cdi_base_path: str = "/srv/icpac_monthly_netcdf",
                 n_cdi_months: int = DEFAULT_CDI_MONTHS):

        if not HAS_GEO_DEPS:
            raise ImportError("Geospatial dependencies not installed")

        self.boundaries_path = Path(boundaries_path)
        self.eprob_path = Path(eprob_path)
        self.cdi_loader = MultiMonthCDILoader(cdi_base_path)
        self.n_cdi_months = n_cdi_months

        self.boundaries = self._load_boundaries()
        self.n_boundaries = len(self.boundaries)

        self.eprob_ds = None
        self.target_grid = None
        self._regridder_cache = {}

    def _load_boundaries(self) -> gpd.GeoDataFrame:
        """Load and prepare admin boundaries."""
        gdf = gpd.read_file(self.boundaries_path)

        if 'GID_1' in gdf.columns:
            gdf['id'] = gdf['GID_1']
        else:
            gdf['id'] = [f'ADMIN_{i:03d}' for i in range(len(gdf))]

        if 'NAME_1' in gdf.columns:
            gdf['name'] = gdf['NAME_1']
        else:
            gdf['name'] = gdf['id']

        if 'GID_1' in gdf.columns:
            gdf['country_code'] = gdf['GID_1'].str.split('.').str[0]
            gdf['country'] = gdf['country_code'].map(COUNTRY_CODE_MAP).fillna('Unknown')
        else:
            gdf['country'] = 'Unknown'

        gdf['centroid_lon'] = gdf.geometry.centroid.x
        gdf['centroid_lat'] = gdf.geometry.centroid.y

        print(f"Loaded {len(gdf)} admin boundaries from {self.boundaries_path.name}")
        return gdf

    def load_empirical_probability(self, init_time: str = None, lead: int = None) -> xr.Dataset:
        """Load empirical probability data."""
        if self.eprob_ds is None:
            self.eprob_ds = xr.open_dataset(self.eprob_path)
            print(f"Loaded empirical probabilities: {dict(self.eprob_ds.sizes)}")

        ds = self.eprob_ds.copy()

        if init_time is not None:
            ds = ds.sel(init=init_time, method='nearest')

        if lead is not None:
            ds = ds.sel(lead=lead)

        if self.target_grid is None:
            self.target_grid = xr.Dataset(coords={'lat': ds.lat, 'lon': ds.lon})

        return ds

    def regrid_cdi_to_eprob(self, cdi: xr.DataArray) -> xr.DataArray:
        """Regrid CDI to match eprob grid using xesmf."""
        if self.target_grid is None:
            raise ValueError("Load empirical probability first")

        target_lat = self.target_grid.lat.values
        target_lon = self.target_grid.lon.values

        cdi_vals = cdi.values
        n_y, n_x = cdi_vals.shape

        # Estimate CDI coordinates
        cdi_y = np.linspace(23.10, -11.69, n_y)
        cdi_x = np.linspace(21.91, 51.40, n_x)

        if HAS_XESMF:
            if cdi_y[0] > cdi_y[-1]:
                cdi_y = cdi_y[::-1]
                cdi_vals = cdi_vals[::-1, :]

            source_ds = xr.Dataset(
                {'cdi': (['lat', 'lon'], cdi_vals)},
                coords={'lat': cdi_y, 'lon': cdi_x}
            )

            lat_descending = target_lat[0] > target_lat[-1]
            target_lat_sorted = target_lat[::-1] if lat_descending else target_lat

            target_ds = xr.Dataset(
                coords={'lat': target_lat_sorted, 'lon': target_lon}
            )

            regridder = xe.Regridder(source_ds, target_ds, 'conservative', unmapped_to_nan=True)
            regridded = regridder(source_ds)['cdi']

            if lat_descending:
                regridded = regridded.isel(lat=slice(None, None, -1))
                regridded = regridded.assign_coords(lat=target_lat)

            return regridded
        else:
            # Fallback block-mean
            regridded = np.full((len(target_lat), len(target_lon)), np.nan)
            for i, lat in enumerate(target_lat):
                for j, lon in enumerate(target_lon):
                    lat_mask = (cdi_y >= lat - 0.5) & (cdi_y < lat + 0.5)
                    lon_mask = (cdi_x >= lon - 0.5) & (cdi_x < lon + 0.5)
                    if np.any(lat_mask) and np.any(lon_mask):
                        lat_idx = np.where(lat_mask)[0]
                        lon_idx = np.where(lon_mask)[0]
                        block = cdi_vals[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
                        valid = block[~np.isnan(block)]
                        if len(valid) > 0:
                            regridded[i, j] = np.nanmean(valid)

            return xr.DataArray(regridded, dims=['lat', 'lon'],
                              coords={'lat': target_lat, 'lon': target_lon}, name='cdi')

    def _extract_at_centroid(self, data: xr.DataArray, lat: float, lon: float) -> float:
        """Extract value at nearest grid point to centroid."""
        try:
            val = float(data.sel(lat=lat, lon=lon, method='nearest').values)
            return val if not np.isnan(val) else 0.0
        except:
            return 0.0

    def _extract_boundary_value(self, data: xr.DataArray, boundary_mask: np.ndarray,
                                 centroid_lat: float, centroid_lon: float,
                                 n_pixels: int) -> float:
        """Extract value using area or centroid method."""
        if n_pixels > 0:
            masked = data.values[boundary_mask]
            return float(np.nanmean(masked))
        else:
            return self._extract_at_centroid(data, centroid_lat, centroid_lon)

    def prepare_analysis_data(self,
                              init_year: int = 2024,
                              init_month: int = 12,
                              target_lead: int = 5) -> Tuple[List[Dict], Dict]:
        """
        Prepare data for all boundaries with multi-month CDI support.
        """
        print("="*60)
        print(f"PREPARING ANALYSIS DATA (V4 - Multi-Month CDI)")
        print(f"Init: {init_year}-{init_month:02d}, Lead: {target_lead}")
        print(f"CDI months requested: {self.n_cdi_months}")
        print("="*60)

        # Load forecast data
        init_time = f"{init_year}-{init_month:02d}-01"
        eprob_ds = self.load_empirical_probability(init_time=init_time, lead=target_lead)
        season_name = get_season_from_lead(init_month, target_lead)
        print(f"Target season: {season_name} SPI3")

        # Load multi-month CDI
        print(f"\nLoading CDI data for last {self.n_cdi_months} months...")
        month_sequence = self.cdi_loader.get_month_sequence(init_year, init_month, self.n_cdi_months)
        cdi_data_raw, available_months = self.cdi_loader.load_multi_month_cdi(
            init_year, init_month, self.n_cdi_months
        )

        print(f"Available CDI months: {len(available_months)}/{self.n_cdi_months}")
        for year, month in available_months:
            print(f"  - {year}-{month:02d}")

        if not available_months:
            raise ValueError("No CDI data available in requested period")

        # Regrid all available CDI data
        print("\nRegridding CDI data...")
        cdi_data_regridded = {}
        for (year, month), cdi_raw in cdi_data_raw.items():
            cdi_data_regridded[(year, month)] = self.regrid_cdi_to_eprob(cdi_raw)

        # Get return period probabilities
        rp_vars = ['eprob_3yr', 'eprob_5yr', 'eprob_10yr', 'eprob_20yr', 'eprob_50yr']
        eprob_data = {v: eprob_ds[v] for v in rp_vars if v in eprob_ds}

        # Create regionmask
        lat = eprob_ds.lat.values
        lon = eprob_ds.lon.values
        regions = regionmask.from_geopandas(self.boundaries, names='id', abbrevs='id')
        mask = regions.mask(lon, lat)

        # Compute temporal weights
        available_indices = [month_sequence.index(m) for m in available_months]
        weights = compute_temporal_weights(self.n_cdi_months, available_indices)

        # Process all boundaries
        boundaries_data = []
        area_extracted = 0
        centroid_extracted = 0

        print(f"\nExtracting data for {self.n_boundaries} boundaries...")

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

            # Extract multi-month CDI values
            cdi_time_series = []
            for year, month in reversed(available_months):  # oldest to newest
                cdi_data = cdi_data_regridded[(year, month)]
                cdi_val = self._extract_boundary_value(
                    cdi_data, boundary_mask, centroid_lat, centroid_lon, n_pixels
                )
                date_str = f"{year}-{month:02d}"
                cdi_time_series.append((date_str, cdi_val if not np.isnan(cdi_val) else 0.0))

            # Compute weighted mean CDI
            weighted_cdi_sum = 0.0
            for (year, month), _ in zip(available_months, range(len(available_months))):
                idx_in_sequence = month_sequence.index((year, month))
                if idx_in_sequence in weights:
                    cdi_data = cdi_data_regridded[(year, month)]
                    cdi_val = self._extract_boundary_value(
                        cdi_data, boundary_mask, centroid_lat, centroid_lon, n_pixels
                    )
                    weighted_cdi_sum += weights[idx_in_sequence] * (cdi_val if not np.isnan(cdi_val) else 0.0)

            # Compute trend and persistence
            trend, trend_magnitude = compute_trend(cdi_time_series)
            persistence = compute_persistence(cdi_time_series)

            # Compute confidence
            max_lag = available_indices[0] if available_indices else self.n_cdi_months
            confidence = compute_data_confidence(len(available_months), self.n_cdi_months, max_lag)

            # Create CDI time series object
            cdi_ts = CDITimeSeries(
                values={d: v for d, v in cdi_time_series},
                weighted_mean=weighted_cdi_sum,
                trend=trend,
                trend_magnitude=trend_magnitude,
                persistence_months=persistence,
                n_available=len(available_months),
                n_requested=self.n_cdi_months,
                confidence=confidence,
                oldest_date=cdi_time_series[0][0] if cdi_time_series else '',
                newest_date=cdi_time_series[-1][0] if cdi_time_series else ''
            )

            # Get return period probabilities
            rp_probs = {}
            for var_name, var_data in eprob_data.items():
                rp_probs[var_name] = self._extract_boundary_value(
                    var_data, boundary_mask, centroid_lat, centroid_lon, n_pixels
                )

            # Spatial coverage
            if n_pixels > 0:
                eprob_5yr_masked = eprob_data['eprob_5yr'].values[boundary_mask]
                spatial_coverage = float(np.sum(eprob_5yr_masked > 0.5) / n_pixels)
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
                # Multi-month CDI fields
                'cdi_weighted_mean': cdi_ts.weighted_mean,
                'cdi_category': categorize_cdi(cdi_ts.weighted_mean),
                'cdi_trend': cdi_ts.trend,
                'cdi_trend_magnitude': cdi_ts.trend_magnitude,
                'cdi_persistence_months': cdi_ts.persistence_months,
                'cdi_n_available': cdi_ts.n_available,
                'cdi_n_requested': cdi_ts.n_requested,
                'cdi_confidence': cdi_ts.confidence,
                'cdi_date_range': f"{cdi_ts.oldest_date} to {cdi_ts.newest_date}",
                # Individual month values (for reference)
                'cdi_values': cdi_ts.values,
                # Forecast fields
                'spatial_coverage': spatial_coverage,
                'severity_index': severity_index,
                **rp_probs
            }
            boundaries_data.append(boundary_data)

        print(f"\nExtracted {len(boundaries_data)} boundaries:")
        print(f"  Area-based: {area_extracted}")
        print(f"  Centroid-based: {centroid_extracted}")

        metadata = {
            'init_time': f"{init_year}-{init_month:02d}",
            'target_season': season_name,
            'target_lead': target_lead,
            'cdi_months_requested': self.n_cdi_months,
            'cdi_months_available': len(available_months),
            'cdi_date_range': f"{available_months[-1][0]}-{available_months[-1][1]:02d} to {available_months[0][0]}-{available_months[0][1]:02d}",
            'n_boundaries': len(boundaries_data),
            'area_extracted': area_extracted,
            'centroid_extracted': centroid_extracted
        }

        return boundaries_data, metadata

    def _compute_severity_index(self, rp_probs: Dict[str, float]) -> float:
        """Compute severity index from multiple return period probabilities."""
        base_prob = rp_probs.get('eprob_5yr', 0.0)

        boost = 0.0
        if rp_probs.get('eprob_10yr', 0) > 0.3:
            boost += 0.05
        if rp_probs.get('eprob_20yr', 0) > 0.2:
            boost += 0.05
        if rp_probs.get('eprob_50yr', 0) > 0.1:
            boost += 0.05

        return min(1.0, base_prob + boost)


class DroughtBayesianNetworkV4:
    """
    Enhanced Bayesian Network with data confidence and temporal trend nodes.

    Network structure:
        antecedent_condition ──┐
                               │
        exceedance_prob ───────┼──► risk_level ──► action
                               │
        spatial_coverage ──────┤
                               │
        temporal_trend ────────┤
                               │
        data_confidence ───────┘

    New nodes:
    - temporal_trend: improving, stable, worsening
    - data_confidence: low, medium, high (affects certainty of antecedent)
    """

    def __init__(self, include_confidence_node: bool = True):
        """
        Initialize BN structure.

        Args:
            include_confidence_node: If True, include data_confidence node
        """
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

        # 1. Antecedent Condition (from weighted CDI) - 5 states
        antecedent_cpd = TabularCPD(
            'antecedent_condition', 5,
            [[0.30], [0.25], [0.20], [0.15], [0.10]],
            state_names={'antecedent_condition':
                        ['No_Drought', 'Mild', 'Moderate', 'Severe', 'Extreme']}
        )

        # 2. Exceedance Probability - 5 states
        exceed_cpd = TabularCPD(
            'exceedance_prob', 5,
            [[0.30], [0.25], [0.20], [0.15], [0.10]],
            state_names={'exceedance_prob':
                        ['Very_Low', 'Low', 'Medium', 'High', 'Very_High']}
        )

        # 3. Spatial Coverage - 3 states
        spatial_cpd = TabularCPD(
            'spatial_coverage', 3,
            [[0.40], [0.35], [0.25]],
            state_names={'spatial_coverage': ['Localized', 'Moderate', 'Widespread']}
        )

        # 4. Temporal Trend - 3 states (NEW in V4)
        trend_cpd = TabularCPD(
            'temporal_trend', 3,
            [[0.30], [0.45], [0.25]],
            state_names={'temporal_trend': ['Improving', 'Stable', 'Worsening']}
        )

        cpds = [antecedent_cpd, exceed_cpd, spatial_cpd, trend_cpd]

        # 5. Data Confidence - 3 states (optional)
        if self.include_confidence_node:
            confidence_cpd = TabularCPD(
                'data_confidence', 3,
                [[0.20], [0.50], [0.30]],
                state_names={'data_confidence': ['Low', 'Medium', 'High']}
            )
            cpds.append(confidence_cpd)

        # 6. Risk Level CPT
        risk_cpd = self._create_risk_cpt()
        cpds.append(risk_cpd)

        # 7. Action CPD
        action_cpd = TabularCPD(
            'action', 4,
            np.array([
                [0.95, 0.10, 0.00, 0.00, 0.00],  # Monitor
                [0.05, 0.85, 0.15, 0.00, 0.00],  # Be_Aware
                [0.00, 0.05, 0.80, 0.20, 0.05],  # Be_Prepared
                [0.00, 0.00, 0.05, 0.80, 0.95],  # Take_Action
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

        # Store CPT statistics
        n_risk_combinations = 5 * 5 * 3 * 3  # antecedent × exceed × spatial × trend
        if self.include_confidence_node:
            n_risk_combinations *= 3  # × confidence

        self.cpt_stats = {
            'n_nodes': len(self.model.nodes()),
            'n_edges': len(self.model.edges()),
            'risk_level_combinations': n_risk_combinations,
            'include_confidence_node': self.include_confidence_node
        }

    def _create_risk_cpt(self) -> TabularCPD:
        """
        Create CPT for risk_level.

        With confidence node: 5 × 5 × 3 × 3 × 3 = 675 combinations
        Without: 5 × 5 × 3 × 3 = 225 combinations
        """
        if self.include_confidence_node:
            n_combinations = 5 * 5 * 3 * 3 * 3  # 675
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
            n_combinations = 5 * 5 * 3 * 3  # 225
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
            for conf in range(3):  # Low, Medium, High
                for trend in range(3):  # Improving, Stable, Worsening
                    for spatial in range(3):
                        for exceed in range(5):
                            for antecedent in range(5):
                                probs = self._compute_risk_probs_v4(
                                    antecedent, exceed, spatial, trend, conf
                                )
                                cpt[:, idx] = probs
                                idx += 1
        else:
            for trend in range(3):
                for spatial in range(3):
                    for exceed in range(5):
                        for antecedent in range(5):
                            probs = self._compute_risk_probs_v4(
                                antecedent, exceed, spatial, trend, confidence=2  # assume high
                            )
                            cpt[:, idx] = probs
                            idx += 1

        return TabularCPD(
            'risk_level', 5,
            cpt,
            evidence=evidence,
            evidence_card=evidence_card,
            state_names=state_names
        )

    def _compute_risk_probs_v4(self, antecedent: int, exceed: int, spatial: int,
                                trend: int, confidence: int = 2) -> np.ndarray:
        """
        Compute risk probabilities with temporal trend and data confidence.

        Args:
            antecedent: 0=No_Drought to 4=Extreme
            exceed: 0=Very_Low to 4=Very_High
            spatial: 0=Localized to 2=Widespread
            trend: 0=Improving, 1=Stable, 2=Worsening
            confidence: 0=Low, 1=Medium, 2=High

        Returns:
            Array of [Minimal, Low, Moderate, High, Extreme] probabilities
        """
        # Base risk score
        base_risk = (antecedent * 0.35 + exceed * 0.50)

        # Spatial adjustment
        if spatial == 2:  # Widespread
            base_risk += 0.5
        elif spatial == 1:  # Moderate
            base_risk += 0.25

        # Trend adjustment (key V4 feature)
        if trend == 2:  # Worsening
            base_risk += 0.3
        elif trend == 0:  # Improving
            base_risk -= 0.25

        # EXPERT RULES

        # Rule 1: Extreme + worsening + high exceedance
        if antecedent == 4 and exceed >= 3 and trend == 2:
            if spatial >= 1:
                probs = np.array([0.0, 0.0, 0.02, 0.18, 0.80])
            else:
                probs = np.array([0.0, 0.0, 0.08, 0.42, 0.50])

        # Rule 2: Severe/Extreme + worsening trend
        elif antecedent >= 3 and trend == 2 and exceed >= 2:
            probs = np.array([0.0, 0.0, 0.10, 0.55, 0.35])

        # Rule 3: Improving trend reduces risk significantly
        elif trend == 0 and antecedent <= 2:
            if exceed <= 1:
                probs = np.array([0.65, 0.30, 0.05, 0.0, 0.0])
            else:
                probs = np.array([0.35, 0.45, 0.15, 0.05, 0.0])

        # Rule 4: Severe antecedent + High exceedance (stable/worsening)
        elif antecedent == 3 and exceed >= 3 and trend >= 1:
            probs = np.array([0.0, 0.05, 0.20, 0.55, 0.20])

        # Rule 5: Moderate with forecast concern
        elif antecedent == 2 and exceed >= 2:
            if spatial == 2:
                probs = np.array([0.0, 0.10, 0.50, 0.35, 0.05])
            else:
                probs = np.array([0.05, 0.20, 0.50, 0.20, 0.05])

        # Rule 6: Low antecedent but high exceedance (onset risk)
        elif antecedent <= 1 and exceed >= 3:
            if trend == 2:  # Worsening - higher concern
                probs = np.array([0.02, 0.18, 0.50, 0.25, 0.05])
            else:
                probs = np.array([0.05, 0.25, 0.50, 0.15, 0.05])

        # Rule 7: Low exceedance with existing drought
        elif exceed <= 1:
            if antecedent >= 3:
                if trend == 0:  # Improving
                    probs = np.array([0.15, 0.50, 0.30, 0.05, 0.0])
                else:
                    probs = np.array([0.10, 0.40, 0.40, 0.10, 0.0])
            elif antecedent >= 1:
                probs = np.array([0.25, 0.55, 0.15, 0.05, 0.0])
            else:
                probs = np.array([0.60, 0.35, 0.05, 0.0, 0.0])

        # Default based on base_risk
        else:
            if base_risk < 1:
                probs = np.array([0.50, 0.40, 0.10, 0.0, 0.0])
            elif base_risk < 2:
                probs = np.array([0.10, 0.35, 0.40, 0.15, 0.0])
            elif base_risk < 3:
                probs = np.array([0.05, 0.15, 0.45, 0.30, 0.05])
            elif base_risk < 4:
                probs = np.array([0.0, 0.05, 0.25, 0.50, 0.20])
            else:
                probs = np.array([0.0, 0.0, 0.10, 0.40, 0.50])

        # Apply confidence-based uncertainty spreading
        if confidence == 0:  # Low confidence - spread probabilities toward uniform
            uniform = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
            probs = 0.5 * probs + 0.5 * uniform
        elif confidence == 1:  # Medium confidence - slight spreading
            uniform = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
            probs = 0.8 * probs + 0.2 * uniform

        # Normalize
        return probs / probs.sum()

    def _categorize_antecedent(self, cdi_category: str) -> str:
        mapping = {
            'No_Drought': 'No_Drought', 'Mild': 'Mild', 'Moderate': 'Moderate',
            'Severe': 'Severe', 'Extreme': 'Extreme', 'Unknown': 'Mild'
        }
        return mapping.get(cdi_category, 'Mild')

    def _categorize_exceedance(self, prob: float) -> str:
        if prob < 0.2: return 'Very_Low'
        elif prob < 0.4: return 'Low'
        elif prob < 0.6: return 'Medium'
        elif prob < 0.8: return 'High'
        else: return 'Very_High'

    def _categorize_spatial(self, coverage: float) -> str:
        if coverage < 0.3: return 'Localized'
        elif coverage < 0.6: return 'Moderate'
        else: return 'Widespread'

    def _categorize_trend(self, trend: str) -> str:
        """Map trend string to BN state."""
        mapping = {
            'improving': 'Improving',
            'stable': 'Stable',
            'worsening': 'Worsening'
        }
        return mapping.get(trend.lower(), 'Stable')

    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence score to BN state."""
        if confidence < 0.4: return 'Low'
        elif confidence < 0.7: return 'Medium'
        else: return 'High'

    def process_boundary(self, boundary_data: Dict) -> Dict:
        """Process single boundary with V4 features."""
        severity_index = boundary_data.get('severity_index', boundary_data.get('eprob_5yr', 0))

        evidence = {
            'antecedent_condition': self._categorize_antecedent(boundary_data['cdi_category']),
            'exceedance_prob': self._categorize_exceedance(severity_index),
            'spatial_coverage': self._categorize_spatial(boundary_data['spatial_coverage']),
            'temporal_trend': self._categorize_trend(boundary_data.get('cdi_trend', 'stable'))
        }

        if self.include_confidence_node:
            evidence['data_confidence'] = self._categorize_confidence(
                boundary_data.get('cdi_confidence', 1.0)
            )

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
            # CDI multi-month fields
            'cdi_weighted_mean': boundary_data.get('cdi_weighted_mean', 0),
            'cdi_category': boundary_data['cdi_category'],
            'cdi_trend': boundary_data.get('cdi_trend', 'stable'),
            'cdi_trend_magnitude': boundary_data.get('cdi_trend_magnitude', 0),
            'cdi_persistence_months': boundary_data.get('cdi_persistence_months', 0),
            'cdi_n_available': boundary_data.get('cdi_n_available', 1),
            'cdi_confidence': boundary_data.get('cdi_confidence', 1.0),
            'cdi_date_range': boundary_data.get('cdi_date_range', ''),
            # Forecast fields
            'severity_index': boundary_data['severity_index'],
            'spatial_coverage_pct': boundary_data['spatial_coverage'] * 100,
            'eprob_3yr': boundary_data.get('eprob_3yr', np.nan),
            'eprob_5yr': boundary_data.get('eprob_5yr', np.nan),
            'eprob_10yr': boundary_data.get('eprob_10yr', np.nan),
            'eprob_20yr': boundary_data.get('eprob_20yr', np.nan),
            'eprob_50yr': boundary_data.get('eprob_50yr', np.nan),
            # BN outputs
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
        for i, boundary_data in enumerate(boundaries_data):
            if (i + 1) % 50 == 0:
                print(f"  Processing {i + 1}/{len(boundaries_data)}...")
            try:
                result = self.process_boundary(boundary_data)
                results.append(result)
            except Exception as e:
                print(f"  Warning: Failed {boundary_data.get('id')}: {e}")

        df = pd.DataFrame(results)
        df['high_risk'] = df['recommended_action'].isin(['Be_Prepared', 'Take_Action'])
        return df

    def get_action_color(self, action: str) -> str:
        return {'Monitor': 'green', 'Be_Aware': 'yellow',
                'Be_Prepared': 'orange', 'Take_Action': 'red'}.get(action, 'gray')


def analyze_drought_v4(boundaries_path: str,
                       eprob_path: str = "/srv/empirical_probability_output/empirical_probabilities.nc",
                       cdi_base_path: str = "/srv/icpac_monthly_netcdf",
                       init_year: int = 2024,
                       init_month: int = 12,
                       target_lead: int = 5,
                       n_cdi_months: int = DEFAULT_CDI_MONTHS,
                       include_confidence_node: bool = True,
                       output_path: str = None) -> pd.DataFrame:
    """
    Run drought analysis V4 with multi-month CDI support.

    Args:
        boundaries_path: Path to admin boundaries GeoJSON
        eprob_path: Path to empirical probability NetCDF
        cdi_base_path: Base path for CDI monthly files
        init_year: Forecast initialization year
        init_month: Forecast initialization month
        target_lead: Lead time in months
        n_cdi_months: Number of CDI months to use (default 6)
        include_confidence_node: Include data confidence in BN
        output_path: Path for CSV output

    Returns:
        DataFrame with risk assessment results
    """
    print("="*70)
    print("DROUGHT BN ANALYSIS V4 - MULTI-MONTH CDI")
    print("="*70)

    loader = DroughtDataLoaderV4(
        boundaries_path=boundaries_path,
        eprob_path=eprob_path,
        cdi_base_path=cdi_base_path,
        n_cdi_months=n_cdi_months
    )

    boundaries_data, metadata = loader.prepare_analysis_data(
        init_year=init_year,
        init_month=init_month,
        target_lead=target_lead
    )

    print(f"\nInitializing Bayesian Network (V4 - Multi-Month CDI)...")
    bn = DroughtBayesianNetworkV4(include_confidence_node=include_confidence_node)
    print(f"  Nodes: {bn.cpt_stats['n_nodes']}")
    print(f"  Risk combinations: {bn.cpt_stats['risk_level_combinations']}")
    print(f"  Confidence node: {bn.cpt_stats['include_confidence_node']}")

    print("\nProcessing boundaries...")
    import time
    start = time.time()
    results = bn.process_all_boundaries(boundaries_data)
    elapsed = time.time() - start

    # Summary
    print(f"\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Target Season: {metadata['target_season']} (Lead {metadata['target_lead']})")
    print(f"Initialization: {metadata['init_time']}")
    print(f"CDI Data: {metadata['cdi_date_range']}")
    print(f"CDI Months: {metadata['cdi_months_available']}/{metadata['cdi_months_requested']}")
    print(f"Boundaries: {len(results)} total")
    print(f"Processing: {elapsed:.2f}s")

    print(f"\nAction Distribution:")
    for action in ['Monitor', 'Be_Aware', 'Be_Prepared', 'Take_Action']:
        count = (results['recommended_action'] == action).sum()
        pct = count / len(results) * 100
        print(f"  {action:15} [{bn.get_action_color(action):6}]: {count:3} ({pct:.1f}%)")

    print(f"\nTrend Distribution:")
    for trend in ['improving', 'stable', 'worsening']:
        count = (results['cdi_trend'] == trend).sum()
        pct = count / len(results) * 100
        print(f"  {trend:12}: {count:3} ({pct:.1f}%)")

    print(f"\nHigh Risk by Country:")
    high_risk = results[results['high_risk']]
    if len(high_risk) > 0:
        for country, count in high_risk['country'].value_counts().items():
            print(f"  {country}: {count}")

    print(f"\nTop 10 Highest Severity (with trends):")
    top = results.nlargest(10, 'severity_index')[[
        'boundary_name', 'country', 'cdi_category', 'cdi_trend',
        'severity_index', 'recommended_action'
    ]]
    print(top.to_string(index=False))

    if output_path:
        cols = ['boundary_id', 'boundary_name', 'country', 'n_pixels', 'extraction_method',
                'cdi_weighted_mean', 'cdi_category', 'cdi_trend', 'cdi_trend_magnitude',
                'cdi_persistence_months', 'cdi_n_available', 'cdi_confidence', 'cdi_date_range',
                'severity_index', 'spatial_coverage_pct',
                'eprob_3yr', 'eprob_5yr', 'eprob_10yr', 'eprob_20yr', 'eprob_50yr',
                'risk_level', 'recommended_action', 'confidence']
        output_cols = [c for c in cols if c in results.columns]
        results[output_cols].to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drought BN IBF V4 - Multi-Month CDI")
    parser.add_argument("--boundaries", default="icpac_adm1v3.geojson")
    parser.add_argument("--eprob", default="/srv/empirical_probability_output/empirical_probabilities.nc")
    parser.add_argument("--cdi-base", default="/srv/icpac_monthly_netcdf")
    parser.add_argument("--init-year", type=int, default=2024)
    parser.add_argument("--init-month", type=int, default=12)
    parser.add_argument("--target-lead", type=int, default=5)
    parser.add_argument("--cdi-months", type=int, default=DEFAULT_CDI_MONTHS,
                       help="Number of CDI months to use (default 6)")
    parser.add_argument("--no-confidence-node", action="store_true",
                       help="Disable data confidence node in BN")
    parser.add_argument("--output", default="drought_bn_v4_results.csv")

    args = parser.parse_args()

    analyze_drought_v4(
        boundaries_path=args.boundaries,
        eprob_path=args.eprob,
        cdi_base_path=args.cdi_base,
        init_year=args.init_year,
        init_month=args.init_month,
        target_lead=args.target_lead,
        n_cdi_months=args.cdi_months,
        include_confidence_node=not args.no_confidence_node,
        output_path=args.output
    )
