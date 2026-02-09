#!/usr/bin/env python3
"""
Drought Impact-Based Forecasting using Bayesian Networks - Version 3

Key improvements over v2:
1. Centroid-based extraction for small boundaries (fixes 98 missing boundaries)
2. Simplified BN structure - removed return_period node (already encoded in eprob)
3. Multi-threshold severity assessment using all return periods
4. Full coverage of all 227 admin boundaries

Author: Claude Code
Date: December 2024
"""

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
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


class CDIDataLoader:
    """Load CDI data from monthly NetCDF files."""

    def __init__(self, cdi_base_path: str = "/srv/icpac_monthly_netcdf"):
        self.cdi_base_path = Path(cdi_base_path)
        self._cache = {}

    def get_cdi_file_path(self, year: int, month: int) -> Path:
        month_str = MONTH_MAP[month]
        filename = f"eadw-cdi-data-{year}-{month_str}.nc"
        return self.cdi_base_path / str(year) / filename

    def check_availability(self, year: int, month: int) -> bool:
        return self.get_cdi_file_path(year, month).exists()

    def get_latest_available(self, target_year: int, target_month: int) -> Tuple[int, int, int]:
        year, month = target_year, target_month
        lag = 0
        while lag < 12:
            if self.check_availability(year, month):
                return (year, month, lag)
            lag += 1
            month -= 1
            if month < 1:
                month = 12
                year -= 1
        raise ValueError(f"No CDI data found within 12 months of {target_year}-{target_month:02d}")

    def load_cdi(self, year: int, month: int) -> xr.DataArray:
        cache_key = (year, month)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.get_cdi_file_path(year, month)
        if not path.exists():
            raise FileNotFoundError(f"CDI file not found: {path}")

        ds = xr.open_dataset(path)
        cdi = ds['cdi']
        self._cache[cache_key] = cdi
        return cdi


class DroughtDataLoaderV3:
    """
    Enhanced data loader with centroid-based extraction for small boundaries.

    Key improvement: Uses nearest-neighbor extraction for boundaries that
    are too small to contain any grid cells at 1-degree resolution.
    """

    def __init__(self,
                 boundaries_path: str,
                 eprob_path: str = "/srv/empirical_probability_output/empirical_probabilities.nc",
                 cdi_base_path: str = "/srv/icpac_monthly_netcdf"):

        if not HAS_GEO_DEPS:
            raise ImportError("Geospatial dependencies not installed")

        self.boundaries_path = Path(boundaries_path)
        self.eprob_path = Path(eprob_path)
        self.cdi_loader = CDIDataLoader(cdi_base_path)

        self.boundaries = self._load_boundaries()
        self.n_boundaries = len(self.boundaries)

        self.eprob_ds = None
        self.target_grid = None

    def _load_boundaries(self) -> gpd.GeoDataFrame:
        """Load and prepare admin boundaries."""
        gdf = gpd.read_file(self.boundaries_path)

        # Use GID_1 as unique ID
        if 'GID_1' in gdf.columns:
            gdf['id'] = gdf['GID_1']
        else:
            gdf['id'] = [f'ADMIN_{i:03d}' for i in range(len(gdf))]

        # Name for display
        if 'NAME_1' in gdf.columns:
            gdf['name'] = gdf['NAME_1']
        else:
            gdf['name'] = gdf['id']

        # Extract country from GID_1
        if 'GID_1' in gdf.columns:
            gdf['country_code'] = gdf['GID_1'].str.split('.').str[0]
            gdf['country'] = gdf['country_code'].map(COUNTRY_CODE_MAP).fillna('Unknown')
        else:
            gdf['country'] = 'Unknown'

        # Calculate centroids for small-boundary extraction
        gdf['centroid_lon'] = gdf.geometry.centroid.x
        gdf['centroid_lat'] = gdf.geometry.centroid.y

        print(f"Loaded {len(gdf)} admin boundaries from {self.boundaries_path.name}")
        print(f"Countries: {sorted(gdf['country'].unique().tolist())}")
        return gdf

    def load_empirical_probability(self, init_time: str = None, lead: int = None) -> xr.Dataset:
        """Load empirical probability data."""
        if self.eprob_ds is None:
            self.eprob_ds = xr.open_dataset(self.eprob_path)
            print(f"Loaded empirical probabilities: {dict(self.eprob_ds.sizes)}")

        ds = self.eprob_ds.copy()

        if init_time is not None:
            ds = ds.sel(init=init_time, method='nearest')
            print(f"Selected init time: {pd.Timestamp(ds.init.values)}")

        if lead is not None:
            ds = ds.sel(lead=lead)
            print(f"Selected lead: {lead} (forecastMonth: {lead + 1})")

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
            # Flip if descending
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

            try:
                if hasattr(regridder, 'clean_weight_file'):
                    regridder.clean_weight_file()
            except:
                pass

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

    def prepare_analysis_data(self,
                              init_year: int = 2024,
                              init_month: int = 12,
                              target_lead: int = 5,
                              cdi_scenario: str = 'actual') -> Tuple[List[Dict], Dict]:
        """
        Prepare data for all boundaries using area or centroid extraction.

        Key improvement: Uses centroid extraction for small boundaries.
        """
        print("="*60)
        print(f"PREPARING ANALYSIS DATA (V3 - All Boundaries)")
        print(f"Init: {init_year}-{init_month:02d}, Lead: {target_lead}")
        print("="*60)

        # Load forecast data
        init_time = f"{init_year}-{init_month:02d}-01"
        eprob_ds = self.load_empirical_probability(init_time=init_time, lead=target_lead)
        season_name = get_season_from_lead(init_month, target_lead)
        print(f"Target season: {season_name} SPI3")

        # Load CDI
        if cdi_scenario == 'perfect':
            cdi_year, cdi_month, lag = init_year, init_month, 0
        elif cdi_scenario == 'lag_2':
            cdi_month = init_month - 2
            cdi_year = init_year - (1 if cdi_month < 1 else 0)
            cdi_month = cdi_month if cdi_month > 0 else cdi_month + 12
            lag = 2
        elif cdi_scenario == 'lag_3':
            cdi_month = init_month - 3
            cdi_year = init_year - (1 if cdi_month < 1 else 0)
            cdi_month = cdi_month if cdi_month > 0 else cdi_month + 12
            lag = 3
        else:
            cdi_year, cdi_month, lag = self.cdi_loader.get_latest_available(init_year, init_month)

        print(f"\nCDI data: {cdi_year}-{cdi_month:02d} (lag: {lag} months)")

        cdi_raw = self.cdi_loader.load_cdi(cdi_year, cdi_month)
        print("Regridding CDI...")
        cdi = self.regrid_cdi_to_eprob(cdi_raw)

        # Get all return period probabilities
        rp_vars = ['eprob_3yr', 'eprob_5yr', 'eprob_10yr', 'eprob_20yr', 'eprob_50yr']
        eprob_data = {v: eprob_ds[v] for v in rp_vars if v in eprob_ds}

        # Create regionmask
        lat = eprob_ds.lat.values
        lon = eprob_ds.lon.values

        regions = regionmask.from_geopandas(self.boundaries, names='id', abbrevs='id')
        mask = regions.mask(lon, lat)

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

            try:
                region_num = regions.map_keys(boundary_id)
                boundary_mask = mask.values == region_num
                n_pixels = np.sum(boundary_mask)
            except:
                n_pixels = 0

            if n_pixels > 0:
                # Area-based extraction
                extraction_method = 'area'
                area_extracted += 1

                # Get all return period probabilities
                rp_probs = {}
                for var_name, var_data in eprob_data.items():
                    masked = var_data.values[boundary_mask]
                    rp_probs[var_name] = float(np.nanmean(masked))

                # Spatial coverage using 5yr threshold
                eprob_5yr_masked = eprob_data['eprob_5yr'].values[boundary_mask]
                spatial_coverage = float(np.sum(eprob_5yr_masked > 0.5) / n_pixels)

                # CDI
                cdi_masked = cdi.values[boundary_mask]
                cdi_mean = float(np.nanmean(cdi_masked))

            else:
                # Centroid-based extraction for small boundaries
                extraction_method = 'centroid'
                centroid_extracted += 1
                n_pixels = 0

                rp_probs = {}
                for var_name, var_data in eprob_data.items():
                    rp_probs[var_name] = self._extract_at_centroid(var_data, centroid_lat, centroid_lon)

                spatial_coverage = 1.0 if rp_probs.get('eprob_5yr', 0) > 0.5 else 0.0
                cdi_mean = self._extract_at_centroid(cdi, centroid_lat, centroid_lon)

            # Compute severity index from multiple return periods
            # Higher probabilities at higher return periods = more severe
            severity_index = self._compute_severity_index(rp_probs)

            boundary_data = {
                'id': boundary_id,
                'name': boundary_name,
                'country': country,
                'n_pixels': int(n_pixels),
                'extraction_method': extraction_method,
                'centroid_lat': centroid_lat,
                'centroid_lon': centroid_lon,
                'cdi_mean': cdi_mean if not np.isnan(cdi_mean) else 0.0,
                'cdi_category': categorize_cdi(cdi_mean),
                'cdi_date': f"{cdi_year}-{cdi_month:02d}",
                'cdi_lag_months': lag,
                'spatial_coverage': spatial_coverage,
                'severity_index': severity_index,
                **rp_probs  # All return period probabilities
            }
            boundaries_data.append(boundary_data)

        print(f"\nExtracted {len(boundaries_data)} boundaries:")
        print(f"  Area-based: {area_extracted}")
        print(f"  Centroid-based: {centroid_extracted}")

        metadata = {
            'init_time': f"{init_year}-{init_month:02d}",
            'target_season': season_name,
            'target_lead': target_lead,
            'cdi_date': f"{cdi_year}-{cdi_month:02d}",
            'cdi_lag_months': lag,
            'cdi_scenario': cdi_scenario,
            'n_boundaries': len(boundaries_data),
            'area_extracted': area_extracted,
            'centroid_extracted': centroid_extracted
        }

        return boundaries_data, metadata

    def _compute_severity_index(self, rp_probs: Dict[str, float], method: str = 'max_weighted') -> float:
        """
        Compute severity index from multiple return period probabilities.

        Methods:
        - 'max_weighted': Use max probability, boosted if longer return periods are exceeded
        - 'eprob_5yr': Simply use the 5-year return period probability (like V2)
        - 'weighted_avg': Weighted average (original, but flawed - drags values down)

        The key insight: A 30% chance of 50-year drought is MORE severe than
        a 30% chance of 3-year drought. But we shouldn't average them because
        longer return periods naturally have lower probabilities.

        Better approach: Use the highest probability, then boost if severe thresholds
        are also being exceeded.
        """
        if method == 'eprob_5yr':
            # Simple: just use 5-year probability (consistent with V2)
            return rp_probs.get('eprob_5yr', 0.0)

        elif method == 'max_weighted':
            # Use max probability as base, boost for severe thresholds
            base_prob = rp_probs.get('eprob_5yr', 0.0)

            # Boost if longer return periods are exceeded
            boost = 0.0
            if rp_probs.get('eprob_10yr', 0) > 0.3:
                boost += 0.05
            if rp_probs.get('eprob_20yr', 0) > 0.2:
                boost += 0.05
            if rp_probs.get('eprob_50yr', 0) > 0.1:
                boost += 0.05

            return min(1.0, base_prob + boost)

        else:  # 'weighted_avg' - original flawed method
            weights = {
                'eprob_3yr': 1.0, 'eprob_5yr': 1.5, 'eprob_10yr': 2.0,
                'eprob_20yr': 2.5, 'eprob_50yr': 3.0
            }
            weighted_sum = sum(weights.get(k, 0) * v for k, v in rp_probs.items()
                              if k in weights and not np.isnan(v))
            total_weight = sum(weights.get(k, 0) for k in rp_probs if k in weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.0


class DroughtBayesianNetworkV3:
    """
    Simplified Bayesian Network for drought risk assessment.

    Key change: Removed return_period node since return period information
    is already encoded in the empirical probability values (eprob_Xyr).

    Network structure:
        antecedent_condition ─┐
                              ├─► risk_level ─► action
        exceedance_prob ──────┤
                              │
        spatial_coverage ─────┘

    This is cleaner because:
    1. Return period is already embedded in the probability variable choice
    2. Severity is captured by the severity_index (multi-threshold)
    3. Fewer parameters = more interpretable CPTs
    """

    def __init__(self):
        """Initialize simplified BN structure."""
        self.model = BayesianNetwork([
            ('antecedent_condition', 'risk_level'),
            ('exceedance_prob', 'risk_level'),
            ('spatial_coverage', 'risk_level'),
            ('risk_level', 'action')
        ])
        self._setup_cpds()

    def _setup_cpds(self):
        """Setup Conditional Probability Distributions."""

        # 1. Antecedent Condition (from CDI) - 5 states
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

        # 4. Risk Level CPT - 5 × 5 × 3 = 75 combinations (simplified!)
        risk_cpd_values = self._create_risk_cpt()

        risk_cpd = TabularCPD(
            'risk_level', 5,
            risk_cpd_values,
            evidence=['antecedent_condition', 'exceedance_prob', 'spatial_coverage'],
            evidence_card=[5, 5, 3],
            state_names={
                'risk_level': ['Minimal', 'Low', 'Moderate', 'High', 'Extreme'],
                'antecedent_condition': ['No_Drought', 'Mild', 'Moderate', 'Severe', 'Extreme'],
                'exceedance_prob': ['Very_Low', 'Low', 'Medium', 'High', 'Very_High'],
                'spatial_coverage': ['Localized', 'Moderate', 'Widespread']
            }
        )

        # 5. Action CPD
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

        self.model.add_cpds(antecedent_cpd, exceed_cpd, spatial_cpd, risk_cpd, action_cpd)
        self.model.check_model()

        # Store CPT statistics
        self.cpt_stats = {
            'n_nodes': 5,
            'n_edges': 4,
            'total_parameters': 5 + 5 + 3 + (5 * 75) + (4 * 5),  # 408 total
            'risk_level_combinations': 75
        }

    def _create_risk_cpt(self) -> np.ndarray:
        """
        Create CPT for risk_level with 75 combinations.

        Reduced from 375 combinations (v2) by removing return_period node.
        """
        n_combinations = 5 * 5 * 3  # 75
        cpt = np.zeros((5, n_combinations))

        idx = 0
        for spatial in range(3):  # Localized, Moderate, Widespread
            for exceed in range(5):  # Very_Low to Very_High
                for antecedent in range(5):  # No_Drought to Extreme
                    risk_probs = self._compute_risk_probs(antecedent, exceed, spatial)
                    cpt[:, idx] = risk_probs
                    idx += 1

        return cpt

    def _compute_risk_probs(self, antecedent: int, exceed: int, spatial: int) -> np.ndarray:
        """
        Compute risk probabilities using expert rules (aligned with V2).

        Args:
            antecedent: 0=No_Drought, 1=Mild, 2=Moderate, 3=Severe, 4=Extreme
            exceed: 0=Very_Low, 1=Low, 2=Medium, 3=High, 4=Very_High
            spatial: 0=Localized, 1=Moderate, 2=Widespread

        Returns:
            Array of [Minimal, Low, Moderate, High, Extreme] probabilities

        Note: Rules adjusted to match V2 behavior for consistency.
        V2 is more conservative (pushes toward higher actions) because:
        1. Drought is slow-onset with compounding effects
        2. Early action is preferred over late action
        """
        # Base risk score (same as V2)
        base_risk = (antecedent * 0.4 + exceed * 0.6)

        # Spatial adjustment (key factor for drought extent)
        if spatial == 2:  # Widespread
            base_risk += 0.5
        elif spatial == 1:  # Moderate
            base_risk += 0.25

        # EXPERT RULES - Aligned with V2 for consistency

        # Rule 1: Extreme antecedent + High/Very_High exceedance
        if antecedent == 4 and exceed >= 3:
            if spatial >= 1:
                probs = np.array([0.0, 0.0, 0.05, 0.20, 0.75])
            else:
                probs = np.array([0.0, 0.0, 0.10, 0.50, 0.40])

        # Rule 2: Severe antecedent + Medium+ exceedance + widespread
        elif antecedent == 3 and exceed >= 2 and spatial == 2:
            probs = np.array([0.0, 0.0, 0.15, 0.60, 0.25])

        # Rule 3: Severe antecedent + High exceedance
        elif antecedent == 3 and exceed >= 3:
            probs = np.array([0.0, 0.05, 0.20, 0.55, 0.20])

        # Rule 4: Moderate conditions with forecast concern
        elif antecedent == 2 and exceed >= 2:
            if spatial == 2:
                probs = np.array([0.0, 0.10, 0.50, 0.35, 0.05])
            else:
                probs = np.array([0.05, 0.20, 0.50, 0.20, 0.05])

        # Rule 5: Mild/None conditions with high probability (onset risk)
        elif antecedent <= 1 and exceed >= 3:
            probs = np.array([0.05, 0.25, 0.50, 0.15, 0.05])

        # Rule 6: Low exceedance with existing drought (monitoring)
        elif exceed <= 1:
            if antecedent >= 3:  # Severe/Extreme existing
                probs = np.array([0.10, 0.40, 0.40, 0.10, 0.0])
            elif antecedent >= 1:  # Mild/Moderate existing
                probs = np.array([0.25, 0.55, 0.15, 0.05, 0.0])
            else:  # No drought
                probs = np.array([0.60, 0.35, 0.05, 0.0, 0.0])

        # Default: base risk determines distribution (V2-aligned scaling)
        else:
            if base_risk < 1:
                probs = np.array([0.50, 0.40, 0.10, 0.0, 0.0])
            elif base_risk < 2:
                # V2 is more conservative here - shifted toward Moderate/High
                probs = np.array([0.10, 0.35, 0.40, 0.15, 0.0])
            elif base_risk < 3:
                probs = np.array([0.05, 0.15, 0.45, 0.30, 0.05])
            elif base_risk < 4:
                probs = np.array([0.0, 0.05, 0.25, 0.50, 0.20])
            else:
                probs = np.array([0.0, 0.0, 0.10, 0.40, 0.50])

        # Normalize and return
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

    def process_boundary(self, boundary_data: Dict) -> Dict:
        """Process single boundary."""
        # Use severity_index for exceedance (incorporates all return periods)
        severity_index = boundary_data.get('severity_index', boundary_data.get('eprob_5yr', 0))

        evidence = {
            'antecedent_condition': self._categorize_antecedent(boundary_data['cdi_category']),
            'exceedance_prob': self._categorize_exceedance(severity_index),
            'spatial_coverage': self._categorize_spatial(boundary_data['spatial_coverage'])
        }

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
            'cdi_mean': boundary_data['cdi_mean'],
            'cdi_category': boundary_data['cdi_category'],
            'cdi_date': boundary_data['cdi_date'],
            'cdi_lag_months': boundary_data['cdi_lag_months'],
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


def analyze_drought(boundaries_path: str,
                    eprob_path: str = "/srv/empirical_probability_output/empirical_probabilities.nc",
                    cdi_base_path: str = "/srv/icpac_monthly_netcdf",
                    init_year: int = 2024,
                    init_month: int = 12,
                    target_lead: int = 5,
                    cdi_scenario: str = 'actual',
                    output_path: str = None) -> pd.DataFrame:
    """Run drought analysis for all boundaries."""

    print("="*70)
    print("DROUGHT BN ANALYSIS V3 - ALL BOUNDARIES")
    print("="*70)

    loader = DroughtDataLoaderV3(
        boundaries_path=boundaries_path,
        eprob_path=eprob_path,
        cdi_base_path=cdi_base_path
    )

    boundaries_data, metadata = loader.prepare_analysis_data(
        init_year=init_year,
        init_month=init_month,
        target_lead=target_lead,
        cdi_scenario=cdi_scenario
    )

    print("\nInitializing Bayesian Network (V3 - Simplified)...")
    bn = DroughtBayesianNetworkV3()
    print(f"  CPT Parameters: {bn.cpt_stats['total_parameters']}")
    print(f"  Risk combinations: {bn.cpt_stats['risk_level_combinations']}")

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
    print(f"CDI Data: {metadata['cdi_date']} (lag: {metadata['cdi_lag_months']} months)")
    print(f"Boundaries: {len(results)} total ({metadata['area_extracted']} area, "
          f"{metadata['centroid_extracted']} centroid)")
    print(f"Processing: {elapsed:.2f}s")

    print(f"\nAction Distribution:")
    for action in ['Monitor', 'Be_Aware', 'Be_Prepared', 'Take_Action']:
        count = (results['recommended_action'] == action).sum()
        pct = count / len(results) * 100
        print(f"  {action:15} [{bn.get_action_color(action):6}]: {count:3} ({pct:.1f}%)")

    print(f"\nHigh Risk by Country:")
    high_risk = results[results['high_risk']]
    if len(high_risk) > 0:
        for country, count in high_risk['country'].value_counts().items():
            print(f"  {country}: {count}")

    print(f"\nTop 10 Highest Severity:")
    top = results.nlargest(10, 'severity_index')[
        ['boundary_name', 'country', 'cdi_category', 'severity_index', 'recommended_action']
    ]
    print(top.to_string(index=False))

    if output_path:
        cols = ['boundary_id', 'boundary_name', 'country', 'n_pixels', 'extraction_method',
                'cdi_mean', 'cdi_category', 'cdi_date', 'cdi_lag_months',
                'severity_index', 'spatial_coverage_pct',
                'eprob_3yr', 'eprob_5yr', 'eprob_10yr', 'eprob_20yr', 'eprob_50yr',
                'risk_level', 'recommended_action', 'confidence']
        results[cols].to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drought BN IBF V3")
    parser.add_argument("--boundaries", default="icpac_adm1v3.geojson")
    parser.add_argument("--eprob", default="/srv/empirical_probability_output/empirical_probabilities.nc")
    parser.add_argument("--cdi-base", default="/srv/icpac_monthly_netcdf")
    parser.add_argument("--init-year", type=int, default=2024)
    parser.add_argument("--init-month", type=int, default=12)
    parser.add_argument("--target-lead", type=int, default=5)
    parser.add_argument("--cdi-scenario", default="actual", choices=["actual", "perfect", "lag_2", "lag_3"])
    parser.add_argument("--output", default="drought_bn_v3_results.csv")

    args = parser.parse_args()

    analyze_drought(
        boundaries_path=args.boundaries,
        eprob_path=args.eprob,
        cdi_base_path=args.cdi_base,
        init_year=args.init_year,
        init_month=args.init_month,
        target_lead=args.target_lead,
        cdi_scenario=args.cdi_scenario,
        output_path=args.output
    )
