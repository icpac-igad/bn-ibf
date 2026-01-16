# Flood BN IBF V1 - Short-term Flood Risk Assessment Workflow

## Overview

Version 1 of the Flood Impact-Based Forecasting system uses Bayesian Networks to assess flood risk for East Africa. The system combines:

1. **Observations**: Last 7 days of IMERG rainfall data (antecedent soil moisture proxy)
2. **Forecasts**: 14-day ensemble precipitation forecasts from ECMWF and GEFS
3. **Empirical Probabilities**: Return period exceedance probabilities from ensemble forecasts

This follows the same Bayesian Network architecture as the drought BN IBF system but adapted for short-term flood forecasting.

---

## Data Sources

### 1. IMERG Daily Precipitation (Observations)

**Product**: GPM_3IMERGDE (Daily Early) or GPM_3IMERGHH (Half-hourly)

| Property | Value |
|----------|-------|
| Source | NASA GES DISC / Earthdata |
| Resolution | 0.1° (~10km) |
| Latency | ~4 hours (Early), ~14 hours (Late) |
| Temporal | Last 7 days rolling window |
| Variable | `precipitation` (mm/day) |

**Purpose**: Characterizes antecedent rainfall conditions (proxy for soil moisture saturation)

**Download Method**: `earthaccess` Python library with Earthdata credentials (`.env` file)

### 2. ECMWF Ensemble Forecast

**Product**: ECMWF ENS (Ensemble Prediction System)

| Property | Value |
|----------|-------|
| Source | ECMWF / Copernicus CDS |
| Resolution | ~0.25° (~25km) |
| Ensemble Members | 51 |
| Forecast Horizon | 15 days |
| Variable | `total_precipitation` (mm) |

**Purpose**: Primary probabilistic precipitation forecast

### 3. GEFS Ensemble Forecast

**Product**: NCEP Global Ensemble Forecast System

| Property | Value |
|----------|-------|
| Source | NOAA / AWS Open Data |
| Resolution | 0.25° |
| Ensemble Members | 31 |
| Forecast Horizon | 16 days |
| Variable | `apcp` (accumulated precipitation) |

**Purpose**: Secondary probabilistic forecast (multi-model approach)

### 4. Future Enhancement: cGAN AI Forecast (V2)

**Product**: Conditional GAN downscaled precipitation forecast

| Property | Value |
|----------|-------|
| Source | Custom ML model |
| Resolution | 0.1° (downscaled) |
| Ensemble Members | Synthetic ensemble from GAN |
| Forecast Horizon | 14 days |

**Purpose**: AI-enhanced high-resolution forecast (future version)

---

## East Africa Domain

### Administrative Boundaries

**File**: `icpac_adm1v3.geojson`

| Property | Value |
|----------|-------|
| Features | 227 Admin-1 boundaries |
| Extent | (21.84°E, -11.75°S) to (51.42°E, 23.15°N) |
| Countries | Burundi, Djibouti, Eritrea, Ethiopia, Kenya, Rwanda, Somalia, South Sudan, Sudan, Tanzania, Uganda |
| ID Field | `GID_1` |
| Name Field | `NAME_1` |

### Spatial Domain

```
West:  21.84°E
East:  51.42°E
South: 11.75°S
North: 23.15°N
```

---

## Temporal Framework

### Observation Window (Antecedent Conditions)

| Period | Days | Description |
|--------|------|-------------|
| Recent | D-1 to D-3 | Most recent rainfall (highest weight) |
| Medium | D-4 to D-5 | Intermediate period |
| Earlier | D-6 to D-7 | Earlier rainfall (lower weight) |

**Temporal Weights (7-day window)**:
```python
TEMPORAL_WEIGHTS_OBS = [0.25, 0.20, 0.15, 0.13, 0.10, 0.09, 0.08]  # Day -1 to Day -7
```

### Forecast Window

| Period | Days | Primary Use |
|--------|------|-------------|
| Short-term | D+1 to D+3 | Flash flood risk |
| Medium-term | D+4 to D+7 | River flood risk |
| Extended | D+8 to D+14 | Preparedness planning |

**Forecast Aggregation Periods**:
- 3-day accumulated precipitation
- 7-day accumulated precipitation
- 14-day accumulated precipitation

---

## Return Period Thresholds

### Precipitation-based Thresholds

Flood risk thresholds based on accumulated precipitation return periods:

| Return Period | Description | Approximate Exceedance |
|---------------|-------------|------------------------|
| 2-year | Moderate rainfall | 50% |
| 5-year | Heavy rainfall | 20% |
| 10-year | Very heavy rainfall | 10% |
| 25-year | Extreme rainfall | 4% |
| 50-year | Exceptional rainfall | 2% |

**Note**: These thresholds should be computed pixel-wise from historical IMERG data (similar to SPI-3 thresholds in drought BN).

### Computing Empirical Probabilities

```python
def compute_flood_eprob(ensemble_precip, thresholds):
    """
    Compute empirical exceedance probability for each return period.

    Args:
        ensemble_precip: (n_members, lat, lon) accumulated precipitation
        thresholds: Dict of return period thresholds (per pixel)

    Returns:
        Dict of exceedance probability DataArrays
    """
    n_members = ensemble_precip.shape[0]
    eprobs = {}

    for rp, threshold in thresholds.items():
        # Count members exceeding threshold
        above = (ensemble_precip >= threshold).sum(dim='member')
        prob = above / n_members
        eprobs[f'eprob_{rp}'] = prob

    return eprobs
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                       │
├───────────────────────────┬─────────────────────────────────────────────┤
│  IMERG Observations       │  Ensemble Forecasts                          │
│  - Last 7 days rainfall   │  - ECMWF ENS (51 members, 15 days)          │
│  - 0.1° resolution        │  - GEFS (31 members, 16 days)               │
│  - Antecedent conditions  │  - [Future: cGAN AI forecast]               │
└───────────────────────────┴─────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FloodDataLoaderV1                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Processing Order:                                                       │
│  1. Load IMERG (10km) → Compute 7-day accumulated precipitation         │
│  2. Load ECMWF ENS → Regrid to 10km, compute 3/7/14-day accum           │
│  3. Load GEFS → Regrid to 10km, compute 3/7/14-day accum                │
│  4. Compute empirical probabilities (per-model and combined)            │
│  5. Extract boundary-level statistics using regionmask                   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FloodBayesianNetworkV1                                │
├─────────────────────────────────────────────────────────────────────────┤
│  Nodes:                                                                  │
│  ├─ antecedent_rainfall (observed) ──┐                                  │
│  ├─ ecmwf_eprob (forecast) ──────────┤                                  │
│  ├─ gefs_eprob (forecast) ───────────┼──► risk_level ──► action         │
│  ├─ spatial_coverage ────────────────┤                                  │
│  ├─ rainfall_trend ──────────────────┤                                  │
│  └─ forecast_agreement ──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT CSV                                       │
│  flood_bn_v1_{YYYY}_{MM}_{DD}.csv                                       │
│  - Per boundary: risk_level, recommended_action, confidence              │
│  - Empirical probabilities from ECMWF and GEFS                          │
│  - Antecedent rainfall statistics                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Bayesian Network Structure

### Network Topology

```
antecedent_rainfall ─────────┐
                             │
ecmwf_exceedance_prob ───────┤
                             │
gefs_exceedance_prob ────────┼──► risk_level ──► action
                             │
spatial_coverage ────────────┤
                             │
rainfall_trend ──────────────┤
                             │
forecast_agreement ──────────┘
```

### Node Definitions

| Node | Type | States | Description |
|------|------|--------|-------------|
| `antecedent_rainfall` | Evidence | Dry, Normal, Wet, Very_Wet, Saturated | 7-day accumulated IMERG rainfall |
| `ecmwf_exceedance_prob` | Evidence | Very_Low, Low, Medium, High, Very_High | ECMWF ensemble exceedance probability |
| `gefs_exceedance_prob` | Evidence | Very_Low, Low, Medium, High, Very_High | GEFS ensemble exceedance probability |
| `spatial_coverage` | Evidence | Localized, Moderate, Widespread | % of boundary with high exceedance |
| `rainfall_trend` | Evidence | Decreasing, Stable, Increasing | 7-day rainfall trend |
| `forecast_agreement` | Evidence | Low, Medium, High | ECMWF-GEFS agreement |
| `risk_level` | Hidden | Minimal, Low, Moderate, High, Extreme | Inferred flood risk |
| `action` | Query | Monitor, Alert, Prepare, Act | Recommended response |

### State Discretization

**Antecedent Rainfall (7-day accumulated, mm)**:

| State | Threshold Range | Description |
|-------|-----------------|-------------|
| Dry | < 10 mm | Below normal, low saturation |
| Normal | 10-30 mm | Average conditions |
| Wet | 30-60 mm | Above normal, moderate saturation |
| Very_Wet | 60-100 mm | High saturation |
| Saturated | > 100 mm | Extreme saturation, high flood risk |

**Exceedance Probability**:

| State | Probability Range |
|-------|-------------------|
| Very_Low | < 0.20 |
| Low | 0.20 - 0.40 |
| Medium | 0.40 - 0.60 |
| High | 0.60 - 0.80 |
| Very_High | > 0.80 |

**Spatial Coverage**:

| State | Coverage Range |
|-------|----------------|
| Localized | < 30% |
| Moderate | 30% - 60% |
| Widespread | > 60% |

**Forecast Agreement**:

| State | Condition |
|-------|-----------|
| Low | |ECMWF - GEFS| > 0.3 (probability difference) |
| Medium | 0.15 < |ECMWF - GEFS| <= 0.3 |
| High | |ECMWF - GEFS| <= 0.15 |

---

## Conditional Probability Tables (CPTs)

### Action CPD (P(Action | Risk_Level))

```
Risk Level:   Minimal   Low    Moderate   High   Extreme
───────────────────────────────────────────────────────
Monitor        0.95    0.15     0.00     0.00    0.00
Alert          0.05    0.80     0.20     0.05    0.00
Prepare        0.00    0.05     0.75     0.25    0.05
Act            0.00    0.00     0.05     0.70    0.95
```

### Risk Level CPD Expert Rules

**Key flood-specific rules**:

1. **Saturated + High Exceedance (both models)**:
   - Widespread → P(Extreme) = 0.85
   - Extreme flood risk when soil is saturated AND heavy rain forecast

2. **Very_Wet + High ECMWF + High GEFS + Increasing trend**:
   - P(High) = 0.65, P(Extreme) = 0.25
   - High confidence in flood risk

3. **Model Disagreement Penalty**:
   - Low agreement → Shift probability toward moderate states
   - Uncertainty reduces confidence in extreme predictions

4. **Dry antecedent + High forecast**:
   - Reduced risk compared to wet antecedent
   - Soil can absorb initial rainfall

5. **Decreasing rainfall trend**:
   - Reduces risk level
   - Indicates improving conditions

---

## Implementation Status

### Phase 1: Data Infrastructure (V1.0) - COMPLETE

1. **IMERG Data Pipeline** ✓
   - [x] Download routine using earthaccess (`download_imerg_daily.py`)
   - [x] 7-day rolling window accumulation
   - [x] Antecedent rainfall categorization
   - [x] Trend computation

2. **GEFS Data Pipeline** ✓
   - [x] GIK streaming from AWS S3 (via `run_gefs_tutorial.py`)
   - [x] Ensemble member extraction (30 members)
   - [x] N-day accumulation (configurable)
   - [x] Empirical probability computation (`gefs_probability.py`)

3. **ECMWF Data Pipeline** (Future V1.1)
   - [ ] CDS API download routine
   - [ ] Ensemble member extraction
   - [ ] Integration with multi-model agreement

### Phase 2: Threshold Development - USING ARBITRARY THRESHOLDS

1. **Arbitrary Precipitation Thresholds** ✓
   - [x] Fixed thresholds: 5, 25, 50, 75, 100, 125 mm
   - [x] Configurable via `PRECIP_THRESHOLDS` dictionary
   - [ ] Future: Pixel-wise return period thresholds

2. **Empirical Probability Computation** ✓
   - [x] GEFS ensemble → exceedance probability (`gefs_probability.py`)
   - [x] Boundary-level extraction
   - [ ] Future: ECMWF integration

### Phase 3: Bayesian Network - COMPLETE

1. **FloodBayesianNetworkV1 Class** ✓
   - [x] Network structure: 6 nodes (antecedent, exceedance, spatial, trend, agreement → risk → action)
   - [x] CPT construction with expert rules
   - [x] Variable Elimination inference

2. **FloodDataLoaderV1 Class** ✓
   - [x] Boundary extraction from GeoJSON
   - [x] IMERG observation loading
   - [x] GEFS probability integration

### Phase 4: Integration & Testing - IN PROGRESS

1. **End-to-end Pipeline**
   - [x] CLI interface for flood_bn_ibf_v1.py
   - [x] Output CSV generation
   - [ ] Visualization support
   - [ ] Daily operational workflow

2. **Validation**
   - [ ] Historical flood event comparison
   - [ ] Skill score assessment
   - [ ] Calibration if needed

### Future: Phase 5 (V2 - AI Enhancement)

1. **cGAN Forecast Integration**
   - [ ] Model deployment
   - [ ] Ensemble generation
   - [ ] Integration with BN

2. **ECMWF Multi-model**
   - [ ] ECMWF ENS download
   - [ ] Multi-model agreement node activation

---

## File Structure

```
flood_ibf/
├── download_imerg_daily.py          # IMERG download routine ✓
├── gefs_probability.py              # GEFS empirical probability computation ✓
├── flood_bn_ibf_v1.py               # Main BN implementation ✓
├── flood_bn_ibf_v1_workflow.md      # This document ✓
├── .env                             # Earthdata credentials
├── .env.example                     # Credential template ✓
├── imerg_data/                      # Downloaded IMERG files ✓
│   └── 3B-DAY-E.MS.MRG.3IMERG.*.nc4
├── gefs_eprob/                      # GEFS probability outputs (generated)
│   └── gefs_eprob_YYYYMMDD_HHz.nc
└── output/                          # BN output CSVs (generated)
    └── flood_bn_v1_YYYYMMDD.csv

# External GEFS infrastructure (in grib-index-kerchunk repo):
grib-index-kerchunk/tutorial/gefs/
├── run_gefs_tutorial.py             # Creates parquet reference files
├── run_gefs_data_streaming_v2.py    # Original streaming/plotting
└── output_parquet/                  # Parquet files for streaming
    └── gepXX_YYYYMMDD_HHz.parquet
```

---

## Usage

### Step 1: Download IMERG Observations

```bash
# Download last 7 days of IMERG data
python download_imerg_daily.py
```

### Step 2: Generate GEFS Parquet Files (if not already done)

```bash
# In grib-index-kerchunk/tutorial/gefs/
python run_gefs_tutorial.py
```

### Step 3: Compute GEFS Exceedance Probabilities

```bash
# Compute 7-day accumulated exceedance probabilities
python gefs_probability.py \
    --parquet-dir /path/to/output_parquet \
    --date 20260110 \
    --run 00 \
    --accum-days 7 \
    --output ./gefs_eprob/gefs_eprob_20260110_00z.nc \
    --boundaries /path/to/icpac_adm1v3.geojson
```

### Step 4: Run Flood Risk Assessment

```bash
# Run flood BN analysis
python flood_bn_ibf_v1.py \
    --boundaries /path/to/icpac_adm1v3.geojson \
    --date 2026-01-10 \
    --imerg-dir ./imerg_data \
    --gefs-dir ./gefs_eprob \
    --obs-days 7 \
    --forecast-accum 7 \
    --output-dir ./output/
```

### Python API

```python
from flood_bn_ibf_v1 import analyze_flood_risk

results = analyze_flood_risk(
    boundaries_path="icpac_adm1v3.geojson",
    target_date="2026-01-10",
    imerg_data_dir="./imerg_data",
    gefs_data_dir="./gefs_eprob",
    obs_days=7,
    forecast_accum_days=7,
    output_dir="./output/"
)
```

### GEFS Probability Python API

```python
from gefs_probability import process_gefs_ensemble, save_probabilities_netcdf

# Compute probabilities
eprobs, metadata = process_gefs_ensemble(
    parquet_dir="./output_parquet",
    target_date="20260110",
    run_hour="00",
    accum_days=7
)

# Save to NetCDF
save_probabilities_netcdf(eprobs, metadata, "./gefs_eprob_20260110.nc")
```

---

## Output CSV Format

| Column | Description |
|--------|-------------|
| `boundary_id` | Admin boundary ID (GID_1) |
| `boundary_name` | Admin boundary name |
| `country` | Country name |
| `antecedent_rainfall_mm` | 7-day accumulated IMERG (mm) |
| `antecedent_category` | Dry/Normal/Wet/Very_Wet/Saturated |
| `rainfall_trend` | Decreasing/Stable/Increasing |
| `ecmwf_eprob_5yr` | ECMWF 5-year return period exceedance |
| `ecmwf_eprob_10yr` | ECMWF 10-year return period exceedance |
| `gefs_eprob_5yr` | GEFS 5-year return period exceedance |
| `gefs_eprob_10yr` | GEFS 10-year return period exceedance |
| `combined_eprob` | Multi-model combined probability |
| `spatial_coverage_pct` | % of area with high exceedance |
| `forecast_agreement` | ECMWF-GEFS agreement level |
| `risk_level` | Minimal/Low/Moderate/High/Extreme |
| `recommended_action` | Monitor/Alert/Prepare/Act |
| `confidence` | Action confidence score |
| `valid_date` | Forecast valid date |

---

## Comparison: Drought BN vs Flood BN

| Feature | Drought BN (V6) | Flood BN (V1) |
|---------|-----------------|---------------|
| **Temporal Scale** | Seasonal (months) | Short-term (days) |
| **Observation** | CDI (Combined Drought Indicator) | IMERG precipitation |
| **Observation Window** | 6 months | 7 days |
| **Forecast Source** | SEAS51 SPI-3 | ECMWF ENS + GEFS |
| **Forecast Horizon** | 1-6 months | 1-14 days |
| **Ensemble Members** | 51 (SEAS51) | 51 (ECMWF) + 31 (GEFS) |
| **Threshold Basis** | SPI-3 return periods | Precipitation return periods |
| **Multi-model** | No (single model) | Yes (ECMWF + GEFS) |
| **Antecedent Node** | `antecedent_condition` (CDI) | `antecedent_rainfall` (IMERG) |
| **Agreement Node** | No | Yes (`forecast_agreement`) |
| **Update Frequency** | Monthly/Seasonal | Daily |

---

## Key Differences from Drought System

### 1. Multi-Model Ensemble

The flood system uses **two independent forecast models** (ECMWF and GEFS) instead of one. This enables:
- Model agreement/disagreement assessment
- More robust probability estimation
- Reduced single-model bias

### 2. Shorter Time Scales

- **Observations**: 7 days vs 6 months
- **Forecasts**: 14 days vs 6 months
- **Update frequency**: Daily vs monthly

### 3. Direct Precipitation vs Index

- **Drought**: Uses derived indices (CDI, SPI-3)
- **Flood**: Uses direct precipitation amounts

### 4. Saturation Focus

- **Drought**: Focus on deficit/dryness
- **Flood**: Focus on saturation/excess

---

## References

- Drought BN IBF V6: `/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/bn-ibf/jupyter_notebooks/drought_bn_ibf_v6.py`
- Drought Workflow: `/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/bn-ibf/jupyter_notebooks/drought_bn_ibf_v5_workflow.md`
- Admin Boundaries: `/home/roller/Documents/08-2023/working_notes_jupyter/ignore_nka_gitrepos/ibf-thresholds-triggers/icpac_adm1v3.geojson`

---

*Document created: January 2026*
*Version: 1.0 (Planning)*
