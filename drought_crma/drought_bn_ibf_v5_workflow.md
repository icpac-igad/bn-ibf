# Drought BN IBF V5 - GCS IceChunk Workflow

## Overview

Version 5 of the Drought Impact-Based Forecasting system integrates with Google Cloud Storage (GCS) IceChunk/Zarr stores for both CDI observation data and SEAS51 SPI3 forecast data. This enables scalable, cloud-native seasonal drought risk assessment across East Africa.

---

## Data Sources

### 1. CDI (Combined Drought Indicator)

**Location:** GCS bucket `cdi_arco`

| Resolution | Path Pattern | Description |
|------------|--------------|-------------|
| 1km | `bn_icpac_cdi_store/1km/{year}/eadw-cdi-data-{year}-{month}.zarr` | Original resolution |
| 10km | `bn_icpac_cdi_store/10km/{year}/eadw-cdi-data-{year}-{month}.zarr` | Regridded (recommended) |

**Source:** ICPAC DroughtWatch FTP → IceChunk via `download_and_process_cdi.py`

**Temporal Coverage:** 2011-present (monthly)

### 2. SPI3 Forecast (SEAS51)

**Location:** GCS bucket `cdi_arco`

| Resolution | Path Pattern | Description |
|------------|--------------|-------------|
| ~100km | `seas51_spi3_raw/{YYYYMM}` | Native SEAS51 resolution |

**Source:** ECMWF SEAS51 → SPI3 calculation → IceChunk via `seas51_spi3_raw_to_icechunk.py`

**Temporal Coverage:** 1981-present (monthly initializations)

**Ensemble Members:**
- 1981-2016: 25 members (original SEAS5)
- 2017-present: 51 members (SEAS5.1)

---

## East Africa Seasons

| Season | Months | Last Month | Primary Use |
|--------|--------|------------|-------------|
| **JFM** | Jan-Feb-Mar | March | Northern regions dry season end |
| **MAM** | Mar-Apr-May | May | Long rains (Kenya, Ethiopia, Somalia) |
| **JJA** | Jun-Jul-Aug | August | Sudan/Eritrea summer rains |
| **SON** | Sep-Oct-Nov | November | Short rains transition |
| **OND** | Oct-Nov-Dec | December | Short rains (East Africa) |

---

## Lead Time Calculation

For SEAS51 6-month forecasts, lead time is calculated from initialization month to the **last month** of the target season.

### Formula
```
lead_time = last_month_of_season - init_month

If lead_time < 0:
    lead_time += 12
    target_year = init_year + 1
```

### Season-to-Init Mapping

Lead time = months from initialization to target (last month of season).

| Season | Target Month | Lead 5 | Lead 4 | Lead 3 | Lead 2 |
|--------|--------------|--------|--------|--------|--------|
| **JFM** | March | Oct* | Nov* | Dec* | Jan |
| **MAM** | May | Dec* | Jan | Feb | Mar |
| **JJA** | August | Mar | Apr | May | Jun |
| **SON** | November | Jun | Jul | Aug | Sep |
| **OND** | December | Jul | Aug | Sep | Oct |

*Note: Previous year for init months before target month.

### Example: MAM 2024 Forecast

To forecast the MAM 2024 season (targeting SPI3-May = Mar+Apr+May precipitation):

| Init Month | Lead Time | Target | Command |
|------------|-----------|--------|---------|
| Dec 2023 | 5 | May 2024 | `--init-year 2023 --init-month 12` |
| Jan 2024 | 4 | May 2024 | `--init-year 2024 --init-month 1` |
| Feb 2024 | 3 | May 2024 | `--init-year 2024 --init-month 2` |
| Mar 2024 | 2 | May 2024 | `--init-year 2024 --init-month 3` |

**Lead time convention (from AA_WORKFLOW_COMMANDS.md):**
- Lead = number of months from init to target month
- Target = last month of season (captures full SPI3 for season)

---

## Regridding Workflow

V5 uses a **10km common grid** established from CDI data:

### Step 1: Establish Target Grid from CDI (10km)
```
CDI source: ~0.1° (~10km) resolution
Grid size: ~360 lat × 310 lon (East Africa domain)
```

### Step 2: Regrid SPI3 from ~100km to 10km
```
SPI3 source: ~1° (~100km) resolution, 51 ensemble members
Grid size: ~36 lat × 33 lon

Regridding method: xesmf bilinear interpolation
Process: Regrid each ensemble member separately, then recombine
```

### Step 3: Compute Empirical Probabilities at 10km
```
For each return period threshold:
  eprob = count(member_spi3 <= threshold) / n_members
```

This ensures all analysis is performed at 10km resolution, matching the CDI observation grid.

---

## Empirical Probability Calculation

### Method
```python
eprob = (n_members_below_threshold) / (total_members) × 100%
```

### Return Period Thresholds

| Return Period | SPI Threshold | Exceedance Prob |
|---------------|---------------|-----------------|
| 3-year | -0.43 | ~33% |
| 5-year | -0.68 | ~20% |
| 10-year | -0.84 | ~10% |
| 20-year | -1.04 | ~5% |
| 50-year | -1.28 | ~2% |

---

## V5 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GCS BUCKET: cdi_arco                              │
├─────────────────────┬───────────────────────────────────────────────┤
│  CDI Data (10km)    │  SPI3 Data (~100km)                            │
│  bn_icpac_cdi_store │  seas51_spi3_raw                              │
│  /10km/{year}/      │  /{YYYYMM}/                                   │
│  *.zarr             │  IceChunk repo                                │
└─────────────────────┴───────────────────────────────────────────────┘
           │                              │
           │                              │
           ▼                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DroughtDataLoaderV5                               │
├─────────────────────────────────────────────────────────────────────┤
│  Processing Order:                                                   │
│  1. Load CDI (10km) → Establish 10km target grid                    │
│  2. Load SPI3 (~100km) → Regrid to 10km using xesmf bilinear        │
│  3. Compute empirical probabilities at 10km resolution              │
│  4. Extract boundary-level statistics using regionmask              │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DroughtBayesianNetworkV5                          │
├─────────────────────────────────────────────────────────────────────┤
│  Nodes: antecedent_condition, exceedance_prob, spatial_coverage,    │
│         temporal_trend, data_confidence → risk_level → action       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT CSV                                   │
│  drought_bn_v5_{SEASON}_{INIT_YEAR}_{INIT_MONTH}.csv                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Usage Commands

### Single Season Analysis

```bash
# MAM 2024 with December 2023 initialization (lead 5)
python drought_bn_ibf_v5.py \
    --boundaries /srv/empirical_probability_output/icpac_adm1v3.geojson \
    --season MAM \
    --init-year 2023 \
    --init-month 12 \
    --output-dir ./results_2024/

# JJA 2024 with March 2024 initialization (lead 5)
python drought_bn_ibf_v5.py \
    --boundaries /srv/empirical_probability_output/icpac_adm1v3.geojson \
    --season JJA \
    --init-year 2024 \
    --init-month 3 \
    --output-dir ./results_2024/
```

### Auto Lead Time Selection

If `--init-month` is omitted, the script automatically selects the initialization month for lead 5:

```bash
# Auto-selects init December 2023 for MAM 2024
python drought_bn_ibf_v5.py \
    --boundaries /path/to/boundaries.geojson \
    --season MAM \
    --init-year 2024 \
    --output-dir ./results/
```

### All Seasons for a Year

```bash
# Generate BN outputs for all 5 seasons in 2024
python drought_bn_ibf_v5.py \
    --boundaries /srv/empirical_probability_output/icpac_adm1v3.geojson \
    --year-all-seasons \
    --init-year 2024 \
    --output-dir ./results_2024_all_seasons/
```

---

## 2024 Seasonal Analysis Commands

### JFM 2024 (January-February-March)
```bash
python drought_bn_ibf_v5.py \
    --boundaries icpac_adm1v3.geojson \
    --season JFM \
    --init-year 2023 \
    --init-month 10 \
    --output-dir ./bn_results_2024/
```
**Interpretation:** October 2023 initialization → March 2024 target (Lead 5)

### MAM 2024 (March-April-May)
```bash
python drought_bn_ibf_v5.py \
    --boundaries icpac_adm1v3.geojson \
    --season MAM \
    --init-year 2023 \
    --init-month 12 \
    --output-dir ./bn_results_2024/
```
**Interpretation:** December 2023 initialization → May 2024 target (Lead 5)

### JJA 2024 (June-July-August)
```bash
python drought_bn_ibf_v5.py \
    --boundaries icpac_adm1v3.geojson \
    --season JJA \
    --init-year 2024 \
    --init-month 3 \
    --output-dir ./bn_results_2024/
```
**Interpretation:** March 2024 initialization → August 2024 target (Lead 5)

### SON 2024 (September-October-November)
```bash
python drought_bn_ibf_v5.py \
    --boundaries icpac_adm1v3.geojson \
    --season SON \
    --init-year 2024 \
    --init-month 6 \
    --output-dir ./bn_results_2024/
```
**Interpretation:** June 2024 initialization → November 2024 target (Lead 5)

### OND 2024 (October-November-December)
```bash
python drought_bn_ibf_v5.py \
    --boundaries icpac_adm1v3.geojson \
    --season OND \
    --init-year 2024 \
    --init-month 7 \
    --output-dir ./bn_results_2024/
```
**Interpretation:** July 2024 initialization → December 2024 target (Lead 5)

### Batch Command (All Seasons)
```bash
python drought_bn_ibf_v5.py \
    --boundaries icpac_adm1v3.geojson \
    --year-all-seasons \
    --init-year 2024 \
    --output-dir ./bn_results_2024_all/
```

---

## Output CSV Format

| Column | Description |
|--------|-------------|
| `boundary_id` | Admin boundary ID (GID_1) |
| `boundary_name` | Admin boundary name |
| `country` | Country name |
| `cdi_weighted_mean` | Weighted mean CDI (last 6 months) |
| `cdi_category` | CDI category (No_Drought, Mild, Moderate, Severe, Extreme) |
| `cdi_trend` | Temporal trend (improving, stable, worsening) |
| `cdi_confidence` | Data confidence score (0-1) |
| `severity_index` | Combined forecast severity |
| `spatial_coverage_pct` | % of area with high exceedance |
| `eprob_3yr` | 3-year return period exceedance probability |
| `eprob_5yr` | 5-year return period exceedance probability |
| `eprob_10yr` | 10-year return period exceedance probability |
| `eprob_20yr` | 20-year return period exceedance probability |
| `eprob_50yr` | 50-year return period exceedance probability |
| `risk_level` | BN-inferred risk (Minimal, Low, Moderate, High, Extreme) |
| `recommended_action` | Recommended response |
| `confidence` | Action confidence score |

---

## Service Account Configuration

The script requires a GCS service account JSON file with read access to the `cdi_arco` bucket.

**Default search paths:**
1. `/scratch/notebook/coiled-data-e4drr_202505.json`
2. `/home/roller/Documents/08-2023/impact_weather_icpac/lab/icpac_gcp/e4drr/gcp-coiled-sa-20250310/coiled-data-e4drr_202505.json`

**Manual specification:**
```bash
python drought_bn_ibf_v5.py \
    --service-account /path/to/your/service_account.json \
    ...
```

---

## Comparison: V4 vs V5

| Feature | V4 | V5 |
|---------|----|----|
| CDI Source | Local NetCDF | GCS IceChunk/Zarr |
| SPI3 Source | Local NetCDF (pre-computed eprob) | GCS IceChunk (raw ensemble) |
| Eprob Computation | Pre-computed | On-the-fly from ensemble |
| Season Support | Manual lead calc | Built-in season mapping |
| Batch Processing | No | Yes (all seasons) |
| Cloud Native | No | Yes |

---

## Troubleshooting

### "No service account file found"
Ensure you have a valid GCS service account JSON file and either:
- Place it in one of the default paths
- Specify with `--service-account`

### "No CDI data available"
Check that CDI data exists in GCS for the requested months:
```python
from drought_bn_ibf_v5 import GCSCDILoader
loader = GCSCDILoader(service_account_file="/path/to/sa.json")
print(loader.check_availability(2024, 12))
```

### "No SPI3 data available"
Check that SPI3 data exists for the initialization month:
```python
from drought_bn_ibf_v5 import GCSSPI3Loader
loader = GCSSPI3Loader(service_account_file="/path/to/sa.json")
print(loader.check_availability(2023, 12))
```

---

## Python API Usage

```python
from drought_bn_ibf_v5 import (
    analyze_season,
    analyze_year_all_seasons,
    get_season_lead_mapping,
    get_recommended_init_months
)

# Single season
results = analyze_season(
    boundaries_path="icpac_adm1v3.geojson",
    season="MAM",
    init_year=2023,
    init_month=12,
    output_dir="./results/"
)

# All seasons for a year
all_results = analyze_year_all_seasons(
    boundaries_path="icpac_adm1v3.geojson",
    year=2024,
    output_dir="./results_2024/"
)

# Get lead time info
info = get_season_lead_mapping("MAM", init_year=2023, init_month=12)
print(info)
# {'lead_time': 5, 'target_year': 2024, 'target_month': 5, ...}

# Get recommended init months for a season
configs = get_recommended_init_months("MAM")
print(configs)
# [(12, 5, -1), (1, 4, 0), (2, 3, 0)]  # (init_month, lead, year_offset)
```

---

*Document created: January 2025*
*Version: 5.0 (GCS IceChunk Integration)*
