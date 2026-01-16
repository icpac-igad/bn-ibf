# Drought Impact-Based Forecasting using Bayesian Networks

## Overview

This document describes the methodology used in the Drought BN IBF system (Version 3) for drought risk assessment using Bayesian Networks with lookup tables (Conditional Probability Tables).

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                                   │
├─────────────────────────────────────────────────────────────────────┤
│  CDI (Combined Drought Indicator)    │  Empirical Probability        │
│  - Monthly NetCDF files              │  - Return period exceedance   │
│  - 0-10 scale                        │  - 3yr, 5yr, 10yr, 20yr, 50yr │
│  - Current drought conditions        │  - SPI3 forecast skill        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA EXTRACTION                                   │
├─────────────────────────────────────────────────────────────────────┤
│  1. Load admin boundaries (GeoJSON)                                  │
│  2. Create region mask using regionmask                              │
│  3. Extract boundary statistics:                                     │
│     - Area-based: mean over all pixels within boundary               │
│     - Centroid-based: nearest-neighbor for small boundaries          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   BAYESIAN NETWORK                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   antecedent_condition ──┐                                          │
│                          ├──► risk_level ──► action                 │
│   exceedance_prob ───────┤                                          │
│                          │                                          │
│   spatial_coverage ──────┘                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                       │
├─────────────────────────────────────────────────────────────────────┤
│  - Risk Level: Minimal, Low, Moderate, High, Extreme                │
│  - Recommended Action: Monitor, Be_Aware, Be_Prepared, Take_Action  │
│  - Confidence scores                                                 │
│  - CSV results per boundary                                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Data Loading Components

### 1.1 CDI Data Loader (`CDIDataLoader`)

Loads Combined Drought Indicator data from monthly NetCDF files.

**File Path Pattern:**
```
/srv/icpac_monthly_netcdf/{year}/eadw-cdi-data-{year}-{month}.nc
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `get_cdi_file_path(year, month)` | Constructs file path for given year/month |
| `check_availability(year, month)` | Returns True if CDI file exists |
| `get_latest_available(target_year, target_month)` | Finds most recent available CDI within 12 months |
| `load_cdi(year, month)` | Loads CDI DataArray with caching |

**CDI Categories:**

| CDI Value | Category |
|-----------|----------|
| 0 | No_Drought |
| 1-2 | Mild |
| 3-4 | Moderate |
| 5-7 | Severe |
| 8-10 | Extreme |

### 1.2 Drought Data Loader (`DroughtDataLoaderV3`)

Enhanced data loader with hybrid extraction strategy.

**Key Features:**
- Area-based extraction for boundaries containing grid cells
- Centroid-based extraction for small boundaries (< 1 pixel at target resolution)
- Automatic regridding of CDI to match empirical probability grid
- Multi-return period probability extraction

**Extraction Methods:**

```python
# Area-based (n_pixels > 0)
rp_probs[var_name] = np.nanmean(var_data.values[boundary_mask])
cdi_mean = np.nanmean(cdi_data.values[boundary_mask])

# Centroid-based (n_pixels == 0)
rp_probs[var_name] = data.sel(lat=centroid_lat, lon=centroid_lon, method='nearest')
```

**Severity Index Calculation:**

The severity index combines multiple return period probabilities:

```python
def compute_severity_index(rp_probs):
    """Max-weighted method: base on 5yr, boost for longer return periods."""
    base_prob = rp_probs.get('eprob_5yr', 0.0)

    boost = 0.0
    if rp_probs.get('eprob_10yr', 0) > 0.3:
        boost += 0.05
    if rp_probs.get('eprob_20yr', 0) > 0.2:
        boost += 0.05
    if rp_probs.get('eprob_50yr', 0) > 0.1:
        boost += 0.05

    return min(1.0, base_prob + boost)
```

---

## 2. Bayesian Network Structure

### 2.1 Network Topology

**Nodes:**

| Node | Type | States | Description |
|------|------|--------|-------------|
| `antecedent_condition` | Evidence | No_Drought, Mild, Moderate, Severe, Extreme | Current drought state from CDI |
| `exceedance_prob` | Evidence | Very_Low, Low, Medium, High, Very_High | Forecast severity (from severity_index) |
| `spatial_coverage` | Evidence | Localized, Moderate, Widespread | Spatial extent of forecast |
| `risk_level` | Hidden | Minimal, Low, Moderate, High, Extreme | Inferred risk state |
| `action` | Query | Monitor, Be_Aware, Be_Prepared, Take_Action | Recommended response |

**Edges:**
- `antecedent_condition → risk_level`
- `exceedance_prob → risk_level`
- `spatial_coverage → risk_level`
- `risk_level → action`

### 2.2 State Discretization

**Exceedance Probability Thresholds:**

| State | Probability Range |
|-------|-------------------|
| Very_Low | < 0.20 |
| Low | 0.20 - 0.40 |
| Medium | 0.40 - 0.60 |
| High | 0.60 - 0.80 |
| Very_High | > 0.80 |

**Spatial Coverage Thresholds:**

| State | Coverage Range |
|-------|----------------|
| Localized | < 30% |
| Moderate | 30% - 60% |
| Widespread | > 60% |

### 2.3 Conditional Probability Tables (CPTs)

#### Action CPD (P(Action | Risk_Level))

```
Risk Level:   Minimal   Low    Moderate   High   Extreme
─────────────────────────────────────────────────────────
Monitor        0.95    0.10     0.00     0.00    0.00
Be_Aware       0.05    0.85     0.15     0.00    0.00
Be_Prepared    0.00    0.05     0.80     0.20    0.05
Take_Action    0.00    0.00     0.05     0.80    0.95
```

#### Risk Level CPD

The risk level CPT has 75 combinations (5 × 5 × 3 = 75).

**Expert Rules for Risk Computation:**

1. **Extreme antecedent + High/Very_High exceedance:**
   - Widespread/Moderate spatial → P(Extreme) = 0.75
   - Localized → P(Extreme) = 0.40

2. **Severe antecedent + Medium+ exceedance + Widespread:**
   - P(High) = 0.60, P(Extreme) = 0.25

3. **Moderate conditions with forecast concern:**
   - Widespread → P(Moderate) = 0.50, P(High) = 0.35

4. **Low exceedance with existing drought:**
   - Maintains monitoring state, gradual recovery

5. **Base risk formula:**
   ```python
   base_risk = antecedent_idx * 0.4 + exceedance_idx * 0.6

   if spatial == 'Widespread':
       base_risk += 0.5
   elif spatial == 'Moderate':
       base_risk += 0.25
   ```

---

## 3. Inference Process

### 3.1 Variable Elimination

The system uses pgmpy's `VariableElimination` for exact inference:

```python
from pgmpy.inference import VariableElimination

inference = VariableElimination(bn.model)

# Set evidence from observations
evidence = {
    'antecedent_condition': categorize_antecedent(cdi_category),
    'exceedance_prob': categorize_exceedance(severity_index),
    'spatial_coverage': categorize_spatial(coverage)
}

# Query posterior distributions
risk_result = inference.query(variables=['risk_level'], evidence=evidence)
action_result = inference.query(variables=['action'], evidence=evidence)
```

### 3.2 Decision Output

The recommended action is the Maximum A Posteriori (MAP) estimate:

```python
recommended_action = action_states[np.argmax(action_probs)]
confidence = float(np.max(action_probs))
```

---

## 4. Data Flow Summary

```
1. INPUTS
   └── CDI NetCDF (year/month) ────────────────► cdi_mean, cdi_category
   └── Empirical Probability NC ───────────────► eprob_3yr, ..., eprob_50yr
   └── Admin Boundaries GeoJSON ───────────────► boundary geometry

2. FEATURE EXTRACTION
   └── cdi_mean ──────────────────────────────► antecedent_condition
   └── severity_index (from eprob) ───────────► exceedance_prob
   └── spatial_coverage (% > 0.5) ────────────► spatial_coverage

3. BAYESIAN INFERENCE
   └── Evidence: {antecedent, exceedance, spatial}
   └── Query: P(risk_level | evidence)
   └── Query: P(action | evidence)

4. OUTPUT
   └── risk_level: Minimal/Low/Moderate/High/Extreme
   └── recommended_action: Monitor/Be_Aware/Be_Prepared/Take_Action
   └── confidence: probability of recommended action
```

---

## 5. File Structure

```
drought_bn_ibf_v3.py
├── Constants
│   ├── MONTH_MAP
│   ├── COUNTRY_CODE_MAP
│   └── CDI_CATEGORIES
├── Helper Functions
│   ├── categorize_cdi()
│   └── get_season_from_lead()
├── CDIDataLoader
│   ├── get_cdi_file_path()
│   ├── check_availability()
│   ├── get_latest_available()
│   └── load_cdi()
├── DroughtDataLoaderV3
│   ├── _load_boundaries()
│   ├── load_empirical_probability()
│   ├── regrid_cdi_to_eprob()
│   ├── _extract_at_centroid()
│   ├── prepare_analysis_data()
│   └── _compute_severity_index()
├── DroughtBayesianNetworkV3
│   ├── _setup_cpds()
│   ├── _create_risk_cpt()
│   ├── _compute_risk_probs()
│   ├── _categorize_* methods
│   ├── process_boundary()
│   └── process_all_boundaries()
└── analyze_drought() - Main entry point
```

---

## 6. Usage

### Command Line

```bash
python drought_bn_ibf_v3.py \
    --boundaries icpac_adm1v3.geojson \
    --eprob /srv/empirical_probability_output/empirical_probabilities.nc \
    --cdi-base /srv/icpac_monthly_netcdf \
    --init-year 2024 \
    --init-month 12 \
    --target-lead 5 \
    --cdi-scenario actual \
    --output drought_bn_v3_results.csv
```

### Python API

```python
from drought_bn_ibf_v3 import analyze_drought

results = analyze_drought(
    boundaries_path="icpac_adm1v3.geojson",
    init_year=2024,
    init_month=12,
    target_lead=5,
    cdi_scenario='actual',
    output_path="results.csv"
)
```

---

## 7. Limitations (V3)

1. **Single CDI observation**: Uses only one month of CDI data
2. **Fixed uncertainty**: Does not account for CDI data lag or quality
3. **Static CPTs**: Expert-defined probabilities, no learning from data
4. **No temporal persistence**: Does not model drought evolution over time
5. **Point-in-time assessment**: No consideration of drought trajectory

These limitations are addressed in V4 with multi-month CDI support and uncertainty propagation.

---

## 8. Version 4 Enhancements

### 8.1 Multi-Month CDI Support

V4 uses up to 6 months of CDI data (configurable) to provide:

1. **Weighted temporal averaging** - Recent months have higher weight
2. **Trend detection** - Identifies improving, stable, or worsening conditions
3. **Persistence scoring** - Counts consecutive drought months
4. **Flexible data handling** - Works with whatever months are available

**Temporal Weights (default):**

| Month Index | Weight | Description |
|-------------|--------|-------------|
| 0 (most recent) | 0.35 | Highest weight |
| 1 | 0.25 | |
| 2 | 0.15 | |
| 3 | 0.12 | |
| 4 | 0.08 | |
| 5 (oldest) | 0.05 | Lowest weight |

### 8.2 CDI Time Series Analysis

**Trend Computation:**
```python
def compute_trend(cdi_values):
    # Simple linear regression on time-ordered values
    slope = np.polyfit(x, values, 1)[0]
    trend_magnitude = np.clip(slope / 2.0, -1.0, 1.0)

    if trend_magnitude < -0.15:
        return 'improving'
    elif trend_magnitude > 0.15:
        return 'worsening'
    else:
        return 'stable'
```

**Persistence Scoring:**
```python
def compute_persistence(cdi_values, threshold=1.0):
    # Count consecutive months from most recent where CDI >= threshold
    persistence = 0
    for cdi in reversed(cdi_values):
        if cdi >= threshold:
            persistence += 1
        else:
            break
    return persistence
```

### 8.3 Data Confidence Scoring

Confidence reflects data quality and availability:

```python
def compute_data_confidence(n_available, n_requested, max_lag_months):
    # Base from availability ratio
    if n_available == 0:
        return 0.1  # Very low but allows BN to work
    elif n_available == 1:
        base = 0.3
    elif n_available == 2:
        base = 0.5
    elif n_available >= 3:
        base = 0.6 + (n_available - 3) * 0.1

    # Penalize for lag (-15% per month)
    lag_penalty = max(0, 1 - max_lag_months * 0.15)

    return min(1.0, base * lag_penalty)
```

**Confidence Categories:**

| Score | Category | Effect on BN |
|-------|----------|--------------|
| < 0.4 | Low | 50% uniform mixing (high uncertainty) |
| 0.4 - 0.7 | Medium | 20% uniform mixing |
| > 0.7 | High | No mixing (trust data) |

### 8.4 Enhanced Bayesian Network Structure

**V4 Network Topology:**

```
antecedent_condition ──┐
                       │
exceedance_prob ───────┼──► risk_level ──► action
                       │
spatial_coverage ──────┤
                       │
temporal_trend ────────┤     (NEW in V4)
                       │
data_confidence ───────┘     (NEW in V4, optional)
```

**New Nodes:**

| Node | States | Description |
|------|--------|-------------|
| `temporal_trend` | Improving, Stable, Worsening | Direction of drought change |
| `data_confidence` | Low, Medium, High | Trust in antecedent data |

**CPT Combinations:**

| Version | Formula | Total Combinations |
|---------|---------|-------------------|
| V3 | 5 × 5 × 3 | 75 |
| V4 (no conf) | 5 × 5 × 3 × 3 | 225 |
| V4 (with conf) | 5 × 5 × 3 × 3 × 3 | 675 |

### 8.5 Expert Rules for Trend Integration

Key V4 expert rules incorporating temporal trend:

1. **Extreme + Worsening + High Exceedance:**
   - Widespread → P(Extreme) = 0.80
   - Most severe scenario, highest alert

2. **Severe/Extreme + Worsening:**
   - P(High/Extreme) significantly elevated
   - Signals deteriorating situation

3. **Improving Trend + Low Antecedent:**
   - P(Minimal) boosted significantly
   - Recovery signal reduces risk

4. **Onset Risk (Low antecedent + High exceedance + Worsening):**
   - Higher concern than stable trend
   - Signals emerging drought

### 8.6 CDI Data Flexibility

V4 gracefully handles missing data:

| Available Months | Behavior |
|------------------|----------|
| 0 | Error - cannot proceed |
| 1-2 | Low confidence, uses available data |
| 3-5 | Medium confidence, partial temporal analysis |
| 6 | Full confidence, complete analysis |

**Example Scenarios:**

```python
# Scenario 1: All 6 months available
# - Full temporal weighting
# - Reliable trend detection
# - High confidence → sharp risk distribution

# Scenario 2: Only 2 months available (recent)
# - Weights renormalized to available months
# - Limited trend detection (may default to 'stable')
# - Low confidence → spread risk distribution toward uniform

# Scenario 3: 4 months available, but 2-month gap
# - Uses available months with adjusted weights
# - Trend computed from available points
# - Medium confidence
```

### 8.7 V4 Usage

**Command Line:**
```bash
python drought_bn_ibf_v4.py \
    --boundaries icpac_adm1v3.geojson \
    --init-year 2024 \
    --init-month 12 \
    --target-lead 5 \
    --cdi-months 6 \
    --output drought_bn_v4_results.csv
```

**With fewer CDI months:**
```bash
python drought_bn_ibf_v4.py \
    --cdi-months 3 \
    --output drought_bn_v4_results_3mo.csv
```

**Without confidence node:**
```bash
python drought_bn_ibf_v4.py \
    --no-confidence-node \
    --output drought_bn_v4_noconf.csv
```

**Python API:**
```python
from drought_bn_ibf_v4 import analyze_drought_v4

results = analyze_drought_v4(
    boundaries_path="icpac_adm1v3.geojson",
    init_year=2024,
    init_month=12,
    target_lead=5,
    n_cdi_months=6,
    include_confidence_node=True,
    output_path="results_v4.csv"
)
```

### 8.8 Output Fields (V4)

V4 adds these fields to the output:

| Field | Type | Description |
|-------|------|-------------|
| `cdi_weighted_mean` | float | Temporally weighted CDI average |
| `cdi_trend` | str | improving/stable/worsening |
| `cdi_trend_magnitude` | float | -1 to +1 trend strength |
| `cdi_persistence_months` | int | Consecutive drought months |
| `cdi_n_available` | int | Number of CDI months loaded |
| `cdi_confidence` | float | 0-1 data confidence score |
| `cdi_date_range` | str | "YYYY-MM to YYYY-MM" |

---

## 9. Comparison: V3 vs V4

| Feature | V3 | V4 |
|---------|----|----|
| CDI months | 1 | 1-6 (configurable) |
| Temporal trend | No | Yes |
| Persistence tracking | No | Yes |
| Data confidence | No | Optional |
| BN nodes | 5 | 6-7 |
| CPT combinations | 75 | 225-675 |
| Missing data handling | Falls back | Graceful degradation |
| Uncertainty quantification | Fixed | Dynamic |

### When to use V3:
- Real-time operations with minimal latency
- Single-month analysis sufficient
- Simpler interpretation needed

### When to use V4:
- Comprehensive drought assessment
- Historical analysis with trend detection
- Operational forecasting with confidence bounds
- Research applications requiring uncertainty quantification
