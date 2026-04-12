`# Flood Impact-Based Forecasting (IBF) System — Technical Documentation

**Version**: v20260412  
**Date**: 12 April 2026  
**System**: Bayesian Network flood risk assessment for East Africa (admin-1 level)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Data Sources and Stores](#2-data-sources-and-stores)
3. [Data Preparation Pipeline (`flood_data_prep.py`)](#3-data-preparation-pipeline)
4. [The Bayesian Network — Mathematical Foundation](#4-the-bayesian-network--mathematical-foundation)
5. [Python BN Implementation (`flood_bn_ibf_v1.py`)](#5-python-bn-implementation)
6. [Julia BN Implementation (`flood_bn_ibf_v1.jl`)](#6-julia-bn-implementation)
7. [RxInfer.jl and Message Passing Inference](#7-rxinferjl-and-message-passing-inference)
8. [Inference Methods Compared: Variable Elimination vs Message Passing vs Direct Matrix](#8-inference-methods-compared)
9. [Orchestration and Summarization](#9-orchestration-and-summarization)
10. [Visualization (`plot_daily_risk_maps.py`)](#10-visualization)
11. [End-to-End Walkthrough: March 1–10, 2026](#11-end-to-end-walkthrough)
12. [Future Directions](#12-future-directions)

---

## 1. System Overview

The Flood IBF system produces daily flood risk assessments for 227 admin-1 boundaries across 11 East African countries (Burundi, Djibouti, Eritrea, Ethiopia, Kenya, Rwanda, Somalia, South Sudan, Sudan, Tanzania, Uganda). It operates by:

1. **Ingesting observations** — 7-day rolling IMERG half-hourly satellite precipitation, summed to daily totals, providing antecedent moisture conditions.
2. **Ingesting forecasts** — ECMWF IFS ensemble (51 members, 7-day lead) total precipitation, accumulated over multiple durations (3h through 7 days).
3. **Computing exceedance probabilities** — comparing ensemble forecast accumulations against pixel-wise climatological return-period thresholds derived from 27 years of CMORPH data.
4. **Running a Bayesian Network** — a discrete expert-elicited BN with 8 nodes that maps the evidence (antecedent conditions, exceedance probability, spatial coverage, rainfall trend, forecast agreement) to a risk level (Minimal → Extreme) and a recommended action (Monitor → Act).
5. **Producing tabular and map outputs** — per-boundary risk CSVs, a 10-day summary table, and admin-1 choropleth maps.

The system is bilingual: a Python implementation using `pgmpy` for variable elimination inference, and a Julia implementation using `RxInfer.jl` for reactive message passing. Both yield numerically identical results on the same inputs. In the operational pipeline deployed for the March 2026 assessment, the **Julia direct-inference path** was used for its speed (sub-millisecond per boundary vs. 10–15 ms in Python).

### Architecture Diagram

```
                        ┌──────────────────────────────┐
                        │   IMERG HH Icechunk Store    │
                        │   source.coop (observations)  │
                        │   451k time steps, 0.1° grid  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                      ┌─────────────────────────────────┐
                      │     flood_data_prep.py          │
                      │     (uv run — Python)           │
   ┌──────────────┐   │                                 │   ┌────────────────────┐
   │ ECMWF TP     │──▶│  • IMERG → 7-day antecedent    │   │ CMORPH RP NetCDF   │
   │ Icechunk     │   │  • ECMWF → duration accums     │◀──│ return_period_precip│
   │ (forecasts)  │   │  • exceedance P(≥ 2yr RP)     │   │ (27yr Gumbel fits)  │
   └──────────────┘   │  • zonal stats → admin-1 CSV   │   └────────────────────┘
                      └──────────────┬──────────────────┘
                                     │ flood_inputs_YYYY-MM-DD.csv
                                     ▼
                      ┌─────────────────────────────────┐
                      │   flood_bn_ibf_v1.jl            │
                      │   (Julia — direct matrix infer) │
                      │                                 │
                      │   • Categorize → discrete states│
                      │   • CPT lookup → risk probs     │  ┌────────────────────────┐
                      │   • CPT × risk → action probs   │  │ icpac_adm1v3.geojson   │
                      │   • argmax → risk, action labels│  │ 227 admin-1 boundaries │
                      └──────────────┬──────────────────┘  └────────────────────────┘
                                     │ flood_bn_v1_YYYY-MM-DD.csv
                                     ▼
                      ┌──────────────────────────────────┐
                      │  summarize_bn.py + plot maps     │
                      │  10-day max-risk summary CSV     │
                      │  choropleth PNGs (panel + daily) │
                      └──────────────────────────────────┘
```

---

## 2. Data Sources and Stores

### 2.1 IMERG Half-Hourly Precipitation (Observations)

| Property | Value |
|----------|-------|
| Store | `s3://e4drr-project/observations/imerg_hh_icechunk` (Icechunk on source.coop) |
| Product | GPM 3IMERGHH v07 Final |
| Variable | `precipitation(time, lat, lon)` |
| Units | mm/hr |
| Resolution | 0.1° (~10 km), half-hourly |
| Time range | 2000-06-01 → 2026-03-10 (451,827 time steps) |
| Grid | 400 lat × 345 lon, lat -14.45→25.45, lon 19.55→53.95 |
| Chunking | (48, 400, 345) — one chunk = one day |
| Access | Anonymous read via Icechunk S3 protocol, path-style addressing |

**Usage in the pipeline**: For each target date D, the pipeline slices `[D-7, D)` (7 days × 48 half-hour steps = 336 time steps), multiplies by 0.5 to convert mm/hr → mm per half-hour, then `resample("1D").sum()` to get 7 daily totals per pixel. These are used for:
- **Antecedent rainfall** — 7-day sum (mm) per admin-1 (area-weighted mean).
- **Trend** — linear regression slope of 7 daily boundary-mean values; classified as Decreasing (<−2 mm/day), Stable, or Increasing (>+2 mm/day).

### 2.2 ECMWF IFS Ensemble Total Precipitation (Forecasts)

| Property | Value |
|----------|-------|
| Store | `s3://e4drr-project/forecasts/ecmwf_ea_tp_icechunk` (Icechunk on source.coop) |
| Variable | `tp(init_date, member, lead_time, lat, lon)` |
| Units | metres (ECMWF native; multiplied by 1000 for mm) |
| Resolution | 0.25° (~25 km) |
| Init dates | 2026-03-01 → 2026-03-31, 00Z daily |
| Members | 51 (`control`, `ens_01` … `ens_50`) |
| Lead times | 53 steps: 3-hourly 0–144 h, then 6-hourly 144–168 h (7 days) |
| Grid | 157 lat × 145 lon, lat **descending** 25→−14, lon 19→55 |
| Chunking | (1, 1, 53, 157, 145) — one chunk = one member of one init_date |

**Critical detail**: `tp` is **cumulative from initialization time**. At lead=0, tp=0 everywhere. To get the accumulation over a window `[0, Δ]`, simply read `tp` at lead=Δ (no differencing needed when the start is the init time). For arbitrary sub-windows `[Δ₁, Δ₂]`, the accumulation is `tp[Δ₂] − tp[Δ₁]`.

**Usage in the pipeline**: For each target date D (init_date=D, 00Z), the pipeline reads `tp` at 7 specific lead times corresponding to the accumulation windows:

| Duration label | Lead time | Accumulation meaning |
|----------------|-----------|---------------------|
| `3hr`  | 3 h   | First 3 hours of forecast |
| `6hr`  | 6 h   | First 6 hours |
| `12hr` | 12 h  | First 12 hours |
| `24hr` | 24 h  | First 24 hours |
| `48hr` | 48 h  | First 2 days |
| `72hr` | 72 h  | First 3 days |
| `7day` | 168 h | Full 7-day forecast |

Each gives a `(51, 157, 145)` array (mm after ×1000 conversion).

### 2.3 CMORPH Return Period Thresholds

| Property | Value |
|----------|-------|
| File | `cmorph_ea_return_periods.nc` (local, ~328 MB) |
| Source | NOAA CDR CMORPH v1.0, processed via Coiled + pencil-chunked Zarr |
| Variables | `return_period_precip(duration, return_period, lat, lon)` — mm |
|           | `dist_location(duration, lat, lon)` — Gumbel location parameter |
|           | `dist_scale(duration, lat, lon)` — Gumbel scale parameter |
|           | `annual_maxima(duration, year, lat, lon)` — raw annual maxima |
| Durations | 30min, 1hr, 3hr, 6hr, 12hr, 24hr, 48hr, 72hr, 7day (9 total) |
| Return periods | 2, 5, 10, 20, 50, 100 years |
| Grid | ~0.073° (~8 km), 550 lat × 474 lon |
| Period | 1998–2024 (27 years), Gumbel method-of-moments fit |

The pipeline uses the **2-year return period** threshold at each of the 7 matching durations (3hr through 7day). These pixel-wise thresholds are nearest-neighbor regridded onto the ECMWF 0.25° grid before comparison. The 2-year RP represents a "moderate" rain event — one expected to be equaled or exceeded once every 2 years on average at each pixel. This is deliberately sensitive; operational deployments might switch to 5-year or higher.

### 2.4 Admin-1 Boundaries

| Property | Value |
|----------|-------|
| File | `icpac_adm1v3.geojson` |
| CRS | EPSG:4326 |
| Features | 227 admin-1 regions |
| Fields | `GID_1` (e.g. "KEN.32_1"), `NAME_1` (e.g. "Nairobi") |
| Extent | 21.84°E → 51.42°E, 11.75°S → 23.15°N |
| Countries | 11 ICPAC member states |

Country is inferred from the ISO 3166-1 alpha-3 prefix of `GID_1`:
`BDI` Burundi, `DJI` Djibouti, `ERI` Eritrea, `ETH` Ethiopia, `KEN` Kenya, `RWA` Rwanda, `SOM` Somalia, `SSD` South Sudan, `SDN` Sudan, `TZA` Tanzania, `UGA` Uganda.

---

## 3. Data Preparation Pipeline

**Script**: `flood_data_prep.py`  
**Runner**: `uv run --with icechunk,xarray,zarr>=3,numpy,pandas,geopandas,regionmask,netcdf4,pyarrow,scipy`

### 3.1 Architecture

The data prep script is a single-file, dependency-isolated pipeline. It uses the `uv` shebang line to install all dependencies into a transient virtual environment at runtime — no persistent `conda`/`pip` env required.

Flow:

```
CLI args (--date D, --rp-years, etc.)
        │
        ├─► Load admin-1 GeoJSON → 227 features
        │
        ├─► Open IMERG icechunk → slice [D-7, D) → sum to 7 daily grids
        │   ├─► Build regionmask on IMERG 0.1° grid
        │   ├─► Per-boundary area-weighted mean of each daily grid → (7 × 227) matrix
        │   ├─► antecedent_mm = row sum per boundary
        │   └─► trend = polyfit slope per boundary, classify by ±2 mm/day
        │
        ├─► Open ECMWF icechunk → select init_date=D
        │   ├─► Read tp at 7 lead times → (51, 157, 145) per duration → ×1000 for mm
        │   ├─► Load CMORPH RP thresholds (2-yr) → nearest-regrid to ECMWF grid
        │   ├─► Per duration: eprob = (accum ≥ threshold).mean(member) → (157, 145)
        │   ├─► p_heavy = max over 7 durations per pixel
        │   ├─► eprob_24h = eprob at 24hr duration (sidecar)
        │   ├─► Build regionmask on ECMWF 0.25° grid
        │   ├─► eprob_heavy_adm = area-weighted mean of p_heavy per boundary
        │   ├─► spatial_coverage = area-weighted fraction of pixels with p_heavy ≥ 0.5
        │   └─► fill_small_boundaries: centroid-nearest fallback for tiny boundaries
        │
        └─► Write CSV: 227 rows × 12 columns
```

### 3.2 Key Functions

| Function | Purpose |
|----------|---------|
| `open_icechunk(prefix)` | Opens an Icechunk repo on source.coop anonymously (path-style S3), returns `xr.Dataset` via `xr.open_zarr()` on the session store |
| `imerg_daily_totals(imerg, D)` | Slices half-hourly IMERG over `[D-7, D)`, converts mm/hr → mm (×0.5 per half-hour), resamples to daily sum |
| `ecmwf_window_accums(ecmwf, D)` | Selects `init_date=D`, reads tp at 7 target lead_times, converts m→mm |
| `load_cmorph_thresholds(path, rp)` | Loads pixel-wise threshold grid for a given return period, per duration |
| `regrid_to(da, lat_target, lon_target)` | Nearest-neighbor regridding via `xr.DataArray.interp()` (requires scipy) |
| `build_mask(gdf, lat, lon)` | Constructs a `regionmask.Regions` object from the GeoDataFrame, returns integer mask on the target grid |
| `zonal_reduce(da, mask, lat, n_regions, thresh=None)` | Area-weighted (cos-lat) zonal mean or fraction-above-threshold per region, looping over region indices |
| `fill_small_boundaries(values, da, gdf, thresh=None)` | For regions with no pixel centroid inside, samples the nearest pixel at the region's geometric centroid |
| `classify_trend(slope, band)` | Maps a regression slope (mm/day) to Increasing / Stable / Decreasing via ±band threshold |

### 3.3 Output CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | GID_1 identifier |
| `name` | str | NAME_1 admin boundary name |
| `country` | str | Country name |
| `antecedent_rainfall_mm` | float | 7-day accumulated IMERG (mm), area-weighted mean |
| `antecedent_category` | str | (left empty — categorized by the BN at inference time) |
| `rainfall_trend` | str | Decreasing / Stable / Increasing |
| `trend_slope_mm_per_day` | float | Raw slope from linear fit |
| `gefs_eprob_heavy` | float | P_heavy: max over 7 durations of P(accum ≥ 2yr-RP), area-weighted mean |
| `eprob_24h` | float | P(24h accum ≥ 2yr-RP threshold), area-weighted mean (sidecar column) |
| `spatial_coverage` | float | Fraction of boundary area where P_heavy ≥ 0.5 |
| `forecast_agreement` | str | Fixed at "Medium" (single-model; ignored when `--no-agreement` used) |
| `target_date` | str | ISO date |

### 3.4 Small-Boundary Fallback

Seven admin-1 regions (e.g. Bujumbura Mairie, Kampala, Dar es Salaam) are small enough that no ECMWF 0.25° pixel centroid falls within their polygon. The `fill_small_boundaries` function handles this by extracting the value at the pixel nearest to the region's geometric centroid using `xr.DataArray.sel(method='nearest')`. This ensures all 227 rows have non-NaN forecast columns.

---

## 4. The Bayesian Network — Mathematical Foundation

### 4.1 Directed Acyclic Graph (DAG)

The flood BN is a discrete Bayesian network with 8 nodes arranged in a two-layer structure:

```
        Evidence layer (5 root nodes)               Hidden          Query
    ┌──────────────────────────────────┐      ┌────────────┐    ┌────────┐
    │ antecedent_rainfall (5 states)   │─────▶│            │    │        │
    │ exceedance_prob     (5 states)   │─────▶│ risk_level │───▶│ action │
    │ spatial_coverage    (3 states)   │─────▶│ (5 states) │    │(4 st.) │
    │ rainfall_trend      (3 states)   │─────▶│            │    │        │
    │ forecast_agreement  (3 states)   │─────▶│            │    │        │
    └──────────────────────────────────┘      └────────────┘    └────────┘
```

The **evidence nodes** are observed (clamped to data at inference time). `risk_level` is the hidden variable we want to infer. `action` is the query variable — the actionable output.

The DAG encodes the following conditional independence assumptions:

- `action` is independent of all evidence given `risk_level` (action depends *only* on risk level, not directly on rainfall data).
- All five evidence nodes are marginally independent of each other (no edges between them). In reality, antecedent rainfall and exceedance probability are correlated, but the BN treats them as conditionally independent parents of risk — the CPT captures their joint effect.

### 4.2 Joint Distribution

By the chain rule of Bayesian networks, the joint distribution factorizes as:

```
P(ant, exc, spa, tre, agr, risk, act)
  = P(ant) · P(exc) · P(spa) · P(tre) · P(agr)
    · P(risk | ant, exc, spa, tre, agr)
    · P(act | risk)
```

Since all evidence nodes are observed (clamped), the marginal priors `P(ant)`, `P(exc)`, etc. cancel out during conditioning. The posterior over risk given evidence reduces to:

```
P(risk = r | ant=a, exc=e, spa=s, tre=t, agr=g)
  = CPT_risk[r, a, e, s, t, g]
```

This is simply a lookup into the Conditional Probability Table. The posterior over action then follows by marginalization:

```
P(act = j | evidence) = Σ_r  P(act=j | risk=r) · P(risk=r | evidence)
                       = Σ_r  CPT_action[j, r] · CPT_risk[r, evidence]
```

Which is a single matrix-vector multiplication: `P(act) = ActionCPT × P(risk|evidence)`.

### 4.3 State Spaces

| Node | States | Cardinality |
|------|--------|:-----------:|
| `antecedent_rainfall` | Dry, Normal, Wet, Very_Wet, Saturated | 5 |
| `exceedance_prob` | Very_Low, Low, Medium, High, Very_High | 5 |
| `spatial_coverage` | Localized, Moderate, Widespread | 3 |
| `rainfall_trend` | Decreasing, Stable, Increasing | 3 |
| `forecast_agreement` | Low, Medium, High | 3 |
| `risk_level` | Minimal, Low, Moderate, High, Extreme | 5 |
| `action` | Monitor, Alert, Prepare, Act | 4 |

Total CPT size for risk: 5 × (5 × 5 × 3 × 3 × 3) = 5 × 675 = 3,375 entries.
Without agreement: 5 × (5 × 5 × 3 × 3) = 5 × 225 = 1,125 entries.
Action CPT: 4 × 5 = 20 entries.

### 4.4 Discretization Thresholds

**Antecedent rainfall** (7-day accumulated mm):

| State | Range |
|-------|-------|
| Dry | < 10 mm |
| Normal | 10 – 30 mm |
| Wet | 30 – 60 mm |
| Very_Wet | 60 – 100 mm |
| Saturated | ≥ 100 mm |

**Exceedance probability** (0–1 scale):

| State | Range |
|-------|-------|
| Very_Low | < 0.20 |
| Low | 0.20 – 0.40 |
| Medium | 0.40 – 0.60 |
| High | 0.60 – 0.80 |
| Very_High | ≥ 0.80 |

**Spatial coverage** (fraction of boundary area with P_heavy ≥ 0.5):

| State | Range |
|-------|-------|
| Localized | < 0.30 |
| Moderate | 0.30 – 0.60 |
| Widespread | ≥ 0.60 |

**Rainfall trend** (7-day regression slope, mm/day):

| State | Condition |
|-------|-----------|
| Decreasing | slope < −2 mm/day |
| Stable | −2 ≤ slope ≤ +2 mm/day |
| Increasing | slope > +2 mm/day |

**Forecast agreement** (|P_ecmwf − P_gefs|):

| State | Condition |
|-------|-----------|
| Low | difference > 0.30 |
| Medium | 0.15 < difference ≤ 0.30 |
| High | difference ≤ 0.15 |

(Currently set to "Medium" by default since only ECMWF is available; the agreement node is skipped via `--no-agreement`.)

### 4.5 Risk CPT: Expert Rules

The CPT for `P(risk_level | parents)` is constructed by an expert-rule function that takes the 5 parent state indices and returns a 5-element probability vector. The logic has two layers:

**Layer 1 — Scenario-specific rules** (checked in priority order):

| Rule | Condition | Risk distribution |
|------|-----------|-------------------|
| R1 | Saturated + High/VeryHigh + Increasing + Moderate/Widespread | [0, 0, 0.05, 0.20, **0.75**] → Extreme |
| R1b | Saturated + High/VeryHigh + Increasing + Localized | [0, 0, 0.10, 0.40, **0.50**] → Extreme |
| R2 | VeryWet/Saturated + High+ + Increasing | [0, 0, 0.10, **0.50**, 0.40] → High |
| R3 | Dry + Low forecast (≤Medium) | [**0.55**, 0.35, 0.10, 0, 0] → Minimal |
| R4 | Decreasing + Low antecedent (≤Wet) + VeryLow/Low exceedance | [**0.65**, 0.30, 0.05, 0, 0] → Minimal |
| R5 | High forecast + Dry/Normal antecedent | [0.10, 0.25, **0.45**, 0.15, 0.05] → Moderate |

**Layer 2 — Base-risk fallback** (continuous score mapped to risk bands):

```
base_risk = antecedent_idx × 0.30 + exceedance_idx × 0.55
          + spatial_modifier + trend_modifier
```

Where `spatial_modifier` ∈ {0, 0.25, 0.5} and `trend_modifier` ∈ {−0.30, 0, +0.35}. Exceedance is weighted nearly twice as heavily as antecedent conditions — the design choice reflects that flood risk is driven more by what is coming (forecast) than by what has already fallen (antecedent), though saturated ground amplifies risk significantly.

The resulting base_risk is thresholded into bands:

| base_risk range | Dominant risk state |
|-----------------|---------------------|
| < 1 | Minimal (50%) / Low (40%) |
| 1 – 2 | Low (35%) / Moderate (40%) |
| 2 – 3 | Moderate (45%) / High (30%) |
| 3 – 4 | High (50%) / Extreme (20%) |
| ≥ 4 | Extreme (50%) / High (40%) |

**Layer 3 — Agreement modifier** (applied last):

If agreement is Low (high model disagreement), the probability vector is blended 50/50 with a uniform distribution [0.2, 0.2, 0.2, 0.2, 0.2], flattening extreme predictions. Medium agreement uses an 80/20 blend. High agreement passes the probabilities through unchanged.

This reflects the epistemic principle that when models disagree, confidence in any extreme prediction should be reduced.

### 4.6 Action CPT

The action CPT is a fixed 4×5 matrix `P(action | risk_level)`:

```
Risk →     Minimal   Low    Moderate   High   Extreme
─────────────────────────────────────────────────────
Monitor     0.95    0.15     0.00     0.00    0.00
Alert       0.05    0.80     0.20     0.05    0.00
Prepare     0.00    0.05     0.75     0.25    0.05
Act         0.00    0.00     0.05     0.70    0.95
```

This CPT encodes a clean diagonal structure: each risk level maps predominantly to one action, with small leakage into adjacent actions to reflect operational uncertainty. The key design choice is that "Moderate" risk leads to "Prepare" (not "Alert") — this aligns with humanitarian readiness protocols where moderate risk should trigger preparatory logistics, not merely an advisory.

---

## 5. Python BN Implementation

**Script**: `flood_bn_ibf_v1.py` (1,091 lines)  
**Library**: `pgmpy` (Probabilistic Graphical Models in Python)

### 5.1 Components

The Python implementation has three major classes:

#### `FloodDataLoaderV1`

Responsible for loading and spatially aggregating input data from local files (IMERG NetCDF + GEFS probability NetCDF). Uses `xarray` for gridded data operations and `regionmask` for zonal statistics. This loader predates the icechunk-based pipeline and works with downloaded files rather than cloud stores.

Key methods:
- `load_imerg_observations()` — Glob-finds IMERG daily files in a directory, concatenates along time.
- `compute_antecedent_stats()` — Sums precipitation over the observation window.
- `load_gefs_probabilities()` — Reads pre-computed exceedance probability NetCDFs.
- `extract_boundary_data()` — Zonal extraction for a single gridded field using regionmask.
- `prepare_analysis_data()` — Orchestrates all the above, returning a list of boundary data dicts.

#### `FloodBayesianNetworkV1`

The BN engine. Wraps a pgmpy `BayesianNetwork` object with edges, CPDs, and an inference routine.

Key methods:
- `__init__()` → `_setup_cpds()` — Constructs the DAG, creates `TabularCPD` objects for every node, adds them to the model, calls `model.check_model()` to verify row-normalization and parent-child consistency.
- `_create_risk_cpt()` — Iterates over all 675 (or 225) parent combinations, calling `_compute_risk_probs()` for each to fill the CPT column-by-column.
- `_compute_risk_probs()` — The expert-rule function (identical logic to Julia's `compute_risk_probs`).
- `_categorize()` — Maps continuous values or string labels to discrete state names for pgmpy evidence.
- `process_boundary()` — For a single boundary: builds evidence dict → creates `VariableElimination` instance → queries risk_level → queries action → returns result dict.
- `process_all_boundaries()` — Loops over all boundaries, calling `process_boundary()` each time.

#### `analyze_flood_risk()` (module-level function)

Top-level entry point that chains the data loader and BN, writes output CSV, and prints summary statistics.

### 5.2 pgmpy Inference: Variable Elimination

pgmpy implements **Variable Elimination (VE)**, an exact inference algorithm for Bayesian networks. For a query `P(X | evidence)`:

1. **Factor construction**: Start with the set of all CPD factors in the network.
2. **Evidence conditioning**: For each observed variable, slice the factor to keep only the observed state.
3. **Variable elimination**: For each hidden variable not in the query, multiply all factors containing it, then sum it out (marginalize). The elimination order affects efficiency but not correctness.
4. **Normalization**: Divide by the sum to get a proper distribution.

For this network, with all evidence observed, VE reduces to exactly the matrix computation described in §4.2 — but pgmpy does not optimize for this case. It still builds factor objects, handles string-keyed states, and creates a fresh `VariableElimination` object per query.

**Performance**: ~10–15 ms per boundary on CPython. For 227 boundaries, this is ~3 seconds — acceptable for batch runs, but 100–300× slower than the Julia direct path.

---

## 6. Julia BN Implementation

**Script**: `flood_bn_ibf_v1.jl` (675 lines)  
**Libraries**: `RxInfer.jl` (optional), `CSV.jl`, `DataFrames.jl`, `LinearAlgebra` (stdlib), `Printf` (stdlib)

### 6.1 Design Philosophy

The Julia implementation takes a dual-track approach:

1. **`infer_direct()`** — A pure matrix-lookup function. No graphical model object. Pre-builds the risk CPT as a `(5 × N_combos)` matrix once, then for each boundary, encodes the 5 parent indices into a single "super-parent" column index and reads the risk probability vector directly. Action probabilities follow from a single matrix-vector multiply `action_cpt * risk_probs`. This is the path used in the operational pipeline.

2. **`@model flood_bn_model`** — A declarative RxInfer.jl model definition using the `@model` macro. This defines the same BN as a factor graph for message-passing inference. It exists for future capabilities (learning, streaming, soft evidence) but is not required for the current fixed-CPT inference.

### 6.2 Type System

Julia uses concrete structs where Python uses dicts:

```julia
struct BoundaryInput
    id::String
    name::String
    country::String
    antecedent_rainfall_mm::Float64
    antecedent_category::String
    rainfall_trend::String
    gefs_eprob_heavy::Float64
    spatial_coverage::Float64
    forecast_agreement::String
end

struct BoundaryResult
    boundary_id::String
    boundary_name::String
    country::String
    antecedent_category::String
    rainfall_trend::String
    risk_level::String
    risk_probabilities::Vector{Float64}
    recommended_action::String
    action_probabilities::Vector{Float64}
    confidence::Float64
end
```

This has two advantages over the Python dict approach:
- **Compile-time type safety** — misspelled fields or wrong types are caught at struct construction, not at runtime deep inside inference.
- **Memory efficiency** — structs are stack-allocated when possible, and fields are tightly packed. A `Vector{BoundaryInput}` is a contiguous array of fixed-size records, vs. Python's list of heap-allocated dicts with string-keyed lookups.

### 6.3 Categorization Functions

Five standalone functions convert continuous values to 1-based state indices:

```julia
categorize_antecedent(rainfall_mm::Float64)::Int   # 1-5
categorize_exceedance(eprob::Float64)::Int          # 1-5
categorize_spatial(coverage::Float64)::Int           # 1-3
categorize_trend(trend::String)::Int                 # 1-3
categorize_agreement(agreement::String)::Int         # 1-3
```

Each uses simple threshold comparisons. The use of `Int` indices (vs. Python's string states) eliminates hash-map lookups during inference.

### 6.4 Super-Parent Encoding

Since `risk_level` has 5 parents, and RxInfer's `DiscreteTransition` takes a single parent, the Julia code introduces a deterministic **encoding function** that maps the 5 parent indices to a single flat index:

```julia
function encode_parents(ant::Int, exc::Int, spa::Int, tre::Int, agr::Int)::Int
    return ((agr - 1) * 3 * 3 * 5 * 5 +
            (tre - 1) * 3 * 5 * 5 +
            (spa - 1) * 5 * 5 +
            (exc - 1) * 5 +
            (ant - 1)) + 1
end
```

The iteration order in `build_risk_cpt()` must match this encoding exactly: agreement (outermost) → trend → spatial → exceedance → antecedent (innermost). This is the row-major flattening convention.

For the `include_agreement=false` case, a simpler 4-parent encoder is used (`encode_parents_no_agreement`), and the CPT shrinks from 675 to 225 columns.

### 6.5 CPT Construction

`build_risk_cpt()` fills a `(5 × N_combos)` Float64 matrix by iterating over all parent combinations and calling `compute_risk_probs()` for each. The matrix is built once and reused for all 227 boundaries — this is the key performance optimization over pgmpy, which rebuilds factor objects per query.

`build_action_cpt()` returns a static `(4 × 5)` Float64 matrix.

### 6.6 Direct Inference

```julia
function infer_direct(ant_idx, exc_idx, spa_idx, tre_idx, agr_idx,
                      risk_cpt, action_cpt; include_agreement=true)
    parent_idx = encode_parents(...)   # → single Int
    risk_probs = risk_cpt[:, parent_idx]   # → Vector{Float64}, length 5
    action_probs = action_cpt * risk_probs  # → Vector{Float64}, length 4
    return risk_probs, action_probs
end
```

This is the **entire inference routine** — one column lookup and one matrix-vector multiply. No factor graphs, no message schedules, no object creation. Compiled Julia executes this in under 50 nanoseconds.

### 6.7 CSV CLI Extension (added for this pipeline)

The `run_csv()` function reads a CSV file produced by `flood_data_prep.py`, constructs `Vector{BoundaryInput}` from the rows, calls `process_all_boundaries(; include_agreement=false)`, and writes a result CSV with the full probability vectors (5 risk + 4 action columns plus labels and confidence).

CLI usage:
```bash
julia --project=. flood_bn_ibf_v1.jl \
    --input-csv bn_inputs/flood_inputs_2026-03-01.csv \
    --output-csv output/flood_bn_v1_2026-03-01.csv \
    --no-agreement
```

### 6.8 RxInfer Optional Loading

RxInfer.jl is a heavy dependency (~200+ packages in its dependency tree). Since the operational pipeline only uses `infer_direct()`, RxInfer is loaded conditionally:

```julia
const HAS_RXINFER = try
    @eval using RxInfer
    true
catch
    false
end
```

The `@model` definition and `infer_rxinfer()` function are only compiled if RxInfer is available. This means the Julia BN can run with just stdlib + CSV + DataFrames — a ~4 second startup vs. ~30 seconds with RxInfer.

---

## 7. RxInfer.jl and Message Passing Inference

### 7.1 What is RxInfer.jl?

RxInfer.jl is a Julia package for **reactive Bayesian inference** through message passing on factor graphs. It was developed at the BIASlab group at TU Eindhoven and differs fundamentally from library-level BN tools like pgmpy:

- **pgmpy** provides a collection of algorithms (VE, BP, sampling) that you call explicitly on a model object.
- **RxInfer.jl** compiles a probabilistic model specification into a **factor graph** — a bipartite graph of variable nodes and factor nodes — and then schedules **messages** to flow along edges until convergence.

### 7.2 Factor Graphs

A factor graph is the computational backbone of message-passing inference. For the flood BN:

```
Variable nodes: ant, exc, spa, tre, agr, risk, act, parent_combo
Factor nodes:   f_prior(parent_combo), f_risk(risk | parent_combo), f_action(act | risk)
```

In the super-parent encoding, the 5 evidence nodes are collapsed into a single `parent_combo` variable node, simplifying the factor graph to:

```
[parent_combo] ──── f_risk ──── [risk] ──── f_action ──── [action]
```

Factor `f_risk` holds the `(5 × 675)` CPT matrix. Factor `f_action` holds the `(4 × 5)` CPT matrix.

### 7.3 The `@model` Macro

RxInfer's `@model` macro is a domain-specific language for defining probabilistic models:

```julia
@model function flood_bn_model(; risk_cpt_matrix, action_cpt_matrix, n_parent_combos)
    parent_combo ~ Categorical(fill(1.0 / n_parent_combos, n_parent_combos))
    risk_level ~ DiscreteTransition(parent_combo, risk_cpt_matrix)
    action ~ DiscreteTransition(risk_level, action_cpt_matrix)
end
```

This reads as:
- `parent_combo` is drawn from a uniform Categorical prior over all 675 (or 225) states.
- `risk_level` is drawn from `Categorical(risk_cpt_matrix * parent_combo_onehot)` — the `DiscreteTransition` node computes `T × x` where `T` is the CPT matrix and `x` is a one-hot encoding of the parent.
- `action` is similarly drawn from `Categorical(action_cpt_matrix * risk_level_onehot)`.

When evidence is provided (clamping `parent_combo` to a specific one-hot vector), the `infer()` function runs message passing to compute the posterior over `risk_level` and `action`.

### 7.4 Message Passing: How It Works

In belief propagation on a factor graph, two types of messages flow:

1. **Variable → Factor messages** (`μ`): A variable node sends to each adjacent factor the product of all incoming messages *except* the one from that factor. For observed variables, this message is a delta (one-hot).

2. **Factor → Variable messages** (`ν`): A factor node sends to each adjacent variable a summary of the factor function marginalized over all other connected variables, weighted by their incoming messages.

For the flood BN factor graph (which is a chain, hence a tree), a single forward-backward pass yields exact posteriors:

**Forward pass** (left → right):
```
μ(parent_combo → f_risk) = δ(parent_combo = observed_index)   [one-hot evidence]
ν(f_risk → risk) = Σ_{pc} CPT_risk[:, pc] · μ(pc)            [= CPT column lookup]
                 = risk_cpt[:, observed_index]                  [exact same as infer_direct]
```

**Backward pass** (right → left, only needed if querying parent posteriors — not needed here):
```
μ(action → f_action) = prior_on_action   [or uniform if unobserved]
ν(f_action → risk) = ...                 [not needed for our query]
```

**Marginal computation**:
```
P(risk | evidence) = ν(f_risk → risk) · μ(risk → f_risk_backward)
```

Since there is no backward evidence on `risk` (it's unobserved), the backward message is uniform, and the marginal equals the forward message exactly — which is the CPT column. The action posterior is then:

```
P(action | evidence) = action_cpt × P(risk | evidence)
```

**This is mathematically identical to `infer_direct()`**. The difference is architectural: RxInfer builds a compiled graph structure that can be extended with soft evidence, learning nodes, and streaming updates, while `infer_direct()` is a hardcoded matrix lookup.

### 7.5 When Message Passing Diverges from Direct Inference

The equivalence holds because:
1. The graph is a tree (no cycles) → belief propagation is exact.
2. All evidence is hard (one-hot) → no iterative convergence needed.
3. CPTs are fixed → no parameter updates.

If any of these conditions change, message passing provides capabilities that direct inference cannot:

| Scenario | Direct inference | Message passing |
|----------|-----------------|-----------------|
| Soft evidence (uncertain observations) | Manual mixture computation | Native — pass probability vector instead of one-hot |
| Loopy graphs (cycles, e.g. spatial neighbors) | Not applicable | Loopy BP iterates until convergence |
| Learning CPT parameters from data | Not applicable | Add Dirichlet priors, run variational message passing |
| Streaming / online updates | Rebuild and re-query | Incremental message updates |
| Missing evidence (partial observation) | Sum over missing variable manually | Automatic via uninformative prior message |

### 7.6 Performance Characteristics

| Metric | `infer_direct()` | RxInfer `infer()` | pgmpy VE |
|--------|:-----------------:|:------------------:|:--------:|
| Setup (first call) | ~100 μs (CPT build) | ~5 ms (graph compilation) | ~50 ms (model + VE creation) |
| Per-boundary inference | ~50 ns | ~500 μs | ~10 ms |
| 227 boundaries total | ~0.01 ms | ~110 ms | ~2,300 ms |
| Memory per query | 0 allocations | Factor graph objects | Factor objects + elimination tree |

For the operational pipeline, `infer_direct()` is the clear choice. RxInfer becomes worthwhile when the model evolves beyond fixed expert CPTs.

---

## 8. Inference Methods Compared

### 8.1 Variable Elimination (Python/pgmpy)

**Algorithm**: Eliminates hidden variables one at a time by multiplying and marginalizing factors. For this tree-structured BN with all evidence observed, the elimination order is trivial (eliminate `risk_level` to get `action`, or query `risk_level` directly from the conditioned CPT).

**Strengths**: Exact. Well-understood. Handles arbitrary DAG structures.

**Weaknesses**: Creates intermediate factor objects per query. Python overhead dominates for small models. No incremental updates — every query starts from scratch.

### 8.2 Belief Propagation / Message Passing (Julia/RxInfer)

**Algorithm**: Messages flow along factor graph edges. For trees, a single pass yields exact marginals. For loopy graphs, iterate until convergence (approximate).

**Strengths**: Natural fit for streaming data. Supports soft evidence natively. Extensible to parameter learning (variational message passing). Compiled factor graph is reusable.

**Weaknesses**: Overkill for fixed-CPT inference on a 2-layer tree. Graph compilation has startup cost. Approximate for loopy graphs.

### 8.3 Direct Matrix Lookup (Julia/`infer_direct`)

**Algorithm**: Encode parent states → index CPT column → matrix multiply for action. No graph object, no message schedule, no factor objects.

**Strengths**: Fastest possible. Zero allocation per query. Trivially parallelizable. Easy to verify (it's just array indexing).

**Weaknesses**: Only works for tree-structured BNs with all evidence observed and fixed CPTs. Cannot do learning, soft evidence, or streaming. Any model change requires rewriting the encoding function.

### 8.4 When to Use Each

| Scenario | Recommended method |
|----------|--------------------|
| Operational daily batch (current) | Direct matrix (`infer_direct`) |
| Rapid prototyping / model exploration | pgmpy Variable Elimination |
| Learning CPTs from historical flood data | RxInfer with Dirichlet priors |
| Real-time dashboard with streaming data | RxInfer streaming inference |
| Uncertainty quantification on inputs | RxInfer with soft evidence |
| Spatial model (neighbor interactions) | RxInfer loopy BP |

---

## 9. Orchestration and Summarization

### 9.1 `run_flood_bn_range.sh`

A bash driver that loops over a date range, running the data-prep and Julia BN for each day:

```bash
./run_flood_bn_range.sh 2026-03-01 2026-03-10
```

For each date D:
1. Calls `flood_data_prep.py` via `uv run` with inline deps → writes `bn_inputs/flood_inputs_D.csv`
2. Calls `julia --project=. flood_bn_ibf_v1.jl --input-csv ... --output-csv ... --no-agreement` → writes `output/flood_bn_v1_D.csv`

Environment variable `RP_YEARS` controls the return period (default 2).

### 9.2 `summarize_bn.py`

Stacks the 10 daily result CSVs into a single long dataframe, then aggregates per admin-1:

| Output column | Computation |
|---------------|-------------|
| `max_risk_level` | Highest risk seen across 10 days |
| `day_of_max` | First date that risk level was reached |
| `action_at_peak` | Recommended action on `day_of_max` |
| `mean_confidence` | Average action probability across all 10 days |
| `days_at_alert_or_worse` | Count of days with risk ≥ Low |
| `days_at_moderate_or_worse` | Count of days with risk ≥ Moderate |

---

## 10. Visualization

### `plot_daily_risk_maps.py`

Joins each daily BN result CSV to the admin-1 GeoDataFrame and renders choropleth maps with a 5-level discrete colormap:

| Risk level | Color | Hex |
|------------|-------|-----|
| Minimal | Green | `#1a9850` |
| Low | Pale yellow-green | `#d9ef8b` |
| Moderate | Amber | `#fee08b` |
| High | Orange-red | `#f46d43` |
| Extreme | Dark red | `#a50026` |

Produces:
- **Combined panel** (`output/flood_bn_v1_risk_maps_panel.png`) — 2×5 grid showing all 10 days side by side for temporal comparison.
- **Per-day maps** (`output/maps/flood_bn_v1_risk_YYYY-MM-DD.png`) — standalone PNGs with legend.

Boundaries with missing data render as light grey.

---

## 11. End-to-End Walkthrough: March 1–10, 2026

### Context

March 2026 is the onset of the long rains (Masika) in East Africa. Antecedent conditions vary: equatorial regions (Kenya Highlands, Lake Victoria basin, coastal Tanzania) have moderate accumulated rainfall from the preceding weeks, while the Horn of Africa (Somalia, Djibouti, northern Kenya) remains dry.

### Pipeline execution

1. `run_flood_bn_range.sh 2026-03-01 2026-03-10` ran all 10 days.
2. For each day, the data prep read ~336 IMERG half-hourly time steps (7 days × 48) and 7 ECMWF lead-time slices across 51 ensemble members.
3. Julia BN inference processed 227 boundaries in under 100 ms per day.
4. Total wall time: ~10 minutes (dominated by IMERG icechunk I/O — each day reads ~6.5 MB across the network).

### Results summary

| Metric | Value |
|--------|-------|
| Total boundary-days | 2,270 |
| Boundaries with Moderate peak | 64 (28%) |
| Boundaries always Minimal | 163 (72%) |
| No boundaries at High or Extreme | 0 |

**Geographic distribution of Moderate-peak boundaries**:
- Kenya: 31 (concentrated in Western Highlands, Rift Valley, Coastal)
- Tanzania: 18 (Arusha, Kilimanjaro, Mbeya, Lake Zone)
- Uganda: 8 (Eastern Uganda — Kapchorwa, Sironko, Moroto)
- Burundi: 6
- Rwanda: 1

**Temporal pattern**: Risk peaks were front-loaded. March 1–2 accounted for 36 of the 64 Moderate peaks, driven by higher antecedent rainfall from February and modest (but non-zero) exceedance probabilities. By March 7–10, drying antecedent conditions reduced most boundaries back to Minimal.

**Interpretation**: The 2-year return-period threshold is sensitive — it flags a broad swath of equatorial East Africa as Moderate during the rains onset. At a 5-year RP, the Moderate count would shrink to roughly 10–15 boundaries (those with the highest ensemble exceedance). The absence of High/Extreme risk reflects the modest exceedance probabilities (max P_heavy = 0.294 on March 1, in Bujumbura Rural) — the 51-member ECMWF ensemble did not produce a strong convergence toward heavy precipitation in this period.

---

## 12. Future Directions

### 12.1 Multi-Model Agreement

Adding GEFS (31 members) to the forecast pipeline would activate the `forecast_agreement` node. Agreement between ECMWF and GEFS narrows or widens the risk distribution: when both models agree on heavy precipitation, extreme outcomes become more probable; when they disagree, the BN pulls predictions toward the center.

### 12.2 Learning CPTs from Data

The expert-elicited CPTs encode domain knowledge but have no empirical calibration. By replacing fixed CPT columns with Dirichlet priors in RxInfer, the system could learn from historical flood events (EMDAT, FloodList, ICPAC flood reports) while retaining the expert priors as informative starting points. This is the highest-value use of the RxInfer architecture.

### 12.3 Soft Evidence

IMERG observations have pixel-level uncertainty. At category boundaries (e.g., 59 mm — is it Wet or Very_Wet?), hard discretization discards information. Soft evidence would propagate this uncertainty through the BN, producing wider risk distributions near thresholds.

### 12.4 Spatial Smoothing

Currently, each boundary is processed independently. A hierarchical or Markov Random Field extension could share information between neighbors — a flood warning in Bungoma should influence the prior for adjacent Busia, since rainfall systems are spatially coherent.

### 12.5 AI-Enhanced Forecasts

The workflow document mentions a future cGAN-downscaled precipitation forecast at 0.1° resolution. This would provide a third forecast source, further strengthening the multi-model agreement signal and improving spatial detail for the exceedance computation.

### 12.6 Operationalization

The current pipeline runs retrospectively (hindcast for March 2026). Operational deployment would require:
- A cron-triggered runner executing daily at ~06:00 UTC (after ECMWF 00Z data lands).
- IMERG Early Run instead of Final (lower latency: ~4 h vs. ~3 months).
- Integration with ICPAC's dissemination platform (maps, CSV feeds, API).
- Forecast verification loop: comparing predicted risk levels against observed flood impacts to continuously calibrate the BN.

---

*This document was generated from the operational pipeline files as of 12 April 2026. All code referenced is in `/scratch/notebook/bn-ibf/flood_ibf/`.*
