# Drought IBF — Bayesian Network parallel of `flood_ibf/`

Monthly impact-based forecasting for **drought** at admin-1 across East
Africa, mirroring the architecture of `bn-ibf/flood_ibf/`.

## Pipeline

```
ERA5 SPI3 obs (zarr)        ┐
SEAS5 SPI3 forecast (zarr)  ├──> drought_data_prep.py ──> drought_inputs_YYYY-MM-01.csv
ERA5 SPI return-period (icechunk) ┤
ICPAC adm1 GeoJSON          ┘                                │
                                                             ▼
                                drought_bn_ibf_v1.py ──> drought_bn_v1_YYYY-MM-01.csv
                                                            (227 rows × ~15 cols)
```

**Source.coop locations** (all anonymous-readable):

| Role | Path | Format | Cadence |
|---|---|---|---|
| SPI obs | `e4drr-project/observations/era5_ecmwf_pencil` | plain Zarr | monthly, 1940-now |
| SPI3 forecast | `e4drr-project/forecasts/seas51_spi3_10km_pencil_zarr` | plain Zarr | 6-month, 51-member |
| RP thresholds | `e4drr-project/observations/era5_ecmwf_rp_icechunk` | icechunk 2.0.3 | static |

The forecast comes from this repo's parent pipeline (`download_seas51_tp_to_icechunk.py process` + `seas51_spi3_pencil_zarr.py`); see  
`/scratch/notebook/bn-ibf/drought_crma/SEAS51_ICECHUNK_README.md`.

## How it differs from flood IBF

| | **flood_ibf** | **drought_ibf** |
|---|---|---|
| Cadence | Daily | **Monthly** |
| Obs | IMERG half-hourly → 7-day antecedent rain (mm) | ERA5 SPI3 monthly → most-recent month |
| Forecast | ECMWF TP, 51 ensemble × 7-day, 7 durations (3h…7d) | SEAS5 SPI3, 51 ensemble × 6 lead months |
| RP | CMORPH NetCDF, 7 durations × 5 RPs (mm) | ERA5 fitted-normal SPI thresholds (per-pixel SPI) |
| Threshold | TP ≥ RP (heavy rain) | SPI ≤ RP (severe drought) |
| Tail | `p95(ens_max / RP)` per boundary | `p5(ens_min SPI)` per boundary |
| Trend | 7-day slope, mm/day; ±2 mm/day band | last-N-month slope, SPI/month; ±0.1 band |

The BN topology is identical (5 parents → `risk_level` → `action`), so
`drought_bn_ibf_v1.py` is structurally a renamed `flood_bn_ibf_v1.py` with:
- **Renamed parent nodes**: `antecedent_rainfall`→`current_spi3`,
  `exceedance_prob`→`deficit_prob`, `rainfall_trend`→`spi3_trend`.
- **Different bin edges and parent-state labels** (e.g.
  `Severe_Drought…Above_Normal` instead of `Dry…Saturated`).
- **Different CPT logic** in `compute_drought_risk_probs` — drought stress
  weights, calibrated against ECMWF SPI return-period semantics
  (3-yr ≈ -0.43, 5-yr ≈ -0.84, 10-yr ≈ -1.28, 20-yr ≈ -1.64, 50-yr ≈ -2.05).

The CRMA decision rule, soft-evidence Gaussian binning, and inference
engines (`pgmpy` + direct tensor contraction) are unchanged.

## BN structure

```
current_spi3      ──┐
deficit_prob      ──┤
spatial_coverage  ──┼──► risk_level ──► action
spi3_trend        ──┤
tail_risk         ──┘
```

| Node | States (size) | Bin cutoffs / source |
|---|---|---|
| `current_spi3` | Severe_Drought / Moderate_Drought / Mild_Drought / Normal / Above_Normal (5) | SPI obs at -1.5, -1.0, -0.5, 0.5 |
| `deficit_prob` | Very_Low … Very_High (5) | P(SPI ≤ -1.0) across leads, bins 0.2/0.4/0.6/0.8 |
| `spatial_coverage` | Localized / Moderate / Widespread (3) | max(deficit-mask fraction, hotspot fraction), bins 0.3/0.6 |
| `spi3_trend` | Deteriorating / Stable / Improving (3) | last-6-month SPI slope, ±0.1 SPI/month |
| `tail_risk` | High / Moderate / Low / Nil (4) | p5(ens_min SPI) per boundary at -1.5/-1.0/-0.5 |
| `risk_level` | Minimal … Extreme (5) | hidden child |
| `action` | Monitor / Alert / Prepare / Act (4) | derived from risk |

## Usage

### 1. Hard evidence (categorical)

```bash
# Adm1 GeoJSON: needs to be on disk (e.g. icpac_adm1v3.geojson with GID_1, NAME_1)
uv run drought_data_prep.py \
    --date 2026-04 \
    --rp-years 5 \
    --adm1 icpac_adm1v3.geojson \
    --out bn_inputs/drought_inputs_2026-04.csv

uv run drought_bn_ibf_v1.py \
    --input  bn_inputs/drought_inputs_2026-04.csv \
    --output output/drought_bn_v1_2026-04.csv
```

### 2. Soft evidence (Gaussian-binned probability vectors)

```bash
uv run drought_data_prep.py \
    --date 2026-04 --rp-years 5 \
    --adm1 icpac_adm1v3.geojson \
    --out bn_inputs/drought_inputs_2026-04.csv \
    --soft-evidence

uv run drought_bn_ibf_v1.py \
    --input  bn_inputs/drought_inputs_2026-04.csv \
    --output output/drought_bn_v1_2026-04.csv \
    --soft-evidence
```

Soft evidence emits 5+5+3+3+4 = 20 columns (`{cur,def,spa,trn,tail}_p[1..K]`)
which the BN consumes via direct tensor contraction (no pgmpy required).

### 3. Per-member sidecar (storyline runs)

```bash
uv run drought_data_prep.py --date 2026-04 \
    --adm1 icpac_adm1v3.geojson \
    --out bn_inputs/drought_inputs_2026-04.csv \
    --member-evidence-sidecar bn_inputs/drought_member_evidence_2026-04.csv
```

Produces `n_boundaries × 51` rows for per-member BN sweeps (mirrors
`flood_data_prep.py --member-evidence-sidecar`).

## Tuning knobs (where the drought / flood logic differ most)

- **`--rp-years`** (3, 5, 10, 20, 50). Default 5. Mild = 3, Moderate = 5,
  Severe = 10, Extreme = 20, Exceptional = 50 yr (per
  `ibf-thresholds-triggers/thresholds/ecmwf_spi/ecmwf_spi_return_periods.py`).
- **`--deficit-spi`** (default -1.0). The SPI cutoff that defines a
  "deficit" event for the deficit-prob parent. -1.0 = McKee moderate
  drought; -1.5 = severe; -2.0 = extreme.
- **`--trend-band`** (default 0.1 SPI/month). Slope ±band that maps to
  "Stable"; outside that, "Improving" / "Deteriorating".
- **`--rp-prefer`** (`fitted` | `empirical` | `standard`). The default
  `fitted` is per-pixel; `standard` uses the theoretical Φ⁻¹(1/T)
  which is constant in space.
- **`--gamma`** in `drought_bn_ibf_v1.py` (default 0.20). CRMA cost-loss
  threshold; same semantics as the flood pipeline.

The CPT weights in `WEIGHTS` (`drought_bn_ibf_v1.py`) and the per-state
stress scores in `_state_score` are the main domain levers — they are
intentionally exposed as plain dicts so a domain expert can tune them
without touching the pgmpy code path.

## SPI clipping convention

Standard SPI analysis caps values at **±4** before downstream use. The gamma
fit underlying SPI can produce ill-conditioned tails at arid pixels (very
many zero-precip months → small effective sample for the lower tail), which
inflates the fitted CDF inverse and emits SPI values like ±6 or ±8 that are
not physically interpretable. Capping at ±4 keeps the index in the meaningful
range without losing real-drought signal (a -4 SPI is already a 1-in-30k
event).

`drought_data_prep.py` applies this clip by default (both obs and forecast):

```bash
--clip-spi 4.0   # default; pass 0 to disable
```

A spot-check confirmed the upstream
`e4drr-project/observations/era5_ecmwf_pencil` SPI3 has ~9 % of cells outside
|SPI|>3 at 2026-01 and a 1991-2020 climatology std of 1.43 (not 1.0),
i.e. the gamma fit is leaky in arid regions. Clipping at ±4 mitigates that
without re-running the upstream SPI calculation.

## Caveats / TODO

- **Lead handling**. The current data prep collapses across leads
  (max-prob deficit, ens-min over all leads). For multi-lead storylines
  (month-1 vs month-3 vs month-6), the prep would emit one CSV per lead
  or extend the schema with `*_lead{1..6}_*` blocks.
- **Calibration**. Bin cutoffs and `WEIGHTS` are uncalibrated initial
  values mirroring flood semantics. They should be reviewed with a
  drought specialist (ICPAC team) and re-tuned against historical
  events (e.g. 2010-2011, 2016-2017, 2020-2022 Horn of Africa droughts).
- **DBN temporal coupling**. The flood pipeline blends the previous day's
  posterior into today's evidence (`α=0.6`); the drought analogue would
  blend month-to-month. Not implemented in v1.
- **Julia port**. `flood_bn_ibf_v1.jl` (RxInfer.jl) has a few features not
  in `drought_bn_ibf_v1.py` (DBN, per-member storylines). The Python
  reference here is sufficient for evaluation; a Julia port can follow
  once the calibration is settled.
- **Observation lag**. ERA5 SPI3 obs lags by ~1-2 months; for a target
  of 2026-04, the most recent obs may be 2026-01 or 2026-02. The script
  uses the latest available month ≤ target as `current_spi3`.

## File map

```
drought_data_prep.py          # CSV builder (drought analogue of flood_data_prep.py)
drought_bn_ibf_v1.py          # BN + CRMA decision (Python; pgmpy or tensor-contract)
README.md                     # this file
```

## See also

- `bn-ibf/flood_ibf/README.md` — flood IBF reference; identical BN topology.
- `bn-ibf/flood_ibf/flood_bn_ibf_v1.jl` — Julia/RxInfer reference for DBN
  temporal coupling and per-member storylines (drought port pending).
- `ibf-thresholds-triggers/thresholds/ecmwf_spi/ecmwf_spi_return_periods.py` —
  source for the RP icechunk store (SPI thresholds).
- `bn-ibf/drought_crma/SEAS51_ICECHUNK_README.md` — upstream pipeline that
  publishes the SPI3 forecast pencil zarr.
