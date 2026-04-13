# Flood IBF — Bayesian Network for East Africa

Daily admin-1 flood risk assessment for the ICPAC domain (11 East African
countries, 227 admin-1 regions). Combines IMERG satellite rainfall
observations with ECMWF ensemble forecasts through an expert-elicited
Bayesian Network to produce per-boundary risk levels and recommended
humanitarian actions.

Operational run for 2026-03-01 → 2026-03-10 successfully flagged the
Nairobi River flash-flood event of 6–7 March 2026 with a 2-day lead time
(Moderate / Prepare on Mar 4, Moderate / Alert on Mar 6).

---

## Method at a glance

```
    IMERG HH icechunk          ECMWF TP icechunk            CMORPH return
  (observations, 0.1°)      (51-member ens, 0.25°)        periods (0.073°)
          │                          │                           │
          ▼                          ▼                           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      flood_data_prep.py                          │
    │  • 7-day antecedent mm + trend slope  (IMERG → admin-1)          │
    │  • 7 duration accumulations (3h…7d)  (ECMWF → admin-1)           │
    │  • P(any member ≥ 2yr RP) per pixel-duration  (eprob grid)       │
    │  • P_heavy = max over durations                                  │
    │  • ens_max_ratio = p95 of (ens_max / threshold) per boundary     │
    │  • hotspot_fraction = pixels where any member exceeds threshold  │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ flood_inputs_YYYY-MM-DD.csv
    ┌─────────────────────────────────────────────────────────────────┐
    │                    flood_bn_ibf_v1.jl                            │
    │                                                                  │
    │  Evidence nodes (6):                                             │
    │    antecedent (Dry…Saturated)    ─────┐                          │
    │    exceedance_prob               ─────┤                          │
    │    spatial_coverage              ─────┼─▶ risk_level ─▶ action   │
    │    rainfall_trend                ─────┤                          │
    │    forecast_agreement (optional) ─────┤                          │
    │    tail_risk (ens-max signal)    ─────┘                          │
    │                                                                  │
    │  Expert rules + base-risk score + agreement blend → risk CPT     │
    │  Direct matrix lookup (~50ns / boundary)                         │
    └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ flood_bn_v1_YYYY-MM-DD.csv
                         Summary, maps, per-day DAGs
```

### Key design choices

| Decision | Value | Rationale |
|----------|-------|-----------|
| Target dates | D ∈ {Mar 1…Mar 10 2026} | 10-day hindcast window |
| Observation window | `[D-7, D)` IMERG daily totals | Antecedent moisture (7d) |
| Forecast init | `init_date = D 00Z` | Latest init available for D |
| Durations evaluated | 3h, 6h, 12h, 24h, 48h, 72h, 7d | Matches CMORPH RP durations |
| Return period threshold | **2-year** (pixel-wise from CMORPH) | Sensitive IBF trigger |
| Tail-risk aggregation | **pixel 95th-percentile** of max_ratio | Preserves localized hotspots |
| Spatial coverage | `max(P_heavy≥0.5 mask, hotspot_fraction)` | Captures sub-boundary hot pixels |
| Trend threshold | ±2 mm/day on 7-day regression slope | Decreasing / Stable / Increasing |
| Agreement node | Disabled (`--no-agreement`) | ECMWF-only, no GEFS available |

---

## Pipeline scripts

All Python scripts use the `uv run` shebang — dependencies install
transiently at first execution (no persistent conda env needed).

### Data pipeline

| Script | Role |
|--------|------|
| `flood_data_prep.py` | Per-day data preparation. Reads IMERG + ECMWF icechunk stores and CMORPH RP NetCDF; writes `bn_inputs/flood_inputs_YYYY-MM-DD.csv` (227 rows, 17 columns including `ens_max_ratio`, `hotspot_fraction`, `ens_min/mean/max_24h_mm`). |
| `flood_bn_ibf_v1.jl` | Julia BN inference engine. Reads prep CSV, runs direct-matrix inference, writes `output/flood_bn_v1_YYYY-MM-DD.csv` with risk/action probability vectors. CLI: `--input-csv IN --output-csv OUT [--no-agreement] [--tail-risk]`. |
| `flood_bn_ibf_v1.py` | Python (pgmpy) reference implementation of the same BN. Used for cross-validation; not on the operational path. |
| `run_flood_bn_range.sh` | Driver: loops over `[START, END]`, runs prep → Julia BN each day. Usage: `./run_flood_bn_range.sh 2026-03-01 2026-03-10`. `RP_YEARS` env var overrides the default 2-yr threshold. |
| `summarize_bn.py` | Stacks the 10 daily result CSVs into a per-boundary summary (`max_risk_level`, `day_of_max`, `mean_confidence`, `days_at_alert_or_worse`). |

### Plot routines

| Script | Output | Purpose |
|--------|--------|---------|
| `plot_daily_risk_maps.py` | `output/flood_bn_v1_risk_maps_panel.png` (2×5 panel) and `output/maps/flood_bn_v1_risk_YYYY-MM-DD.png` (10 standalone) | Admin-1 choropleth maps of the daily BN risk level using a 5-step discrete colormap (Minimal → Extreme). |
| `plot_nairobi_diagnostic.py` | `output/nairobi_diagnostic.png` | 4-panel timeline for a single boundary (default Nairobi): antecedent moisture, ECMWF ensemble spread vs CMORPH 2-yr threshold, tail-risk ratio, and BN risk/action per day. The reported flood window (Mar 6–7) is shaded red. |
| `plot_bn_dag_per_day.py` | `output/bn_dags/bn_dag_<boundary>_<date>.png` | Per-day per-boundary DAG visualization showing every evidence node with its raw value + discretized state, the risk_level posterior (5-bar), and the action posterior (4-bar). Useful for diagnosing why a given risk call was made. |
| `flood_bn_dag.drawio` | — | Editable draw.io schematic of the BN structure (8 nodes, state cardinalities, edge weights, legend). Open in [app.diagrams.net](https://app.diagrams.net). |

### Documentation files

| File | Contents |
|------|----------|
| `README.md` | This file. |
| `flood_bn_ibf_system_v20260412.md` | Deep technical doc (~850 lines): BN math, CPT construction, expert rules, variable elimination vs message passing vs direct matrix, factor graphs, RxInfer.jl mechanics, end-to-end walkthrough. |
| `flood_bn_ibf_v1_workflow.md` | Original design document with data source definitions, threshold tables, and the `FloodDataLoaderV1` / `FloodBayesianNetworkV1` class structure. |
| `python_julia_bn_comparison.md` | Detailed comparison of the Python (pgmpy) and Julia (RxInfer) implementations — what is identical, what differs, and where Julia enables capabilities Python cannot (streaming, parameter learning, soft evidence, hierarchical models). |

---

## Data sources

Only local files listed here; the two icechunk stores are read directly
from `s3://e4drr-project/*` on source.coop via anonymous access.

| Source | Path / URL | Used for |
|--------|-----------|----------|
| IMERG half-hourly | `s3://e4drr-project/observations/imerg_hh_icechunk` | Antecedent 7-day rainfall, trend |
| ECMWF IFS ensemble TP | `s3://e4drr-project/forecasts/ecmwf_ea_tp_icechunk` | Forecast exceedance & tail risk |
| CMORPH return periods | `cmorph_ea_return_periods.nc` | Pixel-wise 2-yr RP thresholds |
| Admin-1 boundaries | `icpac_adm1v3.geojson` | 227 admin-1 polygons |

---

## Bayesian Network structure

8 discrete nodes organized as 6 evidence → 1 hidden (risk_level) → 1
query (action):

| Node | States | Role | Source |
|------|:------:|------|--------|
| `antecedent_rainfall` | Dry, Normal, Wet, Very_Wet, Saturated (5) | Evidence | IMERG 7-day sum |
| `exceedance_prob` | Very_Low, Low, Medium, High, Very_High (5) | Evidence | ECMWF P_heavy (ensemble mean) |
| `spatial_coverage` | Localized, Moderate, Widespread (3) | Evidence | `max(P_heavy≥0.5 mask, hotspot_fraction)` |
| `rainfall_trend` | Decreasing, Stable, Increasing (3) | Evidence | 7-day IMERG slope ±2 mm/day |
| `forecast_agreement` | Low, Medium, High (3) | Optional | ECMWF-GEFS difference (disabled) |
| `tail_risk` | None, Low, Moderate, High (4) | Evidence | p95 of (ens_max / 2-yr RP) per pixel |
| `risk_level` | Minimal, Low, Moderate, High, Extreme (5) | Hidden | Inferred via CPT lookup |
| `action` | Monitor, Alert, Prepare, Act (4) | Query | `action_cpt × P(risk)` |

**Inference**: direct matrix lookup on a pre-built CPT (5 × 900 matrix
with tail_risk enabled). One column indexed per boundary, one matrix-vector
multiply for action. Sub-millisecond for 227 boundaries.

**Why tail_risk matters**: the mean exceedance probability `P_heavy`
under-represents cases where only 1–2 of 51 ensemble members produce
extreme rainfall — this rounds to ~2–4% and discretizes to "Very_Low",
missing the tail signal entirely. The `tail_risk` node captures the ratio
of the ensemble maximum to the threshold; when any single
member-pixel combination hits ≥ the 2-yr threshold, the node fires
Moderate or High and directly elevates the risk posterior.

**Why pixel p95 matters**: the boundary-mean of `ens_max_ratio` smooths
over localized hotspots. For a small admin region (e.g. Nairobi,
~5 ECMWF pixels), a single hot pixel gets diluted 5×. Using the 95th
percentile of pixel-level ratios preserves the worst-case signal while
being robust to single outlier pixels.

---

## Running the pipeline

### One-shot for the Mar 1–10 window
```bash
./run_flood_bn_range.sh 2026-03-01 2026-03-10
./summarize_bn.py
./plot_daily_risk_maps.py
./plot_nairobi_diagnostic.py
./plot_bn_dag_per_day.py --boundary Nairobi --start 2026-03-01 --end 2026-03-10
```

### Single day, single boundary diagnostic
```bash
./flood_data_prep.py --date 2026-03-06 --out bn_inputs/flood_inputs_2026-03-06.csv
julia --project=. flood_bn_ibf_v1.jl \
    --input-csv bn_inputs/flood_inputs_2026-03-06.csv \
    --output-csv output/flood_bn_v1_2026-03-06.csv \
    --no-agreement --tail-risk
./plot_bn_dag_per_day.py --date 2026-03-06 --boundary Nairobi
```

### Environment
- Python deps install via `uv run --with …` on first call; no persistent env required.
- Julia deps tracked in `Project.toml` / `Manifest.toml` (CSV.jl, DataFrames.jl; RxInfer.jl is optional).
- Install Julia via `curl -fsSL https://install.julialang.org | sh -s -- --yes`.

---

## Outputs directory layout

```
flood_ibf/
├── bn_inputs/                                  # daily evidence CSVs (per-run)
│   └── flood_inputs_2026-03-DD.csv             # 227 rows × 17 cols
├── output/
│   ├── flood_bn_v1_2026-03-DD.csv              # 227 rows × risk/action probs
│   ├── flood_bn_v1_2026-03-01_to_10_summary.csv  # per-boundary peak summary
│   ├── flood_bn_v1_risk_maps_panel.png         # 2×5 combined map
│   ├── nairobi_diagnostic.png                  # 4-panel Nairobi timeline
│   ├── maps/
│   │   └── flood_bn_v1_risk_2026-03-DD.png     # per-day standalone map
│   └── bn_dags/
│       └── bn_dag_<boundary>_2026-03-DD.png    # per-day BN DAG
├── flood_data_prep.py                          # data pipeline
├── flood_bn_ibf_v1.jl                          # Julia BN (operational)
├── flood_bn_ibf_v1.py                          # Python BN (reference)
├── run_flood_bn_range.sh                       # orchestrator
├── summarize_bn.py                             # 10-day aggregator
├── plot_daily_risk_maps.py                     # choropleth maps
├── plot_nairobi_diagnostic.py                  # boundary-specific timeline
├── plot_bn_dag_per_day.py                      # per-day DAG with posteriors
├── flood_bn_dag.drawio                         # editable BN schematic
├── Project.toml / Manifest.toml                # Julia environment
├── icpac_adm1v3.geojson                        # 227 admin-1 polygons
├── cmorph_ea_return_periods.nc                 # pixel-wise RP thresholds
└── *.md                                        # documentation files
```

---

## Results for 2026-03-01 to 2026-03-10

**Mar 1–10, 2026 (pixel-p95 + tail_risk enabled)**:

| Metric | Count |
|--------|:-----:|
| Boundaries at Moderate+ peak | 126 / 227 (55%) |
| Boundaries reaching High | 6 |
| Days with Act action fired | 3 (Mar 7, 9, 10) |
| Days with Prepare action | 10 (every day) |

**Nairobi case validation** — Nairobi River flash flood overnight
6–7 March 2026 (≥25 initial deaths, ≥108 across Kenya by month-end,
per [Wikipedia: 2026 Kenya floods](https://en.wikipedia.org/wiki/2026_Kenya_floods)):

| Date | Risk | Action | Tail ratio | Notes |
|------|:----:|:------:|:----------:|-------|
| Mar 1 | Moderate | Alert | 0.84 | Saturated ground (135 mm antecedent) |
| Mar 4 | **Moderate** | **Prepare** | **1.11** | **2-day lead; tail crosses threshold** |
| Mar 6 | Moderate | Alert | 0.54 | Day before / of flood onset |
| Mar 7 | Minimal | Monitor | 0.44 | Post-event dry-out |

---

## Known limitations

1. **Short forecast lead**: init=D 00Z means each day's forecast only covers D→D+7. Multi-init fusion (pooling forecasts from D-3, D-2, D-1, D) could extend effective lead time.
2. **Scale mismatch**: ECMWF 0.25° (~28 km) averages over entire urban areas. CMORPH thresholds at 0.073° are point-scale. Convective rainfall is unpredictable at this scale — the ensemble spread captures it but the mean undersells.
3. **Single-model forecasts**: only ECMWF available. Adding GEFS would activate the `forecast_agreement` node and provide multi-model robustness.
4. **Fixed expert CPTs**: the risk CPT encodes domain knowledge but has no empirical calibration. A future Bayesian-learning upgrade (Dirichlet priors on CPT columns, fed by historical flood events) would refine these.
5. **Independent boundaries**: no spatial smoothing between neighbors — a hotspot in Bungoma does not elevate the prior for adjacent Busia. Hierarchical/MRF extensions possible in RxInfer.

See `flood_bn_ibf_system_v20260412.md` for discussion of each upgrade path.

---

## Citation / provenance

- **Data**: ICPAC IBF team — IMERG from NASA GES DISC, ECMWF IFS ensemble, CMORPH v1.0 (NOAA CDR), admin boundaries from GADM/ICPAC.
- **Stores**: public anonymous read from [source.coop/e4drr-project](https://source.coop/e4drr-project).
- **Method**: Expert BN originally by ICPAC IBF team (Python/pgmpy); Julia/RxInfer port and pixel-level upgrades in this repo.
