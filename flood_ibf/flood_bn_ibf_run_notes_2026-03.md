# Flood BN IBF — March 2026 Run Notes

**Run period**: 2026-03-01 → 2026-03-10  
**Domain**: ICPAC East Africa, 227 admin-1 boundaries (11 countries)  
**Pipeline version**: `flood_bn_ibf_v1.jl` (Julia/RxInfer) + `flood_data_prep.py`

---

## 1. Pipeline steps

### Step 1 — Per-day data preparation (`flood_data_prep.py`)

For each target date D the script:

1. Opens the **IMERG half-hourly icechunk store** (`observations/imerg_hh_icechunk`) and accumulates the 7-day window `[D-7, D)` into daily totals per pixel.
2. Opens the **ECMWF 51-member TP icechunk store** (`forecasts/ecmwf_ea_tp_icechunk`) and computes window accumulations at 3h, 6h, 12h, 24h, 48h, 72h, and 7-day lead times from init `D 00Z`.
3. Reads the **CMORPH 2-year return-period NetCDF** (`cmorph_ea_return_periods.nc`) for pixel-wise exceedance thresholds at each duration.
4. Masks all grids to admin-1 polygons (`icpac_adm1v3.geojson`) using `regionmask`.
5. For each boundary computes:
   - `antecedent_rainfall_mm` — mean 7-day IMERG accumulation
   - `trend_slope_mm_per_day` — linear regression slope over the 7-day window
   - `gefs_eprob_heavy` — `max` over durations of `P(any member ≥ 2yr RP)` pixel fraction
   - `ens_max_ratio` — 95th-percentile across pixels of `(ensemble-max / threshold)`
   - `hotspot_fraction` — fraction of pixels where any member exceeds the RP threshold
   - `spatial_coverage` — `max(eprob ≥ 0.5 mask, hotspot_fraction)`
   - Gaussian soft-bin columns: `ant_p1..p5`, `exc_p1..p5`, `spa_p1..p3`, `trn_p1..p3`, `tail_p1..p4`

Output: `bn_inputs/flood_inputs_YYYY-MM-DD_soft.csv` (227 rows, ~40 columns).

**Run command:**
```bash
uv run python flood_data_prep.py --date YYYY-MM-DD --rp-years 2 \
    --out bn_inputs/flood_inputs_YYYY-MM-DD.csv
```

All 10 input CSVs are committed at `bn_inputs/flood_inputs_2026-03-{01..10}_soft.csv`.

---

### Step 2 — Static BN inference per day (`flood_bn_ibf_v1.jl`, single-day mode)

The Julia script reads one input CSV and runs the BN via fast tensor contraction (`infer_soft_matmul`). This is the **static** (no temporal coupling) baseline.

```bash
julia --project=. flood_bn_ibf_v1.jl \
    --input-csv bn_inputs/flood_inputs_YYYY-MM-DD_soft.csv \
    --output-csv output/flood_bn_v1_YYYY-MM-DD_soft.csv \
    --no-agreement --tail-risk --cost-loss-ratio 0.2
```

The driver script `run_flood_bn_range.sh` loops this for the full range:
```bash
./run_flood_bn_range.sh 2026-03-01 2026-03-10
```

Static outputs: `output/flood_bn_v1_2026-03-{01..10}_soft.csv`.

---

### Step 3 — Dynamic BN sequence (`run_dbn_sequence` in Julia)

The DBN step chains the 10 daily CSVs sequentially. Yesterday's risk posterior is blended into today's inference as a virtual-evidence prior:

```
P(risk_t | evidence_t) ∝ P(evidence_t | risk_t) × [α · P(risk_{t-1}) + (1-α) · Uniform]
```

Parameters: `α = 0.6` (temporal decay), lookback `L = 7` days.

Called from Julia directly (not from the shell script):

```julia
using CSV, DataFrames
include("flood_bn_ibf_v1.jl")

soft_csvs = sort(filter(f -> occursin("_soft.csv", f),
                        readdir("bn_inputs", join=true)))
dbn_df = run_dbn_sequence(soft_csvs;
    include_tail_risk = true,
    cost_loss_ratio   = 0.20,
    temporal_decay    = 0.60,
    lookback          = 7,
)
CSV.write("output/flood_bn_v1_dbn_10day.csv", dbn_df)
```

Per-day DBN outputs are in `output/dbn/flood_bn_v1_2026-03-{01..10}.csv`.  
Combined file: `output/flood_bn_v1_dbn_10day.csv` (2 270 rows, 10 days × 227 boundaries).

---

### Step 4 — Per-member storyline selection

51 ECMWF members are each run through the BN separately (using `run_per_member_bn`), then the worst / median / best member is selected per boundary-day by `P(High ∪ Extreme)`.

Output: `output/flood_bn_v1_storylines_10day.csv`.

---

### Step 5 — Web output generation

After the Julia runs, two Python scripts convert the DBN CSVs into formats consumed by the CRMA web application:

```bash
# Parquet files for calendar heatmap + choropleth map API
uv run python3 generate_bn_parquet.py \
    --input-dir output/dbn --out-dir output/

# Per-day boundary BN-DAG JSON for the interactive DAG panel
uv run python3 generate_bn_dag_json.py \
    --input-dir bn_inputs --dbn-dir output/dbn --out-dir output/bn-dag/
```

Outputs uploaded to `gs://crma-mdx-store`:
- `parquet/flood_bn_ibf_daily.parquet` (10 rows, 1/day)
- `parquet/flood_bn_ibf_boundary_daily.parquet` (2 270 rows)
- `bn-dag/bn-dag-2026-03-{01..10}.json` (~122 KB each, 227 boundaries/file)

---

## 2. Key results — March 2026

### CRMA state distribution across 227 boundaries

| Date       | Actionable_Risk | Assess | Evaluate | Monitor |
|------------|----------------:|-------:|---------:|--------:|
| 2026-03-01 | 54              | 24     | 13       | 136     |
| 2026-03-02 | 28              | 39     | 29       | 131     |
| 2026-03-03 | 7               | 59     | 26       | 135     |
| 2026-03-04 | 3               | 54     | 34       | 136     |
| 2026-03-05 | 1               | 50     | 36       | 140     |
| 2026-03-06 | 1               | 56     | 39       | 131     |
| 2026-03-07 | 1               | 61     | 27       | 138     |
| 2026-03-08 | 18              | 29     | 28       | 152     |
| 2026-03-09 | 23              | 12     | 16       | 176     |
| 2026-03-10 | 22              | 8      | 11       | 186     |

High `Actionable_Risk` counts on Mar 1 (54 boundaries) reflect elevated antecedent moisture
across the domain from a wet February. The signal decays through Mar 3–7, then a
secondary episode lifts counts again on Mar 8–10.

### Nairobi — DBN signal (Nairobi River flash-flood, 6–7 March 2026)

| Date       | Risk level | CRMA state      | P(High∪Extreme) |
|------------|-----------|-----------------|-----------------|
| 2026-03-01 | Moderate  | **Assess**      | 0.166           |
| 2026-03-02 | Moderate  | **Assess**      | 0.115           |
| 2026-03-03 | Low       | Evaluate        | 0.039           |
| 2026-03-04 | Moderate  | **Assess**      | 0.100           |
| 2026-03-05 | Low       | Evaluate        | 0.012           |
| 2026-03-06 | Low       | Evaluate        | 0.037           |
| 2026-03-07 | Low       | **Assess**      | 0.058           |
| 2026-03-08 | Moderate  | **Assess**      | 0.149           |
| 2026-03-09 | Low       | Evaluate        | 0.087           |
| 2026-03-10 | Minimal   | Monitor         | 0.026           |

The system issued **Assess (Orange)** on Mar 1–2 and again on Mar 4, providing a 2-day
lead before the Nairobi River peaked on Mar 6–7. An ensemble-mean threshold approach
would have missed this signal because mean exceedance probability (P_heavy) was low;
the Assess trigger came from the combination of high antecedent moisture (Very_Wet)
and tail-risk (at least one ensemble member exceeding the 2-yr return period threshold).

### Nairobi — Storyline decomposition on March 4 (2-day lead)

| Storyline | Risk level | CRMA state      | P(High∪Extreme) | P(world ≥ this bad) |
|-----------|-----------|-----------------|-----------------|---------------------|
| Worst     | Moderate  | Actionable_Risk | 0.315           | 2%                  |
| Median    | Minimal   | Monitor         | 0.003           | 51%                 |
| Best      | Minimal   | Monitor         | 0.003           | 100%                |

The worst-member storyline reached **Actionable_Risk** on Mar 4 with P(High∪Extreme) = 0.315,
driven by the single ensemble member with the highest ensemble-max / RP ratio. The
median member was benign. This asymmetry is the diagnostic value of per-member
storyline selection: it surfaces the plausible-worst-case that the ensemble mean
suppresses.

---

## 3. Configuration reference

| Parameter              | Value  | Notes                                              |
|------------------------|--------|----------------------------------------------------|
| Return period          | 2 yr   | Pixel-wise CMORPH threshold; sensitive IBF trigger |
| Tail-risk aggregation  | p95    | 95th-percentile of max_ratio across pixels         |
| Agreement node         | Off    | `--no-agreement`; enables 5-parent RxInfer path    |
| Soft-binning σ         | ant=10 mm, exc=0.05, spa=0.05, trn=1 mm/d, tail=0.15 | |
| DBN temporal decay α   | 0.60   | 60% yesterday + 40% uniform                       |
| DBN lookback L         | 7 days | Reset to uniform prior after 7 days                |
| Cost-loss ratio γ      | 0.20   | Actionable_Risk trigger when P(H∪E) ≥ 0.20        |
| Assess threshold       | 0.40   | P(M∪H∪E) ≥ max(2γ, 0.40)                         |
| Evaluate threshold     | 0.30   | P(L∪M∪H∪E) ≥ max(3γ, 0.30)                      |
| Ensemble members       | 51     | ECMWF ENS                                          |
| Boundaries             | 227    | ICPAC admin-1, 11 countries                        |

---

## 4. Output file inventory

| File | Description |
|------|-------------|
| `bn_inputs/flood_inputs_2026-03-{01..10}_soft.csv` | Per-day evidence inputs with Gaussian soft-bin columns |
| `output/flood_bn_v1_2026-03-{01..10}_soft.csv` | Static BN outputs (no temporal coupling) |
| `output/dbn/flood_bn_v1_2026-03-{01..10}.csv` | DBN outputs (with α=0.6 temporal prior) |
| `output/flood_bn_v1_dbn_10day.csv` | Combined DBN output, all 10 days (2 270 rows) |
| `output/flood_bn_v1_storylines_10day.csv` | Worst/median/best storylines per boundary-day |
| `output/flood_bn_ibf_daily.parquet` | Calendar API parquet (10 rows) |
| `output/flood_bn_ibf_boundary_daily.parquet` | Choropleth API parquet (2 270 rows) |
| `output/bn-dag/bn-dag-2026-03-{01..10}.json` | BN-DAG JSON for CRMA boundary panel (~122 KB each) |
