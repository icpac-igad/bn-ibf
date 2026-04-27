# SEAS5 → source.coop icechunk pipeline

Two icechunk stores on source.coop, both under `e4drr-project/forecasts`:

| Store | What | Dims |
|---|---|---|
| `seas51_tp_icechunk_v2` | Raw SEAS5 system-51 total precipitation (`tprate`) | `number=51, forecastMonth=6, time=N, latitude=35, longitude=32` (1° EA) |
| `seas51_spi3_10km_icechunk_v2` | SPI-3 from those forecasts, regridded to 10 km EA | `lead=6, member=51, init=N, lat=351, lon=321` (0.1° EA) |

Both are produced by `download_seas51_tp_to_icechunk.py` (subcommands `download` and `process`). The `download` path runs under `uv run` with PEP-723 deps. The `process` path needs xesmf + xclim (conda-only) and runs under a micromamba env.

---

## One-time setup

```bash
# Repo
cd /scratch/notebook/bn-ibf/drought_crma

# Credentials
cp env.example .env
# Edit .env: paste your CDS api key + source.coop write keys.
# .env is covered by the repo's .gitignore.

# uv installed at /usr/local/bin/uv  (already present)
# micromamba env with the conda deps (xesmf, xclim, esmpy, icechunk, ...)
micromamba install -n aifs-etl -c conda-forge xesmf xclim esmpy -y
```

source.coop S3 layout (already encoded in `.env`):

```
SOURCE_COOP_BUCKET=us-west-2.opendata.source.coop
SOURCE_COOP_PREFIX=e4drr-project/forecasts/seas51_tp_icechunk_v2  # or *_spi3_10km_*
SOURCE_COOP_REGION=us-west-2
```

`AWS_*` keys are temporary STS credentials from the source.coop UI — they expire (typically ~1 h). Regenerate at  
`https://source.coop/repositories/e4drr-project/forecasts/manage`.

Quick check that the keys can read + write:
```bash
uv run probe_sourcecoop.py
```

---

## Reproducing the initial build (1981 → April 2026)

### 1. Download full historical SEAS5 to `seas51_tp_icechunk_v2`

```bash
uv run download_seas51_tp_to_icechunk.py download --full-historical
```

What it does:
- Calls `download_seas5()` from `../../ibf-thresholds-triggers/00-download-data.py`
- Pulls 1981-01 → 2025-12, all 12 months × 6 lead months, 51 ensemble members → ~222 MB GRIB
- Opens with cfgrib → writes to icechunk on source.coop with `mode='w'`
- Wall time: ~12 min (CDS queue + transfer)

### 2. Append the partial current year

When CDS releases new monthly forecasts (around the 13th of each month), append them:

```bash
# Whatever months are released for the current year; example for Jan-Apr 2026:
uv run download_seas51_tp_to_icechunk.py download --months 1-4 --year 2026 --append
```

`--append` reads the existing store, concatenates along `time`, dedupes (new wins on duplicates), sorts, then rewrites.

### 3. Compute SPI-3 + regrid → `seas51_spi3_10km_icechunk_v2`

This is the heavy step (~35 min).

```bash
micromamba run -n aifs-etl python download_seas51_tp_to_icechunk.py process
```

What it does:
- Reads `seas51_tp_icechunk_v2` from source.coop (eager-loads ~750 MB)
- For each lead 1..6: SPI-3 per ensemble member (parameter-transfer pattern from `01-run-process-spi.py` — members 0-24 use cal window 1991-01..2018-01, members 25-50 use 2017-01..2024-01, both with `dist="gamma", method="APP", floc=0`)
- Bilinear regrid 1° → 0.1° (351 × 321 EA grid: lat -12..23, lon 21..53)
- Writes to icechunk via init+fill: empty NaN template first, then per-(lead, init-chunk=25) `region` writes with one commit each. Memory peak ≈ 1.5 GB (fits in 8 GB RAM).

If the STS token expires mid-run, regenerate it in `.env` and resume:

```bash
micromamba run -n aifs-etl python download_seas51_tp_to_icechunk.py process --resume
```

`--resume` skips the template init and skips any `(lead, init-chunk)` slice that's already non-NaN in the store.

---

## Monthly update procedure

Every month, after ECMWF releases the new SEAS5 forecast (~13th of month):

### 1. Refresh source.coop write keys
Generate fresh STS keys at `https://source.coop/repositories/e4drr-project/forecasts/manage`, replace the three `export AWS_*` lines in `.env`.
```bash
uv run probe_sourcecoop.py   # confirms read+write
```

### 2. Append the new init-month to the raw store

Suppose it's June 2026 and the May 2026 init has just become available. The currently-released months for 2026 are 1-5:

```bash
uv run download_seas51_tp_to_icechunk.py download \
    --months 1-5 --year 2026 --append
```

`--append` is safe to re-run: any inits already in the store are de-duplicated (new wins). The store grows from `time=N` to `time=N+1` (one new init per month).

### 3. Refresh the SPI-3 store

The SPI-3 store has fixed dimensions baked into its template (it can't grow `init` after init). For monthly updates, the simplest correct path is **rewrite from scratch** (the template adopts the current `init` length from the source store):

```bash
# Optional: free the old store name first via source.coop UI if you don't want
# to keep the old commit chain. Otherwise the new template overwrites.
micromamba run -n aifs-etl python download_seas51_tp_to_icechunk.py process
```

Wall time: ~35 min for full EA at 0.1°. Each lead is ~70 s SPI + 22 region writes; each STS-token refresh covers ~2 leads, so plan to refresh keys 2-3 times per run.

If a run is interrupted by token expiry:
```bash
# Refresh keys, then:
micromamba run -n aifs-etl python download_seas51_tp_to_icechunk.py process --resume
```

> **Note on append-style SPI updates** — the script does **not** support appending one new init to the existing SPI store, because SPI is computed across the full time series per (lead, member); changing the calibration window or the data tail changes earlier values too. For consistent SPI definitions, recompute the whole thing each month.

### 4. Sanity-check the result

```bash
micromamba run -n aifs-etl python -c "
import icechunk, xarray as xr
storage = icechunk.s3_storage(
    bucket='us-west-2.opendata.source.coop',
    prefix='e4drr-project/forecasts/seas51_spi3_10km_icechunk_v2',
    region='us-west-2', anonymous=True,
)
ds = xr.open_zarr(
    icechunk.Repository.open(storage).readonly_session('main').store,
    consolidated=False,
)
print(ds)
print('init:', ds.init.values[0], '->', ds.init.values[-1])
print('NaN cells in spi3 (should be 0 once filled):',
      int(ds.spi3.isnull().sum().compute()))
"
```

---

## File map

```
download_seas51_tp_to_icechunk.py   # the main script (download + process subcommands)
env.example                          # credential template
.env                                 # real credentials (gitignored)
probe_sourcecoop.py                  # quick read+write check on the source.coop creds
SEAS51_ICECHUNK_README.md            # this file
data/                                # local GRIB cache from CDS downloads
```

The script imports `download_seas5` / `download_current_month_seas5` from  
`../../ibf-thresholds-triggers/00-download-data.py` via `importlib.util` — that file's leading `00-` makes it un-importable as a normal module.

---

## Read access (anonymous)

The published stores are public on source.coop:

```python
import icechunk, xarray as xr

# Raw forecast precipitation
tp = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="e4drr-project/forecasts/seas51_tp_icechunk_v2",
    region="us-west-2", anonymous=True,
)
ds_tp = xr.open_zarr(
    icechunk.Repository.open(tp).readonly_session("main").store,
    consolidated=False,
)

# Regridded SPI-3
spi = icechunk.s3_storage(
    bucket="us-west-2.opendata.source.coop",
    prefix="e4drr-project/forecasts/seas51_spi3_10km_icechunk_v2",
    region="us-west-2", anonymous=True,
)
ds_spi = xr.open_zarr(
    icechunk.Repository.open(spi).readonly_session("main").store,
    consolidated=False,
)
```
