#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "cdsapi",
#   "cfgrib",
#   "icechunk>=0.2.0",
#   "python-dotenv",
#   "xarray",
#   "zarr>=3",
# ]
# ///
"""
SEAS5 (system 51) total precipitation -> source.coop icechunk pipeline.

Two subcommands:

  download   — fetch GRIB from CDS and write/append to seas51_tp_icechunk_v2.
  process    — read seas51_tp_icechunk_v2, compute SPI-3 with the parameter-
               transfer pattern from ../../ibf-thresholds-triggers/01-run-process-spi.py,
               regrid to a 10 km East-Africa grid with xesmf, and write to
               seas51_spi3_10km_icechunk_v2.

`download` is the default `uv run` path — its dependencies are pinned in the
PEP-723 block above. `process` needs xesmf + xclim, which are conda-only;
run it under a micromamba env, e.g.:

  micromamba run -n geo python download_seas51_tp_to_icechunk.py process \\
      [--target-prefix e4drr-project/forecasts/seas51_spi3_10km_icechunk_v2]

xesmf / xclim / dask are imported lazily inside the process path so that a
plain `uv run download …` does not try to resolve them.

Credentials are loaded from `.env` (see env.example):
  CDSAPI_URL, CDSAPI_KEY
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION
  SOURCE_COOP_BUCKET, SOURCE_COOP_PREFIX, SOURCE_COOP_REGION
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import xarray as xr
from dotenv import load_dotenv
import icechunk

DOWNLOADER_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "ibf-thresholds-triggers"
    / "00-download-data.py"
)

DEFAULT_SOURCE_PREFIX = "e4drr-project/forecasts/seas51_tp_icechunk_v2"
DEFAULT_TARGET_PREFIX = "e4drr-project/forecasts/seas51_spi3_10km_icechunk_v2"


# ─── shared helpers ────────────────────────────────────────────────────────


def load_downloader():
    """Import 00-download-data.py — leading digit blocks regular import."""
    if not DOWNLOADER_PATH.exists():
        sys.exit(f"Downloader not found: {DOWNLOADER_PATH}")
    spec = importlib.util.spec_from_file_location("seas5_downloader", DOWNLOADER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f"Missing env var: {name} (copy env.example -> .env and fill it in)")
    return val


def open_icechunk_repo(prefix: str | None = None,
                       *, read_only: bool = False) -> icechunk.Repository:
    """Open or create an icechunk repo at <SOURCE_COOP_BUCKET>/<prefix>."""
    bucket = require_env("SOURCE_COOP_BUCKET")
    prefix = prefix or require_env("SOURCE_COOP_PREFIX")
    region = (
        os.environ.get("SOURCE_COOP_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-west-2"
    )

    access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("SOURCE_COOP_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("SOURCE_COOP_SECRET_ACCESS_KEY")
    session_token = (
        os.environ.get("AWS_SESSION_TOKEN")
        or os.environ.get("SOURCE_COOP_SESSION_TOKEN")
        or None
    )

    if access_key and secret_key:
        storage = icechunk.s3_storage(
            bucket=bucket, prefix=prefix, region=region,
            access_key_id=access_key, secret_access_key=secret_key,
            session_token=session_token,
        )
    elif read_only:
        storage = icechunk.s3_storage(
            bucket=bucket, prefix=prefix, region=region, anonymous=True,
        )
    else:
        sys.exit("Missing AWS credentials for write access (set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env)")

    location = f"s3://{bucket}/{prefix}"
    try:
        repo = icechunk.Repository.open(storage=storage)
        print(f"Opened {'(ro) ' if read_only else 'existing '}icechunk repo: {location}")
    except Exception:
        if read_only:
            raise
        repo = icechunk.Repository.create(storage=storage)
        print(f"Created new icechunk repo:    {location}")
    return repo


def write_dataset(ds: xr.Dataset, repo: icechunk.Repository, message: str) -> None:
    session = repo.writable_session("main")
    ds.to_zarr(session.store, mode="w", consolidated=False)
    commit_id = session.commit(message)
    print(f"Committed {commit_id}: {message}")


# ─── download subcommand ────────────────────────────────────────────────────


def open_grib(paths) -> xr.Dataset:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    datasets = [
        xr.open_dataset(
            p,
            engine="cfgrib",
            backend_kwargs=dict(time_dims=("forecastMonth", "time")),
        )
        for p in paths
    ]
    if len(datasets) == 1:
        return datasets[0]
    return xr.concat(datasets, dim="time")


def cmd_download(args: argparse.Namespace) -> None:
    require_env("CDSAPI_URL")
    require_env("CDSAPI_KEY")

    downloader = load_downloader()

    if args.full_historical:
        result = downloader.download_seas5(output_dir=args.output_dir)
        message = "SEAS5 total precipitation: full historical (1981-2025, all months)"
    else:
        # Upstream parse_month_input handles ranges ("1-6"), comma lists,
        # int, or list[int] — not a bare string like "1". Coerce single-int
        # CLI arg to int so it takes the integer branch.
        month_input = args.months
        if isinstance(month_input, str) and "-" not in month_input and "," not in month_input:
            month_input = int(month_input)

        result = downloader.download_current_month_seas5(
            output_dir=args.output_dir,
            month_input=month_input,
            year=args.year,
        )
        scope = f"months={args.months}"
        if args.year is not None:
            scope += f" year={args.year}"
        message = f"SEAS5 total precipitation: {scope}"

    if result is None:
        sys.exit("Download failed — see messages above.")
    print(f"Downloaded GRIB: {result}")

    ds = open_grib(result)
    print(f"New GRIB dims: {dict(ds.sizes)}")
    print(f"Variables:     {list(ds.data_vars)}")

    repo = open_icechunk_repo()

    if args.append:
        existing = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
        print(f"Existing store dims: {dict(existing.sizes)}")
        n_existing = existing.sizes.get("time", 0)

        combined = (
            xr.concat([existing, ds], dim="time")
            .drop_duplicates("time", keep="last")
            .sortby("time")
        )
        n_new = combined.sizes["time"] - n_existing
        n_replaced = ds.sizes["time"] - n_new
        print(f"Combined dims:       {dict(combined.sizes)}  "
              f"({n_new} added, {n_replaced} replaced)")

        write_dataset(combined, repo, f"{message} (appended)")
        existing.close()
        combined.close()
    else:
        write_dataset(ds, repo, message)

    ds.close()
    print("Done.")


# ─── process subcommand ────────────────────────────────────────────────────


def _parameter_transfer_spi(member_data, member_idx: int, args, spi_func):
    """Per-member SPI-3 with the cal window from 01-run-process-spi.py."""
    if member_idx < args.member_split:
        cal_s, cal_e = args.cal1_start, args.cal1_end
    else:
        cal_s, cal_e = args.cal2_start, args.cal2_end
    return spi_func(
        member_data,
        freq="MS",
        window=3,
        dist="gamma",
        method="APP",
        cal_start=cal_s,
        cal_end=cal_e,
        fitkwargs={"floc": 0},
    )


def cmd_process(args: argparse.Namespace) -> None:
    # Lazy imports — these need a conda/micromamba env (xesmf, xclim).
    import time as _time
    import warnings
    import numpy as np
    import dask.array as da
    from xclim.indices import standardized_precipitation_index
    import xesmf as xe

    # xclim warns once per fit() call about rechunking. We pre-load the
    # source so the warning is uninformative. Silence it.
    warnings.filterwarnings("ignore", message=".*rechunked to run.*")

    # 1. Source store — load eagerly (~750 MB) so xclim works on numpy arrays
    # (~50× speed-up vs re-reading + rechunking on every (lead, member) call).
    source_repo = open_icechunk_repo(args.source_prefix, read_only=True)
    src_session = source_repo.readonly_session("main")
    src_ds = xr.open_zarr(src_session.store, consolidated=False)

    rename = {c: r for c, r in {"latitude": "lat", "longitude": "lon"}.items()
              if c in src_ds.dims or c in src_ds.coords}
    if rename:
        src_ds = src_ds.rename(rename)

    print(f"Source dims:  {dict(src_ds.sizes)}")
    print(f"Source coords: lat[{float(src_ds.lat.min()):.2f}..{float(src_ds.lat.max()):.2f}] "
          f"lon[{float(src_ds.lon.min()):.2f}..{float(src_ds.lon.max()):.2f}]")
    print("Loading source into memory ...", flush=True)
    src_ds = src_ds.load()
    src_ds["tprate"].attrs["units"] = "mm/month"

    # 2. Target grid (regular lat/lon)
    target_lats = np.arange(args.lat_min, args.lat_max + args.resolution / 2, args.resolution)
    target_lons = np.arange(args.lon_min, args.lon_max + args.resolution / 2, args.resolution)
    n_lat, n_lon = len(target_lats), len(target_lons)
    print(f"Target grid:  {n_lat} × {n_lon} @ {args.resolution}° "
          f"(lat[{target_lats[0]:.2f}..{target_lats[-1]:.2f}] "
          f"lon[{target_lons[0]:.2f}..{target_lons[-1]:.2f}])")
    ds_out = xr.Dataset({
        "lat": (["lat"], target_lats, {"units": "degrees_north"}),
        "lon": (["lon"], target_lons, {"units": "degrees_east"}),
    })

    # 3. Target store
    target_repo = open_icechunk_repo(args.target_prefix)

    # 4. Build regridder once (same source grid for all leads/members/inits)
    src_template = src_ds.isel(forecastMonth=0, number=0, time=0)[["tprate"]]
    print(f"Building {args.regrid_method} regridder ...", flush=True)
    regridder = xe.Regridder(src_template, ds_out, args.regrid_method, periodic=False)

    # 5. Init template store. Full output is 6×51×544×351×321×4B ≈ 75 GB —
    # far bigger than RAM and bigger than icechunk's per-session buffer, so we
    # init an empty NaN array first, then fill `region` slices in a fresh
    # session per slice (commit between writes -> bounded memory). "region"
    # here is zarr/xarray terminology for an array slice; we do process the
    # full geographic East-Africa extent, just one init-chunk at a time.
    leads = sorted(int(v) for v in src_ds.forecastMonth.values.tolist())
    members = [int(m) for m in src_ds.number.values]
    n_lead = len(leads)
    n_member = len(members)
    n_init = src_ds.sizes["time"]
    init_chunk = args.init_chunk

    print(f"Initializing target template: lead={n_lead}, member={n_member}, "
          f"init={n_init}, lat={n_lat}, lon={n_lon} (init_chunk={init_chunk})",
          flush=True)
    template = xr.Dataset(
        {
            "spi3": (
                ("lead", "member", "init", "lat", "lon"),
                da.full(
                    (n_lead, n_member, n_init, n_lat, n_lon),
                    np.nan, dtype=np.float32,
                    chunks=(1, n_member, init_chunk, n_lat, n_lon),
                ),
            )
        },
        coords={
            "lead": ("lead", np.array(leads, dtype=np.int32), {"units": "months"}),
            "member": ("member", np.array(members, dtype=np.int32)),
            "init": ("init", src_ds.time.values),
            "lat": ("lat", target_lats, {"units": "degrees_north"}),
            "lon": ("lon", target_lons, {"units": "degrees_east"}),
        },
        attrs={
            "title": "SEAS5 SPI-3 East Africa, regridded",
            "source_store": args.source_prefix,
            "resolution_deg": args.resolution,
            "regrid_method": args.regrid_method,
            "calibration": (
                f"members<{args.member_split}: {args.cal1_start}..{args.cal1_end}; "
                f"others: {args.cal2_start}..{args.cal2_end}"
            ),
        },
    )
    if args.resume:
        print("Resume mode: skipping template init.", flush=True)
        # Read existing store to determine which (lead, init-chunk) slices
        # have already been filled. We test the first cell — if NaN, slice
        # is unwritten.
        existing_ds = xr.open_zarr(target_repo.readonly_session("main").store, consolidated=False)
        existing_status = np.asarray(
            existing_ds["spi3"].isel(member=0, lat=0, lon=0).values
        )  # shape (lead, init)
        existing_ds.close()
    else:
        session = target_repo.writable_session("main")
        template.to_zarr(session.store, mode="w", compute=False, consolidated=False)
        init_commit = session.commit("init template")
        print(f"Initialized template, commit={init_commit}", flush=True)
        existing_status = None

    # 6. Fill: per (lead, init_chunk), regrid eagerly (~225 MB at 0.1°) and
    # write that slice with a fresh session, committing each slice so memory
    # is freed between writes.
    for li, lead_val in enumerate(leads):
        print(f"\n=== Lead {lead_val} ({li + 1}/{n_lead}) ===", flush=True)

        # Resume: if every chunk of this lead is already non-NaN, skip the
        # entire lead (saves the ~70 s SPI compute).
        if existing_status is not None:
            lead_status = existing_status[li]
            if not np.isnan(lead_status).any():
                print(f"  Lead {lead_val} already complete — skipping", flush=True)
                continue

        lead_ds = src_ds.sel(forecastMonth=lead_val)
        lead_ds["tprate"].attrs["units"] = "mm/month"

        # SPI per member (full time series, eager)
        t0 = _time.time()
        member_spis: list = []
        spi_template = None
        for nsl in members:
            try:
                spi = _parameter_transfer_spi(
                    lead_ds.sel(number=nsl)["tprate"],
                    nsl, args, standardized_precipitation_index,
                )
                member_spis.append(spi)
                if spi_template is None:
                    spi_template = spi
            except Exception as e:
                print(f"  member {nsl}: SPI failed ({e})", flush=True)
                member_spis.append(None)

        if spi_template is None:
            print(f"  No members succeeded for lead {lead_val} — skipping", flush=True)
            continue

        n_failed = sum(1 for s in member_spis if s is None)
        if n_failed:
            nan_da = xr.full_like(spi_template, np.nan)
            member_spis = [s if s is not None else nan_da for s in member_spis]
        print(f"  SPI for {n_member - n_failed}/{n_member} members in "
              f"{_time.time()-t0:.1f}s ({n_failed} padded NaN)", flush=True)

        lead_native = xr.concat(member_spis, dim="member")
        del member_spis

        # Regrid + write per init chunk, committing each chunk so icechunk's
        # in-session buffer is flushed and memory is reclaimed.
        for ic_start in range(0, n_init, init_chunk):
            ic_end = min(ic_start + init_chunk, n_init)
            if existing_status is not None and not np.isnan(
                existing_status[li, ic_start:ic_end]
            ).any():
                print(f"  init [{ic_start:4d}:{ic_end:4d}] already filled — skipping",
                      flush=True)
                continue
            chunk_native = lead_native.isel(time=slice(ic_start, ic_end))
            chunk_regridded = regridder(chunk_native, keep_attrs=True)

            # Reshape for region write: dims (lead=1, member, init, lat, lon).
            # xarray's region write requires every variable in the dataset to
            # share at least one dim with the region dims. Drop ALL coord vars
            # — the template already holds the canonical coord values.
            if "time" in chunk_regridded.dims:
                chunk_regridded = chunk_regridded.rename({"time": "init"})
            chunk_regridded = chunk_regridded.expand_dims("lead", axis=0)
            chunk_ds = chunk_regridded.to_dataset(name="spi3")
            chunk_ds = chunk_ds.drop_vars(list(chunk_ds.coords))

            session = target_repo.writable_session("main")
            chunk_ds.to_zarr(
                session.store,
                region={
                    "lead": slice(li, li + 1),
                    "init": slice(ic_start, ic_end),
                },
                consolidated=False,
            )
            session.commit(f"lead {lead_val} init [{ic_start}:{ic_end}]")
            print(f"  init [{ic_start:4d}:{ic_end:4d}] written", flush=True)

            del chunk_native, chunk_regridded, chunk_ds

        del lead_native

    print("\nProcess complete.", flush=True)


# ─── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    sub = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Fetch GRIB from CDS and write/append to source.coop")
    p_dl.add_argument("--output-dir", default="./data",
                      help="Local dir for the downloaded GRIB (default: ./data)")
    mode = p_dl.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full-historical", action="store_true",
                      help="download_seas5() — 1981-2025, all months")
    mode.add_argument("--months",
                      help="Months for download_current_month_seas5 (e.g. 1-12, 1,3,5, 3)")
    p_dl.add_argument("--year", type=int, help="Single year (with --months)")
    p_dl.add_argument("--append", action="store_true",
                      help="Read existing store, concat new data along init-time, "
                           "dedupe (new wins), and rewrite. Default: overwrite.")

    # process
    p_pr = sub.add_parser(
        "process",
        help="SPI-3 (parameter-transfer) + regrid to 10 km EA. Needs xesmf+xclim "
             "(run under micromamba, not uv).",
    )
    p_pr.add_argument("--source-prefix", default=DEFAULT_SOURCE_PREFIX)
    p_pr.add_argument("--target-prefix", default=DEFAULT_TARGET_PREFIX)
    p_pr.add_argument("--resolution", type=float, default=0.1, help="Degrees (default: 0.1 ≈ 10 km)")
    p_pr.add_argument("--lat-min", type=float, default=-12)
    p_pr.add_argument("--lat-max", type=float, default=23)
    p_pr.add_argument("--lon-min", type=float, default=21)
    p_pr.add_argument("--lon-max", type=float, default=53)
    p_pr.add_argument("--regrid-method", default="bilinear", choices=["bilinear", "conservative"])
    p_pr.add_argument("--cal1-start", default="1991-01-01",
                      help="cal_start for members < --member-split (default: 1991-01-01)")
    p_pr.add_argument("--cal1-end", default="2018-01-01")
    p_pr.add_argument("--cal2-start", default="2017-01-01",
                      help="cal_start for members >= --member-split (default: 2017-01-01)")
    p_pr.add_argument("--cal2-end", default="2024-01-01")
    p_pr.add_argument("--member-split", type=int, default=25,
                      help="01-run-process-spi.py uses 25 (default)")
    p_pr.add_argument("--init-chunk", type=int, default=25,
                      help="Init-times per region write. Caps peak memory per "
                           "regrid+write at ~ n_member × init_chunk × n_lat × "
                           "n_lon × 4B (default: 25 ≈ 550 MB at 0.1° EA).")
    p_pr.add_argument("--resume", action="store_true",
                      help="Skip the template init and skip any (lead, init-chunk) "
                           "slice that is already non-NaN in the target store. Use "
                           "this to continue after an STS-token expiry.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_path = Path(args.env_file)
    if not env_path.exists():
        sys.exit(f".env not found at {env_path}. Copy env.example -> .env and fill in values.")
    load_dotenv(env_path)

    if args.command == "download":
        cmd_download(args)
    elif args.command == "process":
        cmd_process(args)


if __name__ == "__main__":
    main()
