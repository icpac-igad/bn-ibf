#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=0.2.0",
#   "python-dotenv",
#   "xarray",
#   "zarr>=3",
#   "netcdf4",
#   "numpy",
#   "pandas",
#   "requests",
#   "beautifulsoup4",
# ]
# ///
"""
ICPAC East-African Drought Watch (EADW) dekadal CDI NetCDF -> source.coop icechunk.

The ICPAC FTP (HTTP-served) at
    https://droughtwatch.icpac.net/ftp/dekadal/netcdf/{YYYY}/eadw-cdi-data-{YYYY-MM-DD}.nc
publishes one ~78 MB NetCDF per dekad (D=01,11,21) covering 2010-01-01 .. now.

This uploader mirrors them to source.coop as an icechunk store at:
    s3://us-west-2.opendata.source.coop/e4drr-project/observations/icpac_cdi_dekadal_icechunk

Subcommands:
  list      Print the FTP file list. No AWS credentials needed.
  sync      Download every NetCDF that is missing from the icechunk store,
            append it as one timestep, commit, then delete the local file.
            Resumable — re-running picks up exactly where the last run stopped.
  status    Show how many timesteps the icechunk store currently has and what
            the last present timestamp is.

Resume / 1-hour-credential strategy:
  source.coop hands out STS credentials valid for ~1 h. Each `sync` iteration
  is one self-contained {download → append-to-zarr → commit → delete-local}
  cycle. When credentials expire mid-loop, the next icechunk write throws an
  AccessDenied; the script catches it, exits cleanly, and a fresh `sync`
  with refreshed creds resumes from the next missing dekad without re-doing
  any committed work.

Credentials are loaded from .env (see env.example or this script's
--print-env-template flag for the exact lines to add).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from bs4 import BeautifulSoup
from dotenv import load_dotenv

import icechunk

FTP_BASE = "https://droughtwatch.icpac.net/ftp/dekadal/netcdf"
DEFAULT_PREFIX = "e4drr-project/observations/icpac_cdi_dekadal_icechunk"
DEFAULT_BUCKET = "us-west-2.opendata.source.coop"
DEFAULT_REGION = "us-west-2"

# Filename pattern the FTP uses
FNAME_RE = re.compile(r"eadw-cdi-data-(\d{4}-\d{2}-\d{2})\.nc$")

ENV_TEMPLATE = """\
# ---------------------------------------------------------------------------
# source.coop S3 target — ICPAC EADW dekadal CDI mirror
# ---------------------------------------------------------------------------
SOURCE_COOP_BUCKET=us-west-2.opendata.source.coop
SOURCE_COOP_PREFIX=e4drr-project/observations/icpac_cdi_dekadal_icechunk
SOURCE_COOP_REGION=us-west-2

# ---------------------------------------------------------------------------
# source.coop S3 write credentials (issued by source.coop, valid ~1 hour).
# Generate at https://source.coop/repositories/e4drr-project/observations/manage
# python-dotenv strips the `export ` prefix automatically.
# ---------------------------------------------------------------------------
export AWS_ACCESS_KEY_ID="<your-access-key-id>"
export AWS_SECRET_ACCESS_KEY="<your-secret-access-key>"
export AWS_SESSION_TOKEN="<your-session-token>"
export AWS_DEFAULT_REGION="us-west-2"
"""


# ──────────────────────────────────────────────────────────────────────────
#  env / icechunk plumbing (mirrors download_seas51_tp_to_icechunk.py)
# ──────────────────────────────────────────────────────────────────────────


def require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(
            f"Missing env var: {name}\n"
            f"Run: {sys.argv[0]} --print-env-template > env.example "
            f"to see what to add to your .env"
        )
    return val


def open_icechunk_repo(*, read_only: bool = False) -> icechunk.Repository:
    """Open or create the EADW CDI icechunk repo at source.coop."""
    bucket = os.environ.get("SOURCE_COOP_BUCKET", DEFAULT_BUCKET)
    prefix = os.environ.get("SOURCE_COOP_PREFIX", DEFAULT_PREFIX)
    region = (
        os.environ.get("SOURCE_COOP_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
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
        sys.exit(
            "Missing AWS credentials for write access "
            "(set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN in .env)"
        )

    location = f"s3://{bucket}/{prefix}"
    try:
        repo = icechunk.Repository.open(storage=storage)
        print(f"[icechunk] opened {'(ro) ' if read_only else ''}{location}")
    except Exception:
        if read_only:
            raise
        repo = icechunk.Repository.create(storage=storage)
        print(f"[icechunk] created    {location}")
    return repo


def store_existing_times(repo: icechunk.Repository) -> set[pd.Timestamp]:
    """Read the time coord from the existing store; empty set if store is empty."""
    try:
        ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    except Exception:
        return set()
    if "time" not in ds.coords and "time" not in ds.dims:
        ds.close()
        return set()
    times = pd.to_datetime(ds.time.values)
    ds.close()
    return {pd.Timestamp(t) for t in times}


# ──────────────────────────────────────────────────────────────────────────
#  FTP discovery
# ──────────────────────────────────────────────────────────────────────────


def list_ftp_year(year: int, *, timeout: int = 30) -> list[tuple[pd.Timestamp, str]]:
    """Return [(timestamp, url), ...] for every NetCDF in YYYY/."""
    url = f"{FTP_BASE}/{year}/"
    r = requests.get(url, timeout=timeout)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        m = FNAME_RE.search(href)
        if m:
            ts = pd.Timestamp(m.group(1))
            out.append((ts, url + href))
    out.sort(key=lambda kv: kv[0])
    return out


def list_ftp_all(start_year: int = 2010,
                 end_year: int | None = None) -> list[tuple[pd.Timestamp, str]]:
    """Discover every dekad NetCDF on the FTP."""
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
    files = []
    for y in range(start_year, end_year + 1):
        year_files = list_ftp_year(y)
        if year_files:
            print(f"[ftp] {y}: {len(year_files)} dekads")
        files.extend(year_files)
    return files


# ──────────────────────────────────────────────────────────────────────────
#  Download + ingest a single dekad
# ──────────────────────────────────────────────────────────────────────────


def stream_download(url: str, dest: Path, *, timeout: int = 300) -> None:
    """Stream a single NetCDF to disk."""
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def open_dekad(path: Path, ts: pd.Timestamp) -> xr.Dataset:
    """
    Open one EADW NetCDF and return an xr.Dataset with a single-element
    `time` dim equal to `ts`. Discovers (lat, lon) dims by inspection so
    the script works regardless of whether the source uses (lat, lon) or
    (latitude, longitude) naming.
    """
    ds = xr.open_dataset(path, engine="netcdf4", chunks=None)

    # Promote / overwrite a single 'time' coordinate.
    if "time" in ds.dims:
        if ds.sizes["time"] != 1:
            raise SystemExit(f"{path.name} has time dim of size {ds.sizes['time']} (expected 1)")
        ds = ds.assign_coords(time=("time", [ts]))
    else:
        ds = ds.expand_dims(time=[ts])

    # Normalise dim names.
    rename = {}
    for src, dst in [("latitude", "lat"), ("longitude", "lon"),
                     ("Latitude", "lat"), ("Longitude", "lon"),
                     ("Lat", "lat"), ("Lon", "lon"), ("y", "lat"), ("x", "lon")]:
        if src in ds.dims and dst not in ds.dims:
            rename[src] = dst
        if src in ds.coords and dst not in ds.coords:
            rename[src] = dst
    if rename:
        ds = ds.rename(rename)

    return ds


def write_or_append_dekad(
    ds_new: xr.Dataset,
    repo: icechunk.Repository,
    *,
    ts: pd.Timestamp,
    commit_message: str | None = None,
) -> str:
    """
    Append one dekad to the icechunk store along the time axis.

    On the first call (empty store) we create the array with `mode="w"`.
    On subsequent calls we use `append_dim="time"`, which icechunk handles
    natively for unbounded time axes. Each call ends with one commit so
    the next run with fresh credentials can resume.
    """
    session = repo.writable_session("main")
    try:
        existing = xr.open_zarr(session.store, consolidated=False)
        first_write = "time" not in existing.dims
        existing.close()
    except Exception:
        first_write = True

    if first_write:
        ds_new.to_zarr(session.store, mode="w", consolidated=False)
        msg = commit_message or f"init store with first dekad {ts.date()}"
    else:
        ds_new.to_zarr(session.store, mode="a", append_dim="time", consolidated=False)
        msg = commit_message or f"append dekad {ts.date()}"

    commit_id = session.commit(msg)
    return commit_id


# ──────────────────────────────────────────────────────────────────────────
#  Subcommands
# ──────────────────────────────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    files = list_ftp_all(start_year=args.start_year, end_year=args.end_year)
    print(f"\n[ftp] total dekads available: {len(files)}")
    if files:
        print(f"[ftp]   first: {files[0][0].date()}  ({files[0][1]})")
        print(f"[ftp]   last:  {files[-1][0].date()}  ({files[-1][1]})")


def cmd_status(args: argparse.Namespace) -> None:
    try:
        repo = open_icechunk_repo(read_only=True)
    except Exception as exc:
        if "doesn't exist" in str(exc).lower() or "does not exist" in str(exc).lower():
            print("[status] icechunk store doesn't exist yet — run `sync` to create it")
            return
        raise
    times = sorted(store_existing_times(repo))
    if not times:
        print("[status] icechunk store exists but has 0 timesteps")
        return
    print(f"[status] icechunk store: {len(times)} timesteps")
    print(f"[status]   first: {times[0].date()}")
    print(f"[status]   last:  {times[-1].date()}")


def cmd_sync(args: argparse.Namespace) -> None:
    # 1. Discover what's on the FTP.
    ftp_files = list_ftp_all(start_year=args.start_year, end_year=args.end_year)
    if not ftp_files:
        sys.exit("No NetCDFs discovered on the FTP — check connectivity.")
    print(f"[sync] FTP advertises {len(ftp_files)} dekads")

    # 2. Open the icechunk store and read existing timestamps.
    repo = open_icechunk_repo()
    have = store_existing_times(repo)
    print(f"[sync] icechunk already has {len(have)} dekads")

    # 3. Filter to just the missing ones.
    missing = [(ts, url) for ts, url in ftp_files if ts not in have]
    if args.limit:
        missing = missing[: args.limit]
    if not missing:
        print("[sync] nothing to do — store is up to date")
        return
    print(f"[sync] will upload {len(missing)} dekads "
          f"({missing[0][0].date()} .. {missing[-1][0].date()})")

    tmp_root = Path(args.tmp_dir or tempfile.gettempdir()) / "icpac_cdi_dl"
    tmp_root.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for i, (ts, url) in enumerate(missing, start=1):
        local = tmp_root / Path(url).name
        try:
            print(f"[{i}/{len(missing)}] {ts.date()}  download → {local.name}", flush=True)
            stream_download(url, local, timeout=args.download_timeout)

            print(f"            open + assign time={ts.date()}", flush=True)
            ds = open_dekad(local, ts)
            if i == 1 and args.verbose:
                print(f"            schema: dims={dict(ds.sizes)} vars={list(ds.data_vars)}",
                      flush=True)

            print(f"            icechunk append + commit", flush=True)
            commit = write_or_append_dekad(ds, repo, ts=ts)
            print(f"            committed {commit[:12]}…", flush=True)
            ds.close()
            n_done += 1

        except Exception as exc:
            # Clean exit on credential expiry or transient S3 / FTP errors —
            # next run resumes from this dekad.
            msg = str(exc)
            print(f"[sync] aborted after {n_done} successful dekads: {type(exc).__name__}: {msg[:200]}",
                  flush=True)
            print(f"[sync]   re-run with refreshed credentials to resume "
                  f"(next dekad to upload: {ts.date()})", flush=True)
            if local.exists() and not args.keep_files:
                local.unlink()
            sys.exit(1)
        finally:
            if local.exists() and not args.keep_files:
                local.unlink()

    print(f"\n[sync] done. {n_done} new dekads uploaded.")


# ──────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--env-file", default=".env",
                   help="Path to .env file (default: .env in cwd)")
    p.add_argument("--print-env-template", action="store_true",
                   help="Print the .env block to add for this script and exit")

    sub = p.add_subparsers(dest="command")

    p_l = sub.add_parser("list", help="List dekads available on the ICPAC FTP")
    p_l.add_argument("--start-year", type=int, default=2010)
    p_l.add_argument("--end-year",   type=int, default=None,
                     help="Defaults to current year")

    p_s = sub.add_parser("status", help="Show how many dekads the icechunk store holds")

    p_y = sub.add_parser("sync", help="Download missing dekads and upload to icechunk")
    p_y.add_argument("--start-year", type=int, default=2010)
    p_y.add_argument("--end-year",   type=int, default=None)
    p_y.add_argument("--tmp-dir",    default=None,
                     help="Where to drop the downloaded NetCDFs (default: /tmp/icpac_cdi_dl)")
    p_y.add_argument("--limit", type=int, default=None,
                     help="Only upload this many missing dekads (smoke test)")
    p_y.add_argument("--download-timeout", type=int, default=300,
                     help="Per-file download timeout in seconds (default 300)")
    p_y.add_argument("--keep-files", action="store_true",
                     help="Keep local NetCDF copies (default: delete after upload)")
    p_y.add_argument("--verbose", action="store_true",
                     help="Print schema details on the first dekad")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_env_template:
        print(ENV_TEMPLATE)
        return
    if not args.command:
        sys.exit("Choose a subcommand: list | status | sync (--help for details)")

    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)
    elif args.command in ("status", "sync"):
        # status (read-only) might still work with anonymous access; sync needs creds.
        print(f"[warn] .env not found at {env_path}; relying on shell environment", flush=True)

    if args.command == "list":
        cmd_list(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "sync":
        cmd_sync(args)


if __name__ == "__main__":
    main()
