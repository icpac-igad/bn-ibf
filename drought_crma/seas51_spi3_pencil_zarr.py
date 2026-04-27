#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "xarray>=2024.1.0",
#     "numpy>=1.26.0",
#     "icechunk>=0.2.0",
#     "dask>=2024.1.0",
#     "python-dotenv>=1.0.0",
#     "coiled>=1.0.0",
#     "distributed>=2024.1.0",
#     "s3fs>=2024.1.0",
#     "zarr>=3",
# ]
# ///
"""
Rechunk SEAS5 SPI-3 Icechunk store from slab to pencil-tile layout (Zarr).

Mirrors the pattern in
../grib-index-kerchunk/ecmwf/ecmwf_ea_tp_pencil_zarr.py — Coiled Dask P2P
rechunk in us-west-2 (same region as source.coop), writing plain Zarr.

  Slab   : (1, 51, 25, 351, 321)  — 1 lead × all members × 25 inits × full EA
            good for: spatial maps at one (lead, init-batch)
  Pencil : (6, 51, 544, 5, 5)     — all leads × all members × all inits × 5×5 tile
            good for: full time series + ensemble + leads at one pixel block
                      (single chunk read; matches ecmwf_spi_icechunk.py
                      PENCIL_CHUNK_SIZE = (-1, 5, 5)).

Source : s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/seas51_spi3_10km_icechunk_v2
Target : s3://us-west-2.opendata.source.coop/e4drr-project/forecasts/seas51_spi3_10km_pencil_zarr

Usage:
  # Show shapes / chunk sizes (anonymous, no creds needed)
  uv run seas51_spi3_pencil_zarr.py --dry-run

  # Coiled P2P rechunk (default — 15 workers in us-west1)
  uv run seas51_spi3_pencil_zarr.py --coiled

  # Verify the published pencil zarr (anonymous read)
  uv run seas51_spi3_pencil_zarr.py --verify

Author: ICPAC GIK Team
"""

import argparse
import os
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()

SOURCE_COOP_BUCKET      = "us-west-2.opendata.source.coop"
SOURCE_COOP_BASE_PREFIX = "e4drr-project/forecasts"
SOURCE_PREFIX           = "seas51_spi3_10km_icechunk_v2"
PENCIL_ZARR_PREFIX      = "seas51_spi3_10km_pencil_zarr"

VARIABLE = "spi3"

# Default coiled cluster
DEFAULT_WORKSPACE   = "gcp-sewaa-nka"
DEFAULT_REGION      = "us-west1"
DEFAULT_VM_TYPE     = "n2-highmem-4"  # 4 vCPU, 32 GB
DEFAULT_N_WORKERS   = 15


# ─── Credentials ─────────────────────────────────────────────────────────────


def _get_s3_credentials():
    """source.coop S3 creds, falling back to AWS_* env vars."""
    access_key    = os.getenv("SOURCE_COOP_ACCESS_KEY_ID")     or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key    = os.getenv("SOURCE_COOP_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("SOURCE_COOP_SESSION_TOKEN")     or os.getenv("AWS_SESSION_TOKEN")
    return access_key, secret_key, session_token


# ─── Storage helpers ─────────────────────────────────────────────────────────


def _s3_source_storage():
    """Anonymous Icechunk storage for the slab source store."""
    import icechunk

    return icechunk.s3_storage(
        bucket=SOURCE_COOP_BUCKET,
        prefix=f"{SOURCE_COOP_BASE_PREFIX}/{SOURCE_PREFIX}",
        region="us-west-2",
        anonymous=True,
    )


def open_source_store():
    import icechunk
    import xarray as xr

    storage = _s3_source_storage()
    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)
    return ds


# ─── Coiled P2P rechunk → plain Zarr ─────────────────────────────────────────


def rechunk_coiled(chunk_lat: int, chunk_lon: int, chunk_init: int,
                   n_workers: int, workspace: str, region: str,
                   dry_run: bool = False):
    """Rechunk slab → pencil via Coiled Dask P2P, writing plain Zarr."""
    import pickle

    import coiled
    import dask
    import distributed
    import icechunk
    import xarray as xr

    print(f"\n{'='*60}")
    print("Rechunk (Coiled P2P): SEAS5 SPI-3 slab → pencil-tile")
    print(f"{'='*60}")

    # P2P rechunk configuration
    dask.config.set({
        "array.rechunk.method": "p2p",
        "optimization.fuse.active": False,
    })
    print("  Dask config: P2P rechunk enabled, fusion disabled")

    src_label = f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{SOURCE_PREFIX}"
    print(f"\n  Source: {src_label}")

    storage = _s3_source_storage()
    repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
    session = repo.readonly_session("main")

    # IcechunkStore must be pickle-serializable for Dask workers
    try:
        pickle.dumps(session.store)
        print("  IcechunkStore is pickle-serializable")
    except Exception as e:
        print(f"  ERROR: IcechunkStore not serializable: {e}")
        return

    # Open with explicit chunks matching the stored layout
    ds = xr.open_zarr(session.store, consolidated=False)
    source_chunks = {
        d: c for d, c in zip(ds[VARIABLE].dims, ds[VARIABLE].encoding["chunks"])
    }
    ds.close()
    ds = xr.open_zarr(session.store, consolidated=False, chunks=source_chunks)

    print(f"  Shape: {dict(ds.sizes)}")
    print(f"  Source chunks: {source_chunks}")

    n_lead   = ds.sizes["lead"]
    n_member = ds.sizes["member"]
    n_init   = ds.sizes["init"]
    n_lat    = ds.sizes["lat"]
    n_lon    = ds.sizes["lon"]
    size_gb  = n_lead * n_member * n_init * n_lat * n_lon * 4 / (1024 ** 3)

    # Pencil layout: bundle leads + members + inits, tile spatially.
    # Matches ecmwf_spi_icechunk.py PENCIL_CHUNK_SIZE = (-1, 5, 5) — full
    # time series at a (lat-tile × lon-tile) block, in a single chunk read.
    init_chunk_size = n_init if chunk_init <= 0 else min(chunk_init, n_init)
    pencil_chunks = {
        "lead": n_lead,
        "member": n_member,
        "init": init_chunk_size,
        "lat": chunk_lat,
        "lon": chunk_lon,
    }
    chunk_bytes  = n_lead * n_member * init_chunk_size * chunk_lat * chunk_lon * 4
    n_lat_chunks  = -(-n_lat // chunk_lat)
    n_lon_chunks  = -(-n_lon // chunk_lon)
    n_init_chunks = -(-n_init // init_chunk_size)
    n_target_chunks = n_init_chunks * n_lat_chunks * n_lon_chunks

    target_path = (
        f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{PENCIL_ZARR_PREFIX}"
    )
    print(f"\n  Target: {target_path}")
    print(f"  Pencil chunks: (lead={n_lead}, member={n_member}, "
          f"init={init_chunk_size}, lat={chunk_lat}, lon={chunk_lon})")
    print(f"  Chunk size:    {chunk_bytes / (1024**2):.2f} MB")
    print(f"  Spatial tiles: {n_lat_chunks} lat × {n_lon_chunks} lon "
          f"= {n_lat_chunks * n_lon_chunks} tiles per init-batch")
    print(f"  Total chunks:  {n_target_chunks:,} "
          f"({n_init_chunks} init-batches × {n_lat_chunks} lat × {n_lon_chunks} lon)")
    print(f"  Total data:    {size_gb:.1f} GB")
    per_worker_gb = size_gb / n_workers
    print(f"  Workers: {n_workers} × {DEFAULT_VM_TYPE} (32 GB each) in {region}")
    print(f"  Per-worker data: {per_worker_gb:.2f} GB")

    if dry_run:
        print("\n  (dry run — no rechunking)")
        ds.close()
        return

    # Credential check before launching cluster
    access_key, secret_key, _ = _get_s3_credentials()
    if not access_key or not secret_key:
        print("ERROR: S3 credentials required for writing to source.coop")
        return

    # Launch fixed-size Coiled cluster (P2P shuffle requires static cluster)
    print(f"\n  Launching Coiled cluster ({workspace}) ...")
    overall_start = time.time()

    cluster = coiled.Cluster(
        name=f"seas51-spi3-pencil-{int(time.time()) % 10000}",
        n_workers=n_workers,
        worker_vm_types=DEFAULT_VM_TYPE,
        region=region,
        workspace=workspace,
        idle_timeout="30 minutes",
    )
    client = distributed.Client(cluster)
    client.wait_for_workers(n_workers=n_workers, timeout=600)
    print(f"  Cluster ready: {n_workers} workers")
    print(f"  Dashboard: {client.dashboard_link}")

    # Rechunk + write
    print("\n  Starting P2P rechunk + write ...")
    ds_rechunked = ds.chunk(pencil_chunks)

    # Re-read .env right before write to catch the freshest STS token
    load_dotenv(override=True)
    access_key, secret_key, session_token = _get_s3_credentials()
    print(f"  S3 credentials refreshed (key=...{access_key[-4:]})")

    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "token": session_token,
        "client_kwargs": {"region_name": "us-west-2"},
    }
    encoding = {
        VARIABLE: {"chunks": (n_lead, n_member, init_chunk_size, chunk_lat, chunk_lon)}
    }

    ds_rechunked.to_zarr(
        target_path,
        storage_options=storage_options,
        encoding=encoding,
        mode="w",
        consolidated=True,
    )

    print("  Write complete!")
    client.close()
    cluster.close()

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print("RECHUNK COMPLETE: seas51_spi3_10km")
    print(f"  Target: {target_path}")
    print(f"  Chunks: ({n_lead}, {n_member}, {init_chunk_size}, {chunk_lat}, {chunk_lon})")
    print(f"  Total chunks: {n_target_chunks:,}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print(f"{'='*60}")


# ─── Verify ──────────────────────────────────────────────────────────────────


def verify_pencil():
    import xarray as xr

    target_path = (
        f"s3://{SOURCE_COOP_BUCKET}/{SOURCE_COOP_BASE_PREFIX}/{PENCIL_ZARR_PREFIX}"
    )
    print(f"\n  Verifying (plain Zarr): {target_path}")
    ds = xr.open_zarr(target_path, storage_options={"anon": True}, consolidated=True)

    print(f"  Dimensions: {dict(ds.sizes)}")
    chunks = ds[VARIABLE].encoding.get("chunks", "unknown")
    print(f"  {VARIABLE} pencil chunks = {chunks}")

    if "init" in ds.coords:
        print(f"  init: {ds.init.values[0]} → {ds.init.values[-1]} "
              f"({len(ds.init)} steps)")

    # Spot check: full ensemble × all leads at one pixel for one init
    # — should be a single chunk read in pencil layout.
    ts = ds[VARIABLE].isel(init=0, lat=0, lon=0).values
    valid = int(np.count_nonzero(~np.isnan(ts)))
    print(f"  {VARIABLE} ensemble at (init=0, lat=0, lon=0): shape={ts.shape}, "
          f"{valid}/{ts.size} valid ({100*valid/ts.size:.1f}%)")

    ds.close()
    print("  Verification passed.")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunk-lat", type=int, default=5,
                        help="Pencil tile lat size (default: 5; ~16.7 MB/chunk)")
    parser.add_argument("--chunk-lon", type=int, default=5,
                        help="Pencil tile lon size (default: 5)")
    parser.add_argument("--chunk-init", type=int, default=-1,
                        help="Init-batch size per chunk (-1=full, default). "
                             "Use a smaller value (e.g. 100) only if you need "
                             "smaller chunks; full-init keeps the time-pencil "
                             "shape from ecmwf_spi_icechunk.py.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show shapes / chunk sizes only, no rechunk")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the published pencil zarr store")
    parser.add_argument("--coiled", action="store_true",
                        help="Run rechunk on Coiled (default for actual runs)")
    parser.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS,
                        help=f"Coiled workers (default: {DEFAULT_N_WORKERS})")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE,
                        help=f"Coiled workspace (default: {DEFAULT_WORKSPACE})")
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help=f"Coiled VM region (default: {DEFAULT_REGION})")
    args = parser.parse_args()

    if args.verify:
        verify_pencil()
    elif args.dry_run or args.coiled:
        rechunk_coiled(args.chunk_lat, args.chunk_lon, args.chunk_init,
                       args.n_workers, args.workspace, args.region,
                       dry_run=args.dry_run)
    else:
        parser.print_help()
        print("\nNote: pass --coiled to actually run (P2P rechunk requires a Coiled cluster).")
        print("      pass --dry-run to preview shapes and chunk sizes.")

    print("\nDone.")


if __name__ == "__main__":
    main()
