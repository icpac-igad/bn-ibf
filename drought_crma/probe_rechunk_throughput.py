#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "icechunk>=0.2.0",
#   "xarray",
#   "zarr>=3",
#   "dask",
#   "fsspec",
#   "s3fs",
# ]
# ///
"""
Probe: read a small subset of the SPI3 icechunk store on source.coop,
report bytes/sec to estimate the full rechunk time. Anonymous read.
"""
import time
import icechunk
import xarray as xr
import numpy as np

SOURCE = dict(
    bucket="us-west-2.opendata.source.coop",
    prefix="e4drr-project/forecasts/seas51_spi3_10km_icechunk_v2",
    region="us-west-2",
)


def main():
    print(f"Opening {SOURCE['prefix']} (anonymous) ...", flush=True)
    storage = icechunk.s3_storage(**SOURCE, anonymous=True)
    repo = icechunk.Repository.open(storage=storage)
    ds = xr.open_zarr(repo.readonly_session("main").store, consolidated=False)
    print(f"Source dims: {dict(ds.sizes)}")
    print(f"Source chunks: {ds['spi3'].encoding.get('chunks', 'unknown')}")

    # Read one full lead × member × init slice for a small lat/lon block
    # (~ 5.6 MB uncompressed per source chunk; we read enough to span all
    # 22 init-chunks for one lead, one (lat, lon) block).
    test = ds.spi3.isel(lead=0, lat=slice(0, 35), lon=slice(0, 32))
    n_bytes = int(np.prod(test.shape) * 4)
    print(f"Test slice: {dict(test.sizes)}  ({n_bytes / 1e6:.1f} MB)")

    t0 = time.time()
    arr = test.values  # eager fetch
    dt = time.time() - t0
    mbs = n_bytes / 1e6 / dt
    print(f"Read in {dt:.1f}s -> {mbs:.1f} MB/s")

    # Project for full rechunk: read 41.76 GB compressed, write similar
    full_gb = 41.76
    proj_min = 2 * full_gb * 1024 / mbs / 60  # *1024 -> MB; ×2 read+write
    print(f"\nProjection for full rechunk (41.76 GB read + ~42 GB write):")
    print(f"  ~{proj_min:.0f} min at this throughput "
          f"(local 2-core dask, single-stream is the floor).")


if __name__ == "__main__":
    main()
