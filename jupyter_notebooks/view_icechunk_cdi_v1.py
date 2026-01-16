#!/usr/bin/env python3
"""
Script to open and visualize CDI data from IceChunk stores in GCS.
Demonstrates opening both 1km and 10km resolution data.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import icechunk

# Configuration
SERVICE_ACCOUNT_FILE = "/scratch/notebook/coiled-data-e4drr_202505.json"
GCS_BUCKET = "cdi_arco"
GCS_PATH = "bn_icpac_cdi_store"


def open_icechunk_store(year, month_name, resolution="1km"):
    """
    Open an IceChunk store from GCS.

    Parameters:
    -----------
    year : int
        Year of the data (2011-2025)
    month_name : str
        Month name (e.g., 'jan', 'feb', 'mar', etc.)
    resolution : str
        Resolution of the data ('1km' or '10km')

    Returns:
    --------
    xarray.Dataset
        The dataset from the IceChunk store
    """
    # Build the prefix path
    filename = f"eadw-cdi-data-{year}-{month_name}"

    if resolution == "1km":
        prefix = f"{GCS_PATH}/icechunk/{year}/{filename}"
    else:
        prefix = f"{GCS_PATH}/icechunk_10km/{year}/{filename}"

    print(f"Opening IceChunk store: gs://{GCS_BUCKET}/{prefix}")

    # Create GCS storage configuration
    storage = icechunk.gcs_storage(
        bucket=GCS_BUCKET,
        prefix=prefix,
        service_account_file=SERVICE_ACCOUNT_FILE
    )

    # Open existing repository
    repo = icechunk.Repository.open(storage=storage)
    session = repo.readonly_session()
    store = session.store

    # Open as xarray dataset
    ds = xr.open_zarr(store, consolidated=False)

    return ds, repo


def list_commits(repo):
    """List all commits in the repository."""
    print("\nCommit History:")
    print("-" * 50)

    # Get the ancestry (commit history)
    try:
        ancestry = repo.ancestry()
        for i, snapshot in enumerate(ancestry):
            print(f"  {i+1}. ID: {snapshot.id}")
            print(f"     Message: {snapshot.message}")
            print(f"     Written at: {snapshot.written_at}")
            print()
    except Exception as e:
        print(f"  Could not retrieve ancestry: {e}")


def plot_cdi_data(ds, title="CDI Data", save_path=None):
    """
    Plot the CDI data from the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing CDI data
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the figure
    """
    # Get the CDI variable
    if 'cdi' not in ds.data_vars:
        print(f"Available variables: {list(ds.data_vars)}")
        return

    cdi = ds['cdi']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot the data
    # Handle different dimension names
    if 'y' in cdi.dims and 'x' in cdi.dims:
        cdi_plot = cdi.squeeze()
    elif 'lat' in cdi.dims and 'lon' in cdi.dims:
        cdi_plot = cdi.squeeze()
    else:
        cdi_plot = cdi.squeeze()

    # Create the plot
    im = cdi_plot.plot(
        ax=ax,
        cmap='RdYlGn_r',  # Red-Yellow-Green reversed (red = drought)
        robust=True,
        add_colorbar=True,
        cbar_kwargs={'label': 'Combined Drought Index (CDI)', 'shrink': 0.8}
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

    return fig


def compare_resolutions(year, month_name):
    """
    Compare 1km and 10km resolution data side by side.

    Parameters:
    -----------
    year : int
        Year of the data
    month_name : str
        Month name (e.g., 'jan', 'feb', etc.)
    """
    print(f"\nComparing resolutions for {month_name.capitalize()} {year}")
    print("=" * 60)

    # Open both datasets
    print("\nOpening 1km dataset...")
    ds_1km, repo_1km = open_icechunk_store(year, month_name, "1km")

    print("\nOpening 10km dataset...")
    ds_10km, repo_10km = open_icechunk_store(year, month_name, "10km")

    # Print dataset info
    print("\n" + "=" * 60)
    print("1km Dataset Info:")
    print("-" * 40)
    print(f"  Dimensions: {dict(ds_1km.sizes)}")
    print(f"  Variables: {list(ds_1km.data_vars)}")
    print(f"  Coordinates: {list(ds_1km.coords)}")

    print("\n10km Dataset Info:")
    print("-" * 40)
    print(f"  Dimensions: {dict(ds_10km.sizes)}")
    print(f"  Variables: {list(ds_10km.data_vars)}")
    print(f"  Coordinates: {list(ds_10km.coords)}")

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1km data
    cdi_1km = ds_1km['cdi'].squeeze()
    im1 = cdi_1km.plot(
        ax=axes[0],
        cmap='RdYlGn_r',
        robust=True,
        add_colorbar=True,
        cbar_kwargs={'label': 'CDI', 'shrink': 0.8}
    )
    axes[0].set_title(f'1km Resolution ({dict(ds_1km.sizes)})', fontsize=12)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # Plot 10km data
    cdi_10km = ds_10km['cdi'].squeeze()
    im2 = cdi_10km.plot(
        ax=axes[1],
        cmap='RdYlGn_r',
        robust=True,
        add_colorbar=True,
        cbar_kwargs={'label': 'CDI', 'shrink': 0.8}
    )
    axes[1].set_title(f'10km Resolution ({dict(ds_10km.sizes)})', fontsize=12)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

    plt.suptitle(f'CDI Comparison: {month_name.capitalize()} {year}', fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = f"/scratch/notebook/cdi_comparison_{year}_{month_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison figure saved to: {save_path}")

    plt.show()

    # Close datasets
    ds_1km.close()
    ds_10km.close()

    return ds_1km, ds_10km


def plot_single_month(year, month_name, resolution="1km", save_fig=True):
    """
    Plot CDI data for a single month.

    Parameters:
    -----------
    year : int
        Year of the data
    month_name : str
        Month name (e.g., 'jan', 'feb', etc.)
    resolution : str
        Resolution ('1km' or '10km')
    save_fig : bool
        Whether to save the figure
    """
    print(f"\nPlotting {resolution} CDI data for {month_name.capitalize()} {year}")
    print("=" * 60)

    # Open the dataset
    ds, repo = open_icechunk_store(year, month_name, resolution)

    # Print dataset info
    print(f"\nDataset Info:")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables: {list(ds.data_vars)}")

    # Show commit history
    list_commits(repo)

    # Get CDI data
    cdi = ds['cdi'].squeeze()

    # Print statistics
    print(f"\nCDI Statistics:")
    print(f"  Min: {float(cdi.min()):.4f}")
    print(f"  Max: {float(cdi.max()):.4f}")
    print(f"  Mean: {float(cdi.mean()):.4f}")
    print(f"  Std: {float(cdi.std()):.4f}")

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    im = cdi.plot(
        ax=ax,
        cmap='RdYlGn_r',
        robust=True,
        add_colorbar=True,
        cbar_kwargs={
            'label': 'Combined Drought Index (CDI)',
            'shrink': 0.8,
            'extend': 'both'
        }
    )

    ax.set_title(f'CDI - {month_name.capitalize()} {year} ({resolution} resolution)', fontsize=14)
    ax.set_xlabel('Longitude (X)')
    ax.set_ylabel('Latitude (Y)')

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        save_path = f"/scratch/notebook/cdi_{year}_{month_name}_{resolution}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.show()

    ds.close()

    return ds


def main():
    """Main function demonstrating how to use the IceChunk stores."""

    print("=" * 60)
    print("CDI IceChunk Store Viewer")
    print("=" * 60)

    # Example: Plot a single month at 1km resolution
    print("\n\n>>> Example 1: Single month at 1km resolution")
    plot_single_month(2011, "apr", "1km")

    # Example: Plot a single month at 10km resolution
    print("\n\n>>> Example 2: Single month at 10km resolution")
    plot_single_month(2011, "apr", "10km")

    # Example: Compare both resolutions
    print("\n\n>>> Example 3: Compare 1km vs 10km resolution")
    compare_resolutions(2011, "apr")


if __name__ == "__main__":
    main()
