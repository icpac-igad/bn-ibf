#!/usr/bin/env python3
"""
Create animated GIF of CDI (Combined Drought Indicator) data from GCS.

Usage:
    python create_cdi_animation.py --start-year 2024 --start-month 1 --end-year 2024 --end-month 12 --output cdi_2024.gif
    python create_cdi_animation.py -sy 2023 -sm 6 -ey 2024 -em 6 -o cdi_animation.gif
"""

import argparse
import os
import sys
import logging
from typing import Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import imageio.v2 as imageio

try:
    import icechunk
    HAS_ICECHUNK = True
except ImportError:
    HAS_ICECHUNK = False
    print("Warning: icechunk not available, will try regular zarr")

import gcsfs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
GCS_BUCKET = "cdi_arco"
CDI_GCS_PATH = "bn_icpac_cdi_store"

MONTH_MAP = {
    1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr',
    5: 'may', 6: 'jun', 7: 'jul', 8: 'aug',
    9: 'sep', 10: 'oct', 11: 'nov', 12: 'dec'
}

MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}

# Default service account paths
DEFAULT_SERVICE_ACCOUNT_PATHS = [
    "coiled-data-e4drr_202505.json",
]

# Default East Africa boundary
DEFAULT_BOUNDARY_PATH = "ea_ghcf_simple.geojson"

# CDI color scheme (standard drought classification)
CDI_COLORS = [
    (0.0, '#2166AC'),   # -3: Extremely wet
    (0.125, '#67A9CF'), # -2: Very wet
    (0.25, '#D1E5F0'),  # -1: Moderately wet
    (0.375, '#F7F7F7'), # 0: Normal
    (0.5, '#FDDBC7'),   # 1: Abnormally dry
    (0.625, '#F4A582'), # 2: Moderate drought
    (0.75, '#D6604D'),  # 3: Severe drought
    (0.875, '#B2182B'), # 4: Extreme drought
    (1.0, '#67001F'),   # 5: Exceptional drought
]


def find_service_account_file() -> Optional[str]:
    """Find an available service account file."""
    for path in DEFAULT_SERVICE_ACCOUNT_PATHS:
        if os.path.exists(path):
            return path
    return None


def get_cdi_colormap():
    """Create CDI-specific colormap."""
    colors = [c[1] for c in CDI_COLORS]
    positions = [c[0] for c in CDI_COLORS]
    return mcolors.LinearSegmentedColormap.from_list('cdi', list(zip(positions, colors)))


class CDIAnimator:
    """Create animated GIF from CDI data."""

    def __init__(self, service_account_file: str = None,
                 boundary_path: str = DEFAULT_BOUNDARY_PATH,
                 resolution: str = '10km'):
        """
        Initialize CDI Animator.

        Args:
            service_account_file: Path to GCS service account JSON
            boundary_path: Path to GeoJSON boundary file for overlay
            resolution: CDI resolution ('10km' or '1km')
        """
        if service_account_file is None:
            service_account_file = find_service_account_file()
        if service_account_file is None:
            raise ValueError("No service account file found. Please provide one.")

        self.service_account_file = service_account_file
        self.resolution = resolution
        self.fs = gcsfs.GCSFileSystem(token=service_account_file)
        logger.info(f"GCS filesystem initialized with {service_account_file}")

        # Load boundary
        self.boundary = None
        if boundary_path and os.path.exists(boundary_path):
            self.boundary = gpd.read_file(boundary_path)
            logger.info(f"Loaded boundary from {boundary_path}")
        else:
            logger.warning(f"Boundary file not found: {boundary_path}")

    def get_cdi_path(self, year: int, month: int) -> str:
        """Get GCS path for CDI data."""
        month_str = MONTH_MAP[month]
        if self.resolution == '10km':
            return f"{CDI_GCS_PATH}/icechunk_10km/{year}/eadw-cdi-data-{year}-{month_str}"
        else:
            return f"{CDI_GCS_PATH}/icechunk/{year}/eadw-cdi-data-{year}-{month_str}"

    def check_availability(self, year: int, month: int) -> bool:
        """Check if CDI data exists for year/month."""
        path = self.get_cdi_path(year, month)
        try:
            return self.fs.exists(f"{GCS_BUCKET}/{path}")
        except:
            return False

    def load_cdi(self, year: int, month: int) -> Optional[xr.Dataset]:
        """Load CDI dataset from GCS."""
        path = self.get_cdi_path(year, month)

        try:
            logger.info(f"Loading CDI from gs://{GCS_BUCKET}/{path}")

            if HAS_ICECHUNK:
                try:
                    storage = icechunk.gcs_storage(
                        bucket=GCS_BUCKET,
                        prefix=path,
                        service_account_file=self.service_account_file
                    )
                    repo = icechunk.Repository.open(storage=storage)
                    session = repo.readonly_session(branch="main")
                    store = session.store
                    ds = xr.open_zarr(store, consolidated=False)
                    return ds
                except Exception as e:
                    logger.warning(f"IceChunk load failed: {e}, trying regular zarr")

            # Fallback to regular zarr
            store = self.fs.get_mapper(f"{GCS_BUCKET}/{path}")
            ds = xr.open_zarr(store, consolidated=True)
            return ds

        except Exception as e:
            logger.error(f"Failed to load CDI {year}-{month:02d}: {e}")
            return None

    def generate_month_sequence(self, start_year: int, start_month: int,
                                 end_year: int, end_month: int) -> List[Tuple[int, int]]:
        """Generate list of (year, month) tuples from start to end."""
        months = []
        year, month = start_year, start_month

        while (year, month) <= (end_year, end_month):
            months.append((year, month))
            month += 1
            if month > 12:
                month = 1
                year += 1

        return months

    def create_frame(self, cdi_data: xr.DataArray, year: int, month: int,
                     vmin: float = -3, vmax: float = 5,
                     figsize: tuple = (12, 10)) -> np.ndarray:
        """
        Create a single frame for the animation.

        Args:
            cdi_data: CDI DataArray
            year: Year for title
            month: Month for title
            vmin: Minimum value for colorbar
            vmax: Maximum value for colorbar
            figsize: Figure size

        Returns:
            RGB array of the frame
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get coordinates
        if 'lat' in cdi_data.dims:
            lat = cdi_data.lat.values
            lon = cdi_data.lon.values
        elif 'latitude' in cdi_data.dims:
            lat = cdi_data.latitude.values
            lon = cdi_data.longitude.values
        else:
            lat = cdi_data.coords[list(cdi_data.dims)[0]].values
            lon = cdi_data.coords[list(cdi_data.dims)[1]].values

        # Plot CDI
        cmap = get_cdi_colormap()
        im = ax.pcolormesh(lon, lat, cdi_data.values,
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          shading='auto')

        # Add boundary overlay
        if self.boundary is not None:
            self.boundary.boundary.plot(ax=ax, color='black', linewidth=0.8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                           fraction=0.046, pad=0.04)
        cbar.set_label('Combined Drought Indicator (CDI)', fontsize=12)

        # Add legend for CDI categories
        legend_labels = [
            ('Extremely Wet', '#2166AC'),
            ('Very Wet', '#67A9CF'),
            ('Moderately Wet', '#D1E5F0'),
            ('Normal', '#F7F7F7'),
            ('Abnormally Dry', '#FDDBC7'),
            ('Moderate Drought', '#F4A582'),
            ('Severe Drought', '#D6604D'),
            ('Extreme Drought', '#B2182B'),
            ('Exceptional Drought', '#67001F'),
        ]

        # Title
        month_name = MONTH_NAMES[month]
        ax.set_title(f'Combined Drought Indicator (CDI) - {month_name} {year}',
                    fontsize=14, fontweight='bold')

        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)

        # Set extent based on data
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        # Convert to RGB array
        fig.canvas.draw()
        # Get the RGBA buffer and convert to RGB
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = buf[:, :, :3]  # Drop alpha channel

        plt.close(fig)
        return frame

    def create_animation(self, start_year: int, start_month: int,
                         end_year: int, end_month: int,
                         output_path: str,
                         fps: float = 1.0,
                         vmin: float = -3, vmax: float = 5,
                         figsize: tuple = (12, 10),
                         dpi: int = 100) -> str:
        """
        Create animated GIF of CDI data.

        Args:
            start_year: Start year
            start_month: Start month (1-12)
            end_year: End year
            end_month: End month (1-12)
            output_path: Output GIF file path
            fps: Frames per second (controls animation speed)
            vmin: Minimum CDI value for colorbar
            vmax: Maximum CDI value for colorbar
            figsize: Figure size
            dpi: DPI for output

        Returns:
            Path to output GIF
        """
        month_sequence = self.generate_month_sequence(start_year, start_month,
                                                       end_year, end_month)

        logger.info(f"Creating animation for {len(month_sequence)} months: "
                   f"{start_year}-{start_month:02d} to {end_year}-{end_month:02d}")

        frames = []
        available_months = []

        for year, month in month_sequence:
            ds = self.load_cdi(year, month)

            if ds is None:
                logger.warning(f"Skipping {year}-{month:02d}: data not available")
                continue

            # Get CDI variable (try common names)
            cdi_var = None
            for var_name in ['cdi', 'CDI', 'combined_drought_indicator']:
                if var_name in ds.data_vars:
                    cdi_var = ds[var_name]
                    break

            if cdi_var is None:
                # Take first variable if CDI not found
                cdi_var = ds[list(ds.data_vars)[0]]
                logger.info(f"Using variable: {list(ds.data_vars)[0]}")

            # Handle time dimension if present
            if 'time' in cdi_var.dims:
                cdi_var = cdi_var.isel(time=0)

            # Create frame
            logger.info(f"Creating frame for {year}-{month:02d}")
            frame = self.create_frame(cdi_var, year, month,
                                      vmin=vmin, vmax=vmax, figsize=figsize)
            frames.append(frame)
            available_months.append((year, month))

        if not frames:
            raise ValueError("No CDI data available for the specified time range")

        # Create GIF
        logger.info(f"Writing GIF with {len(frames)} frames to {output_path}")
        duration = 1.0 / fps  # seconds per frame
        imageio.mimsave(output_path, frames, duration=duration, loop=0)

        logger.info(f"Animation saved: {output_path}")
        logger.info(f"Included months: {len(available_months)}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create animated GIF of CDI data from GCS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create animation for all of 2024
    python create_cdi_animation.py -sy 2024 -sm 1 -ey 2024 -em 12 -o cdi_2024.gif

    # Create animation from June 2023 to June 2024
    python create_cdi_animation.py --start-year 2023 --start-month 6 \\
                                   --end-year 2024 --end-month 6 \\
                                   --output cdi_animation.gif --fps 0.5

    # Use custom boundary file
    python create_cdi_animation.py -sy 2024 -sm 1 -ey 2024 -em 6 \\
                                   --boundary /path/to/boundary.geojson \\
                                   -o output.gif
        """
    )

    parser.add_argument('-sy', '--start-year', type=int, required=True,
                        help='Start year')
    parser.add_argument('-sm', '--start-month', type=int, required=True,
                        help='Start month (1-12)')
    parser.add_argument('-ey', '--end-year', type=int, required=True,
                        help='End year')
    parser.add_argument('-em', '--end-month', type=int, required=True,
                        help='End month (1-12)')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output GIF file path')
    parser.add_argument('--boundary', type=str, default=DEFAULT_BOUNDARY_PATH,
                        help=f'GeoJSON boundary file for overlay (default: {DEFAULT_BOUNDARY_PATH})')
    parser.add_argument('--service-account', type=str, default=None,
                        help='GCS service account JSON file')
    parser.add_argument('--resolution', type=str, default='10km', choices=['10km', '1km'],
                        help='CDI resolution (default: 10km)')
    parser.add_argument('--fps', type=float, default=1.0,
                        help='Frames per second (default: 1.0, slower = longer display per frame)')
    parser.add_argument('--vmin', type=float, default=-3,
                        help='Minimum CDI value for colorbar (default: -3)')
    parser.add_argument('--vmax', type=float, default=5,
                        help='Maximum CDI value for colorbar (default: 5)')
    parser.add_argument('--figsize', type=str, default='12,10',
                        help='Figure size as width,height (default: 12,10)')

    args = parser.parse_args()

    # Validate inputs
    if not (1 <= args.start_month <= 12):
        parser.error("Start month must be between 1 and 12")
    if not (1 <= args.end_month <= 12):
        parser.error("End month must be between 1 and 12")
    if (args.start_year, args.start_month) > (args.end_year, args.end_month):
        parser.error("Start date must be before or equal to end date")

    # Parse figsize
    try:
        figsize = tuple(map(float, args.figsize.split(',')))
    except:
        parser.error("Invalid figsize format. Use: width,height (e.g., 12,10)")

    # Create animator and generate GIF
    try:
        animator = CDIAnimator(
            service_account_file=args.service_account,
            boundary_path=args.boundary,
            resolution=args.resolution
        )

        output_path = animator.create_animation(
            start_year=args.start_year,
            start_month=args.start_month,
            end_year=args.end_year,
            end_month=args.end_month,
            output_path=args.output,
            fps=args.fps,
            vmin=args.vmin,
            vmax=args.vmax,
            figsize=figsize
        )

        print(f"\nAnimation created successfully: {output_path}")

    except Exception as e:
        logger.error(f"Failed to create animation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
