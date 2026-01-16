"""
Download IMERG Daily Early precipitation data for the last 7 days.
Uses earthaccess for authentication and data access.
Credentials loaded from .env file using python-dotenv.
"""

import os
import earthaccess
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Load credentials from .env file ---
load_dotenv()

# Verify credentials are set
username = os.getenv("EARTHDATA_USERNAME")
password = os.getenv("EARTHDATA_PASSWORD")

if not username or not password:
    raise ValueError(
        "Missing Earthdata credentials!\n"
        "Create a .env file with:\n"
        "  EARTHDATA_USERNAME=your_username\n"
        "  EARTHDATA_PASSWORD=your_password"
    )

# Create .netrc file for download authentication
netrc_path = Path.home() / ".netrc"
netrc_content = f"machine urs.earthdata.nasa.gov login {username} password {password}"
netrc_path.write_text(netrc_content)
netrc_path.chmod(0o600)
print(f"Created/updated {netrc_path} for download authentication")

# --- Configuration ---
OUTPUT_DIR = Path("./imerg_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Spatial bounds: (west, south, east, north) - East Africa
BBOX = (33, -5, 42, 15)

# Last 7 days (accounting for ~4hr latency, use yesterday as end)
END_DATE = datetime.now() - timedelta(days=1)
START_DATE = END_DATE - timedelta(days=6)

print(f"Downloading IMERG Daily Early data")
print(f"Period: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
print(f"Bounding box: {BBOX}")
print("-" * 50)

# --- Authenticate ---
print("Authenticating with Earthdata...")
auth = earthaccess.login(strategy="environment")
print("Authentication successful!")

# --- Search for data ---
print("\nSearching for granules...")
results = earthaccess.search_data(
    short_name="GPM_3IMERGDE",  # Daily Early product
    version="07",
    temporal=(START_DATE.strftime('%Y-%m-%d'), END_DATE.strftime('%Y-%m-%d')),
    bounding_box=BBOX
)
print(f"Found {len(results)} granules")

if len(results) == 0:
    print("No granules found. Check date range and product availability.")
else:
    # --- Download ---
    print(f"\nDownloading to {OUTPUT_DIR}...")
    downloaded_files = earthaccess.download(
        results,
        local_path=str(OUTPUT_DIR)
    )
    print(f"Downloaded {len(downloaded_files)} files")

    # --- Verify with xarray ---
    print("\nOpening dataset with xarray...")
    ds = xr.open_mfdataset(downloaded_files, combine='by_coords')

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")

    # Show precipitation variable info
    if 'precipitation' in ds.data_vars:
        precip = ds['precipitation']
        print(f"\nPrecipitation stats:")
        print(f"  Shape: {precip.shape}")
        print(f"  Units: {precip.attrs.get('units', 'N/A')}")

    print("\n✓ Download and verification complete!")
