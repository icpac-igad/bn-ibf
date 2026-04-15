#!/usr/bin/env -S uv run --with requests --with pandas --with geopandas --with pyarrow --with overpy --with tqdm --with openpyxl
"""
collect_inform_osm.py
=====================
Collect two static exposure/vulnerability layers for the ICPAC East-Africa
region and write them as Parquet files ready for joining against admin-1
boundary-level BN risk outputs.

Layer 1 — INFORM Risk Index (admin-1 vulnerability)
----------------------------------------------------
  Source : INFORM Risk Index 2024, Joint Research Centre (JRC) / OCHA
  URL    : https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Risk
  Direct : https://drmkc.jrc.ec.europa.eu/Portals/0/InfoRM/2024/
           INFORM_Risk_2024_v069.xlsx  (updated annually)

  Columns kept per admin-1:
    iso3, adm1_name, country,
    inform_risk,         — composite risk score  0-10
    hazard_exposure,     — H&E component
    vulnerability,       — SOC + VUL components
    lack_of_coping,      — CC component
    year

  Output: inform_ea_adm1.parquet

Layer 2 — OSM Critical Infrastructure counts (admin-1)
-------------------------------------------------------
  Source : Overpass API (OpenStreetMap)
  Tags collected:
    hospitals / health facilities  (amenity=hospital | healthcare=*)
    schools / education            (amenity=school | building=school)
    markets / food access          (amenity=marketplace)
    road bridges                   (bridge=yes + highway=*)
    emergency shelters             (social_facility=shelter)

  Uses the ICPAC admin-1 GeoJSON bounding boxes to spatially filter; counts
  are then assigned to each admin-1 polygon.

  Output: osm_infra_ea_adm1.parquet

Usage
-----
    # Download INFORM only (no OSM)
    uv run collect_inform_osm.py --inform-only --adm1 ../flood_ibf/icpac_adm1v3.geojson

    # Full collection
    uv run collect_inform_osm.py --adm1 ../flood_ibf/icpac_adm1v3.geojson --out ./

    # Offline test (skip downloads, use cached XLSX)
    uv run collect_inform_osm.py --inform-xlsx ./INFORM_Risk_2024.xlsx \
        --adm1 ../flood_ibf/icpac_adm1v3.geojson
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ── ICPAC countries ───────────────────────────────────────────────────────────
EA_ISO3 = {"BDI", "DJI", "ERI", "ETH", "KEN", "RWA", "SOM", "SSD", "SDN", "TZA", "UGA"}

INFORM_YEAR = 2024

# INFORM download.  The JRC occasionally moves files; fall back to the
# previous year URL if the primary fails.
INFORM_URL_PRIMARY = (
    "https://drmkc.jrc.ec.europa.eu/Portals/0/InfoRM/2024/"
    "INFORM_Risk_2024_v069.xlsx"
)
INFORM_URL_FALLBACK = (
    "https://drmkc.jrc.ec.europa.eu/Portals/0/InfoRM/2023/"
    "INFORM_Risk_2023_v068.xlsx"
)

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM tags to count per admin-1
OSM_QUERIES: dict[str, str] = {
    "hospitals":      'node["amenity"="hospital"]',
    "health":         'node["healthcare"]',
    "schools":        'node["amenity"="school"]',
    "markets":        'node["amenity"="marketplace"]',
    "bridges":        'way["bridge"="yes"]["highway"]',
    "shelters":       'node["social_facility"="shelter"]',
}

# ── INFORM ────────────────────────────────────────────────────────────────────

def _try_download_xlsx(urls: list[str], dest: Path) -> Path | None:
    for url in urls:
        try:
            print(f"[inform] downloading {url}")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"[inform] saved to {dest}")
            return dest
        except Exception as exc:
            print(f"[inform] warn: {exc}")
    return None


def load_inform(xlsx_path: Path | None, cache_dir: Path) -> pd.DataFrame:
    """
    Parse INFORM Risk Index XLSX and return a DataFrame indexed by
    iso3 + adm1_name with relevant component scores.
    """
    if xlsx_path is None:
        dest = cache_dir / f"INFORM_Risk_{INFORM_YEAR}.xlsx"
        if not dest.exists():
            result = _try_download_xlsx(
                [INFORM_URL_PRIMARY, INFORM_URL_FALLBACK], dest
            )
            if result is None:
                raise RuntimeError(
                    "Could not download INFORM XLSX. "
                    "Provide --inform-xlsx <path> manually."
                )
        xlsx_path = dest

    print(f"[inform] reading {xlsx_path}")
    # The INFORM workbook has a sheet called "INFORM Risk 2024" or similar.
    xl = pd.ExcelFile(xlsx_path)
    sheet_name = next(
        (s for s in xl.sheet_names if "risk" in s.lower()),
        xl.sheet_names[0],
    )
    raw = xl.parse(sheet_name, header=1)  # row 0 is merged header group

    # Column names vary by year; normalise to lowercase-stripped
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    print(f"[inform] columns: {list(raw.columns[:20])}")

    # Identify key columns by partial match
    def _find(candidates: list[str]) -> str | None:
        for cand in candidates:
            for col in raw.columns:
                if cand in col:
                    return col
        return None

    col_iso3    = _find(["iso3", "iso"])
    col_country = _find(["country"])
    col_adm1    = _find(["admin1", "adm1", "subnational", "region"])
    col_risk    = _find(["inform_risk", "inform risk", "risk_score"])
    col_haz     = _find(["hazard", "h&e", "hazard_and"])
    col_vul     = _find(["vulnerability", "vul_score", "socio"])
    col_cc      = _find(["coping_capacity", "lack_of_coping", "cc_score"])

    if col_iso3 is None or col_risk is None:
        raise RuntimeError(
            f"Cannot find expected columns in {xlsx_path}. "
            f"Available: {list(raw.columns)}"
        )

    keep = {
        "iso3":           col_iso3,
        "country":        col_country,
        "adm1_name":      col_adm1,
        "inform_risk":    col_risk,
        "hazard_exposure": col_haz,
        "vulnerability":  col_vul,
        "lack_of_coping": col_cc,
    }
    # Drop None-valued columns
    keep = {k: v for k, v in keep.items() if v is not None}

    df = raw[list(keep.values())].copy()
    df.columns = list(keep.keys())

    # Filter to EA countries and admin-1 level rows
    df = df[df["iso3"].isin(EA_ISO3)].copy()

    # If the file has country-level and sub-national rows mixed, keep sub-national
    if "adm1_name" in df.columns:
        # Rows where adm1_name is NaN are country-level aggregates → keep as fallback
        df["adm1_name"] = df["adm1_name"].fillna("(national)")

    df["year"] = INFORM_YEAR
    # Numeric coercion
    for col in ["inform_risk", "hazard_exposure", "vulnerability", "lack_of_coping"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)
    print(f"[inform] {len(df)} rows for EA countries")
    return df


# ── OSM via Overpass ──────────────────────────────────────────────────────────

def _overpass_count(bbox: tuple[float, float, float, float],
                    tag_query: str, retries: int = 3) -> int:
    """
    Count OSM elements matching *tag_query* inside *bbox* (S, W, N, E).
    Uses `[out:count]` for efficiency.
    """
    s, w, n, e = bbox
    bbox_str = f"{s},{w},{n},{e}"
    ql = f"""
[out:json][timeout:30];
(
  {tag_query}({bbox_str});
);
out count;
"""
    for attempt in range(retries):
        try:
            r = requests.post(OVERPASS_URL, data={"data": ql}, timeout=60)
            r.raise_for_status()
            j = r.json()
            total = int(j["elements"][0]["tags"]["total"])
            return total
        except Exception as exc:
            wait = 5 * (attempt + 1)
            print(f"    [osm] retry {attempt+1}/{retries} in {wait}s ({exc})")
            time.sleep(wait)
    return -1   # flag as failed


def collect_osm(adm1: gpd.GeoDataFrame, tags: dict[str, str]) -> pd.DataFrame:
    """
    For each admin-1 polygon, query Overpass for each tag set and record counts.
    Returns a DataFrame with columns: id, name, country, <tag_name>_count, ...
    """
    rows = []
    for idx, row in tqdm(adm1.iterrows(), total=len(adm1), desc="OSM admin-1"):
        geom = row.geometry
        s, w, n, e = geom.bounds[1], geom.bounds[0], geom.bounds[3], geom.bounds[2]
        bbox = (s, w, n, e)
        rec: dict = {
            "id":      row.get("GID_1", str(idx)),
            "name":    row.get("NAME_1", ""),
            "country": row.get("GID_1", "")[:3] if "GID_1" in row else "",
        }
        for tag_name, tag_query in tags.items():
            count = _overpass_count(bbox, tag_query)
            rec[f"{tag_name}_count"] = count
            time.sleep(1.5)   # be polite to the public API
        rows.append(rec)
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    # Derive a simple infrastructure score (0-10 scale per capita — placeholder)
    infra_cols = [c for c in df.columns if c.endswith("_count")]
    df["total_infra_count"] = df[infra_cols].clip(lower=0).sum(axis=1)
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adm1", required=True,
                    help="Path to ICPAC admin-1 GeoJSON (icpac_adm1v3.geojson)")
    ap.add_argument("--out", default=".",
                    help="Directory for output Parquet files")
    ap.add_argument("--cache-dir", default="./inform_cache",
                    help="Cache directory for raw downloads")
    ap.add_argument("--inform-xlsx", default=None,
                    help="Path to existing INFORM XLSX (skip download)")
    ap.add_argument("--inform-only", action="store_true",
                    help="Skip OSM collection")
    ap.add_argument("--osm-only", action="store_true",
                    help="Skip INFORM collection")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    # Filter to EA countries
    adm1 = adm1[adm1["GID_1"].str[:3].isin(EA_ISO3)].reset_index(drop=True)
    print(f"[main] {len(adm1)} EA admin-1 boundaries loaded")

    # ── INFORM ────────────────────────────────────────────────────────────────
    if not args.osm_only:
        xlsx_path = Path(args.inform_xlsx) if args.inform_xlsx else None
        inform_df = load_inform(xlsx_path, cache_dir)
        inform_out = out_dir / "inform_ea_adm1.parquet"
        inform_df.to_parquet(inform_out, index=False)
        print(f"[main] wrote {inform_out}  ({len(inform_df)} rows)")

    # ── OSM ───────────────────────────────────────────────────────────────────
    if not args.inform_only:
        print("[main] querying Overpass API for critical infrastructure...")
        print("[main]   (this takes ~5-10 min for 120 boundaries, be patient)")
        osm_df = collect_osm(adm1, OSM_QUERIES)
        osm_out = out_dir / "osm_infra_ea_adm1.parquet"
        osm_df.to_parquet(osm_out, index=False)
        print(f"[main] wrote {osm_out}  ({len(osm_df)} rows)")

    print("[main] done.")


if __name__ == "__main__":
    main()
