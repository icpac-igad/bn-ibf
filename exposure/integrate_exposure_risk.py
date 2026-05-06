#!/usr/bin/env -S uv run --with icechunk --with xarray --with "zarr>=3" --with numpy --with pandas --with geopandas --with regionmask --with pyarrow --with scipy
"""
integrate_exposure_risk.py
==========================
Join exposure × vulnerability layers against the BN flood-risk CSV outputs
to produce a composite impact-risk score per admin-1 boundary per day.

Inputs
------
  --risk-csv      flood_bn_v1_YYYY-MM-DD.csv    (one per day, or a glob)
  --worldpop      worldpop_ea_icechunk/          (icechunk zarr store)
  --inform        inform_ea_adm1.parquet
  --osm           osm_infra_ea_adm1.parquet      (optional)
  --adm1          icpac_adm1v3.geojson

Output
------
  flood_bn_v1_impact_YYYY-MM-DD.csv   — one row per admin-1 with columns:
    id, name, country,
    flood_risk_prob,       — BN posterior P(flood)
    population,            — WorldPop count in boundary (millions)
    pop_at_risk,           — population × flood_risk_prob
    inform_risk,           — INFORM composite score
    vulnerability,         — INFORM vulnerability component
    lack_of_coping,        — INFORM coping capacity deficit
    hospitals_count,       — OSM infra counts (if available)
    schools_count,
    total_infra_count,
    impact_score,          — composite: risk × exposure × vulnerability (0-10)
    impact_tier,           — Critical / High / Moderate / Low / Negligible
    target_date

The impact_score formula (following INFORM structure):
    impact_score = (flood_risk_prob^(1/3)) * 10
                   × (vulnerability/10)
                   × (lack_of_coping/10)
                   scaled to 0-10

This follows the INFORM composite index logic:
    RISK = HAZARD × VULNERABILITY × LACK_OF_COPING_CAPACITY

Usage
-----
    # Single day
    uv run integrate_exposure_risk.py \\
        --risk-csv ../flood_ibf/output/flood_bn_v1_2026-03-01.csv \\
        --worldpop ./worldpop_ea_icechunk \\
        --inform   ./inform_ea_adm1.parquet \\
        --osm      ./osm_infra_ea_adm1.parquet \\
        --adm1     ../flood_ibf/icpac_adm1v3.geojson \\
        --out      ./output

    # Batch: all days in output folder
    uv run integrate_exposure_risk.py \\
        --risk-csv "../flood_ibf/output/flood_bn_v1_2026-03-*.csv" \\
        --worldpop ./worldpop_ea_icechunk \\
        --inform   ./inform_ea_adm1.parquet \\
        --adm1     ../flood_ibf/icpac_adm1v3.geojson \\
        --out      ./output
"""
from __future__ import annotations

import argparse
import glob as _glob
import warnings
from pathlib import Path

import geopandas as gpd
import icechunk as ic
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
import zarr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Impact thresholds (0-10 scale) ──────────────────────────────────────────
IMPACT_TIERS = [
    (7.5, "Critical"),
    (5.0, "High"),
    (3.0, "Moderate"),
    (1.5, "Low"),
    (0.0, "Negligible"),
]

EA_ISO3 = {"BDI", "DJI", "ERI", "ETH", "KEN", "RWA", "SOM", "SSD", "SDN", "TZA", "UGA"}


def impact_tier(score: float) -> str:
    for threshold, label in IMPACT_TIERS:
        if score >= threshold:
            return label
    return "Negligible"


# ── WorldPop loader ──────────────────────────────────────────────────────────

def load_worldpop_for_adm1(store_dir: Path, adm1: gpd.GeoDataFrame,
                            year: int = 2020) -> pd.Series:
    """
    Read the WorldPop icechunk store and compute total population per admin-1
    boundary using area-weighted zonal sum.

    Returns a pd.Series indexed by GID_1.
    """
    storage = ic.local_filesystem_storage(str(store_dir))
    try:
        repo = ic.Repository.open(storage)
    except Exception as exc:
        raise FileNotFoundError(
            f"WorldPop store not found at {store_dir}. "
            f"Run collect_worldpop_arco.py first. ({exc})"
        )
    ds = xr.open_zarr(
        repo.readonly_session("main").store,
        consolidated=False,
    )

    years_in_store = list(ds["year"].values)
    if year not in years_in_store:
        year = int(years_in_store[-1])
        print(f"[wp] year not found, using {year}")

    pop = ds["population"].sel(year=year)  # (lat, lon)
    lat = pop.lat.values.astype("float64")
    lon = pop.lon.values.astype("float64")

    # Build regionmask
    regions = regionmask.Regions(
        outlines=list(adm1.geometry),
        numbers=list(range(len(adm1))),
        names=list(adm1["NAME_1"]),
        abbrevs=list(adm1["GID_1"]),
        name="adm1",
    )
    mask = regions.mask(lon, lat)
    mask_vals = mask.values
    pop_vals = pop.values  # (lat, lon)

    result = {}
    for r_idx, gid in enumerate(adm1["GID_1"]):
        sel = mask_vals == r_idx
        if sel.any():
            vals = pop_vals[sel]
            vals = vals[np.isfinite(vals)]
            result[gid] = float(np.nansum(vals)) if len(vals) > 0 else np.nan
        else:
            # Fallback: sample centroid pixel
            pt = adm1.iloc[r_idx].geometry.centroid
            try:
                v = float(pop.sel(lat=pt.y, lon=pt.x, method="nearest").values)
            except Exception:
                v = np.nan
            result[gid] = v

    return pd.Series(result, name="population")


# ── INFORM loader ────────────────────────────────────────────────────────────

def load_inform(parquet_path: Path, adm1: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Read the INFORM parquet and match rows to GID_1 using ISO3 code.
    For admin-1 rows present, match by normalised name. For missing sub-national
    records, fall back to the country-level national aggregate row.
    """
    df = pd.read_parquet(parquet_path)
    df["iso3"] = df["iso3"].str.strip().str.upper()

    # Build a lookup: iso3 → (adm1_name → row)
    by_iso3 = {}
    for iso3, grp in df.groupby("iso3"):
        by_iso3[iso3] = grp

    rows = []
    for _, boundary in adm1.iterrows():
        gid = boundary["GID_1"]
        iso3 = gid.split(".")[0]
        name = boundary.get("NAME_1", "")

        if iso3 not in by_iso3:
            rows.append({"id": gid, "inform_risk": np.nan,
                         "vulnerability": np.nan, "lack_of_coping": np.nan,
                         "hazard_exposure": np.nan})
            continue

        grp = by_iso3[iso3]

        # Try to match admin-1 name (normalised)
        if "adm1_name" in grp.columns:
            norm_name = str(name).lower().strip()
            matched = grp[grp["adm1_name"].str.lower().str.strip() == norm_name]
        else:
            matched = pd.DataFrame()

        if len(matched) == 0:
            # Fall back to national row
            nat = grp[grp.get("adm1_name", pd.Series(["(national)"])) == "(national)"]
            matched = nat if len(nat) > 0 else grp.head(1)

        r = matched.iloc[0]
        rows.append({
            "id":              gid,
            "inform_risk":     float(r.get("inform_risk", np.nan)),
            "hazard_exposure": float(r.get("hazard_exposure", np.nan)),
            "vulnerability":   float(r.get("vulnerability", np.nan)),
            "lack_of_coping":  float(r.get("lack_of_coping", np.nan)),
        })

    return pd.DataFrame(rows)


# ── OSM loader ───────────────────────────────────────────────────────────────

def load_osm(parquet_path: Path | None) -> pd.DataFrame | None:
    if parquet_path is None or not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    return df


# ── Impact score ─────────────────────────────────────────────────────────────

def compute_impact(
    risk_df: pd.DataFrame,
    pop_series: pd.Series,
    inform_df: pd.DataFrame,
    osm_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge all layers and compute the composite impact score.

    Impact formula (INFORM-aligned, 0-10 scale):
        impact_score = 10 × H^(1/3) × V/10 × (1 - CC/10)

    where:
        H  = flood_risk_prob (BN posterior P(Flood=High|evidence))
        V  = INFORM vulnerability score (0-10)
        CC = INFORM coping capacity (10-lack_of_coping, so inverted here)
    """
    df = risk_df.copy()

    # ── merge population ────────────────────────────────────────────────────
    df["population"] = df["id"].map(pop_series)

    # ── merge INFORM ─────────────────────────────────────────────────────────
    df = df.merge(inform_df, on="id", how="left")

    # ── merge OSM ────────────────────────────────────────────────────────────
    if osm_df is not None:
        osm_cols = ["id"] + [c for c in osm_df.columns if c != "id"]
        df = df.merge(osm_df[osm_cols], on="id", how="left")

    # ── derive flood risk probability from BN output ──────────────────────
    # The BN CSV has ens_max_ratio as a proxy for tail risk signal;
    # flood_risk_prob = P(Flood ≥ High) is already computed by flood_bn_ibf_v1.jl
    # and stored in the CSV.  If not present, proxy from ecmwf_eprob_heavy.
    if "flood_risk_prob" not in df.columns:
        if "p_flood_high" in df.columns:
            df["flood_risk_prob"] = df["p_flood_high"]
        elif "ecmwf_eprob_heavy" in df.columns:
            df["flood_risk_prob"] = df["ecmwf_eprob_heavy"]
        else:
            df["flood_risk_prob"] = np.nan

    # ── population at risk ────────────────────────────────────────────────
    df["pop_at_risk"] = df["population"] * df["flood_risk_prob"]

    # ── INFORM-style composite impact score ───────────────────────────────
    H  = df["flood_risk_prob"].clip(0, 1).fillna(0).values
    V  = df["vulnerability"].clip(0, 10).fillna(5).values / 10.0
    CC = df["lack_of_coping"].clip(0, 10).fillna(5).values / 10.0

    # H: cube-root to compress the probability range (INFORM convention)
    impact_raw = 10.0 * (H ** (1.0 / 3.0)) * V * CC
    impact_raw = np.clip(impact_raw, 0, 10)
    df["impact_score"] = np.round(impact_raw, 3)
    df["impact_tier"]  = [impact_tier(s) for s in impact_raw]

    # ── optional: infrastructure vulnerability multiplier ─────────────────
    if osm_df is not None and "total_infra_count" in df.columns:
        # Regions with very little infrastructure are more vulnerable.
        # Normalise infra per 100k pop and create a deficit score.
        pop100k = (df["population"].fillna(1) / 1e5).clip(lower=0.1)
        infra_density = df["total_infra_count"].fillna(0) / pop100k
        # Max-normalise across EA
        max_density = infra_density.max() if infra_density.max() > 0 else 1.0
        infra_deficit = 1.0 - (infra_density / max_density).clip(0, 1)
        # Blend: 80% base + 20% infrastructure deficit
        df["impact_score_infra"] = np.round(
            (0.8 * impact_raw + 0.2 * infra_deficit * 10).clip(0, 10), 3
        )
        df["impact_tier_infra"] = [
            impact_tier(s) for s in df["impact_score_infra"].values
        ]

    return df


# ── process one CSV ──────────────────────────────────────────────────────────

def process_risk_csv(
    csv_path: Path,
    pop_series: pd.Series,
    inform_df: pd.DataFrame,
    osm_df: pd.DataFrame | None,
    out_dir: Path,
) -> None:
    risk_df = pd.read_csv(csv_path)
    target_date = risk_df["target_date"].iloc[0] if "target_date" in risk_df.columns else csv_path.stem

    print(f"[integrate] {csv_path.name}  boundaries={len(risk_df)}")

    merged = compute_impact(risk_df, pop_series, inform_df, osm_df)

    out_path = out_dir / f"flood_bn_v1_impact_{target_date}.csv"
    merged.to_csv(out_path, index=False)

    # Summary stats
    n_high = (merged["impact_tier"].isin(["Critical", "High"])).sum()
    top5 = merged.nlargest(5, "impact_score")[["name", "country", "impact_score", "impact_tier"]]
    print(f"  → {out_path.name}")
    print(f"     Critical/High: {n_high}  |  "
          f"pop_at_risk: {merged['pop_at_risk'].sum() / 1e6:.2f}M")
    print(f"     Top 5 by impact:\n{top5.to_string(index=False)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--risk-csv", required=True,
                    help="Path or glob to flood BN risk CSV(s)")
    ap.add_argument("--worldpop", default="./worldpop_ea_icechunk",
                    help="Path to WorldPop icechunk store")
    ap.add_argument("--inform", required=True,
                    help="Path to inform_ea_adm1.parquet")
    ap.add_argument("--osm", default=None,
                    help="Path to osm_infra_ea_adm1.parquet (optional)")
    ap.add_argument("--adm1", required=True,
                    help="Path to ICPAC admin-1 GeoJSON")
    ap.add_argument("--worldpop-year", type=int, default=2020,
                    help="WorldPop year to use (default 2020)")
    ap.add_argument("--out", default="./output",
                    help="Output directory for impact CSVs")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve CSV glob
    csv_paths = sorted(Path(p) for p in _glob.glob(args.risk_csv))
    if not csv_paths:
        raise SystemExit(f"No CSVs matched: {args.risk_csv}")
    print(f"[main] {len(csv_paths)} risk CSV(s) to process")

    # Load admin-1 boundaries
    adm1 = gpd.read_file(args.adm1).reset_index(drop=True)
    adm1 = adm1[adm1["GID_1"].str[:3].isin(EA_ISO3)].reset_index(drop=True)
    print(f"[main] {len(adm1)} EA admin-1 boundaries")

    # Load WorldPop zonal sums (done once, shared across all days)
    print("[main] loading WorldPop zonal sums...")
    pop_series = load_worldpop_for_adm1(
        Path(args.worldpop), adm1, year=args.worldpop_year
    )
    total_pop = pop_series.sum()
    print(f"[main] total EA population: {total_pop / 1e6:.1f}M")

    # Load INFORM
    inform_df = load_inform(Path(args.inform), adm1)

    # Load OSM (optional)
    osm_df = load_osm(Path(args.osm) if args.osm else None)

    # Process each risk CSV
    for csv_path in csv_paths:
        process_risk_csv(csv_path, pop_series, inform_df, osm_df, out_dir)

    print("\n[main] all done.")


if __name__ == "__main__":
    main()
