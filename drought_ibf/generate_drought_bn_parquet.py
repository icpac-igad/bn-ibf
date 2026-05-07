#!/usr/bin/env python3
"""
generate_drought_bn_parquet.py

Reads all `drought_bn_v2_notail_cdi_<init>_<season>.csv` files from
`output_v2_notail_cdi/` (the post-CDI 4-parent BN that drives the paper's
choropleth panels) and writes two parquet files to `--out-dir`:

  - drought_bn_ibf_monthly.parquet
        One row per init-month (YYYY-MM). Carries the per-init severity
        summary so the frontend's DisasterCalendar can colour each month
        cell by peak admin-1 risk and the per-CRMA boundary counts.

  - drought_bn_ibf_boundary_monthly.parquet
        One row per (init-month, admin-1 boundary). Carries the per-
        boundary post-CDI CRMA + risk + pre-CDI fields so the frontend's
        DisasterMap can colour the choropleth and pop a tooltip with
        post-CDI vs pre-CDI deltas.

Init-month convention: each init maps to exactly one target season per
the SEASON_INIT_LEAD table in drought_data_prep.py. The init string
(YYYY-MM) is the unique lookup key used by the frontend / API.

`event_key` matches the convention the API expects: `dr-<YYYY-MM>`
(no stage infix; drought is monthly and only the risk-decisions stage
consumes the BN output today).

Usage:
  uv run python3 generate_drought_bn_parquet.py
      [--input-dir output_v2_notail_cdi]
      [--prefix drought_bn_v2_notail_cdi_]
      [--out-dir .]
"""
import argparse
import glob
import os
import re

import pandas as pd

RISK_LEVEL_INT = {
    "Minimal": 1,
    "Low": 2,
    "Moderate": 3,
    "High": 4,
    "Extreme": 5,
}


def load_all(input_dir: str, prefix: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, f"{prefix}*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    frames = []
    for f in files:
        # Filename: <prefix><YYYY>-<MM>_<SEASON>.csv
        m = re.search(rf"{re.escape(prefix)}(\d{{4}}-\d{{2}})_([A-Z]{{3}})\.csv$",
                      os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        df["init_month"] = m.group(1)        # "YYYY-MM"
        df["target_season"] = m.group(2)     # "MAM" / "JJA" / "OND" / "DJF"
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No files in {pattern} matched the <init>_<season> pattern")
    return pd.concat(frames, ignore_index=True)


def make_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (init_month, target_season). Drives the calendar heatmap."""
    df = df.copy()
    df["risk_level_int"] = df["risk_level"].map(RISK_LEVEL_INT).fillna(1).astype(int)

    rows = []
    for (init_month, season), grp in df.groupby(["init_month", "target_season"]):
        counts = grp["crma_state"].value_counts()
        max_risk = grp["risk_level_int"].max()
        year, month = init_month.split("-")
        rows.append({
            "year":  int(year),
            "month": int(month),
            "init_month":    init_month,                 # "YYYY-MM"
            "target_season": season,                     # "MAM" / "JJA" / "OND" / "DJF"
            "event_key":     f"dr-{init_month}",         # frontend lookup key
            "level":         int(max_risk),              # 1..5 for D3 heatmap
            "n_monitor":         int(counts.get("Monitor", 0)),
            "n_evaluate":        int(counts.get("Evaluate", 0)),
            "n_assess":          int(counts.get("Assess", 0)),
            "n_actionable_risk": int(counts.get("Actionable_Risk", 0)),
        })
    return pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)


def make_boundary_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (init_month, boundary). Drives the choropleth + tooltip."""
    df = df.copy()
    df["risk_level_int"] = df["risk_level"].map(RISK_LEVEL_INT).fillna(1).astype(int)

    # Post-CDI fields; pre_cdi columns are present only on the v2_notail_cdi
    # output, which is what we use by default. We carry them through if
    # available so the front-end can show the CDI delta in a tooltip.
    base_keep = [
        "init_month", "target_season",
        "boundary_id", "boundary_name", "country",
        "current_spi3_category", "spi3_trend",
        "risk_level", "risk_level_int", "crma_state", "traffic_light",
        "risk_minimal", "risk_low", "risk_moderate", "risk_high", "risk_extreme",
    ]
    pre_cdi_keep = [
        "risk_minimal_pre_cdi", "risk_low_pre_cdi", "risk_moderate_pre_cdi",
        "risk_high_pre_cdi", "risk_extreme_pre_cdi",
        "crma_state_pre_cdi", "traffic_light_pre_cdi",
        "cdi_level_recomp", "cdi_class_recomp",
        "cdi_level_eadw", "cdi_class_eadw", "cdi_agreement",
    ]
    keep = base_keep + [c for c in pre_cdi_keep if c in df.columns]
    out = df[keep].copy()

    # Match the flood schema field names so DisasterMap can render the
    # choropleth without any drought-specific code path.
    out["event_key"]  = out["init_month"].map(lambda s: f"dr-{s}")
    out["shapeID"]    = out["boundary_id"]
    out["shapeName"]  = out["boundary_name"]
    out["shapeGroup"] = out["country"]

    # p_high_extreme so the API endpoint can return it without recomputing
    out["p_high_extreme"] = (out["risk_high"] + out["risk_extreme"]).round(6)

    return out.sort_values(["init_month", "boundary_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="output_v2_notail_cdi",
                    help="Dir of drought_bn_v2_notail_cdi_<init>_<season>.csv files. "
                         "Default matches the post-CDI 4-parent BN run.")
    ap.add_argument("--prefix",    default="drought_bn_v2_notail_cdi_",
                    help="Filename prefix in --input-dir.")
    ap.add_argument("--out-dir",   default=".",
                    help="Where to write the two parquet files.")
    args = ap.parse_args()

    print(f"Reading from: {args.input_dir} (prefix={args.prefix!r})")
    df = load_all(args.input_dir, args.prefix)
    print(f"  Loaded {len(df)} rows across {df['init_month'].nunique()} init-months")

    monthly = make_monthly(df)
    boundary_monthly = make_boundary_monthly(df)

    monthly_path  = os.path.join(args.out_dir, "drought_bn_ibf_monthly.parquet")
    boundary_path = os.path.join(args.out_dir, "drought_bn_ibf_boundary_monthly.parquet")

    monthly.to_parquet(monthly_path, index=False)
    boundary_monthly.to_parquet(boundary_path, index=False)

    print(f"  Wrote {monthly_path} ({len(monthly)} rows)")
    print(f"  Wrote {boundary_path} ({len(boundary_monthly)} rows)")

    print("\nMonthly preview:")
    print(monthly.to_string(index=False))


if __name__ == "__main__":
    main()
