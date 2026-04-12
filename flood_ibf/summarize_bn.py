#!/usr/bin/env -S uv run --with pandas --with numpy
"""
Aggregate daily Flood BN IBF v1 result CSVs into a single per-admin-1 summary.

For each admin-1 boundary, report across the 10-day horizon:
  - max_risk_level (worst level reached)
  - day_of_max   (first date it reached that level)
  - mean_confidence
  - peak action (matching day_of_max)

Usage: summarize_bn.py --input-dir output/ --dates 2026-03-01,2026-03-10 --out output/summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RISK_ORDER = {"Minimal": 0, "Low": 1, "Moderate": 2, "High": 3, "Extreme": 4}
RISK_INV = {v: k for k, v in RISK_ORDER.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="output")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-03-10")
    ap.add_argument("--out", default="output/flood_bn_v1_2026-03-01_to_10_summary.csv")
    args = ap.parse_args()

    dates = pd.date_range(args.start, args.end, freq="D")
    frames = []
    missing = []
    for d in dates:
        p = Path(args.input_dir) / f"flood_bn_v1_{d.date()}.csv"
        if not p.exists():
            missing.append(str(p))
            continue
        df = pd.read_csv(p)
        df["target_date"] = str(d.date())
        frames.append(df)
    if missing:
        print(f"[summary] WARNING: missing {len(missing)} daily CSVs: {missing}")
    if not frames:
        raise SystemExit("[summary] no daily CSVs found — nothing to aggregate")

    longdf = pd.concat(frames, ignore_index=True)
    longdf["risk_rank"] = longdf["risk_level"].map(RISK_ORDER)

    agg_rows = []
    for bid, g in longdf.groupby("boundary_id", sort=False):
        g = g.sort_values("target_date")
        max_rank = int(g["risk_rank"].max())
        first_max = g.loc[g["risk_rank"] == max_rank].iloc[0]
        agg_rows.append({
            "boundary_id":   bid,
            "boundary_name": first_max["boundary_name"],
            "country":       first_max["country"],
            "max_risk_level": RISK_INV[max_rank],
            "day_of_max":     first_max["target_date"],
            "action_at_peak": first_max["recommended_action"],
            "mean_confidence": round(float(g["confidence"].mean()), 3),
            "days_at_alert_or_worse": int((g["risk_rank"] >= 1).sum()),
            "days_at_moderate_or_worse": int((g["risk_rank"] >= 2).sum()),
        })
    summary = pd.DataFrame(agg_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)

    # Human summary
    print(f"[summary] aggregated {len(longdf)} rows across {len(frames)} days → {out}")
    print()
    print("Max risk distribution over 10 days:")
    dist = summary["max_risk_level"].value_counts().reindex(list(RISK_ORDER.keys())).fillna(0).astype(int)
    for k, v in dist.items():
        bar = "█" * int(v * 60 / max(dist.max(), 1))
        print(f"  {k:10} {v:4d}  {bar}")
    print()
    moderate_plus = summary[summary["max_risk_level"].isin(["Moderate", "High", "Extreme"])]
    if len(moderate_plus):
        print(f"Boundaries at Moderate+ peak ({len(moderate_plus)}):")
        cols = ["country", "boundary_name", "max_risk_level", "day_of_max",
                "action_at_peak", "mean_confidence"]
        print(moderate_plus[cols].to_string(index=False))


if __name__ == "__main__":
    main()
