#!/usr/bin/env python3
"""
generate_bn_dag_json.py
For each date, merges flood_inputs_YYYY-MM-DD_soft.csv (evidence probabilities)
with flood_bn_v1_YYYY-MM-DD.csv (BN posteriors) and emits one JSON file per day:
  output/bn-dag/bn-dag-YYYY-MM-DD.json

Each JSON is a dict keyed by boundary_id:
{
  "KEN.30_1": {
    "boundary": "Nairobi", "date": "2026-03-04",
    "ant":  {"state": "Very_Wet", "probs": [p1,p2,p3,p4,p5], "raw": "83.6 mm/7d"},
    "exc":  {"state": "Very_Low", "probs": [p1,p2,p3,p4,p5], "raw": "P=0.010"},
    "spa":  {"state": "Moderate", "probs": [p1,p2,p3],       "raw": "50% hotspot"},
    "trn":  {"state": "Decreasing","probs":[p1,p2,p3],       "raw": "-2.4 mm/d"},
    "tail": {"state": "Moderate", "probs": [p1,p2,p3,p4],    "raw": "1.11× RP"},
    "risk": {"probs": [p1,p2,p3,p4,p5], "state": "Moderate"},
    "crma": {"state": "Assess", "p_he": 0.100}
  }, ...
}

Usage:
  uv run python3 generate_bn_dag_json.py [--input-dir bn_inputs] [--dbn-dir output/dbn] [--out-dir output/bn-dag]
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

ANT_STATES  = ["Dry", "Normal", "Wet", "Very_Wet", "Saturated"]
EXC_STATES  = ["Very_Low", "Low", "Moderate", "High", "Very_High"]
SPA_STATES  = ["Local", "Moderate", "Wide"]
TRN_STATES  = ["Decreasing", "Stable", "Increasing"]
TAIL_STATES = ["Nil", "Low", "Moderate", "High"]
RISK_STATES = ["Minimal", "Low", "Moderate", "High", "Extreme"]


def argmax_state(probs: list[float], states: list[str]) -> str:
    return states[max(range(len(probs)), key=lambda i: probs[i])]


def round_probs(probs: list[float]) -> list[float]:
    return [round(p, 6) for p in probs]


def build_raw_ant(row: pd.Series) -> str:
    mm = row.get("antecedent_rainfall_mm", float("nan"))
    if pd.isna(mm):
        return "N/A"
    return f"{mm:.1f} mm/7d"


def build_raw_exc(row: pd.Series) -> str:
    p = row.get("ecmwf_eprob_heavy", float("nan"))
    if pd.isna(p):
        return "N/A"
    return f"P={p:.3f}"


def build_raw_spa(row: pd.Series) -> str:
    hf = row.get("hotspot_fraction", float("nan"))
    if pd.isna(hf):
        return "N/A"
    return f"{hf*100:.0f}% hotspot"


def build_raw_trn(row: pd.Series) -> str:
    slope = row.get("trend_slope_mm_per_day", float("nan"))
    if pd.isna(slope):
        return "N/A"
    sign = "+" if slope >= 0 else ""
    return f"{sign}{slope:.1f} mm/d"


def build_raw_tail(row: pd.Series) -> str:
    ratio = row.get("ens_max_ratio_peak", float("nan"))
    if pd.isna(ratio):
        return "N/A"
    return f"{ratio:.2f}× RP"


def row_to_dag(soft_row: pd.Series, dbn_row: pd.Series) -> dict:
    ant_probs  = [soft_row[f"ant_p{i}"] for i in range(1, 6)]
    exc_probs  = [soft_row[f"exc_p{i}"] for i in range(1, 6)]
    spa_probs  = [soft_row[f"spa_p{i}"] for i in range(1, 4)]
    trn_probs  = [soft_row[f"trn_p{i}"] for i in range(1, 4)]
    tail_probs = [soft_row[f"tail_p{i}"] for i in range(1, 5)]

    risk_probs = [
        dbn_row["risk_minimal"],
        dbn_row["risk_low"],
        dbn_row["risk_moderate"],
        dbn_row["risk_high"],
        dbn_row["risk_extreme"],
    ]

    return {
        "boundary": dbn_row["boundary_name"],
        "date": str(dbn_row["target_date"])[:10],
        "ant": {
            "state": argmax_state(ant_probs, ANT_STATES),
            "probs": round_probs(ant_probs),
            "raw": build_raw_ant(soft_row),
        },
        "exc": {
            "state": argmax_state(exc_probs, EXC_STATES),
            "probs": round_probs(exc_probs),
            "raw": build_raw_exc(soft_row),
        },
        "spa": {
            "state": argmax_state(spa_probs, SPA_STATES),
            "probs": round_probs(spa_probs),
            "raw": build_raw_spa(soft_row),
        },
        "trn": {
            "state": argmax_state(trn_probs, TRN_STATES),
            "probs": round_probs(trn_probs),
            "raw": build_raw_trn(soft_row),
        },
        "tail": {
            "state": argmax_state(tail_probs, TAIL_STATES),
            "probs": round_probs(tail_probs),
            "raw": build_raw_tail(soft_row),
        },
        "risk": {
            "probs": round_probs(risk_probs),
            "state": str(dbn_row["risk_level"]),
        },
        "crma": {
            "state": str(dbn_row["crma_state"]),
            "p_he": round(float(dbn_row["p_high_extreme"]), 6),
        },
    }


def process_date(date_str: str, input_dir: str, dbn_dir: str, out_dir: str):
    soft_path = os.path.join(input_dir, f"flood_inputs_{date_str}_soft.csv")
    dbn_path  = os.path.join(dbn_dir,   f"flood_bn_v1_{date_str}.csv")

    if not os.path.isfile(soft_path):
        print(f"  SKIP {date_str}: soft input not found")
        return
    if not os.path.isfile(dbn_path):
        print(f"  SKIP {date_str}: dbn output not found")
        return

    soft = pd.read_csv(soft_path).set_index("id")
    dbn  = pd.read_csv(dbn_path).set_index("boundary_id")

    result = {}
    for bid in dbn.index:
        if bid not in soft.index:
            continue
        result[bid] = row_to_dag(soft.loc[bid], dbn.loc[bid])

    out_path = os.path.join(out_dir, f"bn-dag-{date_str}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  {out_path}  ({len(result)} boundaries, {size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="bn_inputs")
    parser.add_argument("--dbn-dir",   default="output/dbn")
    parser.add_argument("--out-dir",   default="output/bn-dag")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    soft_files = sorted(glob.glob(os.path.join(args.input_dir, "flood_inputs_*_soft.csv")))
    dates = []
    for f in soft_files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f))
        if m:
            dates.append(m.group(1))

    print(f"Processing {len(dates)} dates → {args.out_dir}/")
    for d in dates:
        process_date(d, args.input_dir, args.dbn_dir, args.out_dir)


if __name__ == "__main__":
    main()
