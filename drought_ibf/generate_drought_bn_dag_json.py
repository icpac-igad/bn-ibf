#!/usr/bin/env -S uv run --with pandas python3
"""
generate_drought_bn_dag_json.py

For each (init-month, target-season) pair, merges the drought prep CSV
(soft-evidence probabilities) with the no-tail BN result CSV and emits one
JSON per init keyed by boundary_id with the 4-parent DAG structure:

  cur (current SPI-3)        5 states
  def (forecast deficit prob) 5 states
  spa (spatial coverage)      3 states
  trn (SPI-3 trend)           3 states
  risk (Minimal..Extreme)     5 states
  crma (Monitor / Evaluate / Assess / Actionable_Risk)

Usage:
    ./generate_drought_bn_dag_json.py --bn-dir output_v2_notail \\
        --prep-dir bn_inputs_v2 --out-dir output_v2_notail/bn-dag
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import pandas as pd

CUR_STATES = ["Above_Normal", "Normal", "Mild_Drought",
              "Moderate_Drought", "Severe_Drought"]
DEF_STATES = ["Very_Low", "Low", "Medium", "High", "Very_High"]
SPA_STATES = ["Localized", "Moderate", "Widespread"]
TRN_STATES = ["Improving", "Stable", "Deteriorating"]
RISK_STATES = ["Minimal", "Low", "Moderate", "High", "Extreme"]


def argmax_state(probs, states):
    return states[max(range(len(probs)), key=lambda i: probs[i])]


def round_probs(probs, n=6):
    return [round(p, n) for p in probs]


def raw_cur(row): v = row.get("current_spi3"); return f"{v:+.2f}" if pd.notna(v) else "N/A"
def raw_def(row): v = row.get("forecast_deficit_prob"); return f"P={v:.2f}" if pd.notna(v) else "N/A"
def raw_spa(row): v = row.get("spatial_coverage"); return f"{v*100:.0f}% cov" if pd.notna(v) else "N/A"
def raw_trn(row):
    s = row.get("trend_slope_spi_per_month")
    if pd.isna(s): return "N/A"
    sign = "+" if s >= 0 else ""
    return f"{sign}{s:.2f}/mo"


def row_to_dag(prep_row: pd.Series, bn_row: pd.Series, init_str: str) -> dict:
    cur_p = [prep_row[f"cur_p{i}"] for i in range(1, 6)]
    def_p = [prep_row[f"def_p{i}"] for i in range(1, 6)]
    spa_p = [prep_row[f"spa_p{i}"] for i in range(1, 4)]
    trn_p = [prep_row[f"trn_p{i}"] for i in range(1, 4)]

    risk_p = [bn_row["risk_minimal"], bn_row["risk_low"], bn_row["risk_moderate"],
              bn_row["risk_high"],   bn_row["risk_extreme"]]
    p_he = float(bn_row["risk_high"]) + float(bn_row["risk_extreme"])

    return {
        "boundary": str(bn_row["boundary_name"]),
        "init":     init_str,
        "cur":  {"state": argmax_state(cur_p, CUR_STATES), "probs": round_probs(cur_p), "raw": raw_cur(prep_row)},
        "def":  {"state": argmax_state(def_p, DEF_STATES), "probs": round_probs(def_p), "raw": raw_def(prep_row)},
        "spa":  {"state": argmax_state(spa_p, SPA_STATES), "probs": round_probs(spa_p), "raw": raw_spa(prep_row)},
        "trn":  {"state": argmax_state(trn_p, TRN_STATES), "probs": round_probs(trn_p), "raw": raw_trn(prep_row)},
        "risk": {"state": str(bn_row["risk_level"]), "probs": round_probs(risk_p)},
        "crma": {"state": str(bn_row["crma_state"]), "p_he": round(p_he, 6)},
    }


def process(init_str: str, prep_dir: str, bn_dir: str, out_dir: str):
    prep_glob = glob.glob(os.path.join(prep_dir, f"drought_inputs_{init_str}_*.csv"))
    bn_glob   = glob.glob(os.path.join(bn_dir,   f"drought_bn_v2_notail_{init_str}_*.csv"))
    if not prep_glob or not bn_glob:
        print(f"  SKIP {init_str}: prep={bool(prep_glob)} bn={bool(bn_glob)}")
        return
    prep = pd.read_csv(prep_glob[0]).set_index("id")
    bn   = pd.read_csv(bn_glob[0]).set_index("boundary_id")
    out = {}
    for bid in bn.index:
        if bid not in prep.index:
            continue
        out[bid] = row_to_dag(prep.loc[bid], bn.loc[bid], init_str)
    out_path = os.path.join(out_dir, f"drought-bn-dag-{init_str}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  {out_path}  ({len(out)} boundaries, {os.path.getsize(out_path)/1024:.1f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-dir", default="bn_inputs_v2")
    ap.add_argument("--bn-dir",   default="output_v2_notail")
    ap.add_argument("--out-dir",  default="output_v2_notail/bn-dag")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.prep_dir, "drought_inputs_*.csv")))
    inits = []
    for f in files:
        m = re.search(r"drought_inputs_(\d{4}-\d{2})_", os.path.basename(f))
        if m:
            inits.append(m.group(1))
    print(f"Processing {len(inits)} init-months → {args.out_dir}/")
    for i in inits:
        process(i, args.prep_dir, args.bn_dir, args.out_dir)


if __name__ == "__main__":
    main()
