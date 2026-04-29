#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""
CDI evidence update — apply the JRC CDI observation as a likelihood update
on the BN risk posterior.

Implements the noisy-channel structure:

    parents (5) ──> risk_level ──> cdi_obs (observed)

with P(cdi | risk_level) encoded as the column-stochastic matrix L below.
The BN's risk posterior P(risk | parents) gets multiplied by L[cdi, :]
and renormalised, giving P(risk | parents, cdi). The CRMA state and
traffic-light columns are recomputed from the updated posterior.

Why a post-hoc update rather than a 6th parent in the BN:

  - Mathematically equivalent to "cdi_obs as a child of risk_level
    with measurement-noise CPT L" — the Bayesian network factorises
    into the 5-parent inference (already done) × the L update.
  - Keeps the existing RxInfer 5-parent @model untouched, so the
    rest of the pipeline (DBN temporal coupling, per-member
    storyline, CRMA decision rule) doesn't need re-validation.
  - Makes the CDI contribution auditable: the BN result CSV gains
    `risk_*_pre_cdi` columns alongside the updated posteriors,
    so an operator can see exactly how the CDI moved the answer.

Usage:
    uv run cdi_evidence_update.py \\
        --bn-csv  /tmp/drought_bn_v1_2026-04.csv \\
        --cdi-csv /tmp/cdi_inputs_2026-04.csv \\
        --out     /tmp/drought_bn_v1_cdi_2026-04.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# CDI level → row index (must match cdi_data_prep.py LEVEL_TO_IDX
# minus 1 to convert 1..6 to 0..5):
LEVEL_NAMES = ["No_drought", "Full_recovery", "Partial_recovery",
               "Watch", "Warning", "Alert"]
RISK_NAMES = ["risk_minimal", "risk_low", "risk_moderate",
              "risk_high", "risk_extreme"]


# Likelihood matrix L[cdi_level, risk_level], 6 × 5.
# Each column is P(cdi | risk = r) for fixed r — column-stochastic.
# Calibration: rows ordered No_drought (least stressed) → Alert (most),
# columns Minimal → Extreme. The diagonal is the dominant trend
# (high risk should typically show Alert; low risk should show
# No_drought) but with realistic measurement noise:
#
#  - Forecast-leading boundaries: BN says High but CDI obs is still
#    Watch (drought hasn't manifested yet). L[Watch, High] = 0.20
#    captures this.
#  - Backwards-diverging: CDI shows Alert but BN says Minimal
#    (forecast saw recovery, ground hasn't caught up). L[Alert, Minimal]
#    = 0.01 keeps this rare but possible.
L = np.array([
    # Min   Low   Mod   High   Ext
    [0.50, 0.30, 0.10, 0.05, 0.02],  # No_drought
    [0.25, 0.25, 0.15, 0.05, 0.03],  # Full_recovery
    [0.10, 0.20, 0.20, 0.10, 0.05],  # Partial_recovery
    [0.10, 0.15, 0.25, 0.20, 0.15],  # Watch
    [0.04, 0.08, 0.20, 0.30, 0.25],  # Warning
    [0.01, 0.02, 0.10, 0.30, 0.50],  # Alert
], dtype=np.float64)
# Normalise columns so each P(cdi | risk=r) sums to 1
L = L / L.sum(axis=0, keepdims=True)


def crma_decide(risk_probs: np.ndarray, gamma: float = 0.20):
    """Cost-loss rule — same logic as drought_bn_ibf_v1.jl's compute_crma_state."""
    p_act      = float(risk_probs[3] + risk_probs[4])
    p_assess   = float(risk_probs[2] + risk_probs[3] + risk_probs[4])
    p_evaluate = float(risk_probs[1] + risk_probs[2] + risk_probs[3] + risk_probs[4])
    θ_act      = gamma
    θ_assess   = max(2 * gamma, 0.40)
    θ_evaluate = max(3 * gamma, 0.30)

    if p_act >= θ_act:
        return ("Actionable_Risk", "Red",
                f"P(High∪Extreme)={p_act:.2f} ≥ C/L={θ_act:.2f}")
    if p_assess >= θ_assess:
        return ("Assess", "Orange",
                f"P(Mod∪High∪Extreme)={p_assess:.2f} ≥ {θ_assess:.2f}")
    if p_evaluate >= θ_evaluate:
        return ("Evaluate", "Yellow",
                f"P(Low∪Mod∪High∪Extreme)={p_evaluate:.2f} ≥ {θ_evaluate:.2f}")
    return ("Monitor", "Green",
            "all conditional masses below thresholds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bn-csv",  required=True, help="BN result CSV (drought_bn_v1_*.csv)")
    ap.add_argument("--cdi-csv", required=True, help="CDI sidecar CSV (cdi_inputs_*.csv)")
    ap.add_argument("--out",     required=True, help="Output CSV with CDI-updated posteriors")
    ap.add_argument("--gamma",   type=float, default=0.20)
    args = ap.parse_args()

    bn = pd.read_csv(args.bn_csv)
    cdi = pd.read_csv(args.cdi_csv).set_index("id")

    n = len(bn)
    print(f"[cdi-update] BN rows: {n}  CDI rows: {len(cdi)}")

    # Cache the original posterior for diagnostics
    for c in RISK_NAMES:
        bn[c + "_pre_cdi"] = bn[c]
    bn["crma_state_pre_cdi"]    = bn["crma_state"]
    bn["traffic_light_pre_cdi"] = bn["traffic_light"]
    bn["cdi_level"]             = "missing"
    bn["cdi_level_idx"]         = 0
    bn["cdi_class"]             = 0

    n_updated = 0
    n_changed_crma = 0
    for i, row in bn.iterrows():
        bid = row["boundary_id"]
        if bid not in cdi.index:
            continue
        cdi_row = cdi.loc[bid]
        cdi_idx_1based = int(cdi_row["cdi_level_idx"])    # 1..6
        if cdi_idx_1based < 1 or cdi_idx_1based > 6:
            continue

        prior      = np.asarray([row[c] for c in RISK_NAMES], dtype=np.float64)
        likelihood = L[cdi_idx_1based - 1, :]
        post       = prior * likelihood
        s = post.sum()
        if s <= 0 or not np.isfinite(s):
            continue
        post /= s

        for j, c in enumerate(RISK_NAMES):
            bn.at[i, c] = round(float(post[j]), 6)
        bn.at[i, "cdi_level"]     = LEVEL_NAMES[cdi_idx_1based - 1]
        bn.at[i, "cdi_level_idx"] = cdi_idx_1based
        bn.at[i, "cdi_class"]     = int(cdi_row["cdi_class"])

        crma, light, expl = crma_decide(post, gamma=args.gamma)
        if crma != row["crma_state"]:
            n_changed_crma += 1
        bn.at[i, "crma_state"]       = crma
        bn.at[i, "traffic_light"]    = light
        bn.at[i, "crma_explanation"] = f"with-CDI: {expl}"
        bn.at[i, "risk_level"]       = ["Minimal", "Low", "Moderate",
                                         "High", "Extreme"][int(np.argmax(post))]
        n_updated += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bn.to_csv(out, index=False)

    pre  = bn["crma_state_pre_cdi"].value_counts().reindex(
        ["Monitor", "Evaluate", "Assess", "Actionable_Risk"], fill_value=0)
    post = bn["crma_state"].value_counts().reindex(
        ["Monitor", "Evaluate", "Assess", "Actionable_Risk"], fill_value=0)
    print(f"[cdi-update] wrote {out}  rows={n}  updated={n_updated}  "
          f"crma-flips={n_changed_crma}")
    print(f"[cdi-update] CRMA before CDI: {pre.to_dict()}")
    print(f"[cdi-update] CRMA after  CDI: {post.to_dict()}")


if __name__ == "__main__":
    main()
