#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""
CDI evidence update — apply the JRC CDI observation(s) as a likelihood
update on the BN risk posterior.

Auto-detects the CDI-CSV schema:

  • Single-source CSV (`cdi_level_idx`, `cdi_class`)
        — Path B-α: posterior ∝ prior · L[cdi_level, :]
  • Multi-source CSV (`cdi_level_idx_recomp` and `cdi_level_idx_eadw`,
    optional `cdi_agreement`)
        — Path B-γ: posterior ∝ prior · L[cdi_recomp, :] · L[cdi_eadw, :]

The two-channel multiplicative update is the principled Bayesian
treatment of two independent noisy measurements of the same latent
risk_level: when the channels agree the product L[k,:]·L[k,:] is more
peaked (sharper update), when they disagree the product spreads mass
across more risk levels (softer update). No tuning knob between the
two regimes — the L matrix's own noise structure carries the weight.

Implements the noisy-channel BN factorisation:

    parents (5) ──> risk_level ──> cdi_obs_{recomp,eadw} (observed)

with P(cdi | risk_level) encoded as the column-stochastic matrix L
below. Keeps the existing RxInfer 5-parent @model untouched.

Diagnostic columns preserved on output:
  - risk_*_pre_cdi      : original BN posterior, before CDI update
  - crma_state_pre_cdi  : CRMA decision before CDI
  - traffic_light_pre_cdi
  - cdi_level / cdi_class                     (single-source mode)
  - cdi_level_recomp / cdi_level_eadw / cdi_agreement
                                              (two-source mode)

Usage:
    # single-source (Path B-α, original behaviour)
    uv run cdi_evidence_update.py \\
        --bn-csv  /tmp/drought_bn_v1_2026-04.csv \\
        --cdi-csv /tmp/cdi_inputs_2026-04.csv \\
        --out     /tmp/drought_bn_v1_cdi_2026-04.csv

    # two-source (Path B-γ) — feed the wide CSV from
    # `cdi_data_prep.py --cdi-source both`
    uv run cdi_evidence_update.py \\
        --bn-csv  /tmp/drought_bn_v1_2026-04.csv \\
        --cdi-csv /tmp/cdi_inputs_both_2026-04.csv \\
        --out     /tmp/drought_bn_v1_cdi_both_2026-04.csv
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


def _detect_sources(cdi_columns: set[str]) -> list[tuple[str, str, str]]:
    """Inspect a CDI CSV's columns and return the list of (source_label,
    level_idx_col, class_col) tuples to apply.

    Returns a single-entry list for single-source CSVs (level_idx_col=
    `cdi_level_idx`) or a two-entry list for the wide CSV produced by
    `cdi_data_prep.py --cdi-source both`.
    """
    has_recomp = "cdi_level_idx_recomp" in cdi_columns
    has_eadw   = "cdi_level_idx_eadw"   in cdi_columns
    has_single = "cdi_level_idx" in cdi_columns
    if has_recomp and has_eadw:
        return [("recomp", "cdi_level_idx_recomp", "cdi_class_recomp"),
                ("eadw",   "cdi_level_idx_eadw",   "cdi_class_eadw")]
    if has_single:
        return [("", "cdi_level_idx", "cdi_class")]
    raise SystemExit(
        "CDI CSV missing recognised level columns "
        "(need cdi_level_idx, OR cdi_level_idx_recomp + cdi_level_idx_eadw)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bn-csv",  required=True, help="BN result CSV (drought_bn_v1_*.csv)")
    ap.add_argument("--cdi-csv", required=True, help="CDI sidecar CSV (cdi_inputs_*.csv)")
    ap.add_argument("--out",     required=True, help="Output CSV with CDI-updated posteriors")
    ap.add_argument("--gamma",   type=float, default=0.20)
    args = ap.parse_args()

    bn  = pd.read_csv(args.bn_csv)
    cdi = pd.read_csv(args.cdi_csv).set_index("id")
    sources = _detect_sources(set(cdi.columns))
    mode = "two-source (B-γ)" if len(sources) == 2 else "single-source (B-α)"
    print(f"[cdi-update] mode: {mode}  channels: {[s[0] or 'cdi' for s in sources]}")

    n = len(bn)
    print(f"[cdi-update] BN rows: {n}  CDI rows: {len(cdi)}")

    # Cache original posterior for diagnostics
    for c in RISK_NAMES:
        bn[c + "_pre_cdi"] = bn[c]
    bn["crma_state_pre_cdi"]    = bn["crma_state"]
    bn["traffic_light_pre_cdi"] = bn["traffic_light"]
    if len(sources) == 1:
        bn["cdi_level"]      = "missing"
        bn["cdi_level_idx"]  = 0
        bn["cdi_class"]      = 0
    else:
        bn["cdi_level_recomp"] = "missing"; bn["cdi_class_recomp"] = 0
        bn["cdi_level_eadw"]   = "missing"; bn["cdi_class_eadw"]   = 0
        bn["cdi_agreement"]    = False

    n_updated = 0
    n_changed_crma = 0
    n_disagree = 0
    for i, row in bn.iterrows():
        bid = row["boundary_id"]
        if bid not in cdi.index:
            continue
        cdi_row = cdi.loc[bid]

        # Build the joint likelihood across all available CDI channels.
        # P(risk | parents, cdi_1, cdi_2, …) ∝ prior · ∏_k L[cdi_k, :]
        # which is the standard product of independent observation models.
        joint_lik = np.ones(5, dtype=np.float64)
        per_channel: list[tuple[str, int, int]] = []  # (label, idx, class)
        for label, idx_col, cls_col in sources:
            level_idx = int(cdi_row[idx_col])
            if level_idx < 1 or level_idx > 6:
                continue
            joint_lik *= L[level_idx - 1, :]
            cls = int(cdi_row[cls_col]) if cls_col in cdi_row.index else 0
            per_channel.append((label, level_idx, cls))

        if not per_channel:
            continue

        prior = np.asarray([row[c] for c in RISK_NAMES], dtype=np.float64)
        post = prior * joint_lik
        s = post.sum()
        if s <= 0 or not np.isfinite(s):
            continue
        post /= s

        # Apply posterior + diagnostic columns
        for j, c in enumerate(RISK_NAMES):
            bn.at[i, c] = round(float(post[j]), 6)

        if len(sources) == 1:
            label, level_idx, cls = per_channel[0]
            bn.at[i, "cdi_level"]     = LEVEL_NAMES[level_idx - 1]
            bn.at[i, "cdi_level_idx"] = level_idx
            bn.at[i, "cdi_class"]     = cls
        else:
            channels = {label: (level_idx, cls) for label, level_idx, cls in per_channel}
            r_idx, r_cls = channels.get("recomp", (0, 0))
            e_idx, e_cls = channels.get("eadw",   (0, 0))
            bn.at[i, "cdi_level_recomp"] = LEVEL_NAMES[r_idx - 1] if r_idx else "missing"
            bn.at[i, "cdi_class_recomp"] = r_cls
            bn.at[i, "cdi_level_eadw"]   = LEVEL_NAMES[e_idx - 1] if e_idx else "missing"
            bn.at[i, "cdi_class_eadw"]   = e_cls
            agree = (r_idx > 0 and e_idx > 0 and r_idx == e_idx)
            bn.at[i, "cdi_agreement"] = agree
            if not agree:
                n_disagree += 1

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
    if len(sources) == 2:
        print(f"[cdi-update] source disagreement: {n_disagree}/{n_updated} boundaries "
              f"({n_disagree / max(n_updated, 1):.0%})")
    print(f"[cdi-update] CRMA before CDI: {pre.to_dict()}")
    print(f"[cdi-update] CRMA after  CDI: {post.to_dict()}")


if __name__ == "__main__":
    main()
