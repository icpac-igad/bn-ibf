#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "pgmpy",
# ]
# ///
"""
Drought BN IBF v1 — Python reference Bayesian Network for monthly drought risk.

Drought analogue of flood_bn_ibf_v1.py. Reads the CSV produced by
drought_data_prep.py and emits per-boundary risk posteriors + CRMA
decisions for the target month.

BN structure (parallel to flood):

    current_spi3      ──┐
    deficit_prob      ──┤
    spatial_coverage  ──┼──> risk_level ──> action
    spi3_trend        ──┤
    tail_risk         ──┘
    [forecast_agreement] (legacy node — disabled by default)

Five parents, one hidden child (risk_level: Minimal / Low / Moderate /
High / Extreme), one terminal action node (Monitor / Alert / Prepare /
Act). The CPT is constructed by `compute_drought_risk_probs()` from
expert rules — see PROBABILISTIC_LOGIC_v* in flood_ibf for the flood
analogue.

Usage:
    # Hard evidence (requires the per-boundary state columns already
    # present in the prep CSV):
    uv run drought_bn_ibf_v1.py \\
        --input bn_inputs/drought_inputs_2026-04-01.csv \\
        --output output/drought_bn_v1_2026-04-01.csv

    # Soft evidence (uses the {cur,def,spa,trn,tail}_p[1..K] columns
    # written by drought_data_prep.py --soft-evidence):
    uv run drought_bn_ibf_v1.py \\
        --input bn_inputs/drought_inputs_2026-04-01.csv \\
        --output output/drought_bn_v1_2026-04-01.csv \\
        --soft-evidence
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# pgmpy is heavy — guarded import so soft-only path can run without it.
try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    from pgmpy.models import DiscreteBayesianNetwork
    HAS_PGMPY = True
except ImportError:  # pragma: no cover
    HAS_PGMPY = False


# ─── BN state spaces ─────────────────────────────────────────────────────────

STATES: Dict[str, List[str]] = {
    "current_spi3":     ["Severe_Drought", "Moderate_Drought", "Mild_Drought",
                         "Normal", "Above_Normal"],
    "deficit_prob":     ["Very_Low", "Low", "Medium", "High", "Very_High"],
    "spatial_coverage": ["Localized", "Moderate", "Widespread"],
    "spi3_trend":       ["Deteriorating", "Stable", "Improving"],
    "tail_risk":        ["High", "Moderate", "Low", "Nil"],   # severity ↓
    "forecast_agreement": ["Low", "Medium", "High"],
    "risk_level":       ["Minimal", "Low", "Moderate", "High", "Extreme"],
    "action":           ["Monitor", "Alert", "Prepare", "Act"],
}

# Soft-evidence column prefixes (k = len(STATES[node]))
SOFT_PREFIX = {
    "current_spi3":     ("cur",  5),
    "deficit_prob":     ("def",  5),
    "spatial_coverage": ("spa",  3),
    "spi3_trend":       ("trn",  3),
    "tail_risk":        ("tail", 4),
}


# ─── categorisation (hard evidence path) ────────────────────────────────────


def categorize_current_spi3(spi: float) -> str:
    if spi < -1.5: return "Severe_Drought"
    if spi < -1.0: return "Moderate_Drought"
    if spi < -0.5: return "Mild_Drought"
    if spi <  0.5: return "Normal"
    return "Above_Normal"


def categorize_deficit_prob(p: float) -> str:
    if p < 0.2: return "Very_Low"
    if p < 0.4: return "Low"
    if p < 0.6: return "Medium"
    if p < 0.8: return "High"
    return "Very_High"


def categorize_spatial_coverage(p: float) -> str:
    if p < 0.3: return "Localized"
    if p < 0.6: return "Moderate"
    return "Widespread"


def categorize_spi3_trend(slope: float, band: float = 0.1) -> str:
    if slope >  band: return "Improving"
    if slope < -band: return "Deteriorating"
    return "Stable"


def categorize_tail_risk(spi_min: float) -> str:
    if spi_min < -1.5: return "High"
    if spi_min < -1.0: return "Moderate"
    if spi_min < -0.5: return "Low"
    return "Nil"


# ─── CPT: deterministic mapping from parents to risk distribution ───────────
# Same shape as flood: each parent contributes a numerical "stress" score; we
# combine and lookup-bin to a risk distribution. The CPT lets pgmpy do the
# rest (proper marginalisation under soft evidence).


def _state_score(node: str, state: str) -> float:
    """Per-parent stress score in [0, 1]. Higher = more drought stress."""
    if node == "current_spi3":
        # Drier states score higher
        return {"Severe_Drought": 1.0, "Moderate_Drought": 0.75,
                "Mild_Drought": 0.45, "Normal": 0.15, "Above_Normal": 0.0}[state]
    if node == "deficit_prob":
        return {"Very_Low": 0.0, "Low": 0.25, "Medium": 0.5,
                "High": 0.75, "Very_High": 1.0}[state]
    if node == "spatial_coverage":
        return {"Localized": 0.2, "Moderate": 0.55, "Widespread": 0.95}[state]
    if node == "spi3_trend":
        return {"Improving": 0.0, "Stable": 0.35, "Deteriorating": 0.85}[state]
    if node == "tail_risk":
        return {"Nil": 0.0, "Low": 0.3, "Moderate": 0.65, "High": 1.0}[state]
    raise ValueError(node)


# Per-parent contribution weight to overall stress. Sum = 1.
WEIGHTS = {
    "current_spi3":     0.27,
    "deficit_prob":     0.27,
    "spatial_coverage": 0.16,
    "spi3_trend":       0.12,
    "tail_risk":        0.18,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def compute_drought_risk_probs(
    current: str, deficit: str, spatial: str, trend: str, tail: str,
) -> List[float]:
    """Return P(risk_level | parents) — 5-vector for [Minimal, Low, Moderate, High, Extreme]."""
    s = (
        WEIGHTS["current_spi3"]     * _state_score("current_spi3", current)
        + WEIGHTS["deficit_prob"]     * _state_score("deficit_prob", deficit)
        + WEIGHTS["spatial_coverage"] * _state_score("spatial_coverage", spatial)
        + WEIGHTS["spi3_trend"]       * _state_score("spi3_trend", trend)
        + WEIGHTS["tail_risk"]        * _state_score("tail_risk", tail)
    )
    # Soft mapping s ∈ [0, 1] → 5-state risk dist via narrow Gaussian centred
    # at s, with bin centres at (0.10, 0.30, 0.50, 0.70, 0.90).
    centres = np.array([0.10, 0.30, 0.50, 0.70, 0.90])
    sigma = 0.18
    w = np.exp(-((centres - s) ** 2) / (2 * sigma ** 2))
    w = w / w.sum()
    return w.tolist()


# ─── action CPT ──────────────────────────────────────────────────────────────


_ACTION_CPT = {
    "Minimal":  [0.95, 0.04, 0.01, 0.00],
    "Low":      [0.55, 0.35, 0.09, 0.01],
    "Moderate": [0.10, 0.45, 0.35, 0.10],
    "High":     [0.02, 0.18, 0.50, 0.30],
    "Extreme":  [0.00, 0.05, 0.25, 0.70],
}  # rows: risk states; cols: action states (Monitor, Alert, Prepare, Act)


# ─── BN construction ─────────────────────────────────────────────────────────


def build_bn() -> "DiscreteBayesianNetwork":
    if not HAS_PGMPY:
        raise RuntimeError("pgmpy not available")

    bn = DiscreteBayesianNetwork([
        ("current_spi3",     "risk_level"),
        ("deficit_prob",     "risk_level"),
        ("spatial_coverage", "risk_level"),
        ("spi3_trend",       "risk_level"),
        ("tail_risk",        "risk_level"),
        ("risk_level",       "action"),
    ])

    # Parent priors: uniform (parents will be observed by the CSV).
    for node in ["current_spi3", "deficit_prob", "spatial_coverage",
                 "spi3_trend", "tail_risk"]:
        n = len(STATES[node])
        bn.add_cpds(TabularCPD(node, n, [[1.0 / n]] * n, state_names={node: STATES[node]}))

    # risk_level CPT (5 states × Π(parent cardinalities))
    parents = ["current_spi3", "deficit_prob", "spatial_coverage",
               "spi3_trend", "tail_risk"]
    p_states = [STATES[p] for p in parents]
    p_cards  = [len(s) for s in p_states]

    # column-major cartesian product (pgmpy convention: last parent varies fastest)
    from itertools import product
    n_cols = int(np.prod(p_cards))
    cpt = np.zeros((len(STATES["risk_level"]), n_cols))
    for col_idx, combo in enumerate(product(*p_states)):
        cpt[:, col_idx] = compute_drought_risk_probs(*combo)

    bn.add_cpds(TabularCPD(
        "risk_level", len(STATES["risk_level"]), cpt,
        evidence=parents, evidence_card=p_cards,
        state_names={"risk_level": STATES["risk_level"], **dict(zip(parents, p_states))},
    ))

    # action CPT
    risk_states = STATES["risk_level"]
    a_card = len(STATES["action"])
    a_cpt = np.zeros((a_card, len(risk_states)))
    for j, rs in enumerate(risk_states):
        a_cpt[:, j] = _ACTION_CPT[rs]
    bn.add_cpds(TabularCPD(
        "action", a_card, a_cpt,
        evidence=["risk_level"], evidence_card=[len(risk_states)],
        state_names={"action": STATES["action"], "risk_level": risk_states},
    ))

    bn.check_model()
    return bn


# ─── inference (hard evidence — categorical states) ────────────────────────


def _hard_evidence_from_row(row: pd.Series, trend_band: float) -> Dict[str, str]:
    return {
        "current_spi3":     categorize_current_spi3(float(row["current_spi3"])),
        "deficit_prob":     categorize_deficit_prob(float(row["forecast_deficit_prob"])),
        "spatial_coverage": categorize_spatial_coverage(float(row["spatial_coverage"])),
        "spi3_trend":       categorize_spi3_trend(float(row["trend_slope_spi_per_month"]), band=trend_band),
        "tail_risk":        categorize_tail_risk(float(row["ens_min_spi"])),
    }


# ─── inference (soft evidence — direct tensor contraction) ─────────────────
# pgmpy's VariableElimination supports `virtual_evidence`; we use it for
# parity with the flood_bn_ibf_v1.py soft path.


def _soft_evidence_from_row(row: pd.Series) -> Dict[str, np.ndarray]:
    out = {}
    for node, (prefix, k) in SOFT_PREFIX.items():
        cols = [f"{prefix}_p{i+1}" for i in range(k)]
        if not all(c in row.index for c in cols):
            return {}                       # one node missing → fall back to hard
        v = row[cols].to_numpy(dtype=float)
        v = np.where(np.isfinite(v), v, 0.0)
        s = v.sum()
        out[node] = v / s if s > 0 else np.full(k, 1.0 / k)
    return out


# Direct tensor-contraction path — no pgmpy needed and matches the
# soft-evidence semantics from the flood reference (legacy_inference path).


def _build_risk_tensor() -> Tuple[np.ndarray, List[str]]:
    """Build the 5D risk tensor T[r, cur, def, spa, trn, tail]."""
    parents = ["current_spi3", "deficit_prob", "spatial_coverage",
               "spi3_trend", "tail_risk"]
    p_states = [STATES[p] for p in parents]
    cards = [len(s) for s in p_states]
    T = np.zeros((len(STATES["risk_level"]), *cards))
    from itertools import product
    for idx in product(*[range(c) for c in cards]):
        combo = tuple(p_states[i][idx[i]] for i in range(len(parents)))
        T[(slice(None), *idx)] = compute_drought_risk_probs(*combo)
    return T, parents


def soft_inference(soft_evidence: Dict[str, np.ndarray],
                   T: np.ndarray, parents: List[str]) -> np.ndarray:
    r = T
    for ax_offset, p in enumerate(parents):
        v = soft_evidence[p]
        r = np.tensordot(r, v, axes=([1], [0]))
    s = r.sum()
    return r / s if s > 0 else r


# ─── CRMA decision (cost-loss thresholds) ───────────────────────────────────


def crma_decide(risk_probs: np.ndarray, gamma: float = 0.20) -> Tuple[str, str, str]:
    """Return (crma_state, traffic_light, explanation)."""
    p_high_extreme = float(risk_probs[3] + risk_probs[4])
    p_mod_up       = float(risk_probs[2] + risk_probs[3] + risk_probs[4])
    p_low_up       = float(risk_probs[1] + risk_probs[2] + risk_probs[3] + risk_probs[4])
    if p_high_extreme >= gamma:
        return ("Actionable_Risk", "Red",
                f"P(High∪Extreme)={p_high_extreme:.3f} ≥ γ={gamma:.2f}")
    if p_mod_up >= max(2 * gamma, 0.40):
        return ("Assess", "Orange",
                f"P(Mod∪High∪Extreme)={p_mod_up:.3f} ≥ {max(2*gamma,0.40):.2f}")
    if p_low_up >= max(3 * gamma, 0.30):
        return ("Evaluate", "Yellow",
                f"P(≥Low)={p_low_up:.3f} ≥ {max(3*gamma,0.30):.2f}")
    return ("Monitor", "Green",
            f"P(High∪Extreme)={p_high_extreme:.3f} < γ; under monitoring")


# ─── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="CSV from drought_data_prep.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--soft-evidence", action="store_true",
                    help="Use {cur,def,spa,trn,tail}_p[1..K] columns if present")
    ap.add_argument("--trend-band", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.20,
                    help="CRMA cost-loss threshold (default 0.20)")
    ap.add_argument("--engine", choices=["pgmpy", "tensor"], default="tensor",
                    help="Inference engine: 'tensor' is the direct contraction "
                         "(fast, no pgmpy); 'pgmpy' uses VariableElimination.")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[bn] read {len(df)} boundaries from {args.input}")

    # Pre-build risk tensor (used for both soft and hard tensor paths)
    T, parents = _build_risk_tensor()

    if args.engine == "pgmpy":
        if not HAS_PGMPY:
            raise SystemExit("pgmpy not installed; rerun with --engine tensor")
        bn = build_bn()
        infer = VariableElimination(bn)
    else:
        bn = None; infer = None

    rows = []
    risk_states  = STATES["risk_level"]
    action_states = STATES["action"]

    for _, row in df.iterrows():
        soft = _soft_evidence_from_row(row) if args.soft_evidence else {}

        if soft:
            risk_probs = soft_inference(soft, T, parents)
        else:
            ev = _hard_evidence_from_row(row, trend_band=args.trend_band)
            if args.engine == "pgmpy":
                q = infer.query(["risk_level"], evidence=ev, show_progress=False)
                risk_probs = np.array([q.values[q.state_names["risk_level"].index(s)]
                                       for s in risk_states])
            else:
                # Build one-hot soft vectors and reuse contraction
                onehot = {}
                for p in parents:
                    s_list = STATES[p]
                    v = np.zeros(len(s_list))
                    v[s_list.index(ev[p])] = 1.0
                    onehot[p] = v
                risk_probs = soft_inference(onehot, T, parents)

        # Action posterior (deterministic CPT × risk)
        a_cpt = np.array([_ACTION_CPT[rs] for rs in risk_states])    # (5, 4)
        action_probs = (risk_probs[:, None] * a_cpt).sum(axis=0)
        action_probs /= action_probs.sum()

        crma_state, traffic_light, explanation = crma_decide(risk_probs,
                                                             gamma=args.gamma)

        out = {
            "id":         row["id"],
            "name":       row["name"],
            "country":    row["country"],
            "target_date": row["target_date"],
        }
        for i, s in enumerate(risk_states):
            out[f"risk_{s.lower()}"] = round(float(risk_probs[i]), 4)
        for i, s in enumerate(action_states):
            out[f"action_{s.lower()}"] = round(float(action_probs[i]), 4)
        out["crma_state"]      = crma_state
        out["traffic_light"]   = traffic_light
        out["crma_explanation"] = explanation
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    counts = out_df["crma_state"].value_counts()
    print(f"[bn] wrote {out_path}  rows={len(out_df)}")
    print(f"[bn] CRMA breakdown: {counts.to_dict()}")
    n_red = int((out_df["traffic_light"] == "Red").sum())
    print(f"[bn] {n_red} boundaries flagged Red (Actionable_Risk)")


if __name__ == "__main__":
    main()
