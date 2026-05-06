#!/usr/bin/env -S uv run --with pandas --with numpy --with matplotlib
"""
Render the Flood BN IBF v1 DAG with per-day evidence and posteriors for a
specific admin-1 boundary. Reads the flood_data_prep CSV (evidence) and the
Julia BN output CSV (risk/action probability vectors).

Each node shows:
  - Evidence (blue boxes): observed continuous value + discretized state
  - risk_level (yellow box): 5-bar posterior with argmax label
  - action (green box): 4-bar posterior with argmax label

Usage:
  ./plot_bn_dag_per_day.py --date 2026-03-06 --boundary Nairobi
  ./plot_bn_dag_per_day.py --boundary Nairobi --start 2026-03-01 --end 2026-03-10
  ./plot_bn_dag_per_day.py --date 2026-03-06 --all-moderate
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RISK_STATES = ["Minimal", "Low", "Moderate", "High", "Extreme"]
ACTION_STATES = ["Monitor", "Alert", "Prepare", "Act"]
RISK_COLORS = {
    "Minimal": "#1a9850", "Low": "#d9ef8b", "Moderate": "#fee08b",
    "High": "#f46d43", "Extreme": "#a50026",
}
ACTION_COLORS = {
    "Monitor": "#4575b4", "Alert": "#fee08b",
    "Prepare": "#f46d43", "Act": "#a50026",
}
# CRMA Layer-1 traffic-light palette
CRMA_STATES = ["Monitor", "Evaluate", "Assess", "Actionable_Risk"]
CRMA_COLORS = {
    "Monitor":         "#1a9850",  # Green
    "Evaluate":        "#fee08b",  # Yellow
    "Assess":          "#f46d43",  # Orange
    "Actionable_Risk": "#a50026",  # Red
}


# ---- discretization (mirrors the Julia categorizers) ----
def cat_antecedent(mm: float) -> str:
    if not np.isfinite(mm): return "Normal"
    if mm < 10: return "Dry"
    if mm < 30: return "Normal"
    if mm < 60: return "Wet"
    if mm < 100: return "Very_Wet"
    return "Saturated"

def cat_exceedance(p: float) -> str:
    if not np.isfinite(p) or p < 0.2: return "Very_Low"
    if p < 0.4: return "Low"
    if p < 0.6: return "Medium"
    if p < 0.8: return "High"
    return "Very_High"

def cat_spatial(c: float) -> str:
    if not np.isfinite(c) or c < 0.3: return "Localized"
    if c < 0.6: return "Moderate"
    return "Widespread"

def cat_tail(r: float) -> str:
    if not np.isfinite(r) or r < 0.5: return "None"
    if r < 1.0: return "Low"
    if r < 2.0: return "Moderate"
    return "High"


def draw_evidence_node(ax, x, y, title, value_str, state,
                       width=0.18, height=0.12, sub_line=None):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.008,rounding_size=0.01",
                         linewidth=1.2, facecolor="#dae8fc",
                         edgecolor="#6c8ebf", zorder=3)
    ax.add_patch(box)
    ax.text(x, y + height/2 - 0.018, title, ha="center", va="top",
            fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(x, y + 0.005, value_str, ha="center", va="center",
            fontsize=8, zorder=4, family="monospace")
    ax.text(x, y - height/2 + 0.020, f"→ {state}", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#003366", zorder=4)
    if sub_line:
        ax.text(x, y - height/2 - 0.012, sub_line, ha="center", va="top",
                fontsize=6.5, color="#555555", zorder=4, style="italic")


def draw_dist_node(ax, x, y, title, states, probs, winner, box_color,
                   border_color, width=0.33, height=0.16):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.008,rounding_size=0.01",
                         linewidth=1.5, facecolor=box_color,
                         edgecolor=border_color, zorder=3)
    ax.add_patch(box)
    ax.text(x, y + height/2 - 0.018, title, ha="center", va="top",
            fontsize=9, fontweight="bold", zorder=4)
    # mini bar plot inside the box
    n = len(states)
    bar_w = (width - 0.025) / n
    base_y = y - height/2 + 0.020
    top_y = y + height/2 - 0.038
    avail = top_y - base_y
    x0 = x - width/2 + 0.015
    for i, (s, p) in enumerate(zip(states, probs)):
        if states is RISK_STATES:
            col = RISK_COLORS.get(s, "#999")
        elif states is CRMA_STATES:
            col = CRMA_COLORS.get(s, "#999")
        else:
            col = ACTION_COLORS.get(s, "#999")
        bar_h = avail * max(p, 0.01)
        ax.add_patch(mpatches.Rectangle((x0 + i * bar_w, base_y),
                                        bar_w * 0.85, bar_h,
                                        facecolor=col,
                                        edgecolor="black", linewidth=0.5, zorder=4))
        # label on top of bar
        ax.text(x0 + i * bar_w + bar_w * 0.42, base_y + bar_h + 0.005,
                f"{p*100:.0f}", ha="center", va="bottom", fontsize=6.5, zorder=5)
        # state name below
        short = s[:4] if len(s) > 5 else s
        ax.text(x0 + i * bar_w + bar_w * 0.42, base_y - 0.006, short,
                ha="center", va="top", fontsize=6.5, rotation=0, zorder=4)
    ax.text(x, y - height/2 - 0.014, f"⇒ {winner}",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color="#990000", zorder=4)


def draw_edge(ax, x1, y1, x2, y2, color="#6c8ebf", width=1.2,
              style="arc3,rad=0.0"):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>",
                            mutation_scale=10,
                            connectionstyle=style,
                            color=color, linewidth=width, zorder=2)
    ax.add_patch(arrow)


def render_day(inp_row: pd.Series, out_row: pd.Series,
               date: str, boundary: str, out_path: Path,
               include_tail_risk: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect(aspect=(7.5 / 12) / 1.0 * 1.0)
    ax.axis("off")

    # ---- header ----
    country = out_row.get("country", "")
    risk = out_row["risk_level"]
    crma = out_row.get("crma_state", out_row.get("recommended_action", "Monitor"))
    traffic = out_row.get("traffic_light", "")
    crma_expl = out_row.get("crma_explanation", "")
    confidence = float(out_row["confidence"])
    title = (f"Flood BN IBF v1  •  {boundary} ({country})  •  {date}\n"
             f"risk = {risk}    CRMA = {crma} ({traffic})    confidence = {confidence:.2f}")
    fig.suptitle(title, fontsize=13, y=0.98)

    # ---- evidence nodes (top row) ----
    ant_mm = float(inp_row["antecedent_rainfall_mm"])
    ant_state = cat_antecedent(ant_mm)

    trend = str(inp_row["rainfall_trend"])
    trend_slope = float(inp_row["trend_slope_mm_per_day"])

    ep = float(inp_row["ecmwf_eprob_heavy"])
    ep_state = cat_exceedance(ep)

    sc = float(inp_row["spatial_coverage"])
    sc_state = cat_spatial(sc)

    ratio = float(inp_row["ens_max_ratio"])
    ratio_state = cat_tail(ratio)

    ens_max_24 = float(inp_row.get("ens_max_24h_mm", np.nan))
    ens_mean_24 = float(inp_row.get("ens_mean_24h_mm", np.nan))
    hotspot = float(inp_row.get("hotspot_fraction", np.nan))

    # five evidence nodes evenly spaced
    y_ev = 0.76
    xs = [0.10, 0.30, 0.50, 0.70, 0.90]

    draw_evidence_node(
        ax, xs[0], y_ev,
        "Antecedent Rainfall",
        f"{ant_mm:.1f} mm (7-day)",
        ant_state,
        sub_line="IMERG boundary mean",
    )
    draw_evidence_node(
        ax, xs[1], y_ev,
        "Exceedance Prob",
        f"P_heavy = {ep:.3f}",
        ep_state,
        sub_line=f"mean 24h: {ens_mean_24:.0f}mm  max 24h: {ens_max_24:.0f}mm",
    )
    draw_evidence_node(
        ax, xs[2], y_ev,
        "Spatial Coverage",
        f"{sc:.2%}",
        sc_state,
        sub_line=(f"hotspot frac: {hotspot:.2f}" if np.isfinite(hotspot) else None),
    )
    draw_evidence_node(
        ax, xs[3], y_ev,
        "Rainfall Trend",
        f"slope = {trend_slope:+.2f} mm/d",
        trend,
        sub_line="7-day linregress",
    )
    draw_evidence_node(
        ax, xs[4], y_ev,
        "Tail Risk",
        f"ens_max/thr = {ratio:.2f}",
        ratio_state,
        sub_line="pixel p95 over durations",
    )

    # ---- risk_level node (middle) ----
    probs_r = np.array([float(out_row[f"risk_{s.lower()}"]) for s in RISK_STATES])
    draw_dist_node(ax, 0.5, 0.42, "P(risk_level | evidence)",
                   RISK_STATES, probs_r, risk,
                   box_color="#fff2cc", border_color="#d6b656")

    # ---- CRMA state box (bottom) — Layer-1 output, deterministic rule ----
    # Cumulative-mass bars per CRMA test threshold
    p_act      = probs_r[3] + probs_r[4]
    p_assess   = probs_r[2] + probs_r[3] + probs_r[4]
    p_evaluate = probs_r[1] + probs_r[2] + probs_r[3] + probs_r[4]
    p_monitor  = probs_r[0]
    crma_bars = np.array([p_monitor, p_evaluate, p_assess, p_act])
    draw_dist_node(ax, 0.5, 0.12,
                   "CRMA state (deterministic from P(risk_level), C/L=0.2)",
                   CRMA_STATES, crma_bars, crma,
                   box_color="#d5e8d4", border_color="#82b366",
                   width=0.38)

    # ---- edges: evidence → risk_level ----
    for xe in xs:
        draw_edge(ax, xe, y_ev - 0.06, 0.5, 0.42 + 0.08,
                  color="#6c8ebf", width=1.0)
    # ---- edge: risk → CRMA ----
    draw_edge(ax, 0.5, 0.42 - 0.08, 0.5, 0.12 + 0.08,
              color="#82b366", width=2.0)

    # ---- annotation bar explaining the case ----
    expl_parts = []
    if ant_state in ("Very_Wet", "Saturated"):
        expl_parts.append(f"ground is {ant_state} ({ant_mm:.0f}mm)")
    if ratio_state in ("Moderate", "High"):
        expl_parts.append(f"tail risk {ratio_state} (≥1 member exceeds 2-yr RP)")
    if ep_state not in ("Very_Low",):
        expl_parts.append(f"mean exceedance {ep_state}")
    if trend == "Increasing":
        expl_parts.append("trend Increasing")
    if sc_state in ("Moderate", "Widespread"):
        expl_parts.append(f"spatial {sc_state}")
    if not expl_parts:
        expl_parts.append("all evidence near baseline")
    expl = "Drivers: " + ", ".join(expl_parts)
    fig.text(0.5, 0.02, expl, ha="center", fontsize=10,
             style="italic", color="#222")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", default="Nairobi")
    ap.add_argument("--date", help="Single date YYYY-MM-DD")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end", default="2026-03-10")
    ap.add_argument("--inp-dir", default="bn_inputs")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--fig-dir", default="output/bn_dags")
    args = ap.parse_args()

    if args.date:
        dates = [pd.Timestamp(args.date)]
    else:
        dates = list(pd.date_range(args.start, args.end, freq="D"))

    for d in dates:
        ds = d.date().isoformat()
        inp_fp = Path(args.inp_dir) / f"flood_inputs_{ds}.csv"
        out_fp = Path(args.out_dir) / f"flood_bn_v1_{ds}.csv"
        if not (inp_fp.exists() and out_fp.exists()):
            print(f"[dag] skip {ds}: missing CSV")
            continue
        inp = pd.read_csv(inp_fp)
        out = pd.read_csv(out_fp)
        ri = inp[inp["name"] == args.boundary]
        ro = out[out["boundary_name"] == args.boundary]
        if ri.empty or ro.empty:
            print(f"[dag] skip {ds}: {args.boundary} not in CSV")
            continue
        fig_out = Path(args.fig_dir) / f"bn_dag_{args.boundary.replace(' ','_')}_{ds}.png"
        render_day(ri.iloc[0], ro.iloc[0], ds, args.boundary, fig_out)
        print(f"[dag] wrote {fig_out}")


if __name__ == "__main__":
    main()
