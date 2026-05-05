#!/usr/bin/env -S uv run --with matplotlib --with numpy --with pandas
"""
plot_bn_dag_compare_5months.py
5-row x 2-col figure comparing drought BN DAGs across months and admin-1
boundaries. Mirrors the flood version but uses the drought node spaces.

Each row picks ONE month and TWO admin1 boundaries that share the same
current_spi3_category yet receive opposite CRMA outcomes — making it
explicit that the BN is doing more than reading the current SPI3, and
that forecast deficit / spatial coverage / trend / tail risk push the
posterior in opposite directions.

  2024-01_MAM  ETH.4_1  Benshangul-Gumaz   vs UGA.24_1 Kibale       (Severe_Drought)
  2024-05_JJA  SDN.17_1 West Kurdufan      vs DJI.1_2 Ali Sabieh    (Above_Normal)
  2024-08_OND  KEN.21_1 Lamu               vs KEN.2_1 Bomet         (Normal)
  2025-02_MAM  SSD.8_1  Warap              vs UGA.1_1 Adjumani      (Severe_Drought)
  2026-04_JJA  SOM.5_1  Bay                vs KEN.35_1 Nyandarua    (Moderate_Drought)

Reads parent soft-evidence from bn_inputs_v2/drought_inputs_<MONTH>.csv
and posterior risk + crma from output_v2_cdi/drought_bn_v2_cdi_<MONTH>.csv.

Usage:
  uv run python3 plot_bn_dag_compare_5months.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

# Drought BN node states (drought_bn_ibf_v1.py STATES dict)
ANT_STATES  = ["Severe_Drought", "Moderate_Drought", "Mild_Drought", "Normal", "Above_Normal"]
DEF_STATES  = ["Very_Low", "Low", "Medium", "High", "Very_High"]
SPA_STATES  = ["Localized", "Moderate", "Widespread"]
TRN_STATES  = ["Deteriorating", "Stable", "Improving"]
TAIL_STATES = ["High", "Moderate", "Low", "Nil"]   # severity decreases →
RISK_STATES = ["Minimal", "Low", "Moderate", "High", "Extreme"]

# `window` describes the time slice the node summarises. cur/trn are
# observation-side (past months); def/spa/tail are SEAS5 forecast at
# lead = lead_for_season(init_month, target_season). Per-row {season}
# placeholder is filled in render_dag with the row's target season.
PARENT_CFG = [
    dict(key="ant",  title="Current SPI3",    abbr=["SevDr", "ModDr", "MldDr", "Nrm", "Abv"], color="#a16207", window="obs past 3 mo"),
    dict(key="def",  title="Forecast Deficit", abbr=["VLo", "Lo", "Med", "Hi", "VHi"],         color="#0ea5e9", window="SEAS5 → {season}"),
    dict(key="spa",  title="Spatial",          abbr=["Loc", "Mod", "Wide"],                    color="#10b981", window="SEAS5 → {season}"),
    dict(key="trn",  title="Trend",            abbr=["Det", "Stb", "Imp"],                     color="#f59e0b", window="obs past 6 mo"),
    dict(key="tail", title="Tail Risk",        abbr=["Hi", "Mod", "Low", "Nil"],               color="#ef4444", window="SEAS5 → {season}"),
]

RISK_LABELS = ["Min", "Low", "Mod", "Hi", "Ext"]
CRMA_COLORS = {"Monitor": "#22c55e", "Evaluate": "#eab308",
               "Assess": "#f97316", "Actionable_Risk": "#dc2626"}

# (month_tag, top_id, ctr_id, shared_spi3_cat, caption)
PAIRINGS = [
    ("2024-01_MAM", "ETH.4_1",  "UGA.24_1",
     "Severe_Drought",
     "Both Severe_Drought: forecast deficit + Wide spatial + Hi tail → AR; "
     "isolated stress alone → Monitor."),
    ("2024-05_JJA", "SDN.17_1", "DJI.1_2",
     "Above_Normal",
     "Both Above_Normal SPI3: forecast deficit + Deteriorating trend overrides "
     "the wet present."),
    ("2024-08_OND", "KEN.21_1", "KEN.2_1",
     "Normal",
     "Same country, same Normal SPI3: tail-risk and forecast deficit drive the "
     "split."),
    ("2025-02_MAM", "SSD.8_1",  "UGA.1_1",
     "Severe_Drought",
     "Both Severe_Drought: Improving trend + Localized spatial defuse the "
     "Monitor case."),
    ("2026-04_JJA", "SOM.5_1",  "KEN.35_1",
     "Moderate_Drought",
     "Same Moderate_Drought: very-high forecast deficit + Wide hotspot push "
     "Bay into AR."),
]


def _draw_prob_bars(ax, x, y, w, h, probs, abbrs, state, parent_color):
    n = len(probs)
    bw = w / n
    max_p = max(probs) if max(probs) > 0 else 1
    bar_h_total = h * 0.45

    for i, (p, label) in enumerate(zip(probs, abbrs)):
        bx = x + i * bw
        ax.add_patch(mpatches.Rectangle((bx + bw * 0.08, y + h * 0.10),
                                         bw * 0.84, bar_h_total,
                                         fc="#1e293b", ec="none", zorder=2))
        filled = bar_h_total * (p / max_p)
        is_mode = (p == max(probs))
        fc = "#2563eb" if is_mode else parent_color
        ax.add_patch(mpatches.Rectangle((bx + bw * 0.08, y + h * 0.10),
                                         bw * 0.84, filled,
                                         fc=fc, ec="none", zorder=3))
        ax.text(bx + bw / 2, y + h * 0.075, label,
                ha="center", va="top", fontsize=7, color="#94a3b8", zorder=4)

    ax.text(x + w / 2, y + h * 0.65, state,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=4)


def _draw_node_box(ax, x, y, w, h, title, raw, probs, abbrs, state, color, window=None):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc="#1e293b", ec=color, lw=1.2, zorder=1))
    ax.text(x + w * 0.05, y + h - 0.015, title,
            ha="left", va="top", fontsize=8.5, color="#94a3b8", zorder=4)
    if window:
        ax.text(x + w * 0.95, y + h - 0.015, window,
                ha="right", va="top", fontsize=6.5, color="#64748b",
                style="italic", zorder=4)
    ax.text(x + w / 2, y + h - 0.045, raw,
            ha="center", va="top", fontsize=7.5, color="#cbd5e1", zorder=4)
    _draw_prob_bars(ax, x, y, w, h, probs, abbrs, state, color)


def _draw_risk_node(ax, x, y, w, h, probs, state):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc="#1e293b", ec="#7c3aed", lw=1.8, zorder=1))
    ax.text(x + w / 2, y + h - 0.012, "RISK", ha="center", va="top",
            fontsize=9, fontweight="bold", color="#a78bfa", zorder=4)
    _draw_prob_bars(ax, x, y, w, h, probs, RISK_LABELS, state, "#7c3aed")


def _draw_crma_badge(ax, x, y, w, h, crma_state, p_he):
    c = CRMA_COLORS.get(crma_state, "#6b7280")
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc=c, ec="white", lw=1.4, zorder=1))
    ax.text(x + w / 2, y + h * 0.65, crma_state.replace("_", " "),
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=4)
    ax.text(x + w / 2, y + h * 0.27, f"P(High∪Ext) = {p_he:.3f}",
            ha="center", va="center", fontsize=7.5, color="white", zorder=4)


def _draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=0.8),
                zorder=0)


def argmax_state(probs, states):
    return states[max(range(len(probs)), key=lambda i: probs[i])]


def build_node(soft: pd.Series, post: pd.Series) -> dict:
    ant_p  = [soft[f"cur_p{i}"]  for i in range(1, 6)]
    def_p  = [soft[f"def_p{i}"]  for i in range(1, 6)]
    spa_p  = [soft[f"spa_p{i}"]  for i in range(1, 4)]
    trn_p  = [soft[f"trn_p{i}"]  for i in range(1, 4)]
    tail_p = [soft[f"tail_p{i}"] for i in range(1, 5)]
    risk_p = [post["risk_minimal"], post["risk_low"], post["risk_moderate"],
              post["risk_high"], post["risk_extreme"]]
    p_he = float(post["risk_high"] + post["risk_extreme"])

    def fmt(v, suffix, prec=2, signed=False):
        if pd.isna(v):
            return "N/A"
        sign = "+" if signed and v >= 0 else ""
        return f"{sign}{v:.{prec}f} {suffix}"

    return {
        "boundary": post["boundary_name"],
        "ant":  {"state": argmax_state(ant_p,  ANT_STATES),  "probs": ant_p,
                 "raw": fmt(soft.get("current_spi3"), "SPI3")},
        "def":  {"state": argmax_state(def_p,  DEF_STATES),  "probs": def_p,
                 "raw": f"P={soft.get('forecast_deficit_prob',float('nan')):.3f}"
                        if pd.notna(soft.get("forecast_deficit_prob")) else "N/A"},
        "spa":  {"state": argmax_state(spa_p,  SPA_STATES),  "probs": spa_p,
                 "raw": (f"{soft.get('hotspot_fraction',float('nan'))*100:.0f}% hotspot"
                         if pd.notna(soft.get("hotspot_fraction")) else "N/A")},
        "trn":  {"state": argmax_state(trn_p,  TRN_STATES),  "probs": trn_p,
                 "raw": fmt(soft.get("trend_slope_spi_per_month"), "SPI/mo", signed=True)},
        "tail": {"state": argmax_state(tail_p, TAIL_STATES), "probs": tail_p,
                 "raw": fmt(soft.get("ens_min_spi_peak"), "SPI", signed=True)},
        "risk": {"state": post["risk_level"], "probs": risk_p},
        "crma": {"state": post["crma_state"], "p_he": p_he},
    }


SEASON_ANCHOR_MONTH = {"MAM": 5, "JJA": 8, "OND": 11, "DJF": 2}


def lead_for(init_month: int, season: str) -> int:
    """Same convention as drought_data_prep.py: 1-based SEAS5 lead from init."""
    anchor = SEASON_ANCHOR_MONTH[season]
    return ((anchor - init_month) % 12) or 12


def render_dag(ax, node: dict, header: str, season: str, lead: int | None = None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    pw, ph = 0.18, 0.26
    gap = 0.005
    parent_y = 0.62
    parent_xs = [0.005 + i * (pw + gap) for i in range(5)]

    rw, rh = 0.32, 0.20
    rx, ry = 0.34, 0.30

    cw, ch = 0.30, 0.11
    cx, cy = 0.35, 0.10

    ax.text(0.5, 0.985, header, ha="center", va="top",
            fontsize=10, fontweight="bold", color="white")

    parent_centers = []
    for i, cfg in enumerate(PARENT_CFG):
        n = node[cfg["key"]]
        x = parent_xs[i]
        win = cfg.get("window")
        if win and "{season}" in win:
            win = win.format(season=season)
        _draw_node_box(ax, x, parent_y, pw, ph,
                       cfg["title"], n["raw"], n["probs"], cfg["abbr"], n["state"],
                       cfg["color"], window=win)
        parent_centers.append((x + pw / 2, parent_y))

    risk = node["risk"]
    _draw_risk_node(ax, rx, ry, rw, rh, risk["probs"], risk["state"])
    for (px, py) in parent_centers:
        _draw_arrow(ax, px, py, rx + rw / 2, ry + rh)

    crma = node["crma"]
    _draw_crma_badge(ax, cx, cy, cw, ch, crma["state"], crma["p_he"])
    _draw_arrow(ax, rx + rw / 2, ry, cx + cw / 2, cy + ch)

    lead_txt = f"lead {lead} mo" if lead is not None else ""
    footer = (f"obs nodes = past 3/6 mo | SEAS5 forecast → {season} season"
              f"{(' (' + lead_txt + ')') if lead_txt else ''}")
    ax.text(0.5, 0.025, footer,
            ha="center", va="bottom", fontsize=6.5, style="italic", color="#64748b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", default="bn_inputs_v2")
    ap.add_argument("--posterior-dir", default="output_v2_cdi")
    ap.add_argument("--posterior-prefix", default="drought_bn_v2_cdi_")
    ap.add_argument("--inputs-prefix", default="drought_inputs_")
    ap.add_argument("--out", default="output_v2_cdi/drought_bn_dag_compare_5months.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(len(PAIRINGS), 2,
                             figsize=(15, 4.0 * len(PAIRINGS)),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.04})
    fig.patch.set_facecolor("#0f172a")
    fig.suptitle("Drought BN DAG comparison — same SPI3 category, opposite CRMA, by month",
                 color="white", fontsize=13, fontweight="bold", y=0.995)

    for row, (month, top_id, ctr_id, spi_cat, caption) in enumerate(PAIRINGS):
        soft_path = Path(args.inputs_dir) / f"{args.inputs_prefix}{month}.csv"
        post_path = Path(args.posterior_dir) / f"{args.posterior_prefix}{month}.csv"
        if not soft_path.exists() or not post_path.exists():
            for col in range(2):
                axes[row][col].text(0.5, 0.5, f"missing {month}",
                                    ha="center", va="center", color="white")
                axes[row][col].axis("off")
            continue
        soft = pd.read_csv(soft_path).set_index("id")
        post = pd.read_csv(post_path).set_index("boundary_id")

        # Parse "YYYY-MM_SEASON" to compute SEAS5 lead from init month.
        try:
            ym, season = month.split("_")
            init_m = int(ym.split("-")[1])
            lead = lead_for(init_m, season)
        except Exception:
            season, lead = "season", None

        for col, bid in enumerate((top_id, ctr_id)):
            ax = axes[row][col]
            if bid not in soft.index or bid not in post.index:
                ax.text(0.5, 0.5, f"missing {bid}", ha="center", va="center", color="white")
                ax.axis("off")
                continue
            node = build_node(soft.loc[bid], post.loc[bid])
            tag = "HIGHEST" if col == 0 else "CONTRAST"
            header = f"[{month}]  {tag}  -  {bid}  {node['boundary']}  ({spi_cat})"
            render_dag(ax, node, header, season=season, lead=lead)

        bbox_left = axes[row][0].get_position()
        bbox_right = axes[row][1].get_position()
        cap_x = (bbox_left.x0 + bbox_right.x1) / 2
        cap_y = bbox_left.y0 - 0.008
        fig.text(cap_x, cap_y, caption,
                 ha="center", va="top", fontsize=10, style="italic",
                 color="#cbd5e1")

    legend_patches = [mpatches.Patch(color=c, label=s.replace("_", " "))
                      for s, c in CRMA_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               frameon=False, fontsize=9, labelcolor="white",
               bbox_to_anchor=(0.5, 0.005))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
