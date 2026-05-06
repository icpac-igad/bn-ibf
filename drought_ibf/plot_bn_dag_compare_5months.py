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
RISK_STATES = ["Minimal", "Low", "Moderate", "High", "Extreme"]

# 4-parent BN structure used by the v2_notail_cdi posterior (the version
# the paper's choropleth panels are built from). The tail-risk node is
# omitted because the underlying BN does not consume it — the DAG figure
# is faithful to the inference that produced the CRMA states shown.
PARENT_CFG = [
    dict(key="ant", title="Current SPI3",     abbr=["SevDr", "ModDr", "MldDr", "Nrm", "Abv"], color="#a16207", window="obs past 3 mo"),
    dict(key="def", title="Forecast Deficit", abbr=["VLo", "Lo", "Med", "Hi", "VHi"],         color="#0ea5e9", window="SEAS5 → {season}"),
    dict(key="spa", title="Spatial",          abbr=["Loc", "Mod", "Wide"],                    color="#10b981", window="SEAS5 → {season}"),
    dict(key="trn", title="Trend",            abbr=["Det", "Stb", "Imp"],                     color="#f59e0b", window="obs past 6 mo"),
]

RISK_LABELS = ["Min", "Low", "Mod", "Hi", "Ext"]
CRMA_COLORS = {"Monitor": "#22c55e", "Evaluate": "#eab308",
               "Assess": "#f97316", "Actionable_Risk": "#dc2626"}

# Base font sizes; multiplied by `scale` in helpers below so that
# single-month (--single) renders can bump everything up uniformly.
FS = {"node_title": 8.5, "node_window": 6.5, "node_raw": 7.5,
      "bar_label": 7, "bar_state": 8, "risk_title": 9,
      "crma_state": 9.5, "crma_phe": 7.5,
      "panel_header": 10, "footer": 6.5}

# White-background theme; same palette as the flood plotter.
THEME = {
    "fig_bg":     "white",
    "ax_bg":      "white",
    "box_bg":     "white",
    "bar_track":  "#e2e8f0",
    "bar_mode":   "#2563eb",
    "title":      "#475569",
    "window":     "#64748b",
    "raw":        "#0f172a",
    "state_dark": "#0f172a",
    "label":      "#475569",
    "header":     "#0f172a",
    "footer":     "#64748b",
    "caption":    "#334155",
    "suptitle":   "#0f172a",
    "risk_title": "#7c3aed",
    "arrow":      "#94a3b8",
}

# DAG layout. box_scale=1.05 in render_dag (5% bigger than v1) and parent
# row centred horizontally so there is no right-side gap.
LAYOUT = {
    "pw":  0.18, "ph": 0.26,
    "gap": 0.008,
    "py":  0.62,
    "rw":  0.32, "rh": 0.20,
    "ry":  0.30,
    "cw":  0.30, "ch": 0.11,
    "cy":  0.10,
}

# (month_tag, top_id, ctr_id, shared_spi3_cat, caption). All 5 pairings
# are picked from the v2_notail_cdi sweep (2025-01..2026-04, 16 inits)
# so the DAG figure matches the paper's choropleth pipeline. Each pair
# shares a current-SPI-3 category but receives opposite CRMA outcomes,
# so the split is forced onto the SEAS5 forecast deficit / spatial /
# trend channels (tail node removed).
PAIRINGS = [
    ("2025-03_JJA", "SDN.14_1", "SDN.2_1",
     "Severe_Drought",
     "Same country (SDN), same Severe_Drought category, identical forecast "
     "deficit (0.06) and Widespread hotspot. Soft-bin SPI3 magnitude (-2.38 "
     "vs -1.50) is the discriminator: deeper drought drives AR (P_HE=0.90); "
     "Al Qadarif stays Monitor."),
    ("2025-08_OND", "KEN.40_1", "ERI.6_1",
     "Normal",
     "Both Normal SPI3 but opposite sides (-0.47 vs +0.18): SEAS5 forecast "
     "deficit (0.24 Low vs 0.08 Very_Low) + Stable-vs-Improving trend split "
     "AR (Tana River) vs Monitor (Anseba)."),
    ("2025-09_DJF", "SOM.3_1",  "BDI.4_1",
     "Above_Normal",
     "Both Above_Normal SPI3 (+2.60 vs +0.75): SEAS5 forecast deficit (High "
     "0.64 vs Low 0.25) + Deteriorating trend over-ride the wet present in "
     "Banaadir, lifting AR despite SPI3 saying wet."),
    ("2025-12_MAM", "KEN.9_1",  "ETH.6_1",
     "Severe_Drought",
     "Both Severe_Drought (-3.55 vs -1.87): Widespread hotspot (1.00 vs 0.27 "
     "Localized) + Low-vs-VeryLow forecast deficit lift Isiolo to AR; Gambela's "
     "localised footprint keeps it at Monitor."),
    ("2026-04_JJA", "SOM.5_1",  "RWA.5_1",
     "Moderate_Drought",
     "Both Moderate_Drought (-1.40 vs -1.11), same Widespread hotspot. Forecast "
     "deficit (0.23 Low vs 0.16 Very_Low) + SPI3 magnitude split Bay (AR) from "
     "Kigali (Monitor); Kigali's Deteriorating trend is not enough on its own."),
]


def _draw_prob_bars(ax, x, y, w, h, probs, abbrs, state, parent_color, scale=1.0):
    n = len(probs)
    bw = w / n
    max_p = max(probs) if max(probs) > 0 else 1
    bar_h_total = h * 0.45

    for i, (p, label) in enumerate(zip(probs, abbrs)):
        bx = x + i * bw
        ax.add_patch(mpatches.Rectangle((bx + bw * 0.08, y + h * 0.10),
                                         bw * 0.84, bar_h_total,
                                         fc=THEME["bar_track"], ec="none", zorder=2))
        filled = bar_h_total * (p / max_p)
        is_mode = (p == max(probs))
        fc = THEME["bar_mode"] if is_mode else parent_color
        ax.add_patch(mpatches.Rectangle((bx + bw * 0.08, y + h * 0.10),
                                         bw * 0.84, filled,
                                         fc=fc, ec="none", zorder=3))
        ax.text(bx + bw / 2, y + h * 0.075, label,
                ha="center", va="top", fontsize=FS["bar_label"] * scale,
                color=THEME["label"], zorder=4)

    ax.text(x + w / 2, y + h * 0.65, state,
            ha="center", va="center", fontsize=FS["bar_state"] * scale,
            fontweight="bold", color=THEME["state_dark"], zorder=4)


def _draw_node_box(ax, x, y, w, h, title, raw, probs, abbrs, state, color,
                   window=None, scale=1.0):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc=THEME["box_bg"], ec=color,
                 lw=1.2 * scale, zorder=1))
    ax.text(x + w * 0.05, y + h - 0.015, title,
            ha="left", va="top", fontsize=FS["node_title"] * scale,
            color=THEME["title"], zorder=4)
    if window:
        ax.text(x + w * 0.95, y + h - 0.015, window,
                ha="right", va="top", fontsize=FS["node_window"] * scale,
                color=THEME["window"], style="italic", zorder=4)
    ax.text(x + w / 2, y + h - 0.045, raw,
            ha="center", va="top", fontsize=FS["node_raw"] * scale,
            color=THEME["raw"], zorder=4)
    _draw_prob_bars(ax, x, y, w, h, probs, abbrs, state, color, scale=scale)


def _draw_risk_node(ax, x, y, w, h, probs, state, scale=1.0):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc=THEME["box_bg"],
                 ec=THEME["risk_title"], lw=1.8 * scale, zorder=1))
    ax.text(x + w / 2, y + h - 0.012, "RISK", ha="center", va="top",
            fontsize=FS["risk_title"] * scale, fontweight="bold",
            color=THEME["risk_title"], zorder=4)
    _draw_prob_bars(ax, x, y, w, h, probs, RISK_LABELS, state,
                    THEME["risk_title"], scale=scale)


def _draw_crma_badge(ax, x, y, w, h, crma_state, p_he, scale=1.0):
    c = CRMA_COLORS.get(crma_state, "#6b7280")
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc=c, ec=c,
                 lw=1.4 * scale, zorder=1))
    ax.text(x + w / 2, y + h * 0.65, crma_state.replace("_", " "),
            ha="center", va="center", fontsize=FS["crma_state"] * scale,
            fontweight="bold", color="white", zorder=4)
    ax.text(x + w / 2, y + h * 0.27, f"P(High∪Ext) = {p_he:.3f}",
            ha="center", va="center", fontsize=FS["crma_phe"] * scale,
            color="white", zorder=4)


def _draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=THEME["arrow"], lw=0.8),
                zorder=0)


def argmax_state(probs, states):
    return states[max(range(len(probs)), key=lambda i: probs[i])]


def build_node(soft: pd.Series, post: pd.Series) -> dict:
    # drought_data_prep.py applies a per-node bin REVERSAL for
    # _REVERSE_NODES = {"cur", "tail", "trn"} so the soft column order in
    # the CSV is the opposite of STATES order. We reverse them back here
    # so the bar / abbreviation indexing matches ANT_STATES / TRN_STATES.
    ant_p  = [soft[f"cur_p{i}"] for i in range(1, 6)][::-1]
    def_p  = [soft[f"def_p{i}"] for i in range(1, 6)]
    spa_p  = [soft[f"spa_p{i}"] for i in range(1, 4)]
    trn_p  = [soft[f"trn_p{i}"] for i in range(1, 4)][::-1]
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
        "ant": {"state": argmax_state(ant_p, ANT_STATES), "probs": ant_p,
                "raw": fmt(soft.get("current_spi3"), "SPI3")},
        "def": {"state": argmax_state(def_p, DEF_STATES), "probs": def_p,
                "raw": f"P={soft.get('forecast_deficit_prob', float('nan')):.3f}"
                       if pd.notna(soft.get("forecast_deficit_prob")) else "N/A"},
        "spa": {"state": argmax_state(spa_p, SPA_STATES), "probs": spa_p,
                "raw": (f"{soft.get('hotspot_fraction', float('nan')) * 100:.0f}% hotspot"
                        if pd.notna(soft.get("hotspot_fraction")) else "N/A")},
        "trn": {"state": argmax_state(trn_p, TRN_STATES), "probs": trn_p,
                "raw": fmt(soft.get("trend_slope_spi_per_month"), "SPI/mo", signed=True)},
        "risk": {"state": post["risk_level"], "probs": risk_p},
        "crma": {"state": post["crma_state"], "p_he": p_he},
    }


SEASON_ANCHOR_MONTH = {"MAM": 5, "JJA": 8, "OND": 11, "DJF": 2}


def lead_for(init_month: int, season: str) -> int:
    """Same convention as drought_data_prep.py: 1-based SEAS5 lead from init."""
    anchor = SEASON_ANCHOR_MONTH[season]
    return ((anchor - init_month) % 12) or 12


def render_dag(ax, node: dict, header: str, season: str,
               lead: int | None = None, scale: float = 1.0,
               box_scale: float = 1.05):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(THEME["ax_bg"])
    ax.axis("off")

    n_parents = len(PARENT_CFG)
    # pw scales as 5/n so 4 parents fill the same row width as 5.
    pw  = LAYOUT["pw"]  * (5.0 / n_parents) * box_scale
    ph  = LAYOUT["ph"]  * box_scale
    gap = LAYOUT["gap"] * box_scale
    rw  = LAYOUT["rw"]  * box_scale
    rh  = LAYOUT["rh"]  * box_scale
    cw  = LAYOUT["cw"]  * box_scale
    ch  = LAYOUT["ch"]  * box_scale
    parent_y = LAYOUT["py"]
    ry = LAYOUT["ry"]
    cy = LAYOUT["cy"]

    row_w = n_parents * pw + (n_parents - 1) * gap
    left = max(0.0, (1.0 - row_w) / 2)
    parent_xs = [left + i * (pw + gap) for i in range(n_parents)]
    rx = (1.0 - rw) / 2
    cx = (1.0 - cw) / 2

    ax.text(0.5, 0.985, header, ha="center", va="top",
            fontsize=FS["panel_header"] * scale, fontweight="bold",
            color=THEME["header"])

    parent_centers = []
    for i, cfg in enumerate(PARENT_CFG):
        n = node[cfg["key"]]
        x = parent_xs[i]
        win = cfg.get("window")
        if win and "{season}" in win:
            win = win.format(season=season)
        _draw_node_box(ax, x, parent_y, pw, ph,
                       cfg["title"], n["raw"], n["probs"], cfg["abbr"], n["state"],
                       cfg["color"], window=win, scale=scale)
        parent_centers.append((x + pw / 2, parent_y))

    risk = node["risk"]
    _draw_risk_node(ax, rx, ry, rw, rh, risk["probs"], risk["state"], scale=scale)
    for (px, py) in parent_centers:
        _draw_arrow(ax, px, py, rx + rw / 2, ry + rh)

    crma = node["crma"]
    _draw_crma_badge(ax, cx, cy, cw, ch, crma["state"], crma["p_he"], scale=scale)
    _draw_arrow(ax, rx + rw / 2, ry, cx + cw / 2, cy + ch)

    lead_txt = f"lead {lead} mo" if lead is not None else ""
    footer = (f"obs nodes = past 3/6 mo | SEAS5 forecast → {season} season"
              f"{(' (' + lead_txt + ')') if lead_txt else ''}")
    ax.text(0.5, 0.025, footer,
            ha="center", va="bottom", fontsize=FS["footer"] * scale,
            style="italic", color=THEME["footer"])


def render_single_pairing(month: str, top_id: str, ctr_id: str, spi_cat: str,
                          caption: str, soft: pd.DataFrame, post: pd.DataFrame,
                          season: str, lead: int | None, out_path: Path,
                          scale: float = 1.7):
    """One month, two admins stacked vertically — each DAG fills full width."""
    panels = [("HIGHEST", top_id), ("CONTRAST", ctr_id)]
    n = len(panels)
    fig = plt.figure(figsize=(16, 7.0 * n))
    fig.patch.set_facecolor(THEME["fig_bg"])
    fig.suptitle(f"Drought BN DAG — {month}  ({spi_cat})",
                 color=THEME["suptitle"], fontsize=15, fontweight="bold", y=0.995)

    gs = fig.add_gridspec(n, 1, hspace=0.22)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(n)]

    for ax, (tag, bid) in zip(axes, panels):
        if bid not in soft.index or bid not in post.index:
            ax.text(0.5, 0.5, f"missing {bid}", ha="center", va="center",
                    color=THEME["state_dark"])
            ax.axis("off")
            continue
        node = build_node(soft.loc[bid], post.loc[bid])
        header = f"{tag}  -  {bid}  {node['boundary']}"
        render_dag(ax, node, header, season=season, lead=lead, scale=scale)

    fig.text(0.5, 0.018, caption, ha="center", va="bottom",
             fontsize=11, style="italic", color=THEME["caption"], wrap=True)

    legend_patches = [mpatches.Patch(color=c, label=s.replace("_", " "))
                      for s, c in CRMA_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               frameon=False, fontsize=10, labelcolor=THEME["state_dark"],
               bbox_to_anchor=(0.5, -0.005))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs-dir", default="bn_inputs_v2")
    ap.add_argument("--posterior-dir", default="output_v2_notail_cdi")
    ap.add_argument("--posterior-prefix", default="drought_bn_v2_notail_cdi_")
    ap.add_argument("--inputs-prefix", default="drought_inputs_")
    ap.add_argument("--out", default="output_v2_notail_cdi/drought_bn_dag_compare_5months.png")
    ap.add_argument("--single", default=None,
                    help="Render only one pairing (MONTH like 2025-03_JJA, or "
                         "'all' to emit one expanded figure per pairing). Each "
                         "DAG is stacked vertically with bigger fonts.")
    ap.add_argument("--single-out-dir", default="output_v2_notail_cdi/bn-dag-single",
                    help="Output directory for --single mode figures.")
    args = ap.parse_args()

    if args.single:
        targets = (list(PAIRINGS) if args.single == "all"
                   else [p for p in PAIRINGS if p[0] == args.single])
        if not targets:
            valid = ", ".join(p[0] for p in PAIRINGS)
            raise SystemExit(f"--single {args.single!r} matched no pairing. "
                             f"Available months: {valid} (or 'all').")
        for month, top_id, ctr_id, spi_cat, caption in targets:
            soft_path = Path(args.inputs_dir) / f"{args.inputs_prefix}{month}.csv"
            post_path = Path(args.posterior_dir) / f"{args.posterior_prefix}{month}.csv"
            if not soft_path.exists() or not post_path.exists():
                print(f"[skip] {month}: missing input or posterior CSV")
                continue
            soft = pd.read_csv(soft_path).set_index("id")
            post = pd.read_csv(post_path).set_index("boundary_id")
            try:
                ym, season = month.split("_")
                init_m = int(ym.split("-")[1])
                lead = lead_for(init_m, season)
            except Exception:
                season, lead = "season", None
            out = Path(args.single_out_dir) / f"drought_bn_dag_{month}.png"
            render_single_pairing(month, top_id, ctr_id, spi_cat, caption,
                                  soft, post, season, lead, out)
        return

    fig, axes = plt.subplots(len(PAIRINGS), 2,
                             figsize=(15, 4.0 * len(PAIRINGS)),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.04})
    fig.patch.set_facecolor(THEME["fig_bg"])
    fig.suptitle("Drought BN DAG comparison — same SPI3 category, opposite CRMA, by month",
                 color=THEME["suptitle"], fontsize=13, fontweight="bold", y=0.995)

    for row, (month, top_id, ctr_id, spi_cat, caption) in enumerate(PAIRINGS):
        soft_path = Path(args.inputs_dir) / f"{args.inputs_prefix}{month}.csv"
        post_path = Path(args.posterior_dir) / f"{args.posterior_prefix}{month}.csv"
        if not soft_path.exists() or not post_path.exists():
            for col in range(2):
                axes[row][col].text(0.5, 0.5, f"missing {month}",
                                    ha="center", va="center",
                                    color=THEME["state_dark"])
                axes[row][col].axis("off")
            continue
        soft = pd.read_csv(soft_path).set_index("id")
        post = pd.read_csv(post_path).set_index("boundary_id")

        try:
            ym, season = month.split("_")
            init_m = int(ym.split("-")[1])
            lead = lead_for(init_m, season)
        except Exception:
            season, lead = "season", None

        for col, bid in enumerate((top_id, ctr_id)):
            ax = axes[row][col]
            if bid not in soft.index or bid not in post.index:
                ax.text(0.5, 0.5, f"missing {bid}", ha="center", va="center",
                        color=THEME["state_dark"])
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
                 color=THEME["caption"])

    legend_patches = [mpatches.Patch(color=c, label=s.replace("_", " "))
                      for s, c in CRMA_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               frameon=False, fontsize=9, labelcolor=THEME["state_dark"],
               bbox_to_anchor=(0.5, 0.005))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
