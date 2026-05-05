#!/usr/bin/env -S uv run --with matplotlib --with numpy
"""
plot_bn_dag_compare_5days.py
5-row x 2-col figure comparing BN DAGs across days and admin-1 boundaries.

Each row is a date. Left column = the highest-marked admin1 of the day
(by P(High v Extreme)). Right column = a contrasting admin1 picked manually
to expose which evidence node(s) drive the difference (saturation alone is
not enough, spatial extent and tail risk matter, etc.).

The five curated pairings each spotlight a different mechanism:
  2026-03-01  Simiyu (TZ)        vs Murang'a (KE)   - Saturated antecedent
                                                       alone does not raise risk
                                                       without Wide spatial + tail.
  2026-03-04  Kilimanjaro (TZ)   vs Kabarole (UG)   - Saturated + High tail
                                                       beats Wide spatial alone.
  2026-03-08  Mtwara (TZ)        vs Katavi (TZ)     - Same country, evidence
                                                       differs on every parent.
  2026-03-09  Lindi (TZ)         vs Cankuzo (BI)    - Both Very_Wet antecedent;
                                                       spatial + tail flip risk.
  2026-03-10  Mtwara (TZ)        vs Tabora (TZ)     - Same country + same
                                                       antecedent; pure spatial
                                                       + tail contrast.

Usage:
  uv run python3 plot_bn_dag_compare_5days.py [--dag-dir output/bn-dag]
                                               [--out output/flood_bn_dag_compare_5days.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# `window` is a short tag describing the time slice the node summarises,
# relative to T = target_date. Antecedent looks BACKWARD over the past
# 7 days of IMERG; the four forecast nodes look FORWARD over the 7-day
# ECMWF ensemble window starting at T.
PARENT_CFG = [
    dict(key="ant",  title="Antecedent",  abbr=["Dry", "Nrm", "Wet", "VWt", "Sat"], color="#6366f1", window="past 7d"),
    dict(key="exc",  title="Exceedance",  abbr=["VLo", "Lo", "Med", "Hi", "VHi"],   color="#0ea5e9", window="fcst T→T+7"),
    dict(key="spa",  title="Spatial",     abbr=["Loc", "Mod", "Wide"],               color="#10b981", window="fcst T→T+7"),
    dict(key="trn",  title="Trend",       abbr=["Dec", "Stb", "Inc"],                color="#f59e0b", window="fcst T→T+7"),
    dict(key="tail", title="Tail Risk",   abbr=["Nil", "Low", "Mod", "Hi"],          color="#ef4444", window="fcst T→T+7"),
]

RISK_COLORS = ["#9ca3af", "#60a5fa", "#34d399", "#f59e0b", "#ef4444"]
RISK_LABELS = ["Min", "Low", "Mod", "Hi", "Ext"]
CRMA_COLORS = {"Monitor": "#22c55e", "Evaluate": "#eab308",
               "Assess": "#f97316", "Actionable_Risk": "#dc2626"}

# Each row: (date, [(tag, admin_id), ...], caption). Most rows have two
# panels; 2026-03-04 has three so Nairobi (KEN.30_1) can be shown alongside
# the day's HIGHEST and a Monitor — Nairobi sat in Assess (P(High∪Ext)≈0.10),
# the BN's intermediate-risk read on a known flood-prone city ahead of the
# real flood event reported on 2026-04-06.
PAIRINGS = [
    ("2026-03-01",
     [("HIGHEST", "TZA.24_1"), ("CONTRAST", "KEN.29_1")],
     "Saturated antecedent alone does not lift risk without Wide spatial + tail."),
    ("2026-03-04",
     [("HIGHEST", "TZA.9_1"), ("CONTRAST", "UGA.13_1"),
      ("NAIROBI - pre-event", "KEN.30_1")],
     "Saturated + High tail → AR (Kilimanjaro). Normal + Wide → Monitor (Kabarole). "
     "Nairobi: Saturated past + Mod hotspot but Decreasing forecast → Assess (P(H∪E)=0.10) "
     "ahead of the 2026-04-06 flood event."),
    ("2026-03-08",
     [("HIGHEST", "TZA.15_1"), ("CONTRAST", "TZA.7_1")],
     "Same country: every evidence node differs."),
    ("2026-03-09",
     [("HIGHEST", "TZA.10_1"), ("CONTRAST", "BDI.5_1")],
     "Both Very_Wet antecedent: spatial coverage + tail flip the posterior."),
    ("2026-03-10",
     [("HIGHEST", "TZA.15_1"), ("CONTRAST", "TZA.26_1")],
     "Same country, same antecedent: pure Wide+Hi vs Local+Low contrast."),
]


def _draw_prob_bars(ax, x, y, w, h, probs, abbrs, state, parent_color):
    """Vertical bars normalised to max prob; mode is highlighted."""
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
    # Title (top-left) and time-window tag (top-right), kept on one band so
    # the lead-time of the underlying evidence is always visible.
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


def render_dag(ax, node: dict, header: str):
    """Render one full DAG inside `ax` (unit-square coordinates)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    # Layout
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
        _draw_node_box(ax, x, parent_y, pw, ph,
                       cfg["title"], n["raw"], n["probs"], cfg["abbr"], n["state"],
                       cfg["color"], window=cfg.get("window"))
        parent_centers.append((x + pw / 2, parent_y))

    risk = node["risk"]
    _draw_risk_node(ax, rx, ry, rw, rh, risk["probs"], risk["state"])
    for (px, py) in parent_centers:
        _draw_arrow(ax, px, py, rx + rw / 2, ry + rh)

    crma = node["crma"]
    _draw_crma_badge(ax, cx, cy, cw, ch, crma["state"], crma["p_he"])
    _draw_arrow(ax, rx + rw / 2, ry, cx + cw / 2, cy + ch)

    # Footer: explicit reminder of T = target date and forecast lead window.
    ax.text(0.5, 0.025, "T = target date | forecast nodes integrate lead +1..+7 d (ECMWF), antecedent = past 7 d (IMERG)",
            ha="center", va="bottom", fontsize=6.5, style="italic", color="#64748b")


def _row_axes(fig, gs_row, n_panels):
    """Create n_panels axes inside one row of a 6-col gridspec.
    2-panel rows: spans 0:3 and 3:6. 3-panel rows: 0:2, 2:4, 4:6."""
    if n_panels == 2:
        return [fig.add_subplot(gs_row[0:3]),
                fig.add_subplot(gs_row[3:6])]
    if n_panels == 3:
        return [fig.add_subplot(gs_row[0:2]),
                fig.add_subplot(gs_row[2:4]),
                fig.add_subplot(gs_row[4:6])]
    raise ValueError(f"unsupported panel count {n_panels}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dag-dir", default="output/bn-dag")
    ap.add_argument("--out",     default="output/flood_bn_dag_compare_5days.png")
    args = ap.parse_args()

    n_rows = len(PAIRINGS)
    fig = plt.figure(figsize=(15, 4.0 * n_rows))
    fig.patch.set_facecolor("#0f172a")
    outer = fig.add_gridspec(n_rows, 1, hspace=0.32)
    fig.suptitle("Flood BN DAG comparison — highest-marked vs contrasting admin-1, by day",
                 color="white", fontsize=13, fontweight="bold", y=0.995)

    for row, (date, panels, caption) in enumerate(PAIRINGS):
        sub_gs = outer[row].subgridspec(1, 6, wspace=0.06)
        axes = _row_axes(fig, sub_gs, len(panels))

        path = Path(args.dag_dir) / f"bn-dag-{date}.json"
        dag = json.load(open(path)) if path.exists() else None

        for ax, (tag, bid) in zip(axes, panels):
            if dag is None or bid not in dag:
                ax.text(0.5, 0.5, f"missing {bid}", ha="center", va="center", color="white")
                ax.axis("off")
                continue
            node = dag[bid]
            header = f"[{date}]  {tag}  -  {bid}  {node['boundary']}"
            render_dag(ax, node, header)

        # Row caption under all panels in the row
        bboxes = [a.get_position() for a in axes]
        cap_x = (bboxes[0].x0 + bboxes[-1].x1) / 2
        cap_y = bboxes[0].y0 - 0.008
        fig.text(cap_x, cap_y, caption,
                 ha="center", va="top", fontsize=10, style="italic",
                 color="#cbd5e1", wrap=True)

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
