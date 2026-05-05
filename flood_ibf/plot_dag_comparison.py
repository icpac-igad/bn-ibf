#!/usr/bin/env -S uv run --with matplotlib --with numpy python3
"""
plot_dag_comparison.py

Render N (date, high_bid, low_bid) DAG pairs as a single N-row x 2-col
figure. Each row is one date; left column = highest-risk admin-1 on that
date, right column = a contrasting low-risk admin-1 in the same country
(same evidence sources, very different posterior). The point is to make
the BN's evidence-to-CRMA logic visually obvious — for each row the
reader can scan across and see exactly which parent node(s) drove the
divergence in the risk posterior.

Usage:
    ./plot_dag_comparison.py \\
        --json-dir output/bn-dag \\
        --pairs '2026-03-01:KEN.43_1:KEN.46_1,2026-03-04:TZA.9_1:TZA.22_1,...' \\
        --out flood_bn_dag_comparison_5days.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

PARENT_CFG = [
    dict(key="ant",  title="Antecedent",  abbr=["Dry","Nrm","Wet","VWt","Sat"], color="#6366f1"),
    dict(key="exc",  title="Exceedance",  abbr=["VLo","Lo","Med","Hi","VHi"],   color="#0ea5e9"),
    dict(key="spa",  title="Spatial",     abbr=["Loc","Mod","Wide"],             color="#10b981"),
    dict(key="trn",  title="Trend",       abbr=["Dec","Stb","Inc"],              color="#f59e0b"),
    dict(key="tail", title="Tail Risk",   abbr=["Nil","Low","Mod","Hi"],         color="#ef4444"),
]

RISK_COLORS = ["#9ca3af", "#60a5fa", "#34d399", "#f59e0b", "#ef4444"]
RISK_LABELS = ["Minimal", "Low", "Moderate", "High", "Extreme"]
CRMA_COLORS = {"Monitor": "#22c55e", "Evaluate": "#eab308",
               "Assess":  "#f97316", "Actionable_Risk": "#dc2626"}


def prob_bars(ax, x, y, w, h, probs, abbrs, state, base_color):
    n = len(probs)
    bw = w / n
    max_p = max(probs) if max(probs) > 0 else 1
    bar_h = h * 0.45
    for i, (p, label) in enumerate(zip(probs, abbrs)):
        bx = x + i * bw
        ax.add_patch(mpatches.Rectangle((bx + bw*0.05, y + h*0.10),
                                         bw*0.90, bar_h,
                                         fc="#1e293b", ec="none", zorder=2))
        filled_h = bar_h * (p / max_p)
        is_mode = (p == max(probs))
        fc = "#2563eb" if is_mode else base_color
        ax.add_patch(mpatches.Rectangle((bx + bw*0.05, y + h*0.10),
                                         bw*0.90, filled_h,
                                         fc=fc, ec="none", zorder=3))
        ax.text(bx + bw/2, y + h*0.07, label,
                ha="center", va="top", fontsize=4.5, color="#94a3b8", zorder=4)
    ax.text(x + w/2, y + h*0.62, state,
            ha="center", va="center", fontsize=5.5, fontweight="bold",
            color="white", zorder=4)


def draw_node_box(ax, x, y, w, h, title, raw, probs, abbrs, state, color):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc="#1e293b", ec=color, lw=1.2, zorder=1))
    ax.text(x + w/2, y + h - 0.012, title,
            ha="center", va="top", fontsize=5.5, color="#94a3b8", zorder=4)
    ax.text(x + w/2, y + h - 0.038, raw,
            ha="center", va="top", fontsize=5, color="#cbd5e1", zorder=4)
    prob_bars(ax, x, y, w, h, probs, abbrs, state, color)


def draw_risk_node(ax, x, y, w, h, probs, state):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc="#1e293b", ec="#7c3aed", lw=1.6, zorder=1))
    ax.text(x + w/2, y + h - 0.012, "RISK LEVEL",
            ha="center", va="top", fontsize=6.5, fontweight="bold", color="#a78bfa", zorder=4)
    prob_bars(ax, x, y, w, h, probs, RISK_LABELS, state, RISK_COLORS[2])


def draw_crma_badge(ax, x, y, w, h, crma_state, p_he):
    c = CRMA_COLORS.get(crma_state, "#6b7280")
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.005", fc=c, ec="white", lw=1.2, zorder=1))
    ax.text(x + w/2, y + h*0.62, crma_state.replace("_", " "),
            ha="center", va="center", fontsize=6.5, fontweight="bold", color="white", zorder=4)
    ax.text(x + w/2, y + h*0.25, f"P(High∪Extreme) = {p_he:.3f}",
            ha="center", va="center", fontsize=5.5, color="white", zorder=4)


def draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=0.9),
                zorder=0)


def render_dag(ax, data: dict, header: str):
    """Render one boundary's DAG into the given matplotlib axes."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#0f172a")

    pw, ph = 0.175, 0.28
    gap = 0.010
    parent_y = 0.62
    parent_xs = [0.01 + i * (pw + gap) for i in range(5)]
    rw, rh = 0.28, 0.22
    rx, ry = 0.36, 0.32
    cw, ch = 0.26, 0.12
    cx, cy = 0.37, 0.08

    ax.text(0.5, 0.97, header,
            ha="center", va="top", fontsize=7.5, fontweight="bold", color="white")

    parent_centers = []
    for i, cfg in enumerate(PARENT_CFG):
        node = data[cfg["key"]]
        x = parent_xs[i]
        draw_node_box(ax, x, parent_y, pw, ph,
                      cfg["title"], node["raw"], node["probs"],
                      cfg["abbr"], node["state"], cfg["color"])
        parent_centers.append((x + pw/2, parent_y))

    risk = data["risk"]
    draw_risk_node(ax, rx, ry, rw, rh, risk["probs"], risk["state"])
    for px, py in parent_centers:
        draw_arrow(ax, px, parent_y, rx + rw/2, ry + rh)

    crma = data["crma"]
    draw_crma_badge(ax, cx, cy, cw, ch, crma["state"], crma["p_he"])
    draw_arrow(ax, rx + rw/2, ry, cx + cw/2, cy + ch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="output/bn-dag")
    ap.add_argument("--pairs", required=True,
                    help="Comma-separated triples DATE:HIGH_BID:LOW_BID, "
                         "e.g. '2026-03-01:KEN.43_1:KEN.46_1,...'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    triples = [t.split(":") for t in args.pairs.split(",")]
    n = len(triples)
    fig, axes = plt.subplots(n, 2, figsize=(20, 4.2 * n), squeeze=False)
    fig.patch.set_facecolor("#0f172a")
    if args.title:
        fig.suptitle(args.title, fontsize=14, color="white", y=0.995)

    for r, (date_str, high_bid, low_bid) in enumerate(triples):
        dag = json.load(open(f"{args.json_dir}/bn-dag-{date_str}.json"))
        if high_bid not in dag or low_bid not in dag:
            print(f"WARN row {r}: {high_bid} or {low_bid} missing in {date_str}")
            continue
        for col, bid, label in [(0, high_bid, "HIGHEST RISK"),
                                 (1, low_bid,  "CONTRAST")]:
            d = dag[bid]
            header = (f"{date_str}  ·  {label}  ·  {bid}  ({d['boundary']})  "
                      f"·  CRMA = {d['crma']['state'].replace('_', ' ')}")
            render_dag(axes[r][col], d, header)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.985, bottom=0.005,
                        wspace=0.02, hspace=0.05)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
