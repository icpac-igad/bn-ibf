#!/usr/bin/env python3
"""
Plot a single boundary's BN DAG from a bn-dag JSON file as a PNG.
Usage: uv run python3 plot_bn_dag_from_json.py [--json output/bn-dag/bn-dag-2026-03-04.json]
                                                [--boundary KEN.30_1]
                                                [--out output/bn-dag-nairobi-2026-03-04.png]
"""
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

PARENT_CFG = [
    dict(key="ant",  title="Antecedent",  abbr=["Dry","Nrm","Wet","VWt","Sat"],  color="#6366f1"),
    dict(key="exc",  title="Exceedance",  abbr=["VLo","Lo","Med","Hi","VHi"],    color="#0ea5e9"),
    dict(key="spa",  title="Spatial",     abbr=["Loc","Mod","Wide"],              color="#10b981"),
    dict(key="trn",  title="Trend",       abbr=["Dec","Stb","Inc"],               color="#f59e0b"),
    dict(key="tail", title="Tail Risk",   abbr=["Nil","Low","Mod","Hi"],          color="#ef4444"),
]

RISK_COLORS  = ["#9ca3af", "#60a5fa", "#34d399", "#f59e0b", "#ef4444"]
RISK_LABELS  = ["Minimal", "Low", "Moderate", "High", "Extreme"]
CRMA_COLORS  = {"Monitor": "#22c55e", "Evaluate": "#eab308",
                "Assess": "#f97316", "Actionable_Risk": "#dc2626"}


def prob_bars(ax, x, y, w, h, probs, colors, abbrs, state):
    """Draw a row of normalised probability bars inside a box."""
    n = len(probs)
    bw = w / n
    max_p = max(probs) if max(probs) > 0 else 1
    bar_h = h * 0.45

    for i, (p, c, label) in enumerate(zip(probs, colors, abbrs)):
        bx = x + i * bw
        # background bar track
        ax.add_patch(mpatches.Rectangle((bx + bw*0.05, y + h*0.10),
                                         bw*0.90, bar_h,
                                         fc="#1e293b", ec="none", zorder=2))
        # filled bar
        filled_h = bar_h * (p / max_p)
        is_mode = (p == max(probs))
        fc = "#2563eb" if is_mode else c
        ax.add_patch(mpatches.Rectangle((bx + bw*0.05, y + h*0.10),
                                         bw*0.90, filled_h,
                                         fc=fc, ec="none", zorder=3))
        # label
        ax.text(bx + bw/2, y + h*0.08, label,
                ha="center", va="top", fontsize=5.5, color="#94a3b8", zorder=4)

    # state pill
    ax.text(x + w/2, y + h * 0.62, state,
            ha="center", va="center", fontsize=6.5, fontweight="bold",
            color="white", zorder=4)


def draw_node_box(ax, x, y, w, h, title, raw, probs, colors, abbrs, state, box_color):
    # Box background
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.01", fc="#1e293b", ec=box_color, lw=1.5, zorder=1))
    # Title
    ax.text(x + w/2, y + h - 0.015, title,
            ha="center", va="top", fontsize=7, color="#94a3b8", zorder=4)
    # Raw value
    ax.text(x + w/2, y + h - 0.045, raw,
            ha="center", va="top", fontsize=6, color="#cbd5e1", zorder=4)
    prob_bars(ax, x, y, w, h, probs, colors, abbrs, state)


def draw_risk_node(ax, x, y, w, h, probs, state):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.01", fc="#1e293b", ec="#7c3aed", lw=2, zorder=1))
    ax.text(x + w/2, y + h - 0.015, "RISK LEVEL",
            ha="center", va="top", fontsize=8, fontweight="bold", color="#a78bfa", zorder=4)
    prob_bars(ax, x, y, w, h, probs, RISK_COLORS, RISK_LABELS, state)


def draw_crma_badge(ax, x, y, w, h, crma_state, p_he):
    c = CRMA_COLORS.get(crma_state, "#6b7280")
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.01", fc=c, ec="white", lw=1.5, zorder=1))
    ax.text(x + w/2, y + h*0.60, crma_state.replace("_", " "),
            ha="center", va="center", fontsize=8, fontweight="bold", color="white", zorder=4)
    ax.text(x + w/2, y + h*0.25, f"P(High∪Extreme) = {p_he:.3f}",
            ha="center", va="center", fontsize=6.5, color="white", zorder=4)


def draw_arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#475569", lw=1.2),
                zorder=0)


def render(data: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Layout constants
    pw, ph = 0.175, 0.28      # parent node width, height
    gap = 0.010
    parent_y = 0.62
    parent_xs = [0.01 + i * (pw + gap) for i in range(5)]

    rw, rh = 0.28, 0.22
    rx = 0.36
    ry = 0.32

    cw, ch = 0.26, 0.12
    cx = 0.37
    cy = 0.08

    # Title
    boundary = data.get("boundary", "")
    date = data.get("date", "")
    ax.text(0.5, 0.97, f"Flood BN DAG — {boundary}   ({date})",
            ha="center", va="top", fontsize=11, fontweight="bold", color="white")

    # Parent nodes
    parent_centers = []
    for i, cfg in enumerate(PARENT_CFG):
        key = cfg["key"]
        node = data[key]
        probs = node["probs"]
        colors = [cfg["color"]] * len(probs)
        x = parent_xs[i]
        draw_node_box(ax, x, parent_y, pw, ph,
                      cfg["title"], node["raw"], probs, colors, cfg["abbr"], node["state"],
                      cfg["color"])
        parent_centers.append((x + pw/2, parent_y))

    # Risk node
    risk = data["risk"]
    draw_risk_node(ax, rx, ry, rw, rh, risk["probs"], risk["state"])
    risk_top_mid = (rx + rw/2, ry + rh)
    risk_bot_mid = (rx + rw/2, ry)

    # Arrows: parents → risk
    for (px, py) in parent_centers:
        draw_arrow(ax, px, parent_y, rx + rw/2, ry + rh)

    # CRMA badge
    crma = data["crma"]
    draw_crma_badge(ax, cx, cy, cw, ch, crma["state"], crma["p_he"])

    # Arrow: risk → crma
    draw_arrow(ax, rx + rw/2, ry, cx + cw/2, cy + ch)

    # Legend for CRMA colours
    legend_patches = [mpatches.Patch(color=c, label=s.replace("_", " "))
                      for s, c in CRMA_COLORS.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=6,
              facecolor="#1e293b", edgecolor="#475569", labelcolor="white",
              title="CRMA states", title_fontsize=6.5)

    plt.tight_layout(pad=0.3)
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",     default="output/bn-dag/bn-dag-2026-03-04.json")
    parser.add_argument("--boundary", default="KEN.30_1")
    parser.add_argument("--out",      default="output/bn-dag-nairobi-2026-03-04.png")
    args = parser.parse_args()

    with open(args.json) as f:
        dag = json.load(f)

    if args.boundary not in dag:
        keys = list(dag.keys())
        print(f"Boundary '{args.boundary}' not found. Available (first 10): {keys[:10]}")
        sys.exit(1)

    render(dag[args.boundary], args.out)


if __name__ == "__main__":
    main()
