#!/usr/bin/env -S uv run --with matplotlib --with numpy --with pandas
"""
plot_nairobi_cluster_diagnostic.py
Nairobi + surrounding-counties diagnostic for the 6 March 2026 flood event.

Reads bn-dag JSONs for 1..10 March 2026 and produces:
  1. output/nairobi_cluster_diagnostic.png — two-panel figure:
       (top) p_he time-series for the 5 counties + cluster-max line
       (bot) tail-risk ratio + hotspot fraction time-series for the 5 counties
  2. output/nairobi_cluster_table.tex — LaTeX longtable with one row per day
       and columns showing each county's CRMA, p_he, and tail× RP.
  3. output/nairobi_cluster_table.csv — same data as plain CSV.

Counties (admin-1, Kenya):
  KEN.30_1 Nairobi      — the city itself (small footprint)
  KEN.13_1 Kiambu       — north & west of Nairobi
  KEN.22_1 Machakos     — east & south
  KEN.10_1 Kajiado      — south & southwest
  KEN.29_1 Murang'a     — north of Kiambu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

COUNTIES = [
    ("KEN.30_1", "Nairobi",  "#dc2626"),  # red, the focal admin
    ("KEN.13_1", "Kiambu",   "#2563eb"),  # blue
    ("KEN.22_1", "Machakos", "#16a34a"),  # green
    ("KEN.10_1", "Kajiado",  "#a855f7"),  # purple
    ("KEN.29_1", "Murang'a", "#f59e0b"),  # orange
]

CRMA_COLORS = {"Monitor": "#22c55e", "Evaluate": "#eab308",
               "Assess": "#f97316", "Actionable_Risk": "#dc2626"}

EVENT_DATE = "2026-03-06"


def load_records(dag_dir: Path, dates: list[str]) -> pd.DataFrame:
    rows = []
    for d in dates:
        path = dag_dir / f"bn-dag-{d}.json"
        if not path.exists():
            continue
        dag = json.loads(path.read_text())
        for bid, name, _ in COUNTIES:
            if bid not in dag:
                continue
            n = dag[bid]
            rows.append(dict(
                date=pd.Timestamp(d), bid=bid, name=name,
                ant_state=n["ant"]["state"], ant_raw=n["ant"]["raw"],
                exc_state=n["exc"]["state"], exc_raw=n["exc"]["raw"],
                spa_state=n["spa"]["state"], spa_raw=n["spa"]["raw"],
                trn_state=n["trn"]["state"], trn_raw=n["trn"]["raw"],
                tail_state=n["tail"]["state"], tail_raw=n["tail"]["raw"],
                # Numeric extracts for plotting
                ant_mm=float(n["ant"]["raw"].split()[0]) if "mm" in n["ant"]["raw"] else float("nan"),
                hotspot=float(n["spa"]["raw"].split("%")[0]) / 100.0 if "%" in n["spa"]["raw"] else float("nan"),
                tail_rp=float(n["tail"]["raw"].split("×")[0]) if "×" in n["tail"]["raw"] else float("nan"),
                risk=n["risk"]["state"], crma=n["crma"]["state"],
                p_he=n["crma"]["p_he"],
            ))
    return pd.DataFrame(rows)


def plot_timeseries(df: pd.DataFrame, out: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor("white")

    # Panel 1: P(High v Extreme) for each county + cluster max
    cluster_max = df.groupby("date")["p_he"].max()
    for bid, name, color in COUNTIES:
        sub = df[df["bid"] == bid].sort_values("date")
        ax1.plot(sub["date"], sub["p_he"], "-o",
                 color=color, label=name, lw=1.6, ms=5)
    ax1.plot(cluster_max.index, cluster_max.values, "--",
             color="#111827", lw=1.4, alpha=0.6, label="cluster max")
    # Decision thresholds
    ax1.axhline(0.20, color="#dc2626", lw=0.8, ls=":", alpha=0.7)
    ax1.text(df["date"].min(), 0.205, " AR threshold (C/L=0.20)",
             color="#dc2626", fontsize=8, va="bottom", ha="left")
    ax1.axhline(0.10, color="#f97316", lw=0.8, ls=":", alpha=0.7)
    ax1.text(df["date"].min(), 0.105, " Assess threshold",
             color="#f97316", fontsize=8, va="bottom", ha="left")

    ax1.axvline(pd.Timestamp(EVENT_DATE), color="#dc2626", lw=1.5, alpha=0.5)
    ax1.text(pd.Timestamp(EVENT_DATE), ax1.get_ylim()[1] * 0.95,
             "  6 March\n  flood event",
             color="#dc2626", fontsize=8, va="top", fontweight="bold")
    ax1.set_ylabel("P(High ∪ Extreme)")
    ax1.set_title("Greater Nairobi cluster — BN flood-risk evolution, 1–10 March 2026",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8, ncol=3, frameon=True, framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Tail-risk ratio (left axis) + hotspot fraction (right axis)
    for bid, name, color in COUNTIES:
        sub = df[df["bid"] == bid].sort_values("date")
        ax2.plot(sub["date"], sub["tail_rp"], "-o",
                 color=color, label=name, lw=1.6, ms=5)
    ax2.axhline(1.0, color="#475569", lw=0.8, ls="-", alpha=0.6)
    ax2.text(df["date"].min(), 1.02, " 2-yr return-period threshold",
             color="#475569", fontsize=8, va="bottom", ha="left")
    ax2.axvline(pd.Timestamp(EVENT_DATE), color="#dc2626", lw=1.5, alpha=0.5)
    ax2.set_ylabel("ECMWF tail risk (× return period)")
    ax2.set_xlabel("date")
    ax2.legend(loc="upper right", fontsize=8, ncol=3, frameon=True, framealpha=0.95)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate(rotation=0, ha="center")

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def write_latex_table(df: pd.DataFrame, out: Path):
    """Wide table: one row per date, columns for each county's CRMA/p_he/tail× RP."""
    # Pivot
    dates = sorted(df["date"].unique())
    lines = []
    lines.append(r"\begin{table*}[!tbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Greater-Nairobi cluster diagnostic, 1--10 March 2026. "
                 r"For each day and each of five neighbouring admin-1 counties (Nairobi "
                 r"and the surrounding Kiambu / Machakos / Kajiado / Murang'a) the table "
                 r"shows the CRMA state, $P(\mathrm{High}\cup\mathrm{Extreme})$, and the "
                 r"ECMWF ensemble-max tail ratio (factor of the 2-yr return-period threshold). "
                 r"AR\,$=$\,Actionable\_Risk, As\,$=$\,Assess, Ev\,$=$\,Evaluate, "
                 r"Mn\,$=$\,Monitor. Bold rows are the 6--7 March 2026 flash-flood event days; "
                 r"the cluster-max column is the per-day maximum across the five boundaries. "
                 r"Nairobi proper never crosses the AR threshold (C/L\,$=$\,0.20); the spatial "
                 r"signal materialises 30--80\,km away in Kiambu and Kajiado on 1 March, then "
                 r"decays as the ECMWF forecast loses the convective-burst signal by 5--6 March.}")
    lines.append(r"\label{tab:nairobi_cluster}")
    # 1 date col + 5 (CRMA / p_he / tail) + cluster-max col
    lines.append(r"\begin{tabular}{@{}l|rrr|rrr|rrr|rrr|rrr|r@{}}")
    lines.append(r"\toprule")
    # Header rows
    head1 = ["Date"]
    for _, name, _ in COUNTIES:
        head1.append(r"\multicolumn{3}{c|}{" + name + r"}")
    head1.append(r"max")
    lines.append(" & ".join(head1) + r" \\")
    head2 = [""]
    for _ in COUNTIES:
        head2.extend(["CRMA", r"$P_{HE}$", r"tail$\times$"])
    head2.append(r"$P_{HE}$")
    lines.append(" & ".join(head2) + r" \\")
    lines.append(r"\midrule")

    crma_short = {"Actionable_Risk": "AR", "Assess": "As",
                  "Evaluate": "Ev", "Monitor": "Mn"}
    for d in dates:
        sub = df[df["date"] == d].set_index("bid")
        is_event = pd.Timestamp(d) in (pd.Timestamp("2026-03-06"), pd.Timestamp("2026-03-07"))
        row = [d.strftime("%m-%d")]
        cluster_max = 0.0
        for bid, _, _ in COUNTIES:
            if bid not in sub.index:
                row.extend(["--", "--", "--"])
                continue
            r = sub.loc[bid]
            row.append(crma_short.get(r["crma"], r["crma"]))
            row.append(f"{r['p_he']:.2f}")
            row.append(f"{r['tail_rp']:.2f}")
            cluster_max = max(cluster_max, r["p_he"])
        row.append(f"{cluster_max:.2f}")
        line = " & ".join(row)
        if is_event:
            line = r"\textbf{" + " & ".join(
                r"\textbf{" + c + "}" for c in row
            ).replace(r"\textbf{\textbf{", r"\textbf{").replace(r"}}", "}") + r"}"
            # Simpler: just bold the row by wrapping each cell
            line = " & ".join(r"\textbf{" + c + "}" for c in row)
        lines.append(line + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dag-dir", default="output/bn-dag")
    ap.add_argument("--start", default="2026-03-01")
    ap.add_argument("--end",   default="2026-03-10")
    ap.add_argument("--out-png",   default="output/nairobi_cluster_diagnostic.png")
    ap.add_argument("--out-tex",   default="output/nairobi_cluster_table.tex")
    ap.add_argument("--out-csv",   default="output/nairobi_cluster_table.csv")
    args = ap.parse_args()

    dates = [d.strftime("%Y-%m-%d")
             for d in pd.date_range(args.start, args.end, freq="D")]
    df = load_records(Path(args.dag_dir), dates)

    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    plot_timeseries(df, Path(args.out_png))
    write_latex_table(df, Path(args.out_tex))


if __name__ == "__main__":
    main()
