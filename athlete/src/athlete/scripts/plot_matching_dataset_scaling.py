"""Plot mean matching success for candidate budgets K=1,2,4,8.

Every curve uses the same candidate-budget experiment. Do not mix in the older
single-trajectory experiment, which used a different sampling distribution.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Candidate-budget output directory containing summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.results / "summary.json").read_text())
    protocol = json.loads((args.results / "protocol.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budgets = [1, 2, 4, 8]
    sizes = np.array([100, 200, 500, 1000])
    indexed = {(r["size"], r["budget"]): r for r in rows}
    plot_rows = []
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fig.subplots_adjust(left=0.12, right=0.975, top=0.91, bottom=0.23)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", "-.", ":"]
    for budget, color, marker, linestyle in zip(budgets, colors, markers, linestyles):
        mean = np.array([100 * indexed[(int(n), budget)]["rate_mean"] for n in sizes])
        assert np.isfinite(mean).all() and ((mean >= 0) & (mean <= 100)).all()
        ax.plot(sizes, mean, label=f"K = {budget}", color=color,
                marker=marker, markersize=5,
                markerfacecolor=color,
                markeredgewidth=1.1, linestyle=linestyle, linewidth=1.8,
                clip_on=False)
        plot_rows.extend({"reference_motions_N": int(n), "candidate_budget_K": budget,
                          "mean_success_percent": float(value),
                          "test_candidate_groups": protocol["groups"]}
                         for n, value in zip(sizes, mean))
    ax.set(xlabel="Number of reference motions N",
           ylabel="Candidate-group matching success (%)",
           xticks=sizes, xlim=(50, 1050), ylim=(0, 105))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, ncols=2, fontsize=9,
              handlelength=2.7, columnspacing=1.4, labelspacing=0.7)
    fig.suptitle("Reference-library size and candidate budget", fontsize=12, y=0.985)
    fig.text(0.12, 0.065,
             f"{protocol['groups']:,} fixed candidate groups; distance threshold = {protocol['threshold_m']:.1f} m.\n"
             "Mean across 3 nested subsets. K = maximum candidate trajectories per group.",
             fontsize=8, color="#4A4A4A")
    fig.savefig(args.output_dir / "matching_scaling.pdf")
    fig.savefig(args.output_dir / "matching_scaling.png", dpi=240)
    plt.close(fig)
    with (args.output_dir / "matching_scaling_plot_data.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(plot_rows[0]))
        writer.writeheader()
        writer.writerows(plot_rows)


if __name__ == "__main__":
    main()
