from __future__ import annotations

"""Render a readable PDF table for the linear-Q hyperparameter sweep."""

import json
import os
import sys
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.plotting import STYLE, save_figure_pdf


def main() -> None:
    summary_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "linear_patch_q_sweep",
        "linear_patch_q_sweep_summary.json",
    )
    output_path = os.path.join(
        PROJECT_ROOT,
        "results",
        "linear_patch_q_sweep",
        "figures",
        "heldout_sweep_rankings_table.png",
    )
    data = json.load(open(summary_path, "r", encoding="utf-8"))
    top_runs = data["top_runs"][:12]

    fig, ax = plt.subplots(figsize=(16.5, 8.5), facecolor=STYLE.background)
    ax.set_axis_off()

    title_box = FancyBboxPatch(
        (0.03, 0.905),
        0.94,
        0.07,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=STYLE.title_bar,
    )
    ax.add_patch(title_box)
    ax.text(
        0.50,
        0.948,
        "Linear-Q Hyperparameter Sweep Rankings",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=STYLE.title_text,
    )
    ax.text(
        0.50,
        0.918,
        "Top held-out configurations for patch3x3_plus linear approximate Q-learning",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#d1d5db",
    )

    headers = [
        "Rank",
        "Episodes",
        "Alpha",
        "Decay",
        "TD Clip",
        "W Norm",
        "Seed",
        "Held Opt",
        "Held Loose",
        "Train Opt",
        "Train Loose",
        "Held Steps",
    ]
    col_widths = [0.06, 0.09, 0.08, 0.08, 0.09, 0.09, 0.06, 0.09, 0.10, 0.09, 0.10, 0.10]
    x0 = 0.03
    table_width = 0.94
    normalized_widths = [width / sum(col_widths) * table_width for width in col_widths]
    col_lefts: List[float] = [x0]
    for width in normalized_widths[:-1]:
        col_lefts.append(col_lefts[-1] + width)

    header_y = 0.84
    row_height = 0.056

    for left, width, header in zip(col_lefts, normalized_widths, headers):
        header_box = FancyBboxPatch(
            (left, header_y),
            width,
            row_height,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            transform=ax.transAxes,
            linewidth=0.8,
            edgecolor=STYLE.panel_edge,
            facecolor="#e7e5e4",
        )
        ax.add_patch(header_box)
        ax.text(
            left + width / 2.0,
            header_y + row_height / 2.0,
            header,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color=STYLE.text_primary,
        )

    def fmt_float(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.3g}"

    for idx, run in enumerate(top_runs, start=1):
        cfg = run["config"]
        held = run["heldout_evaluation"]
        train = run["train_evaluation"]
        values = [
            str(idx),
            str(cfg["episodes"]),
            fmt_float(cfg["alpha"]),
            fmt_float(cfg["epsilon_decay_fraction"]),
            fmt_float(cfg["max_td_error"]),
            fmt_float(cfg["max_weight_norm"]),
            str(cfg["seed"]),
            f"{held['optimal_success_rate']:.2f}",
            f"{held['loose_success_rate']:.2f}",
            f"{train['optimal_success_rate']:.2f}",
            f"{train['loose_success_rate']:.2f}",
            f"{held['mean_steps']:.2f}",
        ]
        y = header_y - idx * row_height
        row_face = "#fffdf8" if idx % 2 == 1 else "#f5f5f4"
        if idx <= 2:
            row_face = "#ecfdf5"
        elif held["optimal_success_rate"] >= 0.5:
            row_face = "#eff6ff"

        for left, width, value in zip(col_lefts, normalized_widths, values):
            cell = FancyBboxPatch(
                (left, y),
                width,
                row_height,
                boxstyle="round,pad=0.004,rounding_size=0.009",
                transform=ax.transAxes,
                linewidth=0.6,
                edgecolor=STYLE.panel_edge,
                facecolor=row_face,
            )
            ax.add_patch(cell)
            ax.text(
                left + width / 2.0,
                y + row_height / 2.0,
                value,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color=STYLE.text_primary,
                fontweight="bold" if idx <= 2 and value in {str(idx), f"{held['optimal_success_rate']:.2f}"} else "normal",
            )

    note_box = FancyBboxPatch(
        (0.03, 0.06),
        0.94,
        0.08,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor=STYLE.panel_edge,
        facecolor=STYLE.panel_bg,
    )
    ax.add_patch(note_box)
    ax.text(
        0.05,
        0.115,
        "Key takeaways: alpha=0.01 and stronger stabilization (TD clip 3.0, weight norm 80.0) dominated the sweep.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color=STYLE.text_primary,
    )
    ax.text(
        0.05,
        0.082,
        "The top 2 configurations both reached held-out optimal success 1.00 on the current 4-map evaluation set.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.5,
        color=STYLE.text_primary,
    )

    save_figure_pdf(fig, output_path)
    print(os.path.splitext(output_path)[0] + ".pdf")


if __name__ == "__main__":
    main()
