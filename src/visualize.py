from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib
import matplotlib.patheffects as pe

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


State = Tuple[int, int]

FREE_CELL_COLOR = "#F8FAFC"
OBSTACLE_COLOR = "#334155"
PATH_CELL_COLOR = "#CCFBF1"
START_COLOR = "#2563EB"
GOAL_COLOR = "#F59E0B"
PATH_LINE_COLOR = "#E11D48"
GRID_LINE_COLOR = "#CBD5E1"


def _save_curve(y: List[float], title: str, ylabel: str, output_path: str) -> None:
    x = np.arange(1, len(y) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#FFFFFF")
    ax.plot(x, y, linewidth=2.0, color="#2563EB")
    ax.set_title(title, fontsize=14, fontweight="semibold")
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(color="#CBD5E1", linestyle="--", linewidth=0.8, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curves(metrics: Dict[str, List[float]], figures_dir: str) -> None:
    _save_curve(
        metrics["episode_reward"],
        "Episode Reward",
        "Reward",
        f"{figures_dir}/reward_curve.png",
    )
    _save_curve(
        metrics["episode_reward_ma"],
        "Reward Moving Average (window=50)",
        "Reward",
        f"{figures_dir}/reward_moving_average.png",
    )
    _save_curve(
        metrics["success_rate"],
        "Success Rate",
        "Success Rate",
        f"{figures_dir}/success_rate_curve.png",
    )
    _save_curve(
        metrics["avg_steps_to_goal"],
        "Average Steps to Goal (cumulative)",
        "Steps",
        f"{figures_dir}/average_steps_curve.png",
    )


def plot_final_path(env, path: List[State], output_path: str) -> None:
    grid = env.render_grid(path)
    cmap = ListedColormap(
        [
            FREE_CELL_COLOR,
            OBSTACLE_COLOR,
            PATH_CELL_COLOR,
            START_COLOR,
            GOAL_COLOR,
        ]
    )

    fig_width = max(7.5, min(10.5, env.cols * 0.8))
    fig_height = max(7.0, min(10.0, env.rows * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(f"Final Greedy Path ({max(len(path) - 1, 0)} steps)", fontsize=15, fontweight="semibold", pad=14)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which="minor", color=GRID_LINE_COLOR, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_facecolor(FREE_CELL_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if path:
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        path_line = ax.plot(
            xs,
            ys,
            color=PATH_LINE_COLOR,
            linewidth=3.0,
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=PATH_LINE_COLOR,
            markeredgewidth=1.2,
            zorder=4,
        )[0]
        path_line.set_path_effects([pe.Stroke(linewidth=5.2, foreground="white"), pe.Normal()])

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            ax.annotate(
                "",
                xy=(c2, r2),
                xytext=(c1, r1),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": PATH_LINE_COLOR,
                    "lw": 1.2,
                    "mutation_scale": 12,
                    "shrinkA": 10,
                    "shrinkB": 10,
                    "alpha": 0.95,
                },
                zorder=5,
            )

    sr, sc = env.start
    gr, gc = env.goal
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontsize=12, fontweight="bold", zorder=6)
    ax.text(gc, gr, "G", ha="center", va="center", color="black", fontsize=12, fontweight="bold", zorder=6)

    legend_handles = [
        Patch(facecolor=FREE_CELL_COLOR, edgecolor=GRID_LINE_COLOR, label="Free Cell"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, label="Obstacle"),
        Patch(facecolor=PATH_CELL_COLOR, edgecolor=GRID_LINE_COLOR, label="Path Cell"),
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, label="Start (S)"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, label="Goal (G)"),
        Line2D(
            [0],
            [0],
            color=PATH_LINE_COLOR,
            linewidth=3.0,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=PATH_LINE_COLOR,
            label="Greedy Path",
        ),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        borderpad=0.8,
        labelspacing=0.8,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#E2E8F0")

    fig.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
