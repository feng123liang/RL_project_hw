from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


State = Tuple[int, int]


def _save_curve(y: List[float], title: str, ylabel: str, output_path: str) -> None:
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
    cmap = plt.cm.get_cmap("tab10", 5)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    plt.title("Final Greedy Path")
    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))
    plt.grid(color="white", linestyle="-", linewidth=0.8)

    # Overlay arrows for readability.
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dr = r2 - r1
        dc = c2 - c1
        plt.arrow(
            c1,
            r1,
            0.6 * dc,
            0.6 * dr,
            head_width=0.15,
            head_length=0.15,
            fc="black",
            ec="black",
            length_includes_head=True,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
