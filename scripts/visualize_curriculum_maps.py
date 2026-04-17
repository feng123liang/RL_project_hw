from __future__ import annotations

"""Render curriculum map layouts from a YAML config into a standalone output folder."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from env import GridWorldEnv, RewardConfig, State  # noqa: E402
from utils import ensure_dir  # noqa: E402


FREE_CELL_COLOR = "#F8FAFC"
OBSTACLE_COLOR = "#334155"
START_COLOR = "#2563EB"
GOAL_COLOR = "#F59E0B"
GRID_LINE_COLOR = "#CBD5E1"


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_obstacles(obstacles: Iterable[Iterable[int]]) -> List[State]:
    return [tuple(int(x) for x in item) for item in obstacles]


def build_env(shared_env_cfg: Dict, obstacles: List[State]) -> GridWorldEnv:
    rewards_cfg = shared_env_cfg["rewards"]
    return GridWorldEnv(
        rows=int(shared_env_cfg["rows"]),
        cols=int(shared_env_cfg["cols"]),
        start=tuple(shared_env_cfg["start"]),
        goal=tuple(shared_env_cfg["goal"]),
        obstacles=obstacles,
        rewards=RewardConfig(
            step=float(rewards_cfg["step"]),
            invalid=float(rewards_cfg["invalid"]),
            goal=float(rewards_cfg["goal"]),
        ),
        max_steps=int(shared_env_cfg["max_steps"]),
    )


def _draw_map_on_axis(ax, env: GridWorldEnv, title: str) -> None:
    grid = env.render_grid(path=None)
    cmap = ListedColormap(
        [
            FREE_CELL_COLOR,
            OBSTACLE_COLOR,
            FREE_CELL_COLOR,
            START_COLOR,
            GOAL_COLOR,
        ]
    )

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4)
    ax.set_title(title, fontsize=14, fontweight="semibold", pad=10)
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

    sr, sc = env.start
    gr, gc = env.goal
    ax.text(sc, sr, "S", ha="center", va="center", color="white", fontsize=12, fontweight="bold", zorder=5)
    ax.text(gc, gr, "G", ha="center", va="center", color="black", fontsize=12, fontweight="bold", zorder=5)


def save_single_map_figure(env: GridWorldEnv, level_name: str, obstacle_count: int, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_map_on_axis(ax, env, f"{level_name} ({obstacle_count} obstacles)")

    legend_handles = [
        Patch(facecolor=FREE_CELL_COLOR, edgecolor=GRID_LINE_COLOR, label="Free Cell"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, label="Obstacle"),
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, label="Start (S)"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, label="Goal (G)"),
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#E2E8F0")

    fig.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_overview_figure(level_envs: List[Tuple[str, GridWorldEnv, int]], output_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for ax, (level_name, env, obstacle_count) in zip(axes_flat, level_envs):
        _draw_map_on_axis(ax, env, f"{level_name} ({obstacle_count} obstacles)")

    legend_handles = [
        Patch(facecolor=FREE_CELL_COLOR, edgecolor=GRID_LINE_COLOR, label="Free Cell"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, label="Obstacle"),
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, label="Start (S)"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, label="Goal (G)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        bbox_to_anchor=(0.5, 0.02),
    )

    fig.suptitle("Curriculum Map Overview", fontsize=18, fontweight="semibold", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_maps(config_path: str, output_dir: str | None = None) -> str:
    cfg = load_config(config_path)
    shared_env_cfg = cfg["shared_env"]
    map_cfg = cfg["maps"]

    config_stem = Path(config_path).stem
    render_dir = output_dir or os.path.join("results", "map_visualizations", config_stem)
    ensure_dir(render_dir)

    ordered_levels = sorted(map_cfg.keys())
    level_envs: List[Tuple[str, GridWorldEnv, int]] = []

    for level_name in ordered_levels:
        obstacles = normalize_obstacles(map_cfg[level_name]["obstacles"])
        env = build_env(shared_env_cfg, obstacles)
        level_envs.append((level_name, env, len(obstacles)))
        save_single_map_figure(
            env,
            level_name=level_name,
            obstacle_count=len(obstacles),
            output_path=os.path.join(render_dir, f"{level_name}.png"),
        )

    save_overview_figure(level_envs, os.path.join(render_dir, "overview.png"))
    return render_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize curriculum map layouts from a config file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum/experiment2.yaml",
        help="Path to the curriculum YAML config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to results/map_visualizations/<config-name>",
    )
    args = parser.parse_args()

    output_dir = render_maps(args.config, args.output_dir or None)
    print(f"Saved map visualizations to {output_dir}")


if __name__ == "__main__":
    main()
