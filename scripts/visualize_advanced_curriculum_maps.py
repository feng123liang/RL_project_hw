from __future__ import annotations

"""Visualize advanced curriculum layouts with mechanism-specific markers."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from advanced_curriculum_experiment import build_env, validate_config  # noqa: E402
from utils import ensure_dir  # noqa: E402


FREE_CELL_COLOR = "#F8FAFC"
OBSTACLE_COLOR = "#334155"
START_COLOR = "#2563EB"
GOAL_COLOR = "#F59E0B"
KEY_COLOR = "#10B981"
HAZARD_COLOR = "#EF4444"
DOOR_COLOR = "#A16207"
BONUS_COLOR = "#F97316"
GRID_LINE_COLOR = "#CBD5E1"


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _draw_level(ax, env, title: str) -> None:
    grid = env.render_grid()
    cmap = ListedColormap(
        [
            FREE_CELL_COLOR,  # 0 free
            OBSTACLE_COLOR,   # 1 obstacle
            FREE_CELL_COLOR,  # 2 unused
            START_COLOR,      # 3 start
            GOAL_COLOR,       # 4 goal
            KEY_COLOR,        # 5 key
            HAZARD_COLOR,     # 6 hazard
            FREE_CELL_COLOR,  # 7 unused
            DOOR_COLOR,       # 8 door
            BONUS_COLOR,      # 9 bonus
        ]
    )
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which="minor", color=GRID_LINE_COLOR, linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    annotations = {
        env.start: ("S", "white"),
        env.goal: ("G", "black"),
    }
    for cell in env.key_cells:
        annotations[cell] = ("K", "white")
    for cell in env.bonus_cells:
        annotations[cell] = ("B", "white")
    for door in env.door_cells:
        annotations[door] = ("D", "white")
    for hazard in env.hazard_cells:
        annotations.setdefault(hazard, ("H", "white"))

    for (r, c), (label, color) in annotations.items():
        ax.text(c, r, label, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)


def render_maps(config_path: str, output_dir: str | None = None) -> str:
    cfg = load_config(config_path)
    validated_levels = validate_config(cfg)
    render_dir = output_dir or os.path.join("results", "advanced_map_visualizations", Path(config_path).stem)
    ensure_dir(render_dir)

    shared_env_cfg = cfg["shared_env"]
    level_envs: List[Tuple[str, object]] = []
    for level_name in sorted(validated_levels.keys()):
        env = build_env(shared_env_cfg, validated_levels[level_name])
        level_envs.append((level_name, env))
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_level(ax, env, level_name)
        fig.tight_layout()
        fig.savefig(os.path.join(render_dir, f"{level_name}.png"), dpi=180)
        plt.close(fig)

    n_levels = len(level_envs)
    n_cols = 2
    n_rows = int(np.ceil(n_levels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, (level_name, env) in zip(axes_flat, level_envs):
        _draw_level(ax, env, level_name)
    for ax in axes_flat[len(level_envs):]:
        ax.axis("off")

    legend_handles = [
        Patch(facecolor=FREE_CELL_COLOR, edgecolor=GRID_LINE_COLOR, label="Free Cell"),
        Patch(facecolor=OBSTACLE_COLOR, edgecolor=OBSTACLE_COLOR, label="Obstacle"),
        Patch(facecolor=START_COLOR, edgecolor=START_COLOR, label="Start"),
        Patch(facecolor=GOAL_COLOR, edgecolor=GOAL_COLOR, label="Goal"),
        Patch(facecolor=KEY_COLOR, edgecolor=KEY_COLOR, label="Key"),
        Patch(facecolor=DOOR_COLOR, edgecolor=DOOR_COLOR, label="Locked Door"),
        Patch(facecolor=BONUS_COLOR, edgecolor=BONUS_COLOR, label="Bonus"),
        Patch(facecolor=HAZARD_COLOR, edgecolor=HAZARD_COLOR, label="Timed Hazard"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Advanced Curriculum Mechanism Maps", fontsize=18, fontweight="semibold", y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(os.path.join(render_dir, "overview.png"), dpi=180)
    plt.close(fig)
    return render_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize advanced curriculum maps")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum/experiment3.yaml",
        help="Path to advanced curriculum YAML config",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Optional custom output directory")
    args = parser.parse_args()

    output_dir = render_maps(args.config, args.output_dir or None)
    print(f"Saved advanced map visualizations to {output_dir}")


if __name__ == "__main__":
    main()
