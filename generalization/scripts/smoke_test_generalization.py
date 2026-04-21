from __future__ import annotations

"""Smoke tests for the generalization infrastructure."""

import json
import os
import sys
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.generator import GeneratorConfig, sample_instance_batch
from src.instance import DeliveryTask, GridDeliveryInstance
from src.oracle import shortest_delivery_path
from src.plotting import plot_delivery_map


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_instance(
    instance: GridDeliveryInstance,
    path,
    title: str,
    output_path: str,
    shortest_path_steps: int | None = None,
) -> Dict[str, str]:
    subtitle = "Randomly generated solvable delivery map with exact shortest-path oracle"
    return plot_delivery_map(
        instance,
        path,
        title=title,
        output_path=output_path,
        subtitle=subtitle,
        path_label="Oracle Path",
        show_direction_arrows=True,
        shortest_path_steps=shortest_path_steps,
        agent_steps=max(0, len(path) - 1) if path else None,
    )


def random_generation_smoke(figures_dir: str) -> List[Dict[str, Any]]:
    cfg = GeneratorConfig(
        rows_min=6,
        rows_max=6,
        cols_min=6,
        cols_max=6,
        obstacle_density_min=0.10,
        obstacle_density_max=0.18,
        task_count_min=1,
        task_count_max=2,
        min_optimal_steps=6,
        max_generation_attempts=200,
    )
    examples = sample_instance_batch(cfg, seeds=[11, 12, 13], split="train")
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(examples, start=1):
        figure_paths = plot_instance(
            item.instance,
            item.oracle.path,
            title=f"Random Delivery Example {idx} ({item.instance.instance_id})",
            output_path=os.path.join(figures_dir, f"random_example_{idx}.png"),
            shortest_path_steps=item.oracle.optimal_steps,
        )
        rows.append(
            {
                "instance": item.instance.to_dict(),
                "oracle": item.oracle.to_dict(),
                "figures": figure_paths,
            }
        )
    return rows


def manual_oracle_smoke(figures_dir: str) -> Dict[str, Any]:
    instance = GridDeliveryInstance(
        rows=4,
        cols=4,
        start=(0, 0),
        goal=(3, 3),
        obstacles=tuple(),
        delivery_tasks=(DeliveryTask(pickup=(0, 3), dropoff=(3, 0)),),
        max_steps=100,
        split="unit",
        instance_id="unit_manual",
        generator_seed=0,
        metadata={},
    )
    result = shortest_delivery_path(instance)

    if not result.solvable:
        raise AssertionError("Manual instance should be solvable")
    if result.path[0] != (0, 0) or result.path[-1] != (3, 3):
        raise AssertionError("Manual path endpoints are incorrect")
    if (0, 3) not in result.path or (3, 0) not in result.path:
        raise AssertionError("Manual path should include pickup and dropoff")
    if not (result.path.index((0, 3)) < result.path.index((3, 0)) < len(result.path) - 1):
        raise AssertionError("Pickup must happen before dropoff, and dropoff before terminal goal")

    figure_paths = plot_instance(
        instance,
        result.path,
        title="Manual Oracle Check",
        output_path=os.path.join(figures_dir, "manual_oracle_check.png"),
        shortest_path_steps=result.optimal_steps,
    )
    return {
        "instance": instance.to_dict(),
        "oracle": result.to_dict(),
        "figures": figure_paths,
        "checks": {
            "pickup_before_dropoff": True,
            "dropoff_before_goal_terminal": True,
        },
    }


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "results", "generalization_smoke")
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    payload = {
        "random_generation_examples": random_generation_smoke(figures_dir),
        "manual_oracle_check": manual_oracle_smoke(figures_dir),
    }
    output_path = os.path.join(output_dir, "smoke_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
