from __future__ import annotations

"""Random benchmark instance generator for shortest-delivery tasks."""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.instance import DeliveryTask, GridDeliveryInstance, Position
from src.oracle import OracleResult, shortest_delivery_path


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for random map generation."""

    rows_min: int = 8
    rows_max: int = 8
    cols_min: int = 8
    cols_max: int = 8
    obstacle_density_min: float = 0.10
    obstacle_density_max: float = 0.20
    task_count_min: int = 1
    task_count_max: int = 2
    max_steps_factor: float = 4.0
    min_optimal_steps: int = 8
    max_generation_attempts: int = 400


@dataclass(frozen=True)
class GeneratedInstance:
    """Container returned by the generator with oracle annotation attached."""

    instance: GridDeliveryInstance
    oracle: OracleResult


def sample_instance(
    config: GeneratorConfig,
    seed: int,
    split: str = "train",
) -> GeneratedInstance:
    """Sample a legal instance and solve it exactly with the oracle."""

    rng = random.Random(int(seed))
    for attempt in range(1, config.max_generation_attempts + 1):
        rows = int(rng.randint(config.rows_min, config.rows_max))
        cols = int(rng.randint(config.cols_min, config.cols_max))
        density = float(rng.uniform(config.obstacle_density_min, config.obstacle_density_max))
        task_count = int(rng.randint(config.task_count_min, config.task_count_max))

        instance = _build_candidate_instance(
            rows=rows,
            cols=cols,
            obstacle_density=density,
            task_count=task_count,
            split=split,
            seed=seed,
            attempt=attempt,
            max_steps_factor=config.max_steps_factor,
            rng=rng,
        )
        oracle = shortest_delivery_path(instance)
        if oracle.solvable and int(oracle.optimal_steps) >= int(config.min_optimal_steps):
            return GeneratedInstance(instance=instance, oracle=oracle)

    raise RuntimeError(
        f"Failed to sample a solvable delivery instance within {config.max_generation_attempts} attempts"
    )


def _build_candidate_instance(
    rows: int,
    cols: int,
    obstacle_density: float,
    task_count: int,
    split: str,
    seed: int,
    attempt: int,
    max_steps_factor: float,
    rng: random.Random,
) -> GridDeliveryInstance:
    cells = [(row, col) for row in range(rows) for col in range(cols)]
    min_reserved = 2 + 2 * task_count
    max_obstacles = max(0, len(cells) - min_reserved)
    obstacle_count = min(max_obstacles, max(0, int(math.floor(len(cells) * obstacle_density))))

    obstacle_cells = set(rng.sample(cells, obstacle_count)) if obstacle_count > 0 else set()
    free_cells = [cell for cell in cells if cell not in obstacle_cells]
    if len(free_cells) < min_reserved:
        raise RuntimeError("Not enough free cells to place start/goal/tasks")

    selected = rng.sample(free_cells, min_reserved)
    start = selected[0]
    goal = selected[1]
    tasks: List[DeliveryTask] = []
    for task_idx in range(task_count):
        pickup = selected[2 + 2 * task_idx]
        dropoff = selected[3 + 2 * task_idx]
        tasks.append(DeliveryTask(pickup=pickup, dropoff=dropoff))

    max_steps = max(int(rows * cols * max_steps_factor), int(rows + cols))
    metadata: Dict[str, float | int | str] = {
        "obstacle_density": float(obstacle_density),
        "requested_task_count": int(task_count),
        "generation_attempt": int(attempt),
    }
    return GridDeliveryInstance(
        rows=int(rows),
        cols=int(cols),
        start=start,
        goal=goal,
        obstacles=tuple(sorted(obstacle_cells)),
        delivery_tasks=tuple(tasks),
        max_steps=int(max_steps),
        split=str(split),
        instance_id=f"{split}_seed{seed}_attempt{attempt}",
        generator_seed=int(seed),
        metadata=metadata,
    )


def sample_instance_batch(
    config: GeneratorConfig,
    seeds: Sequence[int],
    split: str,
) -> List[GeneratedInstance]:
    """Generate a reproducible batch of benchmark instances."""

    return [sample_instance(config=config, seed=int(seed), split=split) for seed in seeds]
