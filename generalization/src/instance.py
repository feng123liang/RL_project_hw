from __future__ import annotations

"""Problem-instance definitions for random shortest-delivery tasks."""

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Tuple


Position = Tuple[int, int]


@dataclass(frozen=True)
class DeliveryTask:
    """A single pickup-to-dropoff delivery request."""

    pickup: Position
    dropoff: Position

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            "pickup": [int(self.pickup[0]), int(self.pickup[1])],
            "dropoff": [int(self.dropoff[0]), int(self.dropoff[1])],
        }


@dataclass(frozen=True)
class GridDeliveryInstance:
    """A concrete shortest-delivery map instance.

    The project-level benchmark will sample many such instances from a map
    generator. This dataclass contains only problem definition data and no
    environment transition logic.
    """

    rows: int
    cols: int
    start: Position
    goal: Position
    obstacles: Tuple[Position, ...]
    delivery_tasks: Tuple[DeliveryTask, ...]
    max_steps: int
    split: str
    instance_id: str
    generator_seed: int
    metadata: Dict[str, float | int | str] = field(default_factory=dict)

    @property
    def obstacle_set(self) -> set[Position]:
        return set(self.obstacles)

    @property
    def pickup_points(self) -> Tuple[Position, ...]:
        return tuple(task.pickup for task in self.delivery_tasks)

    @property
    def dropoff_points(self) -> Tuple[Position, ...]:
        return tuple(task.dropoff for task in self.delivery_tasks)

    @property
    def task_count(self) -> int:
        return len(self.delivery_tasks)

    def free_cells(self) -> List[Position]:
        blocked = self.obstacle_set
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in blocked
        ]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["start"] = [int(self.start[0]), int(self.start[1])]
        payload["goal"] = [int(self.goal[0]), int(self.goal[1])]
        payload["obstacles"] = [[int(row), int(col)] for row, col in self.obstacles]
        payload["delivery_tasks"] = [task.to_dict() for task in self.delivery_tasks]
        return payload


def normalize_positions(items: Iterable[Iterable[int]]) -> Tuple[Position, ...]:
    """Normalize nested integer iterables into immutable grid positions."""

    return tuple(tuple(int(value) for value in item) for item in items)
