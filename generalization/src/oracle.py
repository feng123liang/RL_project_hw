from __future__ import annotations

"""Shortest-path oracle for random delivery instances."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.instance import GridDeliveryInstance, Position


OracleState = Tuple[int, int, int, int]


@dataclass(frozen=True)
class OracleResult:
    """Exact shortest-path information for a delivery instance."""

    solvable: bool
    optimal_steps: int | None
    path: Tuple[Position, ...]
    expanded_states: int
    visited_states: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "solvable": bool(self.solvable),
            "optimal_steps": None if self.optimal_steps is None else int(self.optimal_steps),
            "path": [[int(row), int(col)] for row, col in self.path],
            "expanded_states": int(self.expanded_states),
            "visited_states": int(self.visited_states),
        }


def _neighbors(instance: GridDeliveryInstance, position: Position) -> List[Position]:
    row, col = position
    candidates = [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ]
    blocked = instance.obstacle_set
    return [
        next_pos
        for next_pos in candidates
        if 0 <= next_pos[0] < instance.rows
        and 0 <= next_pos[1] < instance.cols
        and next_pos not in blocked
    ]


def _advance_delivery_state(
    instance: GridDeliveryInstance,
    state: OracleState,
    next_position: Position,
) -> OracleState:
    _, _, active_task, delivered_mask = state
    next_active = int(active_task)
    next_mask = int(delivered_mask)

    if next_active >= 0:
        task = instance.delivery_tasks[next_active]
        if next_position == task.dropoff:
            next_mask |= 1 << next_active
            next_active = -1

    if next_active == -1:
        for task_idx, task in enumerate(instance.delivery_tasks):
            delivered = bool(next_mask & (1 << task_idx))
            if (not delivered) and (next_position == task.pickup):
                next_active = task_idx
                break

    return (int(next_position[0]), int(next_position[1]), int(next_active), int(next_mask))


def is_terminal(instance: GridDeliveryInstance, state: OracleState) -> bool:
    row, col, active_task, delivered_mask = state
    all_done_mask = (1 << instance.task_count) - 1
    return (
        (row, col) == instance.goal
        and int(active_task) == -1
        and int(delivered_mask) == all_done_mask
    )


def shortest_delivery_path(instance: GridDeliveryInstance) -> OracleResult:
    """Solve the instance exactly with BFS in the full task state space."""

    start_state: OracleState = (instance.start[0], instance.start[1], -1, 0)
    queue = deque([start_state])
    parents: Dict[OracleState, OracleState | None] = {start_state: None}
    visited = {start_state}
    expanded_states = 0

    while queue:
        state = queue.popleft()
        expanded_states += 1
        if is_terminal(instance, state):
            path = _reconstruct_path(state, parents)
            return OracleResult(
                solvable=True,
                optimal_steps=len(path) - 1,
                path=path,
                expanded_states=expanded_states,
                visited_states=len(visited),
            )

        current_position = (state[0], state[1])
        for next_position in _neighbors(instance, current_position):
            next_state = _advance_delivery_state(instance, state, next_position)
            if next_state not in visited:
                visited.add(next_state)
                parents[next_state] = state
                queue.append(next_state)

    return OracleResult(
        solvable=False,
        optimal_steps=None,
        path=tuple(),
        expanded_states=expanded_states,
        visited_states=len(visited),
    )


def _reconstruct_path(
    terminal_state: OracleState,
    parents: Dict[OracleState, OracleState | None],
) -> Tuple[Position, ...]:
    chain: List[Position] = []
    cursor: OracleState | None = terminal_state
    while cursor is not None:
        chain.append((int(cursor[0]), int(cursor[1])))
        cursor = parents[cursor]
    chain.reverse()
    return tuple(chain)
