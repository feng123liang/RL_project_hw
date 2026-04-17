from __future__ import annotations

"""Advanced grid environment for curriculum experiments.

This version keeps the task fully discrete for tabular Q-learning while using
only movement, key-door access control, timed hazards, and bonus cells.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


Action = int
Position = Tuple[int, int]
RichState = Tuple[int, ...]


@dataclass
class AdvancedRewardConfig:
    """Reward configuration for the advanced curriculum environment."""

    step: float = -0.5
    invalid: float = -1.0
    hazard: float = -12.0
    bonus: float = 8.0
    goal: float = 50.0


@dataclass
class AdvancedLevelSpec:
    """Per-level map and mechanism specification."""

    obstacles: List[Position]
    key_cells: List[Position]
    door_cells: List[Position]
    hazard_cells: List[Position]
    bonus_cells: List[Position]


class AdvancedGridWorldEnv:
    """Grid task with keys, locked doors, timed hazards, and bonus cells."""

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }
    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        start: Position,
        goal: Position,
        level: AdvancedLevelSpec,
        rewards: AdvancedRewardConfig,
        phase_cycle: int,
        hazard_active_phases: Sequence[int],
    ) -> None:
        self.rows = int(rows)
        self.cols = int(cols)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.rewards = rewards
        self.phase_cycle = int(phase_cycle)
        self.hazard_active_phases = {int(phase) for phase in hazard_active_phases}

        self.obstacles = set(level.obstacles)
        self.key_cells = {tuple(cell) for cell in level.key_cells}
        self.door_cells = {tuple(cell) for cell in level.door_cells}
        self.hazard_cells = {tuple(cell) for cell in level.hazard_cells}
        self.bonus_cells = [tuple(cell) for cell in level.bonus_cells]
        self.bonus_index = {cell: idx for idx, cell in enumerate(self.bonus_cells)}

        if self.start in self.obstacles:
            raise ValueError("start cannot be an obstacle")
        if self.goal in self.obstacles:
            raise ValueError("goal cannot be an obstacle")
        if self.goal in self.door_cells:
            raise ValueError("goal cannot be a locked door cell")

        for label, cells in (
            ("key", self.key_cells),
            ("door", self.door_cells),
            ("hazard", self.hazard_cells),
            ("bonus", set(self.bonus_cells)),
        ):
            for cell in cells:
                if cell in self.obstacles:
                    raise ValueError(f"{label} cannot be inside an obstacle")

        self.position: Position = self.start
        self.has_key = 0
        self.phase = 0
        self.bonus_mask = 0

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    @property
    def bonus_state_size(self) -> int:
        return 1 << len(self.bonus_cells)

    @property
    def state_shape(self) -> Tuple[int, int, int, int, int]:
        """Return the factor sizes used by the factored Q-table."""

        return (
            self.rows,
            self.cols,
            2,
            self.phase_cycle,
            self.bonus_state_size,
        )

    def reset(self) -> RichState:
        """Reset the environment to the canonical initial rich state."""

        self.position = self.start
        self.has_key = 0
        self.phase = 0
        self.bonus_mask = 0
        return self._state()

    def _state(self) -> RichState:
        return (
            self.position[0],
            self.position[1],
            int(self.has_key),
            int(self.phase),
            int(self.bonus_mask),
        )

    def in_bounds(self, position: Position) -> bool:
        r, c = position
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_locked_door(self, position: Position, has_key: int) -> bool:
        return position in self.door_cells and not bool(has_key)

    def is_hazard_active(self, phase: int) -> bool:
        return int(phase) in self.hazard_active_phases

    def transition_from_state(
        self,
        state: RichState,
        action: Action,
    ) -> Tuple[RichState, float, bool, Dict[str, bool]]:
        """Pure transition helper used both by `step` and config validation."""

        if action not in self.ACTIONS:
            raise ValueError(f"invalid action: {action}")

        row, col, has_key, phase, bonus_mask = state
        position = (row, col)
        reward = float(self.rewards.step)
        done = False
        next_position = position
        next_has_key = int(has_key)
        next_phase = (int(phase) + 1) % self.phase_cycle
        next_bonus_mask = int(bonus_mask)

        info = {
            "invalid_move": False,
            "picked_key": False,
            "used_key_door": False,
            "hazard_triggered": False,
            "reached_goal": False,
            "collected_bonus": False,
        }

        dr, dc = self.ACTIONS[action]
        candidate = (position[0] + dr, position[1] + dc)
        if (not self.in_bounds(candidate)) or (candidate in self.obstacles) or self.is_locked_door(
            candidate,
            next_has_key,
        ):
            reward = float(self.rewards.invalid)
            info["invalid_move"] = True
        else:
            next_position = candidate
            if candidate in self.door_cells and next_has_key:
                info["used_key_door"] = True

        if next_position in self.key_cells and not next_has_key:
            next_has_key = 1
            info["picked_key"] = True

        if next_position in self.bonus_index:
            idx = self.bonus_index[next_position]
            bit = 1 << idx
            if not (next_bonus_mask & bit):
                next_bonus_mask |= bit
                reward += float(self.rewards.bonus)
                info["collected_bonus"] = True

        if next_position in self.hazard_cells and self.is_hazard_active(phase):
            reward = float(self.rewards.hazard)
            done = True
            info["hazard_triggered"] = True

        if next_position == self.goal and not done:
            reward = float(self.rewards.goal)
            done = True
            info["reached_goal"] = True

        next_state = (
            next_position[0],
            next_position[1],
            int(next_has_key),
            int(next_phase),
            int(next_bonus_mask),
        )
        return next_state, reward, done, info

    def step(self, action: Action) -> Tuple[RichState, float, bool, Dict[str, bool]]:
        """Advance the live environment by one action."""

        next_state, reward, done, info = self.transition_from_state(self._state(), action)
        self.position = (next_state[0], next_state[1])
        self.has_key = next_state[2]
        self.phase = next_state[3]
        self.bonus_mask = next_state[4]
        return next_state, reward, done, info

    def render_grid(self) -> np.ndarray:
        """Encode the current map layout into integers for plotting."""

        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r, c in self.obstacles:
            grid[r, c] = 1
        for r, c in self.hazard_cells:
            if (r, c) not in self.obstacles:
                grid[r, c] = 6
        for r, c in self.key_cells:
            grid[r, c] = 5
        for r, c in self.bonus_cells:
            grid[r, c] = 9
        for r, c in self.door_cells:
            grid[r, c] = 8

        sr, sc = self.start
        gr, gc = self.goal
        grid[sr, sc] = 3
        grid[gr, gc] = 4
        return grid


def normalize_positions(items: Iterable[Iterable[int]] | None) -> List[Position]:
    if not items:
        return []
    return [tuple(int(x) for x in item) for item in items]
