from __future__ import annotations

"""Environment for random shortest-delivery tasks."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.instance import GridDeliveryInstance, Position


Action = int
EnvState = Tuple[int, int, int, int]


@dataclass(frozen=True)
class RewardConfig:
    """Reward configuration for delivery environments."""

    step: float = -0.2
    invalid: float = -1.0
    pickup: float = 1.0
    dropoff: float = 2.0
    goal: float = 5.0


class GridDeliveryEnv:
    """Deterministic grid delivery environment with sequential task logic."""

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

    def __init__(self, instance: GridDeliveryInstance, rewards: RewardConfig | None = None) -> None:
        self.instance = instance
        self.rewards = rewards or RewardConfig()
        self.n_actions = len(self.ACTIONS)
        self.steps = 0
        self.position: Position = instance.start
        self.active_task = -1
        self.delivered_mask = 0

    @property
    def rows(self) -> int:
        return int(self.instance.rows)

    @property
    def cols(self) -> int:
        return int(self.instance.cols)

    @property
    def state_shape(self) -> Tuple[int, int, int, int]:
        delivered_states = 1 << self.instance.task_count
        active_states = self.instance.task_count + 1  # +1 for idle / carrying none
        return (self.rows, self.cols, active_states, delivered_states)

    def reset(self) -> EnvState:
        self.steps = 0
        self.position = self.instance.start
        self.active_task = -1
        self.delivered_mask = 0
        return self.state()

    def state(self) -> EnvState:
        active_index = self.instance.task_count if self.active_task == -1 else int(self.active_task)
        return (
            int(self.position[0]),
            int(self.position[1]),
            int(active_index),
            int(self.delivered_mask),
        )

    def in_bounds(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_blocked(self, position: Position) -> bool:
        return position in self.instance.obstacle_set

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.int32)
        for action, (dr, dc) in self.ACTIONS.items():
            candidate = (self.position[0] + dr, self.position[1] + dc)
            if self.in_bounds(candidate) and not self.is_blocked(candidate):
                mask[action] = 1
        return mask

    def valid_actions(self) -> List[int]:
        mask = self.action_mask()
        return [action for action in range(self.n_actions) if mask[action] == 1]

    def _advance_task_logic(self, position: Position) -> Dict[str, bool]:
        info = {
            "picked_up": False,
            "dropped_off": False,
            "completed_all_deliveries": False,
        }

        if self.active_task >= 0:
            task = self.instance.delivery_tasks[self.active_task]
            if position == task.dropoff:
                self.delivered_mask |= 1 << self.active_task
                self.active_task = -1
                info["dropped_off"] = True

        if self.active_task == -1:
            for task_idx, task in enumerate(self.instance.delivery_tasks):
                delivered = bool(self.delivered_mask & (1 << task_idx))
                if (not delivered) and position == task.pickup:
                    self.active_task = task_idx
                    info["picked_up"] = True
                    break

        all_done_mask = (1 << self.instance.task_count) - 1
        info["completed_all_deliveries"] = self.delivered_mask == all_done_mask and self.active_task == -1
        return info

    def step(self, action: Action) -> Tuple[EnvState, float, bool, Dict[str, bool | int | str]]:
        if action not in self.ACTIONS:
            raise ValueError(f"invalid action: {action}")

        self.steps += 1
        dr, dc = self.ACTIONS[action]
        candidate = (self.position[0] + dr, self.position[1] + dc)
        reward = float(self.rewards.step)
        done = False
        info: Dict[str, bool | int | str] = {
            "invalid_move": False,
            "picked_up": False,
            "dropped_off": False,
            "completed_all_deliveries": False,
            "reached_goal": False,
            "success": False,
            "time_limit": False,
            "active_task": self.active_task,
            "action_name": self.ACTION_NAMES[action],
        }

        if (not self.in_bounds(candidate)) or self.is_blocked(candidate):
            reward = float(self.rewards.invalid)
            info["invalid_move"] = True
            next_position = self.position
        else:
            next_position = candidate

        self.position = next_position
        task_info = self._advance_task_logic(next_position)
        info.update(task_info)

        if task_info["picked_up"]:
            reward += float(self.rewards.pickup)
        if task_info["dropped_off"]:
            reward += float(self.rewards.dropoff)
        if next_position == self.instance.goal:
            info["reached_goal"] = True
            if task_info["completed_all_deliveries"]:
                reward += float(self.rewards.goal)
                done = True
                info["success"] = True

        if self.steps >= self.instance.max_steps and not done:
            done = True
            info["time_limit"] = True

        info["active_task"] = self.active_task
        return self.state(), float(reward), bool(done), info

    def render_grid(self, path: List[Position] | None = None) -> np.ndarray:
        """Render static map geometry for plotting helpers."""

        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        for row, col in self.instance.obstacles:
            grid[row, col] = 1
        if path:
            for row, col in path:
                if (row, col) not in self.instance.obstacle_set:
                    grid[row, col] = max(grid[row, col], 2)
        sr, sc = self.instance.start
        gr, gc = self.instance.goal
        grid[sr, sc] = 3
        grid[gr, gc] = 4
        return grid
