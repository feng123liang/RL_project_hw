from __future__ import annotations

"""GridWorld 环境定义。

该文件负责描述机器人送货任务所处的二维网格世界，包括：
- 状态与动作的表示方式
- 越界/撞障碍时的转移规则
- 到达目标、非法移动、普通移动的奖励设计
- 路径可视化前所需的网格渲染
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


Action = int
State = Tuple[int, int]


@dataclass
class RewardConfig:
    """集中保存环境中的三类奖励配置。"""

    step: float = -1.0
    invalid: float = -5.0
    goal: float = 50.0


class GridWorldEnv:
    """用于最短路径学习的确定性网格环境。

    状态为 `(row, col)`，动作为上下左右四个离散动作。
    若动作导致越界或撞上障碍，则智能体留在原地并受到非法移动惩罚。
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def __init__(
        self,
        rows: int,
        cols: int,
        start: State,
        goal: State,
        obstacles: List[State],
        rewards: RewardConfig,
        max_steps: int = 100,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles)
        self.rewards = rewards
        self.max_steps = max_steps
        self.n_actions = len(self.ACTIONS)

        if self.start in self.obstacles:
            raise ValueError("start cannot be an obstacle")
        if self.goal in self.obstacles:
            raise ValueError("goal cannot be an obstacle")

        self.state: State = self.start
        self.steps = 0

    def reset(self) -> State:
        """重置到起点并返回初始状态。"""

        self.state = self.start
        self.steps = 0
        return self.state

    def in_bounds(self, state: State) -> bool:
        """检查状态是否仍位于网格边界内。"""

        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_blocked(self, state: State) -> bool:
        """检查状态是否是障碍物位置。"""

        return state in self.obstacles

    def step(self, action: Action) -> Tuple[State, float, bool, Dict[str, bool]]:
        """执行一步环境转移。

        返回：
        - next_state: 执行动作后的状态
        - reward: 本步即时奖励
        - done: 当前 episode 是否结束
        - info: 补充标记，说明是否非法移动、是否到达目标、是否触发步数上限
        """

        if action not in self.ACTIONS:
            raise ValueError(f"invalid action: {action}")

        self.steps += 1
        dr, dc = self.ACTIONS[action]
        nr = self.state[0] + dr
        nc = self.state[1] + dc
        candidate = (nr, nc)

        info = {"invalid_move": False, "reached_goal": False, "time_limit": False}

        if (not self.in_bounds(candidate)) or self.is_blocked(candidate):
            reward = self.rewards.invalid
            next_state = self.state
            info["invalid_move"] = True
        else:
            next_state = candidate
            if next_state == self.goal:
                reward = self.rewards.goal
                info["reached_goal"] = True
            else:
                reward = self.rewards.step

        self.state = next_state
        done = info["reached_goal"]

        if self.steps >= self.max_steps and not done:
            done = True
            info["time_limit"] = True

        return next_state, reward, done, info

    def render_grid(self, path: Optional[List[State]] = None) -> np.ndarray:
        """将当前地图与可选路径编码成整数网格，供可视化模块绘图。"""

        grid = np.zeros((self.rows, self.cols), dtype=np.int32)
        for r, c in self.obstacles:
            grid[r, c] = 1

        if path:
            for r, c in path:
                if (r, c) not in (self.start, self.goal) and (r, c) not in self.obstacles:
                    grid[r, c] = 2

        sr, sc = self.start
        gr, gc = self.goal
        grid[sr, sc] = 3
        grid[gr, gc] = 4
        return grid
