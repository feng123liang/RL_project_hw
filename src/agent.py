from __future__ import annotations

"""Q-learning 智能体实现。

该文件负责：
- 维护表格型 Q 值 `q_table`
- 基于 epsilon-greedy 选择动作
- 按照 Bellman 更新公式进行学习
- 在训练结束后提取一条贪心路径用于展示
"""

from typing import List, Tuple

import numpy as np


State = Tuple[int, int]


class QLearningAgent:
    """表格型 Q-learning 智能体。"""

    def __init__(
        self,
        rows: int,
        cols: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = n_actions
        self.q_table = np.zeros((rows, cols, n_actions), dtype=np.float64)

    def select_action(self, state: State, greedy: bool = False) -> int:
        """按 epsilon-greedy 策略选动作。

        `greedy=True` 时完全利用当前 Q 表，常用于评估或提取最终路径；
        否则在探索和利用之间做随机折中。
        """

        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        """对单个状态-动作对执行一次 Q-learning 更新。"""

        current_q = self.q_table[state][action]
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * next_best
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        """在每个 episode 结束后衰减探索率，但不低于下限。"""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_path(self, env, max_steps: int) -> List[State]:
        """从起点开始按当前最优动作滚动，生成一条最终路径。"""

        state = env.reset()
        path = [state]
        for _ in range(max_steps):
            action = self.select_action(state, greedy=True)
            next_state, _, done, _ = env.step(action)
            path.append(next_state)
            state = next_state
            if done:
                break
        return path
