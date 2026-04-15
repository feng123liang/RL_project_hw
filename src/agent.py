from __future__ import annotations

from typing import List, Tuple

import numpy as np


State = Tuple[int, int]


class QLearningAgent:
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
        current_q = self.q_table[state][action]
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.gamma * next_best
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_path(self, env, max_steps: int) -> List[State]:
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
