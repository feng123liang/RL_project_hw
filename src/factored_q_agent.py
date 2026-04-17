from __future__ import annotations

"""Factored tabular Q-learning agent for richer grid tasks.

This file is intentionally separate from `agent.py` so the original baseline
remains untouched. The agent stores a dense Q-table over a fixed, factorized
state space and supports epsilon-greedy control exactly like the baseline
agent, but on higher-dimensional states.
"""

from typing import List, Sequence, Tuple

import numpy as np


RichState = Tuple[int, ...]


class FactoredQLearningAgent:
    """Tabular Q-learning over a factorized discrete state space."""

    def __init__(
        self,
        state_shape: Sequence[int],
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
    ) -> None:
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.n_actions = int(n_actions)
        self.state_shape = tuple(int(size) for size in state_shape)
        self.q_table = np.zeros((*self.state_shape, self.n_actions), dtype=np.float64)

    def select_action(self, state: RichState, greedy: bool = False) -> int:
        """Choose an action with epsilon-greedy exploration."""

        if (not greedy) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: RichState,
        action: int,
        reward: float,
        next_state: RichState,
        done: bool,
    ) -> None:
        """Apply a single Q-learning update."""

        current_q = self.q_table[state][action]
        next_best = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = float(reward) + self.gamma * next_best
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        """Decay exploration after each episode."""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_rollout(self, env, max_steps: int) -> List[RichState]:
        """Roll out the current greedy policy and collect visited rich states."""

        state = env.reset()
        trajectory = [state]
        for _ in range(max_steps):
            action = self.select_action(state, greedy=True)
            next_state, _, done, _ = env.step(action)
            trajectory.append(next_state)
            state = next_state
            if done:
                break
        return trajectory
