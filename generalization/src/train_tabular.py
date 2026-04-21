from __future__ import annotations

"""Reusable tabular Q-learning utilities for delivery experiments."""

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.env import GridDeliveryEnv, RewardConfig
from src.instance import GridDeliveryInstance


StateEncoder = Callable[[GridDeliveryEnv], object]


def canonicalize_state(encoded_state: object) -> object:
    """Convert nested encoder outputs into hashable tabular keys."""

    if isinstance(encoded_state, dict):
        return tuple(sorted((str(key), canonicalize_state(value)) for key, value in encoded_state.items()))
    if isinstance(encoded_state, (list, tuple)):
        return tuple(canonicalize_state(item) for item in encoded_state)
    if isinstance(encoded_state, np.ndarray):
        return canonicalize_state(encoded_state.tolist())
    if isinstance(encoded_state, np.integer):
        return int(encoded_state)
    if isinstance(encoded_state, np.floating):
        return float(encoded_state)
    return encoded_state


@dataclass(frozen=True)
class QLearningConfig:
    """Configuration for tabular Q-learning."""

    episodes: int = 400
    alpha: float = 0.25
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.70
    seed: int = 0


@dataclass(frozen=True)
class EpisodeSummary:
    """Aggregated information from one training episode."""

    episode_index: int
    epsilon: float
    return_value: float
    steps: int
    success: bool
    reached_goal: bool
    completed_all_deliveries: bool
    invalid_moves: int
    unique_states_visited: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "episode_index": int(self.episode_index),
            "epsilon": float(self.epsilon),
            "return_value": float(self.return_value),
            "steps": int(self.steps),
            "success": bool(self.success),
            "reached_goal": bool(self.reached_goal),
            "completed_all_deliveries": bool(self.completed_all_deliveries),
            "invalid_moves": int(self.invalid_moves),
            "unique_states_visited": int(self.unique_states_visited),
        }


@dataclass(frozen=True)
class TrainingSummary:
    """Serialized view of a Q-learning training run."""

    config: Dict[str, object]
    instance: Dict[str, object]
    encoder_name: str
    episodes: List[EpisodeSummary]
    q_table_size: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": self.config,
            "instance": self.instance,
            "encoder_name": self.encoder_name,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "q_table_size": int(self.q_table_size),
        }


QTable = Dict[object, np.ndarray]


class TabularQPolicy:
    """Greedy policy view over a learned tabular Q-function."""

    def __init__(self, q_table: QTable, encoder: StateEncoder, n_actions: int) -> None:
        self.q_table = q_table
        self.encoder = encoder
        self.n_actions = int(n_actions)

    def encode(self, env: GridDeliveryEnv) -> object:
        return canonicalize_state(self.encoder(env))

    def q_values(self, state_key: object) -> np.ndarray:
        values = self.q_table.get(state_key)
        if values is None:
            return np.zeros(self.n_actions, dtype=np.float64)
        return values

    def act(self, env: GridDeliveryEnv, *, greedy: bool = True, rng: random.Random | None = None) -> int:
        state_key = self.encode(env)
        valid_actions = env.valid_actions()
        if not valid_actions:
            return 0
        q_values = self.q_values(state_key)
        if (not greedy) and rng is not None:
            return int(rng.choice(valid_actions))
        return int(_best_action(q_values, valid_actions))


def _build_env(instance: GridDeliveryInstance, rewards: RewardConfig | None = None) -> GridDeliveryEnv:
    return GridDeliveryEnv(instance=instance, rewards=rewards)


def _epsilon_for_episode(config: QLearningConfig, episode_index: int) -> float:
    if config.episodes <= 1:
        return float(config.epsilon_end)
    decay_steps = max(1, int(math.ceil(config.episodes * config.epsilon_decay_fraction)))
    clipped_index = min(int(episode_index), decay_steps)
    progress = clipped_index / float(decay_steps)
    epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
    return float(max(config.epsilon_end, min(config.epsilon_start, epsilon)))


def _best_action(q_values: np.ndarray, valid_actions: Sequence[int]) -> int:
    best_value = None
    best_action = int(valid_actions[0])
    for action in valid_actions:
        action_value = float(q_values[action])
        if (best_value is None) or (action_value > best_value) or (
            action_value == best_value and action < best_action
        ):
            best_value = action_value
            best_action = int(action)
    return best_action


def _epsilon_greedy_action(
    q_table: QTable,
    state_key: object,
    valid_actions: Sequence[int],
    epsilon: float,
    rng: random.Random,
    n_actions: int,
) -> int:
    if not valid_actions:
        return 0
    if rng.random() < epsilon:
        return int(rng.choice(list(valid_actions)))
    q_values = q_table.get(state_key)
    if q_values is None:
        q_values = np.zeros(n_actions, dtype=np.float64)
        q_table[state_key] = q_values
    return _best_action(q_values, valid_actions)


def train_q_learning(
    instance: GridDeliveryInstance,
    encoder: StateEncoder,
    encoder_name: str,
    config: QLearningConfig,
    rewards: RewardConfig | None = None,
) -> tuple[TabularQPolicy, TrainingSummary]:
    """Train a tabular Q-learning agent on one delivery instance."""

    rng = random.Random(int(config.seed))
    env = _build_env(instance, rewards=rewards)
    q_table: QTable = defaultdict(lambda: np.zeros(env.n_actions, dtype=np.float64))
    episode_logs: List[EpisodeSummary] = []

    for episode_index in range(1, config.episodes + 1):
        epsilon = _epsilon_for_episode(config, episode_index - 1)
        env.reset()
        state_key = canonicalize_state(encoder(env))
        done = False
        return_value = 0.0
        invalid_moves = 0
        visited_states = {state_key}
        final_info: Dict[str, object] = {
            "success": False,
            "reached_goal": False,
            "completed_all_deliveries": False,
        }

        while not done:
            valid_actions = env.valid_actions()
            action = _epsilon_greedy_action(
                q_table=q_table,
                state_key=state_key,
                valid_actions=valid_actions,
                epsilon=epsilon,
                rng=rng,
                n_actions=env.n_actions,
            )
            next_state, reward, done, info = env.step(action)
            final_info = info
            next_state_key = canonicalize_state(encoder(env))
            visited_states.add(next_state_key)

            q_values = q_table[state_key]
            next_q_values = q_table[next_state_key]
            if done:
                bootstrap = 0.0
            else:
                next_valid_actions = env.valid_actions()
                bootstrap = max(float(next_q_values[a]) for a in next_valid_actions) if next_valid_actions else 0.0
            td_target = float(reward) + config.gamma * bootstrap
            q_values[action] += config.alpha * (td_target - float(q_values[action]))

            state_key = next_state_key
            return_value += float(reward)
            if bool(info["invalid_move"]):
                invalid_moves += 1

        episode_logs.append(
            EpisodeSummary(
                episode_index=episode_index,
                epsilon=epsilon,
                return_value=return_value,
                steps=env.steps,
                success=bool(final_info["success"]),
                reached_goal=bool(final_info["reached_goal"]),
                completed_all_deliveries=bool(final_info["completed_all_deliveries"]),
                invalid_moves=invalid_moves,
                unique_states_visited=len(visited_states),
            )
        )

    summary = TrainingSummary(
        config={
            "episodes": int(config.episodes),
            "alpha": float(config.alpha),
            "gamma": float(config.gamma),
            "epsilon_start": float(config.epsilon_start),
            "epsilon_end": float(config.epsilon_end),
            "epsilon_decay_fraction": float(config.epsilon_decay_fraction),
            "seed": int(config.seed),
        },
        instance=instance.to_dict(),
        encoder_name=encoder_name,
        episodes=episode_logs,
        q_table_size=len(q_table),
    )
    policy = TabularQPolicy(q_table=dict(q_table), encoder=encoder, n_actions=env.n_actions)
    return policy, summary


def train_q_learning_on_instances(
    instances: Sequence[GridDeliveryInstance],
    encoder: StateEncoder,
    encoder_name: str,
    config: QLearningConfig,
    rewards: RewardConfig | None = None,
) -> tuple[TabularQPolicy, TrainingSummary]:
    """Train one tabular Q-function across a distribution of instances."""

    if not instances:
        raise ValueError("train_q_learning_on_instances requires at least one instance")

    rng = random.Random(int(config.seed))
    base_env = _build_env(instances[0], rewards=rewards)
    q_table: QTable = defaultdict(lambda: np.zeros(base_env.n_actions, dtype=np.float64))
    episode_logs: List[EpisodeSummary] = []

    for episode_index in range(1, config.episodes + 1):
        epsilon = _epsilon_for_episode(config, episode_index - 1)
        instance = instances[(episode_index - 1) % len(instances)]
        env = _build_env(instance, rewards=rewards)
        env.reset()
        state_key = canonicalize_state(encoder(env))
        done = False
        return_value = 0.0
        invalid_moves = 0
        visited_states = {state_key}
        final_info: Dict[str, object] = {
            "success": False,
            "reached_goal": False,
            "completed_all_deliveries": False,
        }

        while not done:
            valid_actions = env.valid_actions()
            action = _epsilon_greedy_action(
                q_table=q_table,
                state_key=state_key,
                valid_actions=valid_actions,
                epsilon=epsilon,
                rng=rng,
                n_actions=env.n_actions,
            )
            _, reward, done, info = env.step(action)
            final_info = info
            next_state_key = canonicalize_state(encoder(env))
            visited_states.add(next_state_key)

            q_values = q_table[state_key]
            next_q_values = q_table[next_state_key]
            if done:
                bootstrap = 0.0
            else:
                next_valid_actions = env.valid_actions()
                bootstrap = max(float(next_q_values[a]) for a in next_valid_actions) if next_valid_actions else 0.0
            td_target = float(reward) + config.gamma * bootstrap
            q_values[action] += config.alpha * (td_target - float(q_values[action]))

            state_key = next_state_key
            return_value += float(reward)
            if bool(info["invalid_move"]):
                invalid_moves += 1

        episode_logs.append(
            EpisodeSummary(
                episode_index=episode_index,
                epsilon=epsilon,
                return_value=return_value,
                steps=env.steps,
                success=bool(final_info["success"]),
                reached_goal=bool(final_info["reached_goal"]),
                completed_all_deliveries=bool(final_info["completed_all_deliveries"]),
                invalid_moves=invalid_moves,
                unique_states_visited=len(visited_states),
            )
        )

    summary = TrainingSummary(
        config={
            "episodes": int(config.episodes),
            "alpha": float(config.alpha),
            "gamma": float(config.gamma),
            "epsilon_start": float(config.epsilon_start),
            "epsilon_end": float(config.epsilon_end),
            "epsilon_decay_fraction": float(config.epsilon_decay_fraction),
            "seed": int(config.seed),
            "training_mode": "instance_distribution",
            "num_training_instances": int(len(instances)),
        },
        instance={
            "training_instance_ids": [instance.instance_id for instance in instances],
            "num_training_instances": int(len(instances)),
        },
        encoder_name=encoder_name,
        episodes=episode_logs,
        q_table_size=len(q_table),
    )
    policy = TabularQPolicy(q_table=dict(q_table), encoder=encoder, n_actions=base_env.n_actions)
    return policy, summary
