from __future__ import annotations

"""Linear approximate Q-learning utilities for delivery experiments."""

from dataclasses import dataclass
import math
import random
from typing import Callable, Dict, List, Sequence

import numpy as np

from src.env import GridDeliveryEnv, RewardConfig
from src.instance import GridDeliveryInstance


VectorEncoder = Callable[[GridDeliveryEnv], np.ndarray]


@dataclass(frozen=True)
class LinearQLearningConfig:
    """Configuration for linear approximate Q-learning."""

    episodes: int = 800
    alpha: float = 0.05
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.80
    seed: int = 0
    max_td_error: float = 10.0
    max_weight_norm: float = 250.0


@dataclass(frozen=True)
class LinearEpisodeSummary:
    """Aggregated information from one approximate-Q episode."""

    episode_index: int
    epsilon: float
    return_value: float
    steps: int
    success: bool
    reached_goal: bool
    completed_all_deliveries: bool
    invalid_moves: int
    feature_norm: float

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
            "feature_norm": float(self.feature_norm),
        }


@dataclass(frozen=True)
class LinearTrainingSummary:
    """Serialized view of a linear approximate Q-learning run."""

    config: Dict[str, object]
    instance: Dict[str, object]
    encoder_name: str
    episodes: List[LinearEpisodeSummary]
    weight_shape: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": self.config,
            "instance": self.instance,
            "encoder_name": self.encoder_name,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "weight_shape": [int(value) for value in self.weight_shape],
        }


class LinearQPolicy:
    """Greedy policy backed by a linear approximate Q-function."""

    def __init__(self, weights: np.ndarray, encoder: VectorEncoder, n_actions: int) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.encoder = encoder
        self.n_actions = int(n_actions)

    def encode(self, env: GridDeliveryEnv) -> np.ndarray:
        features = np.asarray(self.encoder(env), dtype=np.float64)
        if features.ndim != 1:
            raise ValueError("LinearQPolicy expects a 1D feature vector")
        return features

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.weights @ features

    def act(self, env: GridDeliveryEnv, *, greedy: bool = True, rng: random.Random | None = None) -> int:
        features = self.encode(env)
        valid_actions = env.valid_actions()
        if not valid_actions:
            return 0
        if (not greedy) and rng is not None:
            return int(rng.choice(valid_actions))
        q_values = self.q_values(features)
        return int(_best_action(q_values, valid_actions))


def _build_env(instance: GridDeliveryInstance, rewards: RewardConfig | None = None) -> GridDeliveryEnv:
    return GridDeliveryEnv(instance=instance, rewards=rewards)


def _epsilon_for_episode(config: LinearQLearningConfig, episode_index: int) -> float:
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
    q_values: np.ndarray,
    valid_actions: Sequence[int],
    epsilon: float,
    rng: random.Random,
) -> int:
    if not valid_actions:
        return 0
    if rng.random() < epsilon:
        return int(rng.choice(list(valid_actions)))
    return _best_action(q_values, valid_actions)


def train_linear_q_learning_on_instances(
    instances: Sequence[GridDeliveryInstance],
    encoder: VectorEncoder,
    encoder_name: str,
    config: LinearQLearningConfig,
    rewards: RewardConfig | None = None,
) -> tuple[LinearQPolicy, LinearTrainingSummary]:
    """Train a linear approximate Q-function across a distribution of instances."""

    if not instances:
        raise ValueError("train_linear_q_learning_on_instances requires at least one instance")

    rng = random.Random(int(config.seed))
    base_env = _build_env(instances[0], rewards=rewards)
    base_env.reset()
    feature_dim = int(np.asarray(encoder(base_env), dtype=np.float64).shape[0])
    weights = np.zeros((base_env.n_actions, feature_dim), dtype=np.float64)
    episode_logs: List[LinearEpisodeSummary] = []

    for episode_index in range(1, config.episodes + 1):
        epsilon = _epsilon_for_episode(config, episode_index - 1)
        instance = instances[(episode_index - 1) % len(instances)]
        env = _build_env(instance, rewards=rewards)
        env.reset()
        features = np.asarray(encoder(env), dtype=np.float64)
        done = False
        return_value = 0.0
        invalid_moves = 0
        final_info: Dict[str, object] = {
            "success": False,
            "reached_goal": False,
            "completed_all_deliveries": False,
        }

        while not done:
            valid_actions = env.valid_actions()
            q_values = weights @ features
            action = _epsilon_greedy_action(
                q_values=q_values,
                valid_actions=valid_actions,
                epsilon=epsilon,
                rng=rng,
            )
            _, reward, done, info = env.step(action)
            final_info = info
            next_features = np.asarray(encoder(env), dtype=np.float64)

            if done:
                bootstrap = 0.0
            else:
                next_valid_actions = env.valid_actions()
                next_q_values = weights @ next_features
                bootstrap = max(float(next_q_values[a]) for a in next_valid_actions) if next_valid_actions else 0.0

            prediction = float(q_values[action])
            td_target = float(reward) + config.gamma * bootstrap
            td_error = float(np.clip(td_target - prediction, -config.max_td_error, config.max_td_error))
            weights[action] += config.alpha * td_error * features
            weight_norm = float(np.linalg.norm(weights[action]))
            if weight_norm > config.max_weight_norm:
                weights[action] *= config.max_weight_norm / weight_norm

            features = next_features
            return_value += float(reward)
            if bool(info["invalid_move"]):
                invalid_moves += 1

        episode_logs.append(
            LinearEpisodeSummary(
                episode_index=episode_index,
                epsilon=epsilon,
                return_value=return_value,
                steps=env.steps,
                success=bool(final_info["success"]),
                reached_goal=bool(final_info["reached_goal"]),
                completed_all_deliveries=bool(final_info["completed_all_deliveries"]),
                invalid_moves=invalid_moves,
                feature_norm=float(np.linalg.norm(features)),
            )
        )

    summary = LinearTrainingSummary(
        config={
            "episodes": int(config.episodes),
            "alpha": float(config.alpha),
            "gamma": float(config.gamma),
            "epsilon_start": float(config.epsilon_start),
            "epsilon_end": float(config.epsilon_end),
            "epsilon_decay_fraction": float(config.epsilon_decay_fraction),
            "seed": int(config.seed),
            "max_td_error": float(config.max_td_error),
            "max_weight_norm": float(config.max_weight_norm),
            "training_mode": "instance_distribution",
            "num_training_instances": int(len(instances)),
            "approximation": "linear_q_learning",
        },
        instance={
            "training_instance_ids": [instance.instance_id for instance in instances],
            "num_training_instances": int(len(instances)),
        },
        encoder_name=encoder_name,
        episodes=episode_logs,
        weight_shape=[int(value) for value in weights.shape],
    )
    policy = LinearQPolicy(weights=weights.copy(), encoder=encoder, n_actions=base_env.n_actions)
    return policy, summary
