from __future__ import annotations

"""Small neural-network Q-learning utilities for delivery experiments."""

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from src.env import GridDeliveryEnv, RewardConfig
from src.instance import GridDeliveryInstance


VectorEncoder = Callable[[GridDeliveryEnv], np.ndarray]


@dataclass(frozen=True)
class DQNConfig:
    """Configuration for a replay-based DQN baseline."""

    episodes: int = 2400
    learning_rate: float = 1e-3
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.80
    seed: int = 0
    hidden_dim: int = 64
    max_grad_norm: float = 5.0
    replay_capacity: int = 50_000
    batch_size: int = 64
    warmup_steps: int = 256
    train_every: int = 1
    updates_per_step: int = 1
    target_update_interval: int = 250
    double_dqn: bool = True


@dataclass(frozen=True)
class DQNEpisodeSummary:
    """Aggregated information from one neural-Q episode."""

    episode_index: int
    epsilon: float
    return_value: float
    steps: int
    success: bool
    reached_goal: bool
    completed_all_deliveries: bool
    invalid_moves: int
    feature_norm: float
    loss: float
    replay_size: int
    gradient_updates: int

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
            "loss": float(self.loss),
            "replay_size": int(self.replay_size),
            "gradient_updates": int(self.gradient_updates),
        }


@dataclass(frozen=True)
class DQNTrainingSummary:
    """Serialized view of a DQN-style training run."""

    config: Dict[str, object]
    instance: Dict[str, object]
    encoder_name: str
    episodes: List[DQNEpisodeSummary]
    model_shape: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "config": self.config,
            "instance": self.instance,
            "encoder_name": self.encoder_name,
            "episodes": [episode.to_dict() for episode in self.episodes],
            "model_shape": self.model_shape,
        }


class SmallQNetwork(nn.Module):
    """Tiny MLP for Q-value prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ReplayTransition:
    """One transition stored for mini-batch replay updates."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray


class ReplayBuffer:
    """Fixed-size replay buffer with deterministic sampling from the run RNG."""

    def __init__(self, capacity: int, rng: random.Random) -> None:
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive")
        self.capacity = int(capacity)
        self.rng = rng
        self.buffer: deque[ReplayTransition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: ReplayTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[ReplayTransition]:
        if batch_size <= 0:
            raise ValueError("ReplayBuffer batch_size must be positive")
        if batch_size > len(self.buffer):
            raise ValueError("Cannot sample more transitions than the replay buffer contains")
        return self.rng.sample(list(self.buffer), int(batch_size))


class DQNPolicy:
    """Greedy policy backed by a small neural Q-network."""

    def __init__(self, model: SmallQNetwork, encoder: VectorEncoder, n_actions: int, device: str = "cpu") -> None:
        self.model = model
        self.encoder = encoder
        self.n_actions = int(n_actions)
        self.device = device

    def encode(self, env: GridDeliveryEnv) -> np.ndarray:
        features = np.asarray(self.encoder(env), dtype=np.float32)
        if features.ndim != 1:
            raise ValueError("DQNPolicy expects a 1D feature vector")
        return features

    def q_values(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            values = self.model(x).squeeze(0).cpu().numpy()
        return values

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


def _epsilon_for_episode(config: DQNConfig, episode_index: int) -> float:
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


def _masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Return argmax actions after excluding invalid moves."""

    masked_values = q_values.masked_fill(action_mask <= 0.0, -1.0e9)
    return torch.argmax(masked_values, dim=1)


def _masked_max(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Return max Q-values after excluding invalid moves."""

    masked_values = q_values.masked_fill(action_mask <= 0.0, -1.0e9)
    values = torch.max(masked_values, dim=1).values
    return torch.where(values <= -1.0e8, torch.zeros_like(values), values)


def _optimize_dqn_batch(
    *,
    replay: ReplayBuffer,
    model: SmallQNetwork,
    target_model: SmallQNetwork,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    config: DQNConfig,
    device: str,
) -> float | None:
    """Run one mini-batch replay update and return the scalar loss."""

    if len(replay) < max(int(config.batch_size), int(config.warmup_steps)):
        return None

    batch = replay.sample(int(config.batch_size))
    states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=device)
    actions = torch.as_tensor([item.action for item in batch], dtype=torch.int64, device=device).unsqueeze(1)
    rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=device)
    dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=device)
    next_action_masks = torch.as_tensor(
        np.stack([item.next_action_mask for item in batch]),
        dtype=torch.float32,
        device=device,
    )

    predicted_q = model(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        if config.double_dqn:
            next_actions = _masked_argmax(model(next_states), next_action_masks).unsqueeze(1)
            next_q_values = target_model(next_states).gather(1, next_actions).squeeze(1)
            has_valid_next = torch.max(next_action_masks, dim=1).values > 0.0
            next_q_values = torch.where(has_valid_next, next_q_values, torch.zeros_like(next_q_values))
        else:
            next_q_values = _masked_max(target_model(next_states), next_action_masks)
        targets = rewards + (1.0 - dones) * float(config.gamma) * next_q_values

    loss = loss_fn(predicted_q, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.max_grad_norm))
    optimizer.step()
    return float(loss.item())


def train_dqn_on_instances(
    instances: Sequence[GridDeliveryInstance],
    encoder: VectorEncoder,
    encoder_name: str,
    config: DQNConfig,
    rewards: RewardConfig | None = None,
) -> tuple[DQNPolicy, DQNTrainingSummary]:
    """Train a replay-based neural Q-function across a distribution of instances."""

    if not instances:
        raise ValueError("train_dqn_on_instances requires at least one instance")

    rng = random.Random(int(config.seed))
    np.random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))

    base_env = _build_env(instances[0], rewards=rewards)
    base_env.reset()
    feature_dim = int(np.asarray(encoder(base_env), dtype=np.float32).shape[0])
    device = "cpu"
    model = SmallQNetwork(input_dim=feature_dim, hidden_dim=int(config.hidden_dim), n_actions=base_env.n_actions).to(device)
    target_model = SmallQNetwork(
        input_dim=feature_dim,
        hidden_dim=int(config.hidden_dim),
        n_actions=base_env.n_actions,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    loss_fn = nn.SmoothL1Loss()
    replay = ReplayBuffer(capacity=int(config.replay_capacity), rng=rng)
    episode_logs: List[DQNEpisodeSummary] = []
    global_steps = 0
    gradient_updates = 0

    for episode_index in range(1, config.episodes + 1):
        epsilon = _epsilon_for_episode(config, episode_index - 1)
        instance = instances[(episode_index - 1) % len(instances)]
        env = _build_env(instance, rewards=rewards)
        env.reset()
        features = np.asarray(encoder(env), dtype=np.float32)
        done = False
        return_value = 0.0
        invalid_moves = 0
        final_info: Dict[str, object] = {
            "success": False,
            "reached_goal": False,
            "completed_all_deliveries": False,
        }
        loss_values: List[float] = []
        episode_updates_before = gradient_updates

        while not done:
            global_steps += 1
            with torch.no_grad():
                q_values = model(torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            valid_actions = env.valid_actions()
            action = _epsilon_greedy_action(q_values=q_values, valid_actions=valid_actions, epsilon=epsilon, rng=rng)

            _, reward, done, info = env.step(action)
            final_info = info
            next_features = np.asarray(encoder(env), dtype=np.float32)
            next_action_mask = env.action_mask().astype(np.float32)
            replay.push(
                ReplayTransition(
                    state=features.copy(),
                    action=int(action),
                    reward=float(reward),
                    next_state=next_features.copy(),
                    done=bool(done),
                    next_action_mask=next_action_mask,
                )
            )

            if global_steps % max(1, int(config.train_every)) == 0:
                for _ in range(max(1, int(config.updates_per_step))):
                    loss_value = _optimize_dqn_batch(
                        replay=replay,
                        model=model,
                        target_model=target_model,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        config=config,
                        device=device,
                    )
                    if loss_value is None:
                        break
                    loss_values.append(loss_value)
                    gradient_updates += 1
                    if gradient_updates % max(1, int(config.target_update_interval)) == 0:
                        target_model.load_state_dict(model.state_dict())

            features = next_features
            return_value += float(reward)
            if bool(info["invalid_move"]):
                invalid_moves += 1

        episode_logs.append(
            DQNEpisodeSummary(
                episode_index=episode_index,
                epsilon=epsilon,
                return_value=return_value,
                steps=env.steps,
                success=bool(final_info["success"]),
                reached_goal=bool(final_info["reached_goal"]),
                completed_all_deliveries=bool(final_info["completed_all_deliveries"]),
                invalid_moves=invalid_moves,
                feature_norm=float(np.linalg.norm(features)),
                loss=float(sum(loss_values) / len(loss_values)) if loss_values else 0.0,
                replay_size=len(replay),
                gradient_updates=int(gradient_updates - episode_updates_before),
            )
        )

    summary = DQNTrainingSummary(
        config={
            "episodes": int(config.episodes),
            "learning_rate": float(config.learning_rate),
            "gamma": float(config.gamma),
            "epsilon_start": float(config.epsilon_start),
            "epsilon_end": float(config.epsilon_end),
            "epsilon_decay_fraction": float(config.epsilon_decay_fraction),
            "seed": int(config.seed),
            "hidden_dim": int(config.hidden_dim),
            "max_grad_norm": float(config.max_grad_norm),
            "replay_capacity": int(config.replay_capacity),
            "batch_size": int(config.batch_size),
            "warmup_steps": int(config.warmup_steps),
            "train_every": int(config.train_every),
            "updates_per_step": int(config.updates_per_step),
            "target_update_interval": int(config.target_update_interval),
            "double_dqn": bool(config.double_dqn),
            "training_mode": "instance_distribution",
            "num_training_instances": int(len(instances)),
            "approximation": "replay_target_dqn",
        },
        instance={
            "training_instance_ids": [instance.instance_id for instance in instances],
            "num_training_instances": int(len(instances)),
        },
        encoder_name=encoder_name,
        episodes=episode_logs,
        model_shape={
            "input_dim": int(feature_dim),
            "hidden_dim": int(config.hidden_dim),
            "n_actions": int(base_env.n_actions),
        },
    )
    policy = DQNPolicy(model=model.eval(), encoder=encoder, n_actions=base_env.n_actions, device=device)
    return policy, summary
