from __future__ import annotations

"""Curriculum-style initialization experiment for progressively harder maps."""

import argparse
import os
from collections import deque
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from agent import QLearningAgent
from env import GridWorldEnv, RewardConfig, State
from utils import ensure_dir, save_json, seed_everything


StrategyConfig = Dict[str, Any]
Checkpoint = Dict[str, Any]

COLORS = {
    "direct_l4": "#1f77b4",
    "curriculum_transfer": "#d62728",
    "curriculum_transfer_reset_epsilon": "#2ca02c",
    "reverse_curriculum": "#ff7f0e",
    "unrelated_curriculum": "#8c564b",
}
LINE_STYLES = {
    "direct_l4": {"linestyle": "-", "marker": "o", "markevery": 2, "linewidth": 2.6, "zorder": 2},
    "curriculum_transfer": {"linestyle": "--", "marker": "s", "markevery": 2, "linewidth": 2.2, "zorder": 3},
    "curriculum_transfer_reset_epsilon": {
        "linestyle": ":",
        "marker": "^",
        "markevery": 2,
        "linewidth": 2.2,
        "zorder": 4,
    },
    "reverse_curriculum": {
        "linestyle": "-.",
        "marker": "D",
        "markevery": 2,
        "linewidth": 2.0,
        "zorder": 5,
    },
    "unrelated_curriculum": {
        "linestyle": (0, (5, 1)),
        "marker": "P",
        "markevery": 2,
        "linewidth": 2.0,
        "zorder": 6,
    },
}
SUCCESS_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_reward_config(shared_env_cfg: Dict[str, Any]) -> RewardConfig:
    rewards_cfg = shared_env_cfg["rewards"]
    return RewardConfig(
        step=rewards_cfg["step"],
        invalid=rewards_cfg["invalid"],
        goal=rewards_cfg["goal"],
    )


def build_env(shared_env_cfg: Dict[str, Any], obstacles: List[State]) -> GridWorldEnv:
    return GridWorldEnv(
        rows=shared_env_cfg["rows"],
        cols=shared_env_cfg["cols"],
        start=tuple(shared_env_cfg["start"]),
        goal=tuple(shared_env_cfg["goal"]),
        obstacles=obstacles,
        rewards=make_reward_config(shared_env_cfg),
        max_steps=shared_env_cfg["max_steps"],
    )


def normalize_obstacles(obstacles: Iterable[Iterable[int]]) -> List[State]:
    return [tuple(int(x) for x in item) for item in obstacles]


def is_reachable(shared_env_cfg: Dict[str, Any], obstacles: List[State]) -> bool:
    rows = int(shared_env_cfg["rows"])
    cols = int(shared_env_cfg["cols"])
    start = tuple(shared_env_cfg["start"])
    goal = tuple(shared_env_cfg["goal"])
    blocked = set(obstacles)

    q: deque[State] = deque([start])
    visited = {start}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            nxt = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if nxt in blocked or nxt in visited:
                continue
            visited.add(nxt)
            q.append(nxt)

    return False


def validate_config(cfg: Dict[str, Any]) -> Dict[str, List[State]]:
    shared_env_cfg = cfg["shared_env"]
    rows = int(shared_env_cfg["rows"])
    cols = int(shared_env_cfg["cols"])
    start = tuple(shared_env_cfg["start"])
    goal = tuple(shared_env_cfg["goal"])
    maps_cfg = cfg["maps"]

    normalized_maps: Dict[str, List[State]] = {}
    for level_name, level_cfg in maps_cfg.items():
        obstacles = normalize_obstacles(level_cfg["obstacles"])
        for obstacle in obstacles:
            r, c = obstacle
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(f"{level_name}: obstacle {obstacle} is out of bounds")
        obstacle_set = set(obstacles)
        if start in obstacle_set:
            raise ValueError(f"{level_name}: start cannot be an obstacle")
        if goal in obstacle_set:
            raise ValueError(f"{level_name}: goal cannot be an obstacle")
        if not is_reachable(shared_env_cfg, obstacles):
            raise ValueError(f"{level_name}: goal is not reachable from start")
        normalized_maps[level_name] = obstacles

    direct_cfg = cfg["strategies"]["direct_l4"]
    direct_total = int(direct_cfg["episodes"])
    for strategy_name, strategy_cfg in cfg["strategies"].items():
        if strategy_name == "direct_l4":
            continue
        strategy_total = sum(int(phase["episodes"]) for phase in strategy_cfg["phases"])
        if direct_total != strategy_total:
            raise ValueError(
                f"Budget mismatch: direct_l4 has {direct_total} episodes, "
                f"{strategy_name} has {strategy_total} episodes"
            )

    for strategy_name, strategy_cfg in cfg["strategies"].items():
        if strategy_name == "direct_l4":
            if strategy_cfg["level"] not in normalized_maps:
                raise ValueError(f"Unknown level in {strategy_name}: {strategy_cfg['level']}")
        else:
            for phase in strategy_cfg["phases"]:
                if phase["level"] not in normalized_maps:
                    raise ValueError(f"Unknown level in {strategy_name}: {phase['level']}")

    return normalized_maps


def build_agent(cfg: Dict[str, Any], shared_env_cfg: Dict[str, Any]) -> QLearningAgent:
    agent_cfg = cfg["agent"]
    return QLearningAgent(
        rows=shared_env_cfg["rows"],
        cols=shared_env_cfg["cols"],
        n_actions=4,
        alpha=agent_cfg["alpha"],
        gamma=agent_cfg["gamma"],
        epsilon=agent_cfg["epsilon"],
        epsilon_min=agent_cfg["epsilon_min"],
        epsilon_decay=agent_cfg["epsilon_decay"],
    )


def run_training_episode(agent: QLearningAgent, env: GridWorldEnv) -> float:
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(state, greedy=False)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

    agent.decay_epsilon()
    return float(total_reward)


def evaluate_agent(agent: QLearningAgent, env: GridWorldEnv, episodes: int) -> Dict[str, float]:
    success = 0
    total_rewards: List[float] = []
    steps_success_only: List[int] = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        reached_goal = False

        while not done:
            action = agent.select_action(state, greedy=True)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if info["reached_goal"]:
                reached_goal = True

        total_rewards.append(float(ep_reward))
        if reached_goal:
            success += 1
            steps_success_only.append(ep_steps)

    avg_steps = float(np.mean(steps_success_only)) if steps_success_only else float("nan")
    return {
        "success_rate": float(success / episodes),
        "avg_reward": float(np.mean(total_rewards)),
        "avg_steps_success_only": avg_steps,
    }


def greedy_path_length(agent: QLearningAgent, env: GridWorldEnv) -> int:
    path = agent.greedy_path(env, max_steps=env.max_steps)
    return max(0, len(path) - 1)


def strategy_phases(strategy_name: str, strategy_cfg: StrategyConfig) -> List[Dict[str, Any]]:
    if strategy_name == "direct_l4":
        return [
            {
                "phase_name": "direct_l4",
                "level": strategy_cfg["level"],
                "episodes": int(strategy_cfg["episodes"]),
            }
        ]

    phases: List[Dict[str, Any]] = []
    for idx, phase in enumerate(strategy_cfg["phases"], start=1):
        phases.append(
            {
                "phase_name": f"phase_{idx}_{phase['level']}",
                "level": phase["level"],
                "episodes": int(phase["episodes"]),
            }
        )
    return phases


def maybe_reset_epsilon(agent: QLearningAgent, strategy_cfg: StrategyConfig, phase: Dict[str, Any]) -> None:
    """Optionally override epsilon at the start of a phase for ablation studies."""

    overrides = strategy_cfg.get("phase_start_epsilon_overrides", {})
    if phase["level"] in overrides:
        agent.epsilon = float(overrides[phase["level"]])


def run_strategy(
    cfg: Dict[str, Any],
    shared_env_cfg: Dict[str, Any],
    maps: Dict[str, List[State]],
    strategy_name: str,
    strategy_cfg: StrategyConfig,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seed_everything(seed)
    agent = build_agent(cfg, shared_env_cfg)
    eval_cfg = cfg["evaluation"]
    phases = strategy_phases(strategy_name, strategy_cfg)

    total_episodes = sum(int(phase["episodes"]) for phase in phases)
    global_episode = 0
    level4_start_episode: int | None = None
    episode_rewards: List[Dict[str, Any]] = []
    checkpoints: List[Checkpoint] = []
    phase_boundaries: List[Dict[str, Any]] = []

    for phase_index, phase in enumerate(phases, start=1):
        maybe_reset_epsilon(agent, strategy_cfg, phase)
        epsilon_at_phase_start = float(agent.epsilon)
        env = build_env(shared_env_cfg, maps[phase["level"]])
        start_episode = global_episode + 1
        if phase["level"] == "level4" and level4_start_episode is None:
            level4_start_episode = start_episode

        for _ in range(phase["episodes"]):
            reward = run_training_episode(agent, env)
            global_episode += 1
            episode_rewards.append(
                {
                    "global_episode": global_episode,
                    "phase_index": phase_index,
                    "phase_name": phase["phase_name"],
                    "train_level": phase["level"],
                    "reward": reward,
                }
            )

            if global_episode % int(eval_cfg["eval_every_episodes"]) == 0:
                checkpoint_metrics = evaluate_agent(
                    agent,
                    build_env(shared_env_cfg, maps["level4"]),
                    int(eval_cfg["checkpoint_eval_episodes"]),
                )
                checkpoints.append(
                    {
                        "global_episode": global_episode,
                        "strategy": strategy_name,
                        "seed": seed,
                        "current_phase": phase["phase_name"],
                        "level4_started": level4_start_episode is not None,
                        "level4_training_episodes_elapsed": (
                            0
                            if level4_start_episode is None
                            else global_episode - level4_start_episode + 1
                        ),
                        "l4_success_rate": checkpoint_metrics["success_rate"],
                        "l4_avg_reward": checkpoint_metrics["avg_reward"],
                        "l4_avg_steps_success_only": checkpoint_metrics["avg_steps_success_only"],
                    }
                )

        phase_boundaries.append(
            {
                "phase_index": phase_index,
                "phase_name": phase["phase_name"],
                "train_level": phase["level"],
                "start_episode": start_episode,
                "end_episode": global_episode,
                "epsilon_at_phase_start": epsilon_at_phase_start,
            }
        )

    if global_episode != total_episodes:
        raise RuntimeError(f"{strategy_name} ended at {global_episode}, expected {total_episodes}")

    final_metrics = evaluate_agent(
        agent,
        build_env(shared_env_cfg, maps["level4"]),
        int(eval_cfg["final_eval_episodes"]),
    )
    final_metrics.update(
        {
            "seed": seed,
            "strategy": strategy_name,
            "evaluation_level": "level4",
            "evaluation_episodes": int(eval_cfg["final_eval_episodes"]),
            "path_length_greedy": greedy_path_length(agent, build_env(shared_env_cfg, maps["level4"])),
        }
    )

    train_log = {
        "seed": seed,
        "strategy": strategy_name,
        "total_episodes": total_episodes,
        "q_table_shape": list(agent.q_table.shape),
        "final_epsilon": float(agent.epsilon),
        "level4_start_episode": level4_start_episode,
        "phase_boundaries": phase_boundaries,
        "episode_rewards": episode_rewards,
        "checkpoints": checkpoints,
    }

    return train_log, final_metrics


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def safe_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite))


def threshold_label(threshold: float) -> str:
    return f"success_ge_{int(round(threshold * 100)):02d}"


def summarize_checkpoint_records(records: List[Checkpoint]) -> Dict[str, float] | None:
    filtered_records = [record for record in records if record is not None]
    if not filtered_records:
        return None

    return {
        "num_runs": len(filtered_records),
        "global_episode": safe_mean(record["global_episode"] for record in filtered_records),
        "global_episode_std": safe_std(record["global_episode"] for record in filtered_records),
        "level4_training_episodes_elapsed_mean": safe_mean(
            record["level4_training_episodes_elapsed"] for record in filtered_records
        ),
        "level4_training_episodes_elapsed_std": safe_std(
            record["level4_training_episodes_elapsed"] for record in filtered_records
        ),
        "l4_success_rate_mean": safe_mean(record["l4_success_rate"] for record in filtered_records),
        "l4_success_rate_std": safe_std(record["l4_success_rate"] for record in filtered_records),
        "l4_avg_reward_mean": safe_mean(record["l4_avg_reward"] for record in filtered_records),
        "l4_avg_reward_std": safe_std(record["l4_avg_reward"] for record in filtered_records),
        "l4_avg_steps_success_only_mean": safe_mean(
            record["l4_avg_steps_success_only"] for record in filtered_records
        ),
        "l4_avg_steps_success_only_std": safe_std(
            record["l4_avg_steps_success_only"] for record in filtered_records
        ),
    }


def compute_auc(
    checkpoint_summary: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    horizon: float | None = None,
) -> float:
    pairs = []
    for item in checkpoint_summary:
        x = float(item[x_key])
        y = float(item[y_key])
        if np.isfinite(x) and np.isfinite(y):
            pairs.append((x, y))

    if len(pairs) < 2:
        return float("nan")

    collapsed: Dict[float, float] = {}
    for x, y in pairs:
        collapsed[x] = max(y, collapsed.get(x, -np.inf))

    xs = np.asarray(sorted(collapsed.keys()), dtype=np.float64)
    ys = np.asarray([collapsed[x] for x in xs], dtype=np.float64)
    if horizon is not None:
        finite_horizon = float(horizon)
        mask = xs <= finite_horizon
        xs = xs[mask]
        ys = ys[mask]
    if xs.size < 2:
        return float("nan")
    return float(np.trapezoid(ys, xs))


def total_l4_training_budget(strategy_name: str, strategy_cfg: StrategyConfig) -> int:
    if strategy_name == "direct_l4":
        return int(strategy_cfg["episodes"])
    return sum(int(phase["episodes"]) for phase in strategy_cfg["phases"] if phase["level"] == "level4")


def aggregate_results(
    cfg: Dict[str, Any],
    train_logs: List[Dict[str, Any]],
    final_evals: List[Dict[str, Any]],
) -> Dict[str, Any]:
    strategies = list(cfg["strategies"].keys())
    shared_l4_horizon = min(
        total_l4_training_budget(strategy_name, cfg["strategies"][strategy_name])
        for strategy_name in strategies
    )
    summary: Dict[str, Any] = {
        "seed_list": cfg["seed_list"],
        "shared_l4_auc_horizon": shared_l4_horizon,
        "strategies": {},
    }

    for strategy in strategies:
        strategy_train_logs = [log for log in train_logs if log["strategy"] == strategy]
        strategy_final_evals = [item for item in final_evals if item["strategy"] == strategy]
        checkpoint_episodes = sorted(
            {record["global_episode"] for log in strategy_train_logs for record in log["checkpoints"]}
        )

        checkpoint_summary = []
        for episode in checkpoint_episodes:
            matching = [
                record
                for log in strategy_train_logs
                for record in log["checkpoints"]
                if record["global_episode"] == episode
            ]
            checkpoint_summary.append(
                {
                    "global_episode": episode,
                    "level4_training_episodes_elapsed_mean": safe_mean(
                        record["level4_training_episodes_elapsed"] for record in matching
                    ),
                    "l4_success_rate_mean": safe_mean(record["l4_success_rate"] for record in matching),
                    "l4_success_rate_std": safe_std(record["l4_success_rate"] for record in matching),
                    "l4_avg_reward_mean": safe_mean(record["l4_avg_reward"] for record in matching),
                    "l4_avg_reward_std": safe_std(record["l4_avg_reward"] for record in matching),
                    "l4_avg_steps_success_only_mean": safe_mean(
                        record["l4_avg_steps_success_only"] for record in matching
                    ),
                    "l4_avg_steps_success_only_std": safe_std(
                        record["l4_avg_steps_success_only"] for record in matching
                    ),
                }
            )

        jumpstart_checkpoint = summarize_checkpoint_records(
            [
                next((record for record in log["checkpoints"] if record["level4_started"]), None)
                for log in strategy_train_logs
            ]
        )

        first_nonzero_checkpoint = summarize_checkpoint_records(
            [
                next((record for record in log["checkpoints"] if record["l4_success_rate"] > 0.0), None)
                for log in strategy_train_logs
            ]
        )
        first_full_checkpoint = summarize_checkpoint_records(
            [
                next((record for record in log["checkpoints"] if record["l4_success_rate"] >= 1.0), None)
                for log in strategy_train_logs
            ]
        )
        threshold_milestones = {
            threshold_label(threshold): summarize_checkpoint_records(
                [
                    next((record for record in log["checkpoints"] if record["l4_success_rate"] >= threshold), None)
                    for log in strategy_train_logs
                ]
            )
            for threshold in SUCCESS_THRESHOLDS
        }

        final_metrics_summary = {}
        for metric_name in (
            "success_rate",
            "avg_reward",
            "avg_steps_success_only",
            "path_length_greedy",
        ):
            values = [float(item[metric_name]) for item in strategy_final_evals]
            final_metrics_summary[metric_name] = {
                "mean": safe_mean(values),
                "std": safe_std(values),
                "per_seed": values,
            }

        summary["strategies"][strategy] = {
            "num_runs": len(strategy_final_evals),
            "final_metrics": final_metrics_summary,
            "milestones": {
                "first_nonzero_success_checkpoint": first_nonzero_checkpoint,
                "first_full_success_checkpoint": first_full_checkpoint,
            },
            "threshold_milestones": threshold_milestones,
            "sample_efficiency": {
                "auc_l4_success_rate_vs_global_episode": compute_auc(
                    checkpoint_summary,
                    x_key="global_episode",
                    y_key="l4_success_rate_mean",
                ),
                "auc_l4_success_rate_vs_l4_training_episodes": compute_auc(
                    checkpoint_summary,
                    x_key="level4_training_episodes_elapsed_mean",
                    y_key="l4_success_rate_mean",
                ),
                "normalized_auc_l4_success_rate_shared_horizon": (
                    compute_auc(
                        checkpoint_summary,
                        x_key="level4_training_episodes_elapsed_mean",
                        y_key="l4_success_rate_mean",
                        horizon=float(shared_l4_horizon),
                    )
                    / float(shared_l4_horizon)
                ),
            },
            "checkpoint_summary": checkpoint_summary,
            "transfer_views": {
                "strong_transfer": {
                    "jumpstart": jumpstart_checkpoint,
                    "auc_l4_success_rate_vs_global_episode": compute_auc(
                        checkpoint_summary,
                        x_key="global_episode",
                        y_key="l4_success_rate_mean",
                    ),
                    "time_to_threshold": {
                        key: (
                            None
                            if value is None
                            else {
                                "global_episode": value["global_episode"],
                                "global_episode_std": value["global_episode_std"],
                            }
                        )
                        for key, value in threshold_milestones.items()
                    },
                },
                "weak_transfer": {
                    "jumpstart": jumpstart_checkpoint,
                    "normalized_auc_l4_success_rate_shared_horizon": (
                        compute_auc(
                            checkpoint_summary,
                            x_key="level4_training_episodes_elapsed_mean",
                            y_key="l4_success_rate_mean",
                            horizon=float(shared_l4_horizon),
                        )
                        / float(shared_l4_horizon)
                    ),
                    "shared_l4_auc_horizon": shared_l4_horizon,
                    "time_to_threshold": {
                        key: (
                            None
                            if value is None
                            else {
                                "level4_training_episodes_elapsed_mean": value[
                                    "level4_training_episodes_elapsed_mean"
                                ],
                                "level4_training_episodes_elapsed_std": value[
                                    "level4_training_episodes_elapsed_std"
                                ],
                            }
                        )
                        for key, value in threshold_milestones.items()
                    },
                },
            },
        }

    return summary


def plot_checkpoint_metric(
    summary: Dict[str, Any],
    metric_prefix: str,
    ylabel: str,
    title: str,
    output_path: str,
    x_key: str = "global_episode",
    xlabel: str = "Global Episode",
) -> None:
    plt.figure(figsize=(8, 5))

    for strategy, strategy_summary in summary["strategies"].items():
        checkpoints = strategy_summary["checkpoint_summary"]
        x = np.asarray([item[x_key] for item in checkpoints], dtype=np.float64)
        y = np.asarray([item[f"{metric_prefix}_mean"] for item in checkpoints], dtype=np.float64)
        std = np.asarray([item[f"{metric_prefix}_std"] for item in checkpoints], dtype=np.float64)
        color = COLORS.get(strategy)
        label = strategy.replace("_", " ")
        style = LINE_STYLES.get(strategy, {})

        plt.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker"),
            markevery=style.get("markevery"),
            linewidth=style.get("linewidth", 2.0),
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            zorder=style.get("zorder", 2),
        )
        finite = np.isfinite(y) & np.isfinite(std)
        if np.any(finite):
            plt.fill_between(
                x[finite],
                y[finite] - std[finite],
                y[finite] + std[finite],
                alpha=0.12,
                color=color,
                zorder=1,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_milestone_bars(summary: Dict[str, Any], output_path: str) -> None:
    strategies = list(summary["strategies"].keys())
    labels = [strategy.replace("_", " ") for strategy in strategies]
    x = np.arange(len(strategies))
    width = 0.36

    global_values = []
    l4_values = []
    colors = [COLORS.get(strategy) for strategy in strategies]

    for strategy in strategies:
        milestone = summary["strategies"][strategy]["milestones"]["first_full_success_checkpoint"]
        if milestone is None:
            global_values.append(np.nan)
            l4_values.append(np.nan)
            continue
        global_values.append(float(milestone["global_episode"]))
        l4_values.append(float(milestone["level4_training_episodes_elapsed_mean"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x, global_values, color=colors, alpha=0.85)
    axes[0].set_title("First Full L4 Success by Global Episode")
    axes[0].set_ylabel("Global Episode")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, l4_values, color=colors, alpha=0.85)
    axes[1].set_title("First Full L4 Success by L4 Training Episodes")
    axes[1].set_ylabel("Level4 Training Episodes Elapsed")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_jumpstart_compare(summary: Dict[str, Any], output_path: str) -> None:
    strategies = list(summary["strategies"].keys())
    labels = [strategy.replace("_", " ") for strategy in strategies]
    x = np.arange(len(strategies))
    colors = [COLORS.get(strategy) for strategy in strategies]

    metric_defs = [
        ("global_episode", "Jumpstart Global Episode"),
        ("level4_training_episodes_elapsed_mean", "Jumpstart L4 Episodes"),
        ("l4_success_rate_mean", "Jumpstart Success Rate"),
        ("l4_avg_reward_mean", "Jumpstart Average Reward"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, (metric_key, title) in zip(axes_flat, metric_defs):
        values = []
        for strategy in strategies:
            jumpstart = summary["strategies"][strategy]["transfer_views"]["strong_transfer"]["jumpstart"]
            values.append(float("nan") if jumpstart is None else float(jumpstart[metric_key]))

        ax.bar(x, values, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_threshold_milestones(summary: Dict[str, Any], output_path: str) -> None:
    strategies = list(summary["strategies"].keys())
    labels = [strategy.replace("_", " ") for strategy in strategies]
    x = np.arange(len(strategies))
    colors = [COLORS.get(strategy) for strategy in strategies]

    fig, axes = plt.subplots(2, len(SUCCESS_THRESHOLDS), figsize=(4 * len(SUCCESS_THRESHOLDS), 8))

    for col_index, threshold in enumerate(SUCCESS_THRESHOLDS):
        key = threshold_label(threshold)
        global_values = []
        weak_values = []
        for strategy in strategies:
            milestone = summary["strategies"][strategy]["threshold_milestones"][key]
            global_values.append(float("nan") if milestone is None else float(milestone["global_episode"]))
            weak_values.append(
                float("nan")
                if milestone is None
                else float(milestone["level4_training_episodes_elapsed_mean"])
            )

        axes[0, col_index].bar(x, global_values, color=colors, alpha=0.85)
        axes[0, col_index].set_title(f"Success >= {int(threshold * 100)}%")
        axes[0, col_index].set_ylabel("Global Episode")
        axes[0, col_index].set_xticks(x)
        axes[0, col_index].set_xticklabels(labels, rotation=15)
        axes[0, col_index].grid(axis="y", alpha=0.3)

        axes[1, col_index].bar(x, weak_values, color=colors, alpha=0.85)
        axes[1, col_index].set_title(f"Success >= {int(threshold * 100)}%")
        axes[1, col_index].set_ylabel("Level4 Training Episodes")
        axes[1, col_index].set_xticks(x)
        axes[1, col_index].set_xticklabels(labels, rotation=15)
        axes[1, col_index].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sample_efficiency_bar(summary: Dict[str, Any], output_path: str) -> None:
    strategies = list(summary["strategies"].keys())
    labels = [strategy.replace("_", " ") for strategy in strategies]
    x = np.arange(len(strategies))
    colors = [COLORS.get(strategy) for strategy in strategies]

    strong_values = [
        summary["strategies"][strategy]["sample_efficiency"]["auc_l4_success_rate_vs_global_episode"]
        for strategy in strategies
    ]
    weak_values = [
        summary["strategies"][strategy]["sample_efficiency"]["normalized_auc_l4_success_rate_shared_horizon"]
        for strategy in strategies
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x, strong_values, color=colors, alpha=0.85)
    axes[0].set_title("Strong Transfer: AUC vs Global Episode")
    axes[0].set_ylabel("AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, weak_values, color=colors, alpha=0.85)
    axes[1].set_title("Weak Transfer: Normalized AUC vs Shared L4 Budget")
    axes[1].set_ylabel("Normalized AUC")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_transfer_view_tables(summary: Dict[str, Any]) -> Dict[str, Any]:
    tables: Dict[str, Any] = {
        "shared_l4_auc_horizon": summary["shared_l4_auc_horizon"],
        "strategies": {},
    }

    for strategy, strategy_summary in summary["strategies"].items():
        thresholds: Dict[str, Any] = {}
        for key, milestone in strategy_summary["threshold_milestones"].items():
            thresholds[key] = (
                None
                if milestone is None
                else {
                    "global_episode": milestone["global_episode"],
                    "global_episode_std": milestone["global_episode_std"],
                    "level4_training_episodes_elapsed_mean": milestone[
                        "level4_training_episodes_elapsed_mean"
                    ],
                    "level4_training_episodes_elapsed_std": milestone[
                        "level4_training_episodes_elapsed_std"
                    ],
                }
            )

        tables["strategies"][strategy] = {
            "jumpstart": strategy_summary["transfer_views"]["strong_transfer"]["jumpstart"],
            "time_to_threshold": thresholds,
            "sample_efficiency": strategy_summary["sample_efficiency"],
        }

    return tables


def plot_final_metrics_bar(summary: Dict[str, Any], output_path: str) -> None:
    metric_defs = [
        ("success_rate", "Success Rate"),
        ("avg_reward", "Average Reward"),
        ("avg_steps_success_only", "Avg Steps (Success Only)"),
        ("path_length_greedy", "Greedy Path Length"),
    ]
    strategies = list(summary["strategies"].keys())
    labels = [strategy.replace("_", " ") for strategy in strategies]
    x = np.arange(len(strategies))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()

    for ax, (metric_name, title) in zip(axes_flat, metric_defs):
        means = [
            summary["strategies"][strategy]["final_metrics"][metric_name]["mean"] for strategy in strategies
        ]
        stds = [summary["strategies"][strategy]["final_metrics"][metric_name]["std"] for strategy in strategies]
        colors = [COLORS.get(strategy) for strategy in strategies]

        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_run_outputs(
    base_dir: str,
    strategy_name: str,
    seed: int,
    train_log: Dict[str, Any],
    final_eval: Dict[str, Any],
) -> None:
    run_dir = os.path.join(base_dir, "runs", strategy_name, f"seed_{seed}")
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "train_log.json"), train_log)
    save_json(os.path.join(run_dir, "final_eval.json"), final_eval)


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    shared_env_cfg = cfg["shared_env"]
    maps = validate_config(cfg)

    output_dir = os.path.join("results", "curriculum")
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    ensure_dir(output_dir)
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    train_logs: List[Dict[str, Any]] = []
    final_evals: List[Dict[str, Any]] = []

    for strategy_name, strategy_cfg in cfg["strategies"].items():
        for seed in cfg["seed_list"]:
            print(f"[{strategy_name}] seed={seed} ...")
            train_log, final_eval = run_strategy(
                cfg=cfg,
                shared_env_cfg=shared_env_cfg,
                maps=maps,
                strategy_name=strategy_name,
                strategy_cfg=strategy_cfg,
                seed=int(seed),
            )
            train_logs.append(train_log)
            final_evals.append(final_eval)
            save_run_outputs(output_dir, strategy_name, int(seed), train_log, final_eval)

    summary = aggregate_results(cfg, train_logs, final_evals)
    save_json(os.path.join(output_dir, "summary.json"), summary)
    save_json(os.path.join(tables_dir, "transfer_views.json"), build_transfer_view_tables(summary))

    plot_checkpoint_metric(
        summary,
        metric_prefix="l4_success_rate",
        ylabel="Success Rate",
        title="Level 4 Success Rate During Training",
        output_path=os.path.join(figures_dir, "l4_success_rate_compare.png"),
    )
    plot_checkpoint_metric(
        summary,
        metric_prefix="l4_avg_reward",
        ylabel="Average Reward",
        title="Level 4 Average Reward During Training",
        output_path=os.path.join(figures_dir, "l4_avg_reward_compare.png"),
    )
    plot_checkpoint_metric(
        summary,
        metric_prefix="l4_avg_steps_success_only",
        ylabel="Average Steps (Success Only)",
        title="Level 4 Average Steps During Training",
        output_path=os.path.join(figures_dir, "l4_avg_steps_compare.png"),
    )
    plot_checkpoint_metric(
        summary,
        metric_prefix="l4_success_rate",
        ylabel="Success Rate",
        title="Level 4 Success Rate vs Level4 Training Episodes",
        output_path=os.path.join(figures_dir, "l4_success_rate_vs_l4_training.png"),
        x_key="level4_training_episodes_elapsed_mean",
        xlabel="Level4 Training Episodes Elapsed",
    )
    plot_checkpoint_metric(
        summary,
        metric_prefix="l4_avg_reward",
        ylabel="Average Reward",
        title="Level 4 Average Reward vs Level4 Training Episodes",
        output_path=os.path.join(figures_dir, "l4_avg_reward_vs_l4_training.png"),
        x_key="level4_training_episodes_elapsed_mean",
        xlabel="Level4 Training Episodes Elapsed",
    )
    plot_final_metrics_bar(summary, os.path.join(figures_dir, "final_metrics_bar.png"))
    plot_milestone_bars(summary, os.path.join(figures_dir, "milestone_compare.png"))
    plot_jumpstart_compare(summary, os.path.join(figures_dir, "jumpstart_compare.png"))
    plot_threshold_milestones(summary, os.path.join(figures_dir, "l4_threshold_milestones.png"))
    plot_sample_efficiency_bar(summary, os.path.join(figures_dir, "l4_success_auc_compare.png"))

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curriculum initialization experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum/experiment.yaml",
        help="Path to curriculum experiment config",
    )
    args = parser.parse_args()

    summary = run_experiment(load_config(args.config))
    for strategy_name, strategy_summary in summary["strategies"].items():
        success = strategy_summary["final_metrics"]["success_rate"]["mean"]
        reward = strategy_summary["final_metrics"]["avg_reward"]["mean"]
        steps = strategy_summary["final_metrics"]["avg_steps_success_only"]["mean"]
        print(
            f"{strategy_name}: "
            f"success_rate={success:.3f}, "
            f"avg_reward={reward:.2f}, "
            f"avg_steps_success_only={steps:.2f}"
        )


if __name__ == "__main__":
    main()
