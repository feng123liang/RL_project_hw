from __future__ import annotations

"""Standalone curriculum experiment for the simplified advanced grid task."""

import argparse
import os
from collections import deque
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from advanced_env import (
    AdvancedGridWorldEnv,
    AdvancedLevelSpec,
    AdvancedRewardConfig,
    Position,
    RichState,
    normalize_positions,
)
from factored_q_agent import FactoredQLearningAgent
from utils import ensure_dir, save_json, seed_everything


StrategyConfig = Dict[str, Any]
Checkpoint = Dict[str, Any]
SUCCESS_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_cell_group(level_cfg: Dict[str, Any], singular_key: str, plural_key: str) -> List[Position]:
    if plural_key in level_cfg:
        return normalize_positions(level_cfg.get(plural_key, []))
    value = level_cfg.get(singular_key)
    if value is None:
        return []
    return [tuple(int(x) for x in value)]


def make_reward_config(shared_env_cfg: Dict[str, Any]) -> AdvancedRewardConfig:
    rewards_cfg = shared_env_cfg["rewards"]
    return AdvancedRewardConfig(
        step=float(rewards_cfg["step"]),
        invalid=float(rewards_cfg["invalid"]),
        hazard=float(rewards_cfg["hazard"]),
        bonus=float(rewards_cfg["bonus"]),
        goal=float(rewards_cfg["goal"]),
    )


def level_spec_from_cfg(level_cfg: Dict[str, Any]) -> AdvancedLevelSpec:
    obstacles = normalize_positions(level_cfg.get("obstacles", []))
    return AdvancedLevelSpec(
        obstacles=obstacles,
        key_cells=parse_cell_group(level_cfg, "key_pos", "key_cells"),
        door_cells=normalize_positions(level_cfg.get("door_cells", [])),
        hazard_cells=normalize_positions(level_cfg.get("hazard_cells", [])),
        bonus_cells=parse_cell_group(level_cfg, "bonus_pos", "bonus_cells"),
    )


def build_env(shared_env_cfg: Dict[str, Any], level_cfg: Dict[str, Any]) -> AdvancedGridWorldEnv:
    return AdvancedGridWorldEnv(
        rows=int(shared_env_cfg["rows"]),
        cols=int(shared_env_cfg["cols"]),
        start=tuple(shared_env_cfg["start"]),
        goal=tuple(shared_env_cfg["goal"]),
        level=level_spec_from_cfg(level_cfg),
        rewards=make_reward_config(shared_env_cfg),
        phase_cycle=int(shared_env_cfg["phase_cycle"]),
        hazard_active_phases=shared_env_cfg["hazard_active_phases"],
    )


def is_advanced_level_solvable(env: AdvancedGridWorldEnv) -> bool:
    """Run BFS over the full rich state space to ensure the level is feasible."""

    start_state = env.reset()
    queue: deque[RichState] = deque([start_state])
    visited = {start_state}

    while queue:
        state = queue.popleft()
        for action in range(env.n_actions):
            next_state, _, done, info = env.transition_from_state(state, action)
            if done and info["reached_goal"]:
                return True
            if done:
                continue
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    return False


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    shared_env_cfg = cfg["shared_env"]
    rows = int(shared_env_cfg["rows"])
    cols = int(shared_env_cfg["cols"])
    start = tuple(shared_env_cfg["start"])
    goal = tuple(shared_env_cfg["goal"])
    all_cells = {(r, c) for r in range(rows) for c in range(cols)}

    validated_levels: Dict[str, Dict[str, Any]] = {}
    for level_name, raw_level_cfg in cfg["levels"].items():
        level_cfg = dict(raw_level_cfg)
        if "free_cells" in level_cfg and "obstacles" in level_cfg:
            raise ValueError(f"{level_name}: specify either obstacles or free_cells, not both")
        if "free_cells" in level_cfg:
            free_cells = normalize_positions(level_cfg["free_cells"])
            level_cfg["obstacles"] = [list(cell) for cell in sorted(all_cells - set(free_cells))]

        spec = level_spec_from_cfg(level_cfg)
        points_to_check: List[Tuple[str, Position]] = [("start", start), ("goal", goal)]
        points_to_check.extend(("key", cell) for cell in spec.key_cells)
        points_to_check.extend(("door", cell) for cell in spec.door_cells)
        points_to_check.extend(("hazard", cell) for cell in spec.hazard_cells)
        points_to_check.extend(("bonus", cell) for cell in spec.bonus_cells)

        for label, point in points_to_check:
            if not (0 <= point[0] < rows and 0 <= point[1] < cols):
                raise ValueError(f"{level_name}: {label} cell {point} is out of bounds")
        obstacle_set = set(spec.obstacles)
        if start in obstacle_set or goal in obstacle_set:
            raise ValueError(f"{level_name}: start/goal cannot be obstacles")

        env = build_env(shared_env_cfg, level_cfg)
        if not is_advanced_level_solvable(env):
            raise ValueError(f"{level_name}: no feasible policy reaches the goal")
        validated_levels[level_name] = level_cfg

    direct_total = int(cfg["strategies"]["direct_l4"]["episodes"])
    for strategy_name, strategy_cfg in cfg["strategies"].items():
        if strategy_name == "direct_l4":
            continue
        total = sum(int(phase["episodes"]) for phase in strategy_cfg["phases"])
        if total != direct_total:
            raise ValueError(f"Budget mismatch between direct_l4 and {strategy_name}")
    return validated_levels


def build_agent(cfg: Dict[str, Any], state_shape: Tuple[int, ...], n_actions: int) -> FactoredQLearningAgent:
    agent_cfg = cfg["agent"]
    return FactoredQLearningAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        alpha=float(agent_cfg["alpha"]),
        gamma=float(agent_cfg["gamma"]),
        epsilon=float(agent_cfg["epsilon"]),
        epsilon_min=float(agent_cfg["epsilon_min"]),
        epsilon_decay=float(agent_cfg["epsilon_decay"]),
    )


def rollout_cap(cfg: Dict[str, Any]) -> int:
    shared_env_cfg = cfg["shared_env"]
    rows = int(shared_env_cfg["rows"])
    cols = int(shared_env_cfg["cols"])
    return int(shared_env_cfg.get("rollout_step_cap", rows * cols * 4))


def run_training_episode(
    agent: FactoredQLearningAgent,
    env: AdvancedGridWorldEnv,
    max_steps: int,
) -> Dict[str, Any]:
    state = env.reset()
    done = False
    total_reward = 0.0
    picked_key = False
    used_key_door = False
    collected_bonus = False
    truncated = False

    for _ in range(max_steps):
        action = agent.select_action(state, greedy=False)
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        total_reward += reward
        picked_key = picked_key or info["picked_key"]
        used_key_door = used_key_door or info["used_key_door"]
        collected_bonus = collected_bonus or info["collected_bonus"]
        state = next_state
        if done:
            break
    else:
        truncated = True

    agent.decay_epsilon()
    return {
        "reward": float(total_reward),
        "picked_key": picked_key,
        "used_key_door": used_key_door,
        "collected_bonus": collected_bonus,
        "truncated": truncated,
        "reached_goal": bool(info["reached_goal"]),
    }


def evaluate_agent(
    agent: FactoredQLearningAgent,
    env: AdvancedGridWorldEnv,
    episodes: int,
    max_steps: int,
) -> Dict[str, float]:
    success = 0
    total_rewards: List[float] = []
    steps_success_only: List[int] = []
    key_rate = 0
    door_rate = 0
    bonus_rate = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        picked_key = False
        used_key_door = False
        collected_bonus = False

        for _ in range(max_steps):
            action = agent.select_action(state, greedy=True)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            picked_key = picked_key or info["picked_key"]
            used_key_door = used_key_door or info["used_key_door"]
            collected_bonus = collected_bonus or info["collected_bonus"]
            if done:
                break

        total_rewards.append(float(ep_reward))
        if info["reached_goal"]:
            success += 1
            steps_success_only.append(ep_steps)
        if picked_key:
            key_rate += 1
        if used_key_door:
            door_rate += 1
        if collected_bonus:
            bonus_rate += 1

    avg_steps = float(np.mean(steps_success_only)) if steps_success_only else float("nan")
    return {
        "success_rate": float(success / episodes),
        "avg_reward": float(np.mean(total_rewards)),
        "avg_steps_success_only": avg_steps,
        "key_collection_rate": float(key_rate / episodes),
        "door_usage_rate": float(door_rate / episodes),
        "bonus_collection_rate": float(bonus_rate / episodes),
    }


def strategy_phases(strategy_name: str, strategy_cfg: StrategyConfig) -> List[Dict[str, Any]]:
    if strategy_name == "direct_l4":
        return [{"phase_name": "direct_l4", "level": strategy_cfg["level"], "episodes": int(strategy_cfg["episodes"])}]

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


def maybe_reset_epsilon(agent: FactoredQLearningAgent, strategy_cfg: StrategyConfig, phase: Dict[str, Any]) -> None:
    overrides = strategy_cfg.get("phase_start_epsilon_overrides", {})
    if phase["level"] in overrides:
        agent.epsilon = float(overrides[phase["level"]])


def run_strategy(
    cfg: Dict[str, Any],
    validated_levels: Dict[str, Dict[str, Any]],
    strategy_name: str,
    strategy_cfg: StrategyConfig,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    shared_env_cfg = cfg["shared_env"]
    seed_everything(seed)
    envs_for_shape = [build_env(shared_env_cfg, level_cfg) for level_cfg in validated_levels.values()]
    max_bonus_state = max(env.state_shape[-1] for env in envs_for_shape)
    state_shape = (*envs_for_shape[0].state_shape[:-1], max_bonus_state)
    agent = build_agent(cfg, state_shape=state_shape, n_actions=envs_for_shape[0].n_actions)
    eval_cfg = cfg["evaluation"]
    max_steps = rollout_cap(cfg)
    phases = strategy_phases(strategy_name, strategy_cfg)

    total_episodes = sum(int(phase["episodes"]) for phase in phases)
    global_episode = 0
    level4_start_episode: int | None = None
    episode_logs: List[Dict[str, Any]] = []
    checkpoints: List[Checkpoint] = []

    for phase_index, phase in enumerate(phases, start=1):
        maybe_reset_epsilon(agent, strategy_cfg, phase)
        env = build_env(shared_env_cfg, validated_levels[phase["level"]])
        if phase["level"] == "level4" and level4_start_episode is None:
            level4_start_episode = global_episode + 1

        for _ in range(phase["episodes"]):
            train_result = run_training_episode(agent, env, max_steps)
            global_episode += 1
            episode_logs.append(
                {
                    "global_episode": global_episode,
                    "phase_index": phase_index,
                    "phase_name": phase["phase_name"],
                    "train_level": phase["level"],
                    **train_result,
                }
            )

            if global_episode % int(eval_cfg["eval_every_episodes"]) == 0:
                metrics = evaluate_agent(
                    agent,
                    build_env(shared_env_cfg, validated_levels["level4"]),
                    int(eval_cfg["checkpoint_eval_episodes"]),
                    max_steps,
                )
                checkpoints.append(
                    {
                        "global_episode": global_episode,
                        "strategy": strategy_name,
                        "seed": seed,
                        "current_phase": phase["phase_name"],
                        "level4_training_episodes_elapsed": (
                            0 if level4_start_episode is None else global_episode - level4_start_episode + 1
                        ),
                        "l4_success_rate": metrics["success_rate"],
                        "l4_avg_reward": metrics["avg_reward"],
                        "l4_avg_steps_success_only": metrics["avg_steps_success_only"],
                        "l4_key_collection_rate": metrics["key_collection_rate"],
                        "l4_door_usage_rate": metrics["door_usage_rate"],
                        "l4_bonus_collection_rate": metrics["bonus_collection_rate"],
                    }
                )

    final_metrics = evaluate_agent(
        agent,
        build_env(shared_env_cfg, validated_levels["level4"]),
        int(eval_cfg["final_eval_episodes"]),
        max_steps,
    )
    final_metrics.update(
        {
            "seed": seed,
            "strategy": strategy_name,
            "evaluation_level": "level4",
            "q_table_shape": list(agent.q_table.shape),
        }
    )

    train_log = {
        "seed": seed,
        "strategy": strategy_name,
        "total_episodes": total_episodes,
        "final_epsilon": float(agent.epsilon),
        "q_table_shape": list(agent.q_table.shape),
        "level4_start_episode": level4_start_episode,
        "episodes": episode_logs,
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


def total_l4_training_budget(strategy_name: str, strategy_cfg: StrategyConfig) -> int:
    if strategy_name == "direct_l4":
        return int(strategy_cfg["episodes"])
    return sum(int(phase["episodes"]) for phase in strategy_cfg["phases"] if phase["level"] == "level4")


def aggregate_results(cfg: Dict[str, Any], train_logs: List[Dict[str, Any]], final_evals: List[Dict[str, Any]]) -> Dict[str, Any]:
    strategies = list(cfg["strategies"].keys())
    shared_l4_horizon = min(total_l4_training_budget(name, cfg["strategies"][name]) for name in strategies)
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
                    "l4_key_collection_rate_mean": safe_mean(
                        record["l4_key_collection_rate"] for record in matching
                    ),
                    "l4_key_collection_rate_std": safe_std(
                        record["l4_key_collection_rate"] for record in matching
                    ),
                    "l4_bonus_collection_rate_mean": safe_mean(
                        record["l4_bonus_collection_rate"] for record in matching
                    ),
                    "l4_bonus_collection_rate_std": safe_std(
                        record["l4_bonus_collection_rate"] for record in matching
                    ),
                }
            )

        threshold_milestones = {
            threshold_label(threshold): next(
                (item for item in checkpoint_summary if item["l4_success_rate_mean"] >= threshold),
                None,
            )
            for threshold in SUCCESS_THRESHOLDS
        }

        final_metrics_summary: Dict[str, Any] = {}
        for metric_name in (
            "success_rate",
            "avg_reward",
            "avg_steps_success_only",
            "key_collection_rate",
            "door_usage_rate",
            "bonus_collection_rate",
        ):
            values = [float(item[metric_name]) for item in strategy_final_evals]
            final_metrics_summary[metric_name] = {
                "mean": safe_mean(values),
                "std": safe_std(values),
                "per_seed": values,
            }

        normalized_auc = float("nan")
        xs = np.asarray(
            [item["level4_training_episodes_elapsed_mean"] for item in checkpoint_summary],
            dtype=np.float64,
        )
        ys = np.asarray([item["l4_success_rate_mean"] for item in checkpoint_summary], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys) & (xs <= float(shared_l4_horizon))
        if np.count_nonzero(valid) >= 2:
            normalized_auc = float(np.trapezoid(ys[valid], xs[valid]) / float(shared_l4_horizon))

        summary["strategies"][strategy] = {
            "checkpoint_summary": checkpoint_summary,
            "threshold_milestones": threshold_milestones,
            "sample_efficiency": {"normalized_auc_l4_success_rate_shared_horizon": normalized_auc},
            "final_metrics": final_metrics_summary,
        }
    return summary


def plot_metric(summary: Dict[str, Any], metric_prefix: str, ylabel: str, output_path: str, x_key: str, xlabel: str) -> None:
    plt.figure(figsize=(8, 5))
    for strategy, strategy_summary in summary["strategies"].items():
        checkpoints = strategy_summary["checkpoint_summary"]
        x = np.asarray([item[x_key] for item in checkpoints], dtype=np.float64)
        y = np.asarray([item[f"{metric_prefix}_mean"] for item in checkpoints], dtype=np.float64)
        plt.plot(x, y, label=strategy.replace("_", " "), linewidth=2.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_final_metrics(summary: Dict[str, Any], output_path: str) -> None:
    strategies = list(summary["strategies"].keys())
    labels = [name.replace("_", " ") for name in strategies]
    x = np.arange(len(strategies))
    metric_defs = [
        ("success_rate", "Success Rate"),
        ("avg_reward", "Avg Reward"),
        ("key_collection_rate", "Key Collection"),
        ("bonus_collection_rate", "Bonus Collection"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric_name, title) in zip(axes.flatten(), metric_defs):
        means = [summary["strategies"][name]["final_metrics"][metric_name]["mean"] for name in strategies]
        stds = [summary["strategies"][name]["final_metrics"][metric_name]["std"] for name in strategies]
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_run_outputs(base_dir: str, strategy_name: str, seed: int, train_log: Dict[str, Any], final_eval: Dict[str, Any]) -> None:
    run_dir = os.path.join(base_dir, "runs", strategy_name, f"seed_{seed}")
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "train_log.json"), train_log)
    save_json(os.path.join(run_dir, "final_eval.json"), final_eval)


def run_experiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    validated_levels = validate_config(cfg)
    output_dir = os.path.join("results", "advanced_curriculum")
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    train_logs: List[Dict[str, Any]] = []
    final_evals: List[Dict[str, Any]] = []

    for strategy_name, strategy_cfg in cfg["strategies"].items():
        for seed in cfg["seed_list"]:
            print(f"[{strategy_name}] seed={seed} ...")
            train_log, final_eval = run_strategy(
                cfg=cfg,
                validated_levels=validated_levels,
                strategy_name=strategy_name,
                strategy_cfg=strategy_cfg,
                seed=int(seed),
            )
            train_logs.append(train_log)
            final_evals.append(final_eval)
            save_run_outputs(output_dir, strategy_name, int(seed), train_log, final_eval)

    summary = aggregate_results(cfg, train_logs, final_evals)
    save_json(os.path.join(output_dir, "summary.json"), summary)

    plot_metric(
        summary,
        metric_prefix="l4_success_rate",
        ylabel="L4 Success Rate",
        output_path=os.path.join(figures_dir, "l4_success_rate_compare.png"),
        x_key="global_episode",
        xlabel="Global Episode",
    )
    plot_metric(
        summary,
        metric_prefix="l4_success_rate",
        ylabel="L4 Success Rate",
        output_path=os.path.join(figures_dir, "l4_success_rate_vs_l4_training.png"),
        x_key="level4_training_episodes_elapsed_mean",
        xlabel="Level4 Training Episodes",
    )
    plot_metric(
        summary,
        metric_prefix="l4_bonus_collection_rate",
        ylabel="Greedy Bonus Collection Rate",
        output_path=os.path.join(figures_dir, "l4_bonus_collection_compare.png"),
        x_key="level4_training_episodes_elapsed_mean",
        xlabel="Level4 Training Episodes",
    )
    plot_metric(
        summary,
        metric_prefix="l4_key_collection_rate",
        ylabel="Greedy Key Collection Rate",
        output_path=os.path.join(figures_dir, "l4_key_collection_compare.png"),
        x_key="level4_training_episodes_elapsed_mean",
        xlabel="Level4 Training Episodes",
    )
    plot_final_metrics(summary, os.path.join(figures_dir, "final_metrics_bar.png"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run advanced curriculum experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum/experiment3.yaml",
        help="Path to advanced curriculum config",
    )
    args = parser.parse_args()

    summary = run_experiment(load_config(args.config))
    for strategy_name, strategy_summary in summary["strategies"].items():
        success = strategy_summary["final_metrics"]["success_rate"]["mean"]
        reward = strategy_summary["final_metrics"]["avg_reward"]["mean"]
        bonus_rate = strategy_summary["final_metrics"]["bonus_collection_rate"]["mean"]
        print(
            f"{strategy_name}: success_rate={success:.3f}, avg_reward={reward:.2f}, "
            f"bonus_collection_rate={bonus_rate:.2f}"
        )


if __name__ == "__main__":
    main()
