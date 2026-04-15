from __future__ import annotations

import argparse
import copy
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train import build_env, load_config, train
from utils import ensure_dir, save_json


SWEEP_SPECS = [
    {
        "name": "epsilon_decay",
        "label": "Epsilon Decay",
        "path": ["agent", "epsilon_decay"],
        "values": [0.99, 0.995, 0.999],
    },
    {
        "name": "goal_reward",
        "label": "Goal Reward",
        "path": ["environment", "rewards", "goal"],
        "values": [20, 50, 80],
    },
]

PALETTE = ["#2563EB", "#E11D48", "#059669", "#D97706", "#7C3AED"]


def _set_nested(payload: Dict[str, Any], path: List[str], value: Any) -> None:
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _get_nested(payload: Dict[str, Any], path: List[str]) -> Any:
    cursor = payload
    for key in path:
        cursor = cursor[key]
    return cursor


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return str(value)


def _run_eval(cfg: Dict[str, Any], episodes: int) -> Dict[str, float]:
    env = build_env(cfg)
    q_table_path = os.path.join(cfg["paths"]["models_dir"], "q_table.npy")
    q_table = np.load(q_table_path)

    success = 0
    total_rewards = []
    steps_success_only = []

    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        reached_goal = False

        while not done:
            action = int(np.argmax(q_table[state]))
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            if info["reached_goal"]:
                reached_goal = True

        total_rewards.append(ep_reward)
        if reached_goal:
            success += 1
            steps_success_only.append(ep_steps)

    return {
        "episodes": episodes,
        "success_rate": float(success / episodes),
        "avg_reward": float(np.mean(total_rewards)),
        "avg_steps_success_only": float(np.mean(steps_success_only)) if steps_success_only else float("nan"),
    }


def _build_run_cfg(
    base_cfg: Dict[str, Any],
    output_root: str,
    sweep_name: str,
    path: List[str],
    value: Any,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    _set_nested(cfg, path, value)

    run_dir = os.path.join(output_root, sweep_name, _format_value(value))
    cfg["paths"] = {
        "models_dir": os.path.join(run_dir, "models"),
        "logs_dir": os.path.join(run_dir, "logs"),
        "figures_dir": os.path.join(run_dir, "figures"),
    }
    return cfg


def _plot_sweep_summary(
    sweep_name: str,
    sweep_label: str,
    records: List[Dict[str, Any]],
    output_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("white")

    for idx, record in enumerate(records):
        color = PALETTE[idx % len(PALETTE)]
        label = f"{sweep_label}={record['value']}"
        metrics = record["train_metrics"]
        episodes = np.arange(1, len(metrics["episode_reward_ma"]) + 1)

        axes[0, 0].plot(episodes, metrics["episode_reward_ma"], color=color, linewidth=2.0, label=label)
        axes[0, 1].plot(episodes, metrics["success_rate"], color=color, linewidth=2.0, label=label)

    values = [str(record["value"]) for record in records]
    train_success = [record["train_metrics"]["final_success_rate"] for record in records]
    eval_success = [record["eval_summary"]["success_rate"] for record in records]
    eval_steps = [record["eval_summary"]["avg_steps_success_only"] for record in records]

    axes[1, 0].plot(values, train_success, color="#2563EB", linewidth=2.2, marker="o", label="Train")
    axes[1, 0].plot(values, eval_success, color="#E11D48", linewidth=2.2, marker="o", label="Eval")
    axes[1, 1].bar(values, eval_steps, color="#059669", width=0.55)

    axes[0, 0].set_title("Reward Moving Average")
    axes[0, 1].set_title("Training Success Rate")
    axes[1, 0].set_title("Final Success Rate")
    axes[1, 1].set_title("Eval Avg Steps (Success Only)")

    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Success Rate")
    axes[1, 0].set_xlabel(sweep_label)
    axes[1, 0].set_ylabel("Success Rate")
    axes[1, 1].set_xlabel(sweep_label)
    axes[1, 1].set_ylabel("Steps")

    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False)

    for ax in axes.flat:
        ax.grid(color="#CBD5E1", linestyle="--", linewidth=0.8, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#FFFFFF")

    fig.suptitle(f"Single-Factor Hyperparameter Sweep: {sweep_name}", fontsize=15, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_summary(
    base_cfg: Dict[str, Any],
    summary: Dict[str, Any],
    output_path: str,
) -> None:
    lines = [
        "# Hyperparameter Sweep Summary",
        "",
        "All runs keep the same map, seed, and non-target hyperparameters.",
        "",
        f"- Seed: `{base_cfg['seed']}`",
        f"- Episodes per run: `{base_cfg['training']['episodes']}`",
        f"- Evaluation episodes per run: `{summary['evaluation_episodes']}`",
        "",
    ]

    for sweep in summary["sweeps"]:
        lines.append(f"## {sweep['label']}")
        lines.append("")
        lines.append("| Value | Train Success | Train Avg Steps | Eval Success | Eval Avg Reward | Eval Avg Steps |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for record in sweep["runs"]:
            train_metrics = record["train_metrics"]
            eval_summary = record["eval_summary"]
            lines.append(
                "| "
                f"{record['value']} | "
                f"{train_metrics['final_success_rate']:.3f} | "
                f"{train_metrics['final_avg_steps_success_only']:.2f} | "
                f"{eval_summary['success_rate']:.3f} | "
                f"{eval_summary['avg_reward']:.2f} | "
                f"{eval_summary['avg_steps_success_only']:.2f} |"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_experiments(config_path: str, output_root: str, eval_episodes: int) -> Dict[str, Any]:
    base_cfg = load_config(config_path)
    ensure_dir(output_root)
    ensure_dir(os.path.join(output_root, "figures"))

    summary: Dict[str, Any] = {
        "config_path": config_path,
        "output_root": output_root,
        "evaluation_episodes": eval_episodes,
        "baseline_values": {
            spec["name"]: _get_nested(base_cfg, spec["path"]) for spec in SWEEP_SPECS
        },
        "sweeps": [],
    }

    for spec in SWEEP_SPECS:
        sweep_records = []
        print(f"\n=== Running sweep: {spec['name']} ===")
        for value in spec["values"]:
            run_cfg = _build_run_cfg(base_cfg, output_root, spec["name"], spec["path"], value)
            print(f"[{spec['name']}] value={value}")
            train_metrics = train(run_cfg)
            eval_summary = _run_eval(run_cfg, eval_episodes)
            record = {
                "value": value,
                "paths": run_cfg["paths"],
                "train_metrics": train_metrics,
                "eval_summary": eval_summary,
            }
            sweep_records.append(record)
            save_json(os.path.join(run_cfg["paths"]["logs_dir"], "eval_summary.json"), eval_summary)

        figure_path = os.path.join(output_root, "figures", f"{spec['name']}_comparison.png")
        _plot_sweep_summary(spec["name"], spec["label"], sweep_records, figure_path)
        summary["sweeps"].append(
            {
                "name": spec["name"],
                "label": spec["label"],
                "path": spec["path"],
                "values": spec["values"],
                "figure_path": figure_path,
                "runs": sweep_records,
            }
        )

    save_json(os.path.join(output_root, "summary.json"), summary)
    _write_markdown_summary(base_cfg, summary, os.path.join(output_root, "summary.md"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-factor hyperparameter comparison experiments")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Base yaml config")
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/hyperparam_sweeps",
        help="Directory to store sweep results",
    )
    parser.add_argument("--eval-episodes", type=int, default=200, help="Greedy evaluation episodes per run")
    args = parser.parse_args()

    summary = run_experiments(args.config, args.output_root, args.eval_episodes)
    print("\nExperiment summary saved to:")
    print(f"- {os.path.join(args.output_root, 'summary.json')}")
    print(f"- {os.path.join(args.output_root, 'summary.md')}")
    for sweep in summary["sweeps"]:
        print(f"- {sweep['figure_path']}")


if __name__ == "__main__":
    main()
