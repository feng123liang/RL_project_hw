from __future__ import annotations

"""Run replay-buffer DQN against the tuned linear-Q baseline."""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoders import encode_patch3x3_plus_vector
from src.evaluator import evaluate_policy
from src.generator import GeneratorConfig, sample_instance_batch
from src.plotting import plot_delivery_map, plot_evaluation_summary, plot_training_curves
from src.train_dqn import DQNConfig, train_dqn_on_instances
from src.train_linear import LinearQLearningConfig, train_linear_q_learning_on_instances


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def render_selected_rollouts(
    *,
    figures_dir: str,
    run_name: str,
    split_name: str,
    evaluation_summary,
    generated_items,
) -> List[Dict[str, Any]]:
    instance_lookup = {item.instance.instance_id: item for item in generated_items}
    split_dir = os.path.join(figures_dir, run_name, split_name)
    ensure_dir(split_dir)

    optimal = [rollout for rollout in evaluation_summary.rollout_summaries if rollout.optimal_success]
    loose_only = [
        rollout
        for rollout in evaluation_summary.rollout_summaries
        if rollout.success and not rollout.optimal_success
    ]
    failures = [rollout for rollout in evaluation_summary.rollout_summaries if not rollout.success]
    selected_rollouts = optimal[:3] + loose_only[:2] + failures[:3]

    rendered: List[Dict[str, Any]] = []
    for rollout in selected_rollouts:
        item = instance_lookup[rollout.instance_id]
        status_text = (
            "Optimal Success"
            if rollout.optimal_success
            else ("Loose Success" if rollout.success else "Failure")
        )
        figure_paths = plot_delivery_map(
            item.instance,
            rollout.path,
            title=f"{split_name.replace('_', ' ').title()} ({run_name})",
            subtitle=f"{rollout.instance_id} | {status_text}",
            output_path=os.path.join(split_dir, f"{rollout.instance_id}.png"),
            path_label=f"{run_name} Rollout",
            show_direction_arrows=True,
            shortest_path_steps=item.oracle.optimal_steps,
            agent_steps=rollout.steps,
        )
        rendered.append(
            {
                "instance_id": rollout.instance_id,
                "success": bool(rollout.success),
                "optimal_success": bool(rollout.optimal_success),
                "steps": int(rollout.steps),
                "optimal_steps": rollout.optimal_steps,
                "step_excess": rollout.step_excess,
                "figures": figure_paths,
            }
        )
    return rendered


def summarize_config(config: object) -> Dict[str, Any]:
    return {
        key: value
        for key, value in config.__dict__.items()
        if isinstance(value, (bool, int, float, str))
    }


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "results", "dqn_replay_patch_q_experiment")
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    generator_config = GeneratorConfig(
        rows_min=6,
        rows_max=6,
        cols_min=6,
        cols_max=6,
        obstacle_density_min=0.10,
        obstacle_density_max=0.18,
        task_count_min=1,
        task_count_max=1,
        min_optimal_steps=8,
        max_generation_attempts=300,
    )
    train_seeds = list(range(410, 420))
    heldout_seeds = list(range(610, 710))
    train_batch = sample_instance_batch(generator_config, seeds=train_seeds, split="dist_train_10")
    heldout_batch = sample_instance_batch(generator_config, seeds=heldout_seeds, split="dist_eval_100")

    linear_config = LinearQLearningConfig(
        episodes=2400,
        alpha=0.005,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.75,
        seed=23,
        max_td_error=3.0,
        max_weight_norm=60.0,
    )
    dqn_config = DQNConfig(
        episodes=4200,
        learning_rate=7e-4,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.85,
        seed=23,
        hidden_dim=96,
        max_grad_norm=5.0,
        replay_capacity=40_000,
        batch_size=96,
        warmup_steps=400,
        train_every=1,
        updates_per_step=1,
        target_update_interval=300,
        double_dqn=True,
    )

    runs: List[Tuple[str, str, object, object]] = [
        ("patch3x3_plus_linear_tuned", "linear", train_linear_q_learning_on_instances, linear_config),
        ("patch3x3_plus_dqn_replay", "replay_dqn", train_dqn_on_instances, dqn_config),
    ]
    run_results: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []

    for run_name, run_kind, trainer, config in runs:
        policy, training_summary = trainer(
            instances=[item.instance for item in train_batch],
            encoder=encode_patch3x3_plus_vector,
            encoder_name=run_name,
            config=config,
        )
        train_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in train_batch],
            split_name="train_distribution_10",
            encoder_name=run_name,
        )
        heldout_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in heldout_batch],
            split_name="heldout_distribution_100",
            encoder_name=run_name,
        )

        training_figures = plot_training_curves(
            [episode.to_dict() for episode in training_summary.episodes],
            title=f"Training Curves ({run_name})",
            subtitle=f"{run_kind} on patch3x3_plus vector input",
            output_path=os.path.join(figures_dir, f"training_curves_{run_name}.png"),
        )
        selected_train_rollouts = render_selected_rollouts(
            figures_dir=figures_dir,
            run_name=run_name,
            split_name="train_distribution_10_examples",
            evaluation_summary=train_eval,
            generated_items=train_batch,
        )
        selected_heldout_rollouts = render_selected_rollouts(
            figures_dir=figures_dir,
            run_name=run_name,
            split_name="heldout_distribution_100_examples",
            evaluation_summary=heldout_eval,
            generated_items=heldout_batch,
        )

        run_results.append(
            {
                "run_name": run_name,
                "run_kind": run_kind,
                "config": summarize_config(config),
                "training_summary": training_summary.to_dict(),
                "train_evaluation": train_eval.to_dict(),
                "heldout_evaluation": heldout_eval.to_dict(),
                "figures": {
                    "training_curves": training_figures,
                    "selected_train_rollouts": selected_train_rollouts,
                    "selected_heldout_rollouts": selected_heldout_rollouts,
                },
            }
        )
        evaluation_rows.extend(
            [
                {
                    "encoder_name": run_name,
                    "split_name": "train_10",
                    "loose_success_rate": train_eval.loose_success_rate,
                    "optimal_success_rate": train_eval.optimal_success_rate,
                    "mean_return": train_eval.mean_return,
                },
                {
                    "encoder_name": run_name,
                    "split_name": "heldout_100",
                    "loose_success_rate": heldout_eval.loose_success_rate,
                    "optimal_success_rate": heldout_eval.optimal_success_rate,
                    "mean_return": heldout_eval.mean_return,
                },
            ]
        )

    evaluation_figures = plot_evaluation_summary(
        evaluation_rows,
        title="Replay DQN vs Tuned Linear Q",
        subtitle="Same 10 training maps and 100 held-out maps",
        output_path=os.path.join(figures_dir, "evaluation_summary.png"),
    )

    payload = {
        "experiment_name": "dqn_replay_patch_q_experiment",
        "generator_config": summarize_config(generator_config),
        "num_training_instances": len(train_batch),
        "train_seed_range": [train_seeds[0], train_seeds[-1]],
        "num_heldout_instances": len(heldout_batch),
        "heldout_seed_range": [heldout_seeds[0], heldout_seeds[-1]],
        "training_instances": [
            {
                "instance": item.instance.to_dict(),
                "oracle": item.oracle.to_dict(),
            }
            for item in train_batch
        ],
        "heldout_instances": [
            {
                "instance": item.instance.to_dict(),
                "oracle": item.oracle.to_dict(),
            }
            for item in heldout_batch
        ],
        "runs": run_results,
        "figures": {
            "evaluation_summary": evaluation_figures,
        },
    }

    output_path = os.path.join(output_dir, "dqn_replay_patch_q_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
