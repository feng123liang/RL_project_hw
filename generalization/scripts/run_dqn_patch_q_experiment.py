from __future__ import annotations

"""Run a small neural-network Q-learning experiment for delivery generalization."""

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


def render_rollout_collection(
    *,
    figures_dir: str,
    encoder_name: str,
    split_name: str,
    evaluation_summary,
    generated_items,
) -> List[Dict[str, Any]]:
    instance_lookup = {item.instance.instance_id: item for item in generated_items}
    split_dir = os.path.join(figures_dir, encoder_name, split_name)
    ensure_dir(split_dir)

    rendered: List[Dict[str, Any]] = []
    for rollout in evaluation_summary.rollout_summaries:
        item = instance_lookup[rollout.instance_id]
        status_text = (
            "Optimal Success"
            if rollout.optimal_success
            else ("Loose Success" if rollout.success else "Failure")
        )
        figure_paths = plot_delivery_map(
            item.instance,
            rollout.path,
            title=f"{split_name.replace('_', ' ').title()} Rollout ({encoder_name})",
            subtitle=f"{rollout.instance_id} | {status_text}",
            output_path=os.path.join(split_dir, f"{rollout.instance_id}.png"),
            path_label=f"{split_name.replace('_', ' ').title()} Rollout",
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


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "results", "dqn_patch_q_experiment")
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
    train_batch = sample_instance_batch(generator_config, seeds=list(range(410, 420)), split="dist_train_10")
    heldout_batch = sample_instance_batch(generator_config, seeds=list(range(610, 710)), split="dist_eval_100")

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
        episodes=3200,
        learning_rate=1e-3,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.80,
        seed=23,
        hidden_dim=64,
        max_grad_norm=5.0,
    )

    runs: List[Tuple[str, object, object, str]] = [
        ("patch3x3_plus_linear", train_linear_q_learning_on_instances, linear_config, "linear"),
        ("patch3x3_plus_dqn", train_dqn_on_instances, dqn_config, "dqn"),
    ]
    results: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []

    for encoder_name, trainer, config, training_kind in runs:
        policy, training_summary = trainer(
            instances=[item.instance for item in train_batch],
            encoder=encode_patch3x3_plus_vector,
            encoder_name=encoder_name,
            config=config,
        )
        train_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in train_batch],
            split_name="train_distribution_10",
            encoder_name=encoder_name,
        )
        heldout_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in heldout_batch],
            split_name="heldout_distribution_100",
            encoder_name=encoder_name,
        )

        training_figures = plot_training_curves(
            [episode.to_dict() for episode in training_summary.episodes],
            title=f"Training Curves ({encoder_name})",
            subtitle=f"{training_kind.upper()} over patch3x3_plus vector input",
            output_path=os.path.join(figures_dir, f"training_curves_{encoder_name}.png"),
        )
        representative_rollout = heldout_eval.rollout_summaries[0]
        heldout_instance = heldout_batch[0]
        rollout_figures = plot_delivery_map(
            heldout_instance.instance,
            representative_rollout.path,
            title=f"Held-Out Rollout ({encoder_name})",
            subtitle="Representative held-out rollout for neural-vs-linear comparison",
            output_path=os.path.join(figures_dir, f"heldout_rollout_{encoder_name}.png"),
            path_label="Held-Out Rollout",
            show_direction_arrows=True,
            shortest_path_steps=heldout_instance.oracle.optimal_steps,
            agent_steps=representative_rollout.steps,
        )
        all_train_rollouts = render_rollout_collection(
            figures_dir=figures_dir,
            encoder_name=encoder_name,
            split_name="train_distribution_10",
            evaluation_summary=train_eval,
            generated_items=train_batch,
        )
        all_heldout_rollouts = render_rollout_collection(
            figures_dir=figures_dir,
            encoder_name=encoder_name,
            split_name="heldout_distribution_100",
            evaluation_summary=heldout_eval,
            generated_items=heldout_batch,
        )

        results.append(
            {
                "encoder_name": encoder_name,
                "training_summary": training_summary.to_dict(),
                "train_evaluation": train_eval.to_dict(),
                "heldout_evaluation": heldout_eval.to_dict(),
                "figures": {
                    "training_curves": training_figures,
                    "heldout_rollout": rollout_figures,
                    "all_train_rollouts": all_train_rollouts,
                    "all_heldout_rollouts": all_heldout_rollouts,
                },
            }
        )
        evaluation_rows.extend(
            [
                {
                    "encoder_name": encoder_name,
                    "split_name": "train_10",
                    "loose_success_rate": train_eval.loose_success_rate,
                    "optimal_success_rate": train_eval.optimal_success_rate,
                    "mean_return": train_eval.mean_return,
                },
                {
                    "encoder_name": encoder_name,
                    "split_name": "heldout_100",
                    "loose_success_rate": heldout_eval.loose_success_rate,
                    "optimal_success_rate": heldout_eval.optimal_success_rate,
                    "mean_return": heldout_eval.mean_return,
                },
            ]
        )

    evaluation_figures = plot_evaluation_summary(
        evaluation_rows,
        title="Neural Q-Learning vs Linear Q-Learning",
        subtitle="Comparison on 10 training maps and 100 held-out maps",
        output_path=os.path.join(figures_dir, "evaluation_summary.png"),
    )

    payload = {
        "experiment_name": "dqn_patch_q_experiment",
        "generator_config": {
            "rows_min": generator_config.rows_min,
            "rows_max": generator_config.rows_max,
            "cols_min": generator_config.cols_min,
            "cols_max": generator_config.cols_max,
            "obstacle_density_min": generator_config.obstacle_density_min,
            "obstacle_density_max": generator_config.obstacle_density_max,
            "task_count_min": generator_config.task_count_min,
            "task_count_max": generator_config.task_count_max,
            "min_optimal_steps": generator_config.min_optimal_steps,
        },
        "linear_config": {
            "episodes": linear_config.episodes,
            "alpha": linear_config.alpha,
            "gamma": linear_config.gamma,
            "epsilon_start": linear_config.epsilon_start,
            "epsilon_end": linear_config.epsilon_end,
            "epsilon_decay_fraction": linear_config.epsilon_decay_fraction,
            "seed": linear_config.seed,
            "max_td_error": linear_config.max_td_error,
            "max_weight_norm": linear_config.max_weight_norm,
        },
        "dqn_config": {
            "episodes": dqn_config.episodes,
            "learning_rate": dqn_config.learning_rate,
            "gamma": dqn_config.gamma,
            "epsilon_start": dqn_config.epsilon_start,
            "epsilon_end": dqn_config.epsilon_end,
            "epsilon_decay_fraction": dqn_config.epsilon_decay_fraction,
            "seed": dqn_config.seed,
            "hidden_dim": dqn_config.hidden_dim,
            "max_grad_norm": dqn_config.max_grad_norm,
        },
        "num_training_instances": len(train_batch),
        "num_heldout_instances": len(heldout_batch),
        "encoder_runs": results,
        "figures": {
            "evaluation_summary": evaluation_figures,
        },
    }

    output_path = os.path.join(output_dir, "dqn_patch_q_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
