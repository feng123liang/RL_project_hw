from __future__ import annotations

"""Evaluate a 3x3 local-window state design for tabular Q-learning."""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoders import encode_feature_state, encode_patch3x3_state, encode_path_distance_state
from src.evaluator import evaluate_policy
from src.generator import GeneratorConfig, sample_instance_batch
from src.plotting import plot_delivery_map, plot_evaluation_summary, plot_training_curves
from src.train_tabular import QLearningConfig, train_q_learning_on_instances


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
    output_dir = os.path.join(PROJECT_ROOT, "results", "patch3x3_experiment")
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
    train_batch = sample_instance_batch(generator_config, seeds=[410, 411, 412, 413], split="dist_train")
    heldout_batch = sample_instance_batch(generator_config, seeds=[510, 511, 512, 513], split="dist_eval")

    encoders: List[Tuple[str, object]] = [
        ("feature", encode_feature_state),
        ("distance_state", encode_path_distance_state),
        ("patch3x3", encode_patch3x3_state),
    ]
    training_config = QLearningConfig(
        episodes=800,
        alpha=0.20,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.80,
        seed=17,
    )

    results: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []

    for encoder_name, encoder in encoders:
        policy, training_summary = train_q_learning_on_instances(
            instances=[item.instance for item in train_batch],
            encoder=encoder,
            encoder_name=encoder_name,
            config=training_config,
        )
        train_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in train_batch],
            split_name="train_distribution",
            encoder_name=encoder_name,
        )
        heldout_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in heldout_batch],
            split_name="heldout_distribution",
            encoder_name=encoder_name,
        )

        training_figures = plot_training_curves(
            [episode.to_dict() for episode in training_summary.episodes],
            title=f"Patch3x3 Training Curves ({encoder_name})",
            subtitle="Tabular Q-learning with a 3x3 local observation comparison",
            output_path=os.path.join(figures_dir, f"training_curves_{encoder_name}.png"),
        )
        representative_rollout = heldout_eval.rollout_summaries[0]
        heldout_instance = heldout_batch[0]
        rollout_figures = plot_delivery_map(
            heldout_instance.instance,
            representative_rollout.path,
            title=f"Held-Out Rollout ({encoder_name})",
            subtitle="3x3 local-window comparison on a held-out instance",
            output_path=os.path.join(figures_dir, f"heldout_rollout_{encoder_name}.png"),
            path_label="Held-Out Rollout",
            show_direction_arrows=True,
            shortest_path_steps=heldout_instance.oracle.optimal_steps,
            agent_steps=representative_rollout.steps,
        )

        all_train_rollouts = render_rollout_collection(
            figures_dir=figures_dir,
            encoder_name=encoder_name,
            split_name="train_distribution",
            evaluation_summary=train_eval,
            generated_items=train_batch,
        )
        all_heldout_rollouts = render_rollout_collection(
            figures_dir=figures_dir,
            encoder_name=encoder_name,
            split_name="heldout_distribution",
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
                    "split_name": "train_dist",
                    "loose_success_rate": train_eval.loose_success_rate,
                    "optimal_success_rate": train_eval.optimal_success_rate,
                    "mean_return": train_eval.mean_return,
                },
                {
                    "encoder_name": encoder_name,
                    "split_name": "heldout_dist",
                    "loose_success_rate": heldout_eval.loose_success_rate,
                    "optimal_success_rate": heldout_eval.optimal_success_rate,
                    "mean_return": heldout_eval.mean_return,
                },
            ]
        )

    evaluation_figures = plot_evaluation_summary(
        evaluation_rows,
        title="3x3 Local-Window State Summary",
        subtitle="Comparing a 3x3 observation window against earlier tabular states",
        output_path=os.path.join(figures_dir, "evaluation_summary.png"),
    )

    payload = {
        "experiment_name": "patch3x3_experiment",
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
        "training_config": {
            "episodes": training_config.episodes,
            "alpha": training_config.alpha,
            "gamma": training_config.gamma,
            "epsilon_start": training_config.epsilon_start,
            "epsilon_end": training_config.epsilon_end,
            "epsilon_decay_fraction": training_config.epsilon_decay_fraction,
            "seed": training_config.seed,
        },
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
        "encoder_runs": results,
        "figures": {
            "evaluation_summary": evaluation_figures,
        },
    }

    output_path = os.path.join(output_dir, "patch3x3_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
