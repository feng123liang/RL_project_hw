from __future__ import annotations

"""Run the first tabular Q-learning baseline for the standalone project."""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoders import encode_absolute_state, encode_feature_state
from src.evaluator import evaluate_policy
from src.generator import GeneratorConfig, sample_instance, sample_instance_batch
from src.plotting import plot_delivery_map, plot_evaluation_summary, plot_training_curves
from src.train_tabular import QLearningConfig, train_q_learning


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "results", "tabular_baseline")
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
    train_generated = sample_instance(generator_config, seed=202, split="baseline_train")
    eval_generated = sample_instance_batch(generator_config, seeds=[301, 302, 303, 304], split="baseline_eval")

    encoders: List[Tuple[str, object]] = [
        ("absolute", encode_absolute_state),
        ("feature", encode_feature_state),
    ]
    training_config = QLearningConfig(
        episodes=450,
        alpha=0.25,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.75,
        seed=7,
    )

    results: List[Dict[str, Any]] = []
    evaluation_rows: List[Dict[str, Any]] = []

    for encoder_name, encoder in encoders:
        policy, training_summary = train_q_learning(
            instance=train_generated.instance,
            encoder=encoder,
            encoder_name=encoder_name,
            config=training_config,
        )
        train_eval = evaluate_policy(
            policy=policy,
            instances=[train_generated.instance],
            split_name="train_instance",
            encoder_name=encoder_name,
        )
        heldout_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in eval_generated],
            split_name="heldout_instances",
            encoder_name=encoder_name,
        )

        training_figures = plot_training_curves(
            [episode.to_dict() for episode in training_summary.episodes],
            title=f"Training Curves ({encoder_name})",
            subtitle="Tabular Q-learning on a fixed delivery instance",
            output_path=os.path.join(figures_dir, f"training_curves_{encoder_name}.png"),
        )
        train_rollout = train_eval.rollout_summaries[0]
        rollout_figures = plot_delivery_map(
            train_generated.instance,
            train_rollout.path,
            title=f"Greedy Rollout ({encoder_name})",
            subtitle="Policy rollout after tabular training on the fixed baseline instance",
            output_path=os.path.join(figures_dir, f"rollout_{encoder_name}.png"),
            path_label="Greedy Rollout",
            show_direction_arrows=True,
            shortest_path_steps=train_generated.oracle.optimal_steps,
            agent_steps=train_rollout.steps,
        )

        encoder_result = {
            "encoder_name": encoder_name,
            "training_summary": training_summary.to_dict(),
            "train_evaluation": train_eval.to_dict(),
            "heldout_evaluation": heldout_eval.to_dict(),
            "figures": {
                "training_curves": training_figures,
                "rollout": rollout_figures,
            },
        }
        results.append(encoder_result)
        evaluation_rows.extend(
            [
                {
                    "encoder_name": encoder_name,
                    "split_name": "train",
                    "loose_success_rate": train_eval.loose_success_rate,
                    "optimal_success_rate": train_eval.optimal_success_rate,
                    "mean_return": train_eval.mean_return,
                },
                {
                    "encoder_name": encoder_name,
                    "split_name": "heldout",
                    "loose_success_rate": heldout_eval.loose_success_rate,
                    "optimal_success_rate": heldout_eval.optimal_success_rate,
                    "mean_return": heldout_eval.mean_return,
                },
            ]
        )

    evaluation_figures = plot_evaluation_summary(
        evaluation_rows,
        title="Tabular Baseline Evaluation Summary",
        subtitle="Train-instance performance versus held-out random-instance generalization",
        output_path=os.path.join(figures_dir, "evaluation_summary.png"),
    )

    payload = {
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
        "train_instance": {
            "instance": train_generated.instance.to_dict(),
            "oracle": train_generated.oracle.to_dict(),
        },
        "heldout_instances": [
            {
                "instance": item.instance.to_dict(),
                "oracle": item.oracle.to_dict(),
            }
            for item in eval_generated
        ],
        "encoder_runs": results,
        "figures": {
            "evaluation_summary": evaluation_figures,
        },
    }
    output_path = os.path.join(output_dir, "baseline_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
