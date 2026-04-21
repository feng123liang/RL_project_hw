from __future__ import annotations

"""Validate the best linear-Q configuration on 100 random held-out maps."""

import json
import os
import sys
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoders import encode_patch3x3_plus_vector
from src.evaluator import evaluate_policy
from src.generator import GeneratorConfig, sample_instance_batch
from src.plotting import plot_delivery_map, plot_training_curves
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
    evaluation_summary,
    generated_items,
    split_name: str,
) -> List[Dict[str, Any]]:
    instance_lookup = {item.instance.instance_id: item for item in generated_items}
    split_dir = os.path.join(figures_dir, run_name, split_name)
    ensure_dir(split_dir)

    successful = [rollout for rollout in evaluation_summary.rollout_summaries if rollout.optimal_success]
    failed = [rollout for rollout in evaluation_summary.rollout_summaries if not rollout.optimal_success]

    selected_rollouts = successful[:3] + failed[:3]
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
            title=f"{split_name.replace('_', ' ').title()} Rollout ({run_name})",
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
    output_dir = os.path.join(PROJECT_ROOT, "results", "best_linear_q_100_heldout")
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
    heldout_seeds = list(range(610, 710))
    heldout_batch = sample_instance_batch(generator_config, seeds=heldout_seeds, split="dist_eval_100")

    best_config = LinearQLearningConfig(
        episodes=1200,
        alpha=0.01,
        gamma=0.98,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_fraction=0.75,
        seed=23,
        max_td_error=3.0,
        max_weight_norm=80.0,
    )
    run_name = "best_linear_q_100_heldout"

    policy, training_summary = train_linear_q_learning_on_instances(
        instances=[item.instance for item in train_batch],
        encoder=encode_patch3x3_plus_vector,
        encoder_name="patch3x3_plus_linear",
        config=best_config,
    )
    train_eval = evaluate_policy(
        policy=policy,
        instances=[item.instance for item in train_batch],
        split_name="train_distribution",
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
        title="Best Linear-Q Training Curves",
        subtitle="Best sweep configuration retrained before 100-map held-out validation",
        output_path=os.path.join(figures_dir, "training_curves_best_linear_q.png"),
    )
    selected_rollout_figures = render_selected_rollouts(
        figures_dir=figures_dir,
        run_name=run_name,
        evaluation_summary=heldout_eval,
        generated_items=heldout_batch,
        split_name="heldout_distribution_100_examples",
    )

    payload = {
        "experiment_name": "best_linear_q_100_heldout",
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
        "best_config": {
            "episodes": best_config.episodes,
            "alpha": best_config.alpha,
            "gamma": best_config.gamma,
            "epsilon_start": best_config.epsilon_start,
            "epsilon_end": best_config.epsilon_end,
            "epsilon_decay_fraction": best_config.epsilon_decay_fraction,
            "seed": best_config.seed,
            "max_td_error": best_config.max_td_error,
            "max_weight_norm": best_config.max_weight_norm,
        },
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
        "training_summary": training_summary.to_dict(),
        "train_evaluation": train_eval.to_dict(),
        "heldout_evaluation": heldout_eval.to_dict(),
        "figures": {
            "training_curves": training_figures,
            "selected_heldout_rollouts": selected_rollout_figures,
        },
    }

    output_path = os.path.join(output_dir, "best_linear_q_100_heldout_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
