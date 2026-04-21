from __future__ import annotations

"""Run a hyperparameter sweep for linear approximate Q-learning."""

import itertools
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
from src.plotting import plot_delivery_map, plot_evaluation_summary, plot_training_curves
from src.train_linear import LinearQLearningConfig, train_linear_q_learning_on_instances


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def config_name(config: LinearQLearningConfig) -> str:
    return (
        f"ep{config.episodes}"
        f"_a{str(config.alpha).replace('.', 'p')}"
        f"_edf{str(config.epsilon_decay_fraction).replace('.', 'p')}"
        f"_td{str(config.max_td_error).replace('.', 'p')}"
        f"_wn{str(config.max_weight_norm).replace('.', 'p')}"
        f"_s{config.seed}"
    )


def render_best_rollouts(
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
    output_dir = os.path.join(PROJECT_ROOT, "results", "linear_patch_q_sweep")
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

    alphas = [0.005, 0.01, 0.02]
    episode_options = [1200, 1600]
    decay_options = [0.75, 0.85]
    td_error_options = [3.0, 5.0]
    weight_norm_options = [80.0, 120.0]
    seed_options = [17, 23]

    sweep_results: List[Dict[str, Any]] = []
    best_run: Dict[str, Any] | None = None
    best_score = None

    for episodes, alpha, decay, max_td_error, max_weight_norm, seed in itertools.product(
        episode_options,
        alphas,
        decay_options,
        td_error_options,
        weight_norm_options,
        seed_options,
    ):
        config = LinearQLearningConfig(
            episodes=episodes,
            alpha=alpha,
            gamma=0.98,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_fraction=decay,
            seed=seed,
            max_td_error=max_td_error,
            max_weight_norm=max_weight_norm,
        )
        run_name = config_name(config)

        policy, training_summary = train_linear_q_learning_on_instances(
            instances=[item.instance for item in train_batch],
            encoder=encode_patch3x3_plus_vector,
            encoder_name="patch3x3_plus_linear",
            config=config,
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
            split_name="heldout_distribution",
            encoder_name=run_name,
        )

        score = (
            float(heldout_eval.optimal_success_rate),
            float(heldout_eval.loose_success_rate),
            float(train_eval.optimal_success_rate),
            float(train_eval.loose_success_rate),
            -float(heldout_eval.mean_steps),
        )
        result = {
            "run_name": run_name,
            "config": {
                "episodes": config.episodes,
                "alpha": config.alpha,
                "gamma": config.gamma,
                "epsilon_start": config.epsilon_start,
                "epsilon_end": config.epsilon_end,
                "epsilon_decay_fraction": config.epsilon_decay_fraction,
                "seed": config.seed,
                "max_td_error": config.max_td_error,
                "max_weight_norm": config.max_weight_norm,
            },
            "training_summary": training_summary.to_dict(),
            "train_evaluation": train_eval.to_dict(),
            "heldout_evaluation": heldout_eval.to_dict(),
            "score_tuple": list(score),
        }
        sweep_results.append(result)
        if best_score is None or score > best_score:
            best_score = score
            best_run = result

    if best_run is None:
        raise RuntimeError("Sweep produced no runs")

    best_training_figures = plot_training_curves(
        best_run["training_summary"]["episodes"],
        title=f"Best Linear-Q Training Curves ({best_run['run_name']})",
        subtitle="Best configuration from the linear approximate Q-learning hyperparameter sweep",
        output_path=os.path.join(figures_dir, "best_training_curves.png"),
    )

    best_run_obj = next(item for item in sweep_results if item["run_name"] == best_run["run_name"])
    best_train_eval = best_run_obj["train_evaluation"]
    best_heldout_eval = best_run_obj["heldout_evaluation"]

    # Re-wrap JSON payload into a shape compatible with the rendering helper.
    def _to_object_rollouts(payload):
        class _Eval:
            def __init__(self, p):
                self.rollout_summaries = []
                for rollout in p["rollout_summaries"]:
                    class _Rollout:
                        pass
                    obj = _Rollout()
                    for key, value in rollout.items():
                        setattr(obj, key, value)
                    self.rollout_summaries.append(obj)
        return _Eval(payload)

    best_train_rollouts = render_best_rollouts(
        figures_dir=figures_dir,
        run_name=best_run["run_name"],
        evaluation_summary=_to_object_rollouts(best_train_eval),
        generated_items=train_batch,
        split_name="train_distribution",
    )
    best_heldout_rollouts = render_best_rollouts(
        figures_dir=figures_dir,
        run_name=best_run["run_name"],
        evaluation_summary=_to_object_rollouts(best_heldout_eval),
        generated_items=heldout_batch,
        split_name="heldout_distribution",
    )

    top_runs = sorted(
        sweep_results,
        key=lambda item: (
            item["heldout_evaluation"]["optimal_success_rate"],
            item["heldout_evaluation"]["loose_success_rate"],
            item["train_evaluation"]["optimal_success_rate"],
            item["train_evaluation"]["loose_success_rate"],
            -item["heldout_evaluation"]["mean_steps"],
        ),
        reverse=True,
    )[:12]
    evaluation_rows = [
        {
            "encoder_name": run["run_name"],
            "split_name": "heldout",
            "loose_success_rate": run["heldout_evaluation"]["loose_success_rate"],
            "optimal_success_rate": run["heldout_evaluation"]["optimal_success_rate"],
            "mean_return": run["heldout_evaluation"]["mean_return"],
        }
        for run in top_runs
    ]

    evaluation_figures = plot_evaluation_summary(
        evaluation_rows,
        title="Linear-Q Hyperparameter Sweep (Top 12)",
        subtitle="Top held-out configurations for patch3x3_plus linear approximate Q-learning",
        output_path=os.path.join(figures_dir, "heldout_sweep_summary.png"),
    )

    payload = {
        "experiment_name": "linear_patch_q_sweep",
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
        "search_space": {
            "episodes": episode_options,
            "alpha": alphas,
            "epsilon_decay_fraction": decay_options,
            "max_td_error": td_error_options,
            "max_weight_norm": weight_norm_options,
            "seed": seed_options,
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
        "num_runs": len(sweep_results),
        "runs": sweep_results,
        "top_runs": top_runs,
        "best_run": {
            **best_run,
            "figures": {
                "training_curves": best_training_figures,
                "best_train_rollouts": best_train_rollouts,
                "best_heldout_rollouts": best_heldout_rollouts,
            },
        },
        "figures": {
            "heldout_sweep_summary": evaluation_figures,
        },
    }

    output_path = os.path.join(output_dir, "linear_patch_q_sweep_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
