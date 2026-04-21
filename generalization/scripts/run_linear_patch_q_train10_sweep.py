from __future__ import annotations

"""Run a focused hyperparameter sweep for linear Q-learning with 10 training maps."""

import itertools
import json
import os
import sys
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoders import encode_patch3x3_plus_vector
from src.evaluator import evaluate_policy
from src.generator import GeneratorConfig, sample_instance_batch
from src.plotting import STYLE, plot_delivery_map, plot_training_curves, save_figure_pdf
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


def compact_run_record(
    *,
    run_name: str,
    config: LinearQLearningConfig,
    training_summary,
    train_eval,
    heldout_eval,
) -> Dict[str, Any]:
    tail = training_summary.episodes[-50:] if len(training_summary.episodes) >= 50 else training_summary.episodes
    tail_success = sum(1.0 for ep in tail if ep.success) / len(tail) if tail else 0.0
    final_episode = training_summary.episodes[-1].to_dict() if training_summary.episodes else {}
    return {
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
        "train_metrics": {
            "loose_success_rate": train_eval.loose_success_rate,
            "optimal_success_rate": train_eval.optimal_success_rate,
            "mean_return": train_eval.mean_return,
            "mean_steps": train_eval.mean_steps,
        },
        "heldout_metrics": {
            "loose_success_rate": heldout_eval.loose_success_rate,
            "optimal_success_rate": heldout_eval.optimal_success_rate,
            "mean_return": heldout_eval.mean_return,
            "mean_steps": heldout_eval.mean_steps,
        },
        "last_50_training_success_rate": tail_success,
        "final_episode": final_episode,
    }


def score_run(run: Dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(run["heldout_metrics"]["optimal_success_rate"]),
        float(run["heldout_metrics"]["loose_success_rate"]),
        float(run["train_metrics"]["optimal_success_rate"]),
        float(run["train_metrics"]["loose_success_rate"]),
        -float(run["heldout_metrics"]["mean_steps"]),
    )


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
    loose_only = [rollout for rollout in failed if rollout.success]
    hard_fail = [rollout for rollout in failed if not rollout.success]

    selected_rollouts = successful[:3] + loose_only[:2] + hard_fail[:2]
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


def render_top_runs_table(top_runs: List[Dict[str, Any]], output_path: str) -> Dict[str, str]:
    fig, ax = plt.subplots(figsize=(16.5, 8.5), facecolor=STYLE.background)
    ax.set_axis_off()

    title_box = FancyBboxPatch(
        (0.03, 0.905),
        0.94,
        0.07,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        transform=ax.transAxes,
        linewidth=0,
        facecolor=STYLE.title_bar,
    )
    ax.add_patch(title_box)
    ax.text(
        0.50,
        0.948,
        "Train-10 Linear-Q Sweep Rankings",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=STYLE.title_text,
    )
    ax.text(
        0.50,
        0.918,
        "Top held-out configurations for patch3x3_plus linear approximate Q-learning with 10 training maps",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.0,
        color="#d1d5db",
    )

    headers = [
        "Rank",
        "Episodes",
        "Alpha",
        "Decay",
        "W Norm",
        "Seed",
        "Held Opt",
        "Held Loose",
        "Train Opt",
        "Train Loose",
        "Held Steps",
        "Last50 Train",
    ]
    col_widths = [0.06, 0.09, 0.08, 0.08, 0.09, 0.06, 0.09, 0.10, 0.09, 0.10, 0.10, 0.11]
    x0 = 0.03
    table_width = 0.94
    normalized_widths = [width / sum(col_widths) * table_width for width in col_widths]
    col_lefts: List[float] = [x0]
    for width in normalized_widths[:-1]:
        col_lefts.append(col_lefts[-1] + width)

    header_y = 0.84
    row_height = 0.056

    for left, width, header in zip(col_lefts, normalized_widths, headers):
        header_box = FancyBboxPatch(
            (left, header_y),
            width,
            row_height,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            transform=ax.transAxes,
            linewidth=0.8,
            edgecolor=STYLE.panel_edge,
            facecolor="#e7e5e4",
        )
        ax.add_patch(header_box)
        ax.text(
            left + width / 2.0,
            header_y + row_height / 2.0,
            header,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color=STYLE.text_primary,
        )

    def fmt_float(value: float) -> str:
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.3g}"

    for idx, run in enumerate(top_runs, start=1):
        cfg = run["config"]
        held = run["heldout_metrics"]
        train = run["train_metrics"]
        values = [
            str(idx),
            str(cfg["episodes"]),
            fmt_float(cfg["alpha"]),
            fmt_float(cfg["epsilon_decay_fraction"]),
            fmt_float(cfg["max_weight_norm"]),
            str(cfg["seed"]),
            f"{held['optimal_success_rate']:.2f}",
            f"{held['loose_success_rate']:.2f}",
            f"{train['optimal_success_rate']:.2f}",
            f"{train['loose_success_rate']:.2f}",
            f"{held['mean_steps']:.2f}",
            f"{run['last_50_training_success_rate']:.2f}",
        ]
        y = header_y - idx * row_height
        row_face = "#fffdf8" if idx % 2 == 1 else "#f5f5f4"
        if idx <= 2:
            row_face = "#ecfdf5"
        elif held["optimal_success_rate"] >= 0.5:
            row_face = "#eff6ff"

        for left, width, value in zip(col_lefts, normalized_widths, values):
            cell = FancyBboxPatch(
                (left, y),
                width,
                row_height,
                boxstyle="round,pad=0.004,rounding_size=0.009",
                transform=ax.transAxes,
                linewidth=0.6,
                edgecolor=STYLE.panel_edge,
                facecolor=row_face,
            )
            ax.add_patch(cell)
            ax.text(
                left + width / 2.0,
                y + row_height / 2.0,
                value,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color=STYLE.text_primary,
            )

    note_box = FancyBboxPatch(
        (0.03, 0.06),
        0.94,
        0.08,
        boxstyle="round,pad=0.012,rounding_size=0.016",
        transform=ax.transAxes,
        linewidth=0.8,
        edgecolor=STYLE.panel_edge,
        facecolor=STYLE.panel_bg,
    )
    ax.add_patch(note_box)
    ax.text(
        0.05,
        0.115,
        "This sweep keeps TD clip fixed at 3.0 and retunes training budget / learning rate / decay for the 10-map training setup.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.4,
        color=STYLE.text_primary,
    )
    ax.text(
        0.05,
        0.082,
        "Use this table as the main readable sweep figure; the earlier bar-style sweep figure was intentionally replaced here.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10.4,
        color=STYLE.text_primary,
    )
    return save_figure_pdf(fig, output_path)


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "results", "linear_patch_q_train10_sweep")
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

    episode_options = [1600, 2400, 3200]
    alphas = [0.003, 0.005, 0.01]
    decay_options = [0.75, 0.85]
    weight_norm_options = [60.0, 80.0]
    seed_options = [17, 23]
    max_td_error = 3.0

    sweep_results: List[Dict[str, Any]] = []
    best_run: Dict[str, Any] | None = None
    best_config: LinearQLearningConfig | None = None

    for episodes, alpha, decay, max_weight_norm, seed in itertools.product(
        episode_options,
        alphas,
        decay_options,
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
            split_name="train_distribution_10",
            encoder_name=run_name,
        )
        heldout_eval = evaluate_policy(
            policy=policy,
            instances=[item.instance for item in heldout_batch],
            split_name="heldout_distribution_100",
            encoder_name=run_name,
        )

        compact = compact_run_record(
            run_name=run_name,
            config=config,
            training_summary=training_summary,
            train_eval=train_eval,
            heldout_eval=heldout_eval,
        )
        compact["score_tuple"] = list(score_run(compact))
        sweep_results.append(compact)

        if best_run is None or score_run(compact) > score_run(best_run):
            best_run = compact
            best_config = config

    if best_run is None or best_config is None:
        raise RuntimeError("Sweep produced no runs")

    top_runs = sorted(sweep_results, key=score_run, reverse=True)[:12]

    best_policy, best_training_summary = train_linear_q_learning_on_instances(
        instances=[item.instance for item in train_batch],
        encoder=encode_patch3x3_plus_vector,
        encoder_name="patch3x3_plus_linear",
        config=best_config,
    )
    best_train_eval = evaluate_policy(
        policy=best_policy,
        instances=[item.instance for item in train_batch],
        split_name="train_distribution_10",
        encoder_name=best_run["run_name"],
    )
    best_heldout_eval = evaluate_policy(
        policy=best_policy,
        instances=[item.instance for item in heldout_batch],
        split_name="heldout_distribution_100",
        encoder_name=best_run["run_name"],
    )

    best_training_figures = plot_training_curves(
        [episode.to_dict() for episode in best_training_summary.episodes],
        title=f"Best Train-10 Linear-Q Curves ({best_run['run_name']})",
        subtitle="Best configuration from the 10-training-map focused sweep",
        output_path=os.path.join(figures_dir, "best_training_curves.png"),
    )
    selected_rollout_figures = render_selected_rollouts(
        figures_dir=figures_dir,
        run_name=best_run["run_name"],
        evaluation_summary=best_heldout_eval,
        generated_items=heldout_batch,
        split_name="heldout_distribution_100_examples",
    )
    ranking_table_figures = render_top_runs_table(
        top_runs,
        output_path=os.path.join(figures_dir, "train10_sweep_rankings_table.png"),
    )

    payload = {
        "experiment_name": "linear_patch_q_train10_sweep",
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
            "max_td_error": [max_td_error],
            "max_weight_norm": weight_norm_options,
            "seed": seed_options,
        },
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
        "num_runs": len(sweep_results),
        "runs": sweep_results,
        "top_runs": top_runs,
        "best_run": {
            "run_name": best_run["run_name"],
            "config": best_run["config"],
            "train_evaluation": best_train_eval.to_dict(),
            "heldout_evaluation": best_heldout_eval.to_dict(),
            "training_summary": best_training_summary.to_dict(),
            "score_tuple": best_run["score_tuple"],
            "figures": {
                "training_curves": best_training_figures,
                "selected_heldout_rollouts": selected_rollout_figures,
                "ranking_table": ranking_table_figures,
            },
        },
    }

    output_path = os.path.join(output_dir, "linear_patch_q_train10_sweep_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
