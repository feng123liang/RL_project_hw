from __future__ import annotations

"""Smoke test for delivery environment and encoders."""

import json
import os
import sys
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoders import encode_absolute_state, encode_feature_state, encode_patch_state
from src.env import GridDeliveryEnv
from src.generator import GeneratorConfig, sample_instance
from src.plotting import plot_delivery_map


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def action_from_step(current, nxt) -> int:
    dr = int(nxt[0] - current[0])
    dc = int(nxt[1] - current[1])
    mapping = {
        (-1, 0): 0,
        (1, 0): 1,
        (0, -1): 2,
        (0, 1): 3,
    }
    return mapping[(dr, dc)]


def rollout_oracle_path(env: GridDeliveryEnv, path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    env.reset()
    records.append(
        {
            "step": 0,
            "position": list(env.position),
            "absolute_state": list(encode_absolute_state(env)),
            "feature_state": encode_feature_state(env),
            "patch_state": encode_patch_state(env),
            "action_mask": env.action_mask().tolist(),
        }
    )

    for step_idx in range(len(path) - 1):
        current = path[step_idx]
        nxt = path[step_idx + 1]
        action = action_from_step(current, nxt)
        _, reward, done, info = env.step(action)
        records.append(
            {
                "step": step_idx + 1,
                "position": list(env.position),
                "action": int(action),
                "action_name": str(info["action_name"]),
                "reward": float(reward),
                "done": bool(done),
                "info": info,
                "absolute_state": list(encode_absolute_state(env)),
                "feature_state": encode_feature_state(env),
                "patch_state": encode_patch_state(env),
                "action_mask": env.action_mask().tolist(),
            }
        )
    return records


def plot_rollout(instance, path, output_path: str, shortest_path_steps: int | None = None) -> Dict[str, str]:
    return plot_delivery_map(
        instance,
        path,
        title=f"Environment Oracle Rollout ({instance.instance_id})",
        output_path=output_path,
        subtitle="Environment dynamics replayed step-by-step using the exact shortest route",
        path_label="Rollout",
        show_direction_arrows=True,
        shortest_path_steps=shortest_path_steps,
        agent_steps=max(0, len(path) - 1) if path else None,
    )


def main() -> None:
    output_dir = os.path.join(PROJECT_ROOT, "results", "delivery_env_smoke")
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(output_dir)
    ensure_dir(figures_dir)

    cfg = GeneratorConfig(
        rows_min=6,
        rows_max=6,
        cols_min=6,
        cols_max=6,
        obstacle_density_min=0.12,
        obstacle_density_max=0.18,
        task_count_min=2,
        task_count_max=2,
        min_optimal_steps=8,
        max_generation_attempts=200,
    )
    generated = sample_instance(cfg, seed=101, split="smoke")
    env = GridDeliveryEnv(generated.instance)
    rollout_records = rollout_oracle_path(env, generated.oracle.path)

    final_info = rollout_records[-1]["info"]
    payload = {
        "instance": generated.instance.to_dict(),
        "oracle": generated.oracle.to_dict(),
        "rollout": rollout_records,
        "checks": {
            "oracle_rollout_success": bool(final_info["success"]),
            "completed_all_deliveries": bool(final_info["completed_all_deliveries"]),
            "goal_reached": bool(final_info["reached_goal"]),
            "final_step_count_matches_optimal": int(rollout_records[-1]["step"]) == int(generated.oracle.optimal_steps),
        },
        "figures": plot_rollout(
            generated.instance,
            generated.oracle.path,
            os.path.join(figures_dir, "delivery_env_oracle_rollout.png"),
            shortest_path_steps=generated.oracle.optimal_steps,
        ),
    }
    output_path = os.path.join(output_dir, "env_smoke_summary.json")
    save_json(output_path, payload)
    print(output_path)


if __name__ == "__main__":
    main()
