from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import yaml

from train import build_env
from utils import ensure_dir, save_json


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(config_path: str, episodes: int) -> Dict:
    cfg = load_config(config_path)
    env = build_env(cfg)

    model_path = os.path.join(cfg["paths"]["models_dir"], "q_table.npy")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first: python src/train.py --config {config_path}"
        )

    q_table = np.load(model_path)

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

    summary = {
        "episodes": episodes,
        "success_rate": float(success / episodes),
        "avg_reward": float(np.mean(total_rewards)),
        "avg_steps_success_only": float(np.mean(steps_success_only)) if steps_success_only else float("nan"),
    }

    logs_dir = cfg["paths"]["logs_dir"]
    ensure_dir(logs_dir)
    save_json(os.path.join(logs_dir, "eval_summary.json"), summary)

    print("Evaluation finished.")
    print(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Q-learning baseline")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to yaml config")
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate(args.config, args.episodes)


if __name__ == "__main__":
    main()
