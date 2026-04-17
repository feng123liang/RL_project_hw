from __future__ import annotations

"""评估入口。

该文件负责加载训练好的 Q 表，并在评估阶段使用纯贪心策略运行多个 episode，
统计成功率、平均奖励和成功 episode 的平均步数。
"""

import argparse
import os
from typing import Dict

import numpy as np
import yaml

from train import build_env
from utils import ensure_dir, save_json


def load_config(config_path: str) -> Dict:
    """读取 YAML 配置文件。"""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(config_path: str, episodes: int) -> Dict:
    """加载已训练模型并执行多轮评估。

    与训练阶段不同，这里不再探索，也不更新 Q 表，
    每一步都直接选择当前状态下 Q 值最大的动作。
    """

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
    """解析命令行参数并启动评估。"""

    parser = argparse.ArgumentParser(description="Evaluate Q-learning baseline")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to yaml config")
    parser.add_argument("--episodes", type=int, default=200, help="Number of evaluation episodes")
    args = parser.parse_args()

    evaluate(args.config, args.episodes)


if __name__ == "__main__":
    main()
