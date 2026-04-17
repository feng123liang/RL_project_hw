from __future__ import annotations

"""训练入口。

该文件负责把配置、环境、智能体、训练循环、结果保存和训练后可视化串起来，
是整个 baseline 的主流程入口。
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import yaml

from agent import QLearningAgent
from env import GridWorldEnv, RewardConfig
from utils import ensure_dir, moving_average, save_json, seed_everything
from visualize import plot_final_path, plot_training_curves


def load_config(config_path: str) -> Dict:
    """读取 YAML 配置文件。"""

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(cfg: Dict) -> GridWorldEnv:
    """根据配置字典构建 GridWorld 环境实例。"""

    ecfg = cfg["environment"]
    rcfg = ecfg["rewards"]
    return GridWorldEnv(
        rows=ecfg["rows"],
        cols=ecfg["cols"],
        start=tuple(ecfg["start"]),
        goal=tuple(ecfg["goal"]),
        obstacles=[tuple(x) for x in ecfg["obstacles"]],
        rewards=RewardConfig(step=rcfg["step"], invalid=rcfg["invalid"], goal=rcfg["goal"]),
        max_steps=ecfg["max_steps"],
    )


def build_agent(cfg: Dict, env: GridWorldEnv) -> QLearningAgent:
    """根据配置和环境尺寸构建 Q-learning 智能体。"""

    acfg = cfg["agent"]
    return QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        n_actions=env.n_actions,
        alpha=acfg["alpha"],
        gamma=acfg["gamma"],
        epsilon=acfg["epsilon"],
        epsilon_min=acfg["epsilon_min"],
        epsilon_decay=acfg["epsilon_decay"],
    )


def train(cfg: Dict) -> Dict[str, List[float]]:
    """执行完整训练流程并保存模型、日志和图像结果。

    训练中每个 episode 都会：
    - 从起点开始与环境交互
    - 用 Q-learning 公式更新 Q 表
    - 记录奖励、成功率、成功时步数等指标
    """

    seed_everything(cfg["seed"])

    env = build_env(cfg)
    agent = build_agent(cfg, env)
    episodes = cfg["training"]["episodes"]

    rewards: List[float] = []
    success_flags: List[int] = []
    steps_to_goal_raw: List[float] = []

    success_count = 0
    successful_steps_sum = 0

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        ep_steps = 0
        reached_goal = False

        # 单个 episode 内持续交互，直到到达目标或触发时间上限。
        while not done:
            action = agent.select_action(state, greedy=False)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            ep_steps += 1
            state = next_state

            if info["reached_goal"]:
                reached_goal = True

        if reached_goal:
            success_count += 1
            successful_steps_sum += ep_steps
            steps_to_goal_raw.append(float(ep_steps))
        else:
            steps_to_goal_raw.append(float("nan"))

        rewards.append(float(total_reward))
        success_flags.append(1 if reached_goal else 0)
        agent.decay_epsilon()

        if (ep + 1) % 200 == 0:
            print(
                f"Episode {ep + 1}/{episodes} | "
                f"reward={total_reward:.1f} | "
                f"success_rate={success_count / (ep + 1):.3f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    success_rate_curve = (np.cumsum(success_flags) / np.arange(1, episodes + 1)).tolist()
    avg_steps_curve = []
    running_success = 0
    running_steps = 0
    # 只在成功 episode 上统计平均步数，避免失败轨迹干扰该指标。
    for i, s in enumerate(success_flags):
        if s == 1:
            running_success += 1
            running_steps += int(steps_to_goal_raw[i])
        if running_success == 0:
            avg_steps_curve.append(float("nan"))
        else:
            avg_steps_curve.append(running_steps / running_success)

    paths = cfg["paths"]
    models_dir = paths["models_dir"]
    logs_dir = paths["logs_dir"]
    figures_dir = paths["figures_dir"]
    ensure_dir(models_dir)
    ensure_dir(logs_dir)
    ensure_dir(figures_dir)

    np.save(os.path.join(models_dir, "q_table.npy"), agent.q_table)

    metrics = {
        "episode_reward": rewards,
        "episode_reward_ma": moving_average(rewards, window=50),
        "success_rate": success_rate_curve,
        "avg_steps_to_goal": avg_steps_curve,
        "final_success_rate": float(success_count / episodes),
        "final_avg_steps_success_only": float(successful_steps_sum / max(success_count, 1)),
    }
    save_json(os.path.join(logs_dir, "train_metrics.json"), metrics)

    greedy_path = agent.greedy_path(env, max_steps=env.max_steps)
    plot_training_curves(metrics, figures_dir)
    plot_final_path(env, greedy_path, os.path.join(figures_dir, "final_path.png"))

    print("Training finished.")
    print(f"Final success rate: {metrics['final_success_rate']:.3f}")
    print(f"Final average steps (success episodes): {metrics['final_avg_steps_success_only']:.2f}")

    return metrics


def main() -> None:
    """解析命令行参数并启动训练。"""

    parser = argparse.ArgumentParser(description="Train Q-learning baseline in GridWorld")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
