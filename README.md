# RL Final Project Baseline

This baseline implements **tabular Q-learning** for a robot-delivery GridWorld task.

Current default config:
- `10x10` GridWorld
- fixed start `S=(0, 0)` and goal `G=(9, 9)`
- static obstacle walls with narrow gaps
- tabular Q-learning + epsilon-greedy exploration

## Structure

```text
.
|-- configs/
|   `-- base.yaml
|-- results/
|   |-- figures/
|   |-- hyperparam_sweeps/
|   |-- logs/
|   `-- models/
`-- src/
    |-- agent.py
    |-- env.py
    |-- evaluate.py
    |-- hyperparam_experiments.py
    |-- train.py
    |-- utils.py
    `-- visualize.py
```

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Train baseline

```bash
python src/train.py --config configs/base.yaml
```

Outputs:
- `results/models/q_table.npy`
- `results/logs/train_metrics.json`
- `results/figures/reward_curve.png`
- `results/figures/reward_moving_average.png`
- `results/figures/success_rate_curve.png`
- `results/figures/average_steps_curve.png`
- `results/figures/final_path.png`

## 3) Evaluate trained model

```bash
python src/evaluate.py --config configs/base.yaml --episodes 200
```

Outputs:
- `results/logs/eval_summary.json`

## 4) Run single-factor hyperparameter experiments

This script runs controlled comparisons where **only one hyperparameter changes at a time**.

Sweeps included by default:
- `epsilon_decay = 0.99 / 0.995 / 0.999`
- `goal_reward = 20 / 50 / 80`

```bash
python src/hyperparam_experiments.py --config configs/base.yaml --output-root results/hyperparam_sweeps --eval-episodes 200
```

Outputs:
- `results/hyperparam_sweeps/summary.json`
- `results/hyperparam_sweeps/summary.md`
- `results/hyperparam_sweeps/figures/epsilon_decay_comparison.png`
- `results/hyperparam_sweeps/figures/goal_reward_comparison.png`

## Notes

- The baseline training run saves the latest Q-table under `results/models/q_table.npy`.
- The hyperparameter sweep stores each run in its own subdirectory under `results/hyperparam_sweeps/`.
- By default, `results/` is ignored by git. If you want to version selected result files, add them explicitly with `git add -f`.
