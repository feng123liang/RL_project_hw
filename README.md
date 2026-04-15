# RL Final Project Baseline

This baseline implements **tabular Q-learning** for a robot-delivery GridWorld task.

## Structure

```text
.
|-- configs/
|   `-- base.yaml
|-- results/
|   |-- figures/
|   |-- logs/
|   `-- models/
`-- src/
    |-- agent.py
    |-- env.py
    |-- evaluate.py
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
