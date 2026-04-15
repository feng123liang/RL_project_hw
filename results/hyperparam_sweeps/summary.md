# Hyperparameter Sweep Summary

All runs keep the same map, seed, and non-target hyperparameters.

- Seed: `42`
- Episodes per run: `2000`
- Evaluation episodes per run: `200`

## Epsilon Decay

| Value | Train Success | Train Avg Steps | Eval Success | Eval Avg Reward | Eval Avg Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.99 | 0.964 | 34.72 | 1.000 | 23.00 | 28.00 |
| 0.995 | 0.956 | 36.02 | 1.000 | 23.00 | 28.00 |
| 0.999 | 0.893 | 54.80 | 1.000 | 23.00 | 28.00 |

## Goal Reward

| Value | Train Success | Train Avg Steps | Eval Success | Eval Avg Reward | Eval Avg Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 0.955 | 35.99 | 1.000 | -7.00 | 28.00 |
| 50 | 0.956 | 36.02 | 1.000 | 23.00 | 28.00 |
| 80 | 0.957 | 35.96 | 1.000 | 53.00 | 28.00 |
