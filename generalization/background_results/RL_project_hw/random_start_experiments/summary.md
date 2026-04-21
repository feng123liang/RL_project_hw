# Random-Start Experiment Summary

This experiment keeps the goal fixed and compares two training settings:

- fixed-start training
- random-start training

Evaluation is deterministic and uses greedy action selection.

- Grid size: `10x10`
- Goal: `(9, 9)`
- Episodes per run: `2000`

## Summary Table

| Training Setup | Train Success | Train Avg Steps | Fixed-Start Eval Steps | All-Starts Eval Success | All-Starts Avg Steps | All-Starts Avg Excess Steps | All-Starts Avg Step Ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Start Training | 0.956 | 36.02 | 28.00 | 0.861 | 15.44 | 0.00 | 1.000 |
| Random Start Training | 0.957 | 22.01 | 28.00 | 1.000 | 16.81 | 0.00 | 1.000 |

## Takeaways

- Best all-start success rate: `Random Start Training`
- Best all-start path efficiency: `Fixed Start Training`