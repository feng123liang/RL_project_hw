# Q-Learning 最短路径配送泛化性实验报告

本文档总结目前已经完成的所有关键实验，目标是让后来读代码或看结果的人能快速理解：我们为什么从固定地图 Q-learning 走到随机地图泛化实验，以及每一步实验得到了什么结论。

项目主目录：

`.`

详细工作日志：

`WORK_LOG.md`

## 1. 研究目标

本项目当前主线是探索 Q-learning 在最短路径配送任务中的泛化性。

核心问题是：

Q-learning 能不能从“只会记住某一张固定地图上的状态-动作表”，逐渐变成“能在随机生成的新地图上仍然走出最短配送路径”的方法？

因此，我们关心的不是单纯在训练地图上成功，而是：

- 能不能迁移到未见过的地图。
- 能不能在完成配送约束后到达终点。
- 能不能以 oracle 最短路径步数到达终点。

## 2. 当前任务定义

当前 `generalization` 代码库使用的是随机网格配送任务。

- 智能体从起点 `S` 出发。
- 地图上有障碍物。
- 智能体需要完成 pickup/dropoff 配送任务。
- 完成所有配送任务之后，智能体需要到达终点 `G`。
- 每张地图由随机生成器生成，并用 oracle 过滤，确保地图可解。
- oracle 通过 BFS 在完整配送状态空间中求解最短合法路径。

当前主实验的地图设置如下：

| 设置 | 值 |
| --- | ---: |
| 地图大小 | `6 x 6` |
| 障碍物密度 | `0.10` 到 `0.18` |
| 配送任务数 | `1` |
| 最短路径长度下限 | `8` |
| 最大步数系数 | `4.0` |
| 最终训练地图数 | `10`，seeds `410..419` |
| 最终 held-out 测试地图数 | `100`，seeds `610..709` |

核心代码文件：

- `src/instance.py`
- `src/generator.py`
- `src/env.py`
- `src/oracle.py`
- `src/evaluator.py`

## 3. 评价指标

当前实验统一使用两个成功率。

| 指标 | 定义 |
| --- | --- |
| 宽松成功率 | 在 `max_steps` 内完成所有配送任务并到达终点。 |
| 最短路径成功率 | 宽松成功，并且 rollout 步数严格等于 oracle 最短路径步数。 |

第二个指标更严格，也是本项目最重要的指标。因为智能体可能绕路后仍然到达终点，但这不满足“最短距离配送”的目标。

## 4. 主线切换前的背景实验

在重写 `generalization` 独立代码库之前，我们已经在旧项目目录中做过一些实验。这些实验不是当前主代码库的一部分，但它们解释了为什么后来主线会转向“泛化性”和“最短路径成功率”。

### 4.1 固定地图超参数搜索

结果文件：

`background_results/RL_project_hw/hyperparam_sweeps/summary.md`

主要结果：

- 在固定地图上，多个 epsilon decay 和 goal reward 设置都可以达到 `1.000` evaluation success。
- 这说明 Q-learning 能解固定地图，但不能说明它能泛化到变化后的地图。

### 4.2 随机起点泛化实验

结果文件：

`background_results/RL_project_hw/random_start_experiments/summary.md`

结果摘要：

| 训练设置 | 全起点评估成功率 | 全起点平均多走步数 |
| --- | ---: | ---: |
| 固定起点训练 | `0.861` | `0.00` |
| 随机起点训练 | `1.000` | `0.00` |

结论：

- 在训练阶段随机化起点能显著提升对不同起点的泛化。
- 这启发我们进一步随机化地图，而不仅仅随机化起点。

### 4.3 不同难度 level 的超参数搜索

结果文件：

`background_results/RL_project_hw_advanced_sweep/advanced_level_hyperparam_sweeps/summary.md`

主要结果：

- 宽松成功率经常很高，很多设置达到 `1.000`。
- 但是最短路径成功率更敏感。
- 在带 bonus 或复杂约束的 level 上，宽松成功率可以很高，但最短路径成功率仍然接近 `0.000`。

结论：

- 只看“是否到达终点”会掩盖路径效率问题。
- 因此后续实验必须同时报告宽松成功率和最短路径成功率。

### 4.4 Level3 必经中间点实验

结果文件：

`background_results/RL_project_hw_advanced_sweep/level3_mandatory_waypoint/summary.md`

结果摘要：

| 指标 | 平均值 |
| --- | ---: |
| 宽松到达终点成功率 | `1.000` |
| 严格经过中间点成功率 | `1.000` |
| 最短路径成功率 | `0.200` |
| 严格成功平均步数 | `47.400` |
| oracle 全中间点最短路径 | `45` |

结论：

- 智能体能学会满足任务约束，但不一定能学到最短路径。
- 这进一步强化了“最短路径成功率”作为主指标的重要性。

## 5. 独立 generalization 代码库

当前主实验都在独立目录中完成：

`.`

主要模块如下：

| 模块 | 作用 |
| --- | --- |
| `instance.py` | 定义不可变的配送地图实例。 |
| `generator.py` | 随机生成可解地图。 |
| `oracle.py` | 精确最短路径 oracle。 |
| `env.py` | 网格配送环境和奖励逻辑。 |
| `encoders.py` | 不同状态表示。 |
| `train_tabular.py` | Tabular Q-learning。 |
| `train_linear.py` | 线性近似 Q-learning。 |
| `train_dqn.py` | 神经网络 Q-learning 和 replay DQN。 |
| `evaluator.py` | rollout 评估和指标统计。 |
| `plotting.py` | PDF-only 可视化。 |

可视化也已经统一：

- 只保存 PDF。
- 地图有统一风格。
- 显示起点、终点、pickup/dropoff、障碍物和路线箭头。
- 右侧卡片显示 oracle 最短步数和 agent rollout 步数。

## 6. 尝试过的状态表示

| 状态表示 | 设计思路 | 结果 |
| --- | --- | --- |
| `absolute` | 原始 row/col/task 状态。 | 能解固定图，不泛化。 |
| `feature` | 坐标、墙、目标相对方向等特征。 | 能解训练图，held-out 仍为 `0`。 |
| `feature_v2` | 减少绝对坐标依赖，加入拓扑特征。 | held-out 仍为 `0`。 |
| `distance_state` | 使用到当前目标的最短路距离和各动作距离变化。 | tabular 下训练表现好，但不泛化。 |
| `patch3x3` | 当前位置周围 `3x3` 局部窗口加任务信息。 | 训练图可完成，held-out 不泛化。 |
| `patch3x3_plus` | `3x3` 局部窗口加拓扑、最短路距离、动作级 progress 特征。 | tabular 仍失败，但配合函数近似效果很好。 |
| `patch3x3_plus_vector` | 向量化后的 `patch3x3_plus`，用于线性 Q 和 DQN。 | 当前最重要的成功状态表示。 |

关键结论：

状态设计有帮助，但如果学习器仍然是 tabular，泛化仍然很弱。真正的转折点是引入函数近似，让不同地图中的相似局部结构共享参数。

## 7. 主要实验过程

### 7.1 固定地图 tabular baseline

脚本：

`scripts/run_tabular_baseline.py`

结果：

`results/tabular_baseline/baseline_summary.json`

| Encoder | Train loose | Train optimal | Held-out loose | Held-out optimal |
| --- | ---: | ---: | ---: | ---: |
| `absolute` | `1.00` | `1.00` | `0.00` | `0.00` |
| `feature` | `1.00` | `1.00` | `0.00` | `0.00` |

结论：

- 固定地图可以被 tabular Q-learning 记住并解决。
- 一旦换 held-out 地图，成功率直接归零。
- 这说明固定地图结果主要是记忆，不是泛化。

### 7.2 多地图 tabular 训练

脚本：

`scripts/run_distributional_tabular_generalization.py`

结果：

`results/distributional_tabular_generalization/distributional_summary.json`

| Encoder | Train loose | Train optimal | Held-out loose | Held-out optimal |
| --- | ---: | ---: | ---: | ---: |
| `absolute` | `0.00` | `0.00` | `0.00` | `0.00` |
| `feature` | `1.00` | `1.00` | `0.00` | `0.00` |
| `feature_v2` | `1.00` | `0.75` | `0.00` | `0.00` |

结论：

- 多地图训练让 `feature` 类状态能解决训练分布。
- 但是 tabular 表仍然不能迁移到 unseen maps。
- 问题不是训练地图太少这么简单，而是 tabular 表不能共享相似状态的经验。

### 7.3 更复杂的 tabular 状态设计

脚本：

- `scripts/run_distance_state_experiment.py`
- `scripts/run_patch3x3_experiment.py`
- `scripts/run_patch3x3_plus_experiment.py`

结果：

- `results/distance_state_experiment/distance_state_summary.json`
- `results/patch3x3_experiment/patch3x3_summary.json`
- `results/patch3x3_plus_experiment/patch3x3_plus_summary.json`

代表性结果：

| Encoder | Train loose | Train optimal | Held-out loose | Held-out optimal |
| --- | ---: | ---: | ---: | ---: |
| `distance_state` | `1.00` | `1.00` | `0.00` | `0.00` |
| `patch3x3` | `1.00` | `0.75` | `0.00` | `0.00` |
| `patch3x3_plus` | `1.00` | `0.75` | `0.00` | `0.00` |

结论：

- 局部窗口、拓扑和最短路距离特征都能帮助训练图表现。
- 但 tabular Q-learning 仍然无法在 held-out 地图上泛化。

### 7.4 线性近似 Q-learning

脚本：

`scripts/run_linear_patch_q_experiment.py`

结果：

`results/linear_patch_q_experiment/linear_patch_q_summary.json`

| Method | Train loose | Train optimal | Held-out loose | Held-out optimal |
| --- | ---: | ---: | ---: | ---: |
| `distance_state_tabular` | `1.00` | `1.00` | `0.00` | `0.00` |
| `patch3x3_plus_tabular` | `1.00` | `0.75` | `0.00` | `0.00` |
| `patch3x3_plus_linear` | `0.50` | `0.50` | `0.25` | `0.25` |

结论：

- 这是当前重写代码库里第一次出现非零 held-out 泛化。
- 即使训练表现还不完美，线性函数近似已经开始把跨地图的相似结构共享起来。

### 7.5 线性 Q 超参数搜索

脚本：

`scripts/run_linear_patch_q_sweep.py`

结果：

`results/linear_patch_q_sweep/linear_patch_q_sweep_summary.json`

可读表格 PDF：

`results/linear_patch_q_sweep/figures/heldout_sweep_rankings_table.pdf`

4 张训练地图设置下的最佳超参数：

| 超参数 | 值 |
| --- | ---: |
| Episodes | `1200` |
| Alpha | `0.01` |
| Gamma | `0.98` |
| Epsilon start | `1.0` |
| Epsilon end | `0.05` |
| Epsilon decay fraction | `0.75` |
| Max TD error | `3.0` |
| Max weight norm | `80.0` |
| Seed | `23` |

小 held-out set 上的最佳结果：

| Train loose | Train optimal | Held-out loose | Held-out optimal |
| ---: | ---: | ---: | ---: |
| `1.00` | `1.00` | `1.00` | `1.00` |

结论：

- 线性 Q 对超参数非常敏感。
- 学习率、TD error clip、weight norm clip 都很关键。
- 小 held-out set 上的完美结果需要更大的测试集验证。

### 7.6 最优线性 Q 在 100 张 held-out 地图上验证

脚本：

`scripts/run_best_linear_q_100_heldout.py`

结果：

`results/best_linear_q_100_heldout/best_linear_q_100_heldout_summary.json`

| Train loose | Train optimal | Held-out loose | Held-out optimal |
| ---: | ---: | ---: | ---: |
| `1.00` | `1.00` | `0.94` | `0.90` |

结论：

- 线性 Q 的泛化能力是真实存在的，不是小测试集偶然现象。
- 但仍有一部分 held-out 地图无法以最短路径完成。

### 7.7 训练地图从 4 张扩展到 10 张

第一次直接复用旧超参数：

`scripts/run_best_linear_q_train10_eval100.py`

结果：

`results/best_linear_q_train10_eval100/best_linear_q_train10_eval100_summary.json`

| Train loose | Train optimal | Held-out loose | Held-out optimal |
| ---: | ---: | ---: | ---: |
| `0.50` | `0.40` | `0.57` | `0.45` |

重新针对 10 张训练地图调参：

`scripts/run_linear_patch_q_train10_sweep.py`

结果：

`results/linear_patch_q_train10_sweep/linear_patch_q_train10_sweep_summary.json`

10 张训练地图设置下的最佳超参数：

| 超参数 | 值 |
| --- | ---: |
| Episodes | `2400` |
| Alpha | `0.005` |
| Gamma | `0.98` |
| Epsilon start | `1.0` |
| Epsilon end | `0.05` |
| Epsilon decay fraction | `0.75` |
| Max TD error | `3.0` |
| Max weight norm | `60.0` |
| Seed | `23` |

重新调参后的结果：

| Train loose | Train optimal | Held-out loose | Held-out optimal |
| ---: | ---: | ---: | ---: |
| `1.00` | `1.00` | `0.95` | `0.93` |

结论：

- 增加训练地图本身不是问题。
- 之前性能下降主要是因为超参数不适配更大的训练分布。
- 重新调参后，10 张训练地图略优于 4 张训练地图。

### 7.8 第一版神经网络 Q-learning

脚本：

`scripts/run_dqn_patch_q_experiment.py`

结果：

`results/dqn_patch_q_experiment/dqn_patch_q_summary.json`

网络结构：

`Linear(48, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, 4)`

训练方式：

- 在线单步 Q-learning。
- SmoothL1 loss。
- Adam optimizer。
- Gradient clipping。
- 没有 replay buffer。
- 没有 target network。

结果：

| 方法 | Held-out loose | Held-out optimal |
| --- | ---: | ---: |
| Tuned linear Q | `0.95` | `0.93` |
| First online DQN | `0.87` | `0.87` |

结论：

- 神经网络 Q-learning 能工作，但第一版在线 DQN 不如调好的线性 Q。
- 问题大概率不只是网络表达能力，而是训练 recipe 不够稳定。

### 7.9 Replay-buffer DQN

脚本：

`scripts/run_dqn_replay_patch_q_experiment.py`

结果：

`results/dqn_replay_patch_q_experiment/dqn_replay_patch_q_summary.json`

对比图：

`results/dqn_replay_patch_q_experiment/figures/evaluation_summary.pdf`

网络结构：

`Linear(48, 96) -> ReLU -> Linear(96, 96) -> ReLU -> Linear(96, 4)`

训练方式：

- Replay buffer。
- Mini-batch 更新。
- Target network。
- Double-DQN target action selection。
- Bootstrap 时 mask 非法动作。
- SmoothL1 loss。
- Adam optimizer。
- Gradient clipping。

Replay DQN 配置：

| 超参数 | 值 |
| --- | ---: |
| Episodes | `4200` |
| Learning rate | `0.0007` |
| Gamma | `0.98` |
| Epsilon start | `1.0` |
| Epsilon end | `0.05` |
| Epsilon decay fraction | `0.85` |
| Hidden dim | `96` |
| Replay capacity | `40000` |
| Batch size | `96` |
| Warmup steps | `400` |
| Target update interval | `300` |
| Double DQN | `True` |

最终结果：

| 方法 | Train loose | Train optimal | Held-out loose | Held-out optimal |
| --- | ---: | ---: | ---: | ---: |
| Tuned linear Q | `1.00` | `1.00` | `0.95` | `0.93` |
| Replay DQN | `1.00` | `1.00` | `1.00` | `1.00` |

结论：

- 这是目前第一次明确超过 tuned linear Q 的神经网络方法。
- 提升不只是来自 MLP，而是来自更成熟的 DQN 训练机制。
- Replay 提高样本利用率。
- Target network 稳定 bootstrap。
- Double-DQN 和非法动作 mask 减少 overestimation 以及非法动作带来的问题。

## 8. 总结果表

| 阶段 | 方法 | Held-out 设置 | Held-out loose | Held-out optimal | 主要结论 |
| --- | --- | --- | ---: | ---: | --- |
| 固定地图 baseline | Tabular absolute / feature | 小 held-out set | `0.00` | `0.00` | 固定地图学习主要是记忆。 |
| 多地图 tabular | Tabular feature variants | 小 held-out set | `0.00` | `0.00` | 多地图训练仍不能解决 tabular 泛化。 |
| 手工状态设计 | Distance / patch states, tabular | 小 held-out set | `0.00` | `0.00` | 状态设计有帮助，但 tabular 仍失败。 |
| 初版线性 Q | Linear over `patch3x3_plus_vector` | 4 张 held-out | `0.25` | `0.25` | 函数近似首次带来非零泛化。 |
| 线性 Q sweep | Tuned linear Q | 4 张 held-out | `1.00` | `1.00` | 超参数非常关键。 |
| 线性 Q 大测试 | Tuned linear Q | 100 张 held-out | `0.94` | `0.90` | 线性近似有稳定泛化能力。 |
| 10 图训练调参 | Tuned linear Q | 100 张 held-out | `0.95` | `0.93` | 更多训练图在调参后有帮助。 |
| 第一版 DQN | MLP，无 replay/target | 100 张 held-out | `0.87` | `0.87` | 神经网络能学，但 recipe 不够成熟。 |
| Replay DQN | MLP + replay + target | 100 张 held-out | `1.00` | `1.00` | 当前最强结果。 |

## 9. 主要结论

### 9.1 Tabular Q-learning 在这个任务上泛化很弱

Tabular agent 可以解决固定地图或训练地图，但对未见过的随机地图基本失败。即使加入局部墙信息、目标方向、最短路距离、`3x3` window 等手工特征，tabular 方法仍然无法稳定泛化。

核心原因是：

- Tabular Q-learning 对每个离散 state key 单独存 Q 值。
- 不同地图中相似的局部结构不会自然共享参数。
- 新地图会产生训练中没见过或很少见过的状态组合。

### 9.2 状态设计有用，但函数近似才是转折点

`patch3x3_plus_vector` 是目前最重要的状态表示，因为它同时包含：

- 局部障碍布局。
- 局部拓扑。
- 当前任务进度。
- 当前目标模式。
- 到目标和终点的最短路距离信息。
- 每个动作对目标距离的影响。

但是，这个状态只有和线性近似 Q 或 DQN 结合之后才真正发挥作用。

### 9.3 线性 Q 是一个很强的 baseline

调参后的线性 Q 在 `10` 张训练地图和 `100` 张 held-out 地图上达到：

- held-out 宽松成功率 `0.95`。
- held-out 最短路径成功率 `0.93`。

这说明 Q-learning 在合适状态表示和函数近似下确实能泛化。

### 9.4 Replay DQN 是目前最强方法

Replay DQN 在当前 `100` 张 held-out 地图上达到：

- held-out 宽松成功率 `1.00`。
- held-out 最短路径成功率 `1.00`。

这是目前最强结果，说明更成熟的神经网络 Q-learning 训练方式可以学习到可迁移的最短配送策略。

### 9.5 训练分布和超参数必须一起看

直接把 4 图训练的最佳线性 Q 超参数搬到 10 图训练时，结果下降到：

- held-out 宽松成功率 `0.57`。
- held-out 最短路径成功率 `0.45`。

重新调参后恢复到：

- held-out 宽松成功率 `0.95`。
- held-out 最短路径成功率 `0.93`。

所以不能简单说“训练图更多就一定更好”或“一定更差”，必须同时控制训练预算和超参数。

## 10. 重要结果路径

最终 Replay DQN 对比：

- `results/dqn_replay_patch_q_experiment/dqn_replay_patch_q_summary.json`
- `results/dqn_replay_patch_q_experiment/figures/evaluation_summary.pdf`
- `results/dqn_replay_patch_q_experiment/figures/training_curves_patch3x3_plus_dqn_replay.pdf`
- `results/dqn_replay_patch_q_experiment/figures/patch3x3_plus_dqn_replay/heldout_distribution_100_examples`

最优线性 Q 结果：

- `results/linear_patch_q_train10_sweep/linear_patch_q_train10_sweep_summary.json`
- `results/linear_patch_q_train10_sweep/figures/train10_sweep_rankings_table.pdf`
- `results/best_linear_q_100_heldout/best_linear_q_100_heldout_summary.json`

早期 tabular 和状态设计结果：

- `results/tabular_baseline/baseline_summary.json`
- `results/distributional_tabular_generalization/distributional_summary.json`
- `results/distance_state_experiment/distance_state_summary.json`
- `results/patch3x3_experiment/patch3x3_summary.json`
- `results/patch3x3_plus_experiment/patch3x3_plus_summary.json`

旧代码库背景实验：

- `background_results/RL_project_hw/hyperparam_sweeps/summary.md`
- `background_results/RL_project_hw/random_start_experiments/summary.md`
- `background_results/RL_project_hw_advanced_sweep/advanced_level_hyperparam_sweeps/summary.md`
- `background_results/RL_project_hw_advanced_sweep/level3_mandatory_waypoint/summary.md`

## 11. 后续建议

当前 Replay DQN 的结果很强，但还需要鲁棒性验证。

建议下一步：

- 换一批新的 `100` 或 `500` 张 held-out 地图测试 Replay DQN。
- 用多个训练随机种子重复 Replay DQN，估计方差。
- 扫 `learning_rate`、`hidden_dim`、`batch_size`、`target_update_interval`。
- 提高地图难度，例如更大地图、更多障碍、多个配送任务。
- 做 ablation：只开 replay、只开 target network、关闭 Double-DQN、关闭非法动作 mask。
- 做一个更严格实验：不使用 oracle-derived shortest-path features，只用局部观测，看 DQN 是否还能泛化。

## 12. 一句话总结

目前实验说明：tabular Q-learning 主要是在记忆地图，单纯手工状态设计不足以泛化；线性函数近似首次带来稳定迁移，而加入 replay buffer、target network 和 Double-DQN 的成熟版 DQN 在当前 `100` 张 held-out 随机地图上达到了 `1.00 / 1.00` 的宽松成功率和最短路径成功率。
