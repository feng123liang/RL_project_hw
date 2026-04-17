# RL Project

This repository contains a tabular Q-learning baseline and a set of curriculum /
transfer-learning experiments for grid-based robot navigation tasks.

The codebase currently has two main tracks:

- `baseline`: the original 2D GridWorld with position-only state and standard Q-learning
- `advanced curriculum`: a richer GridWorld with `key-door`, `timed hazard`, and `bonus` mechanisms, used for curriculum transfer experiments

The current advanced experiment of record is:

- `configs/curriculum/experiment6.yaml`

It is the most up-to-date simplified advanced setting in this repository.

## Repository Layout

```text
project1/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   └── curriculum/
│       ├── experiment1.yaml
│       ├── experiment2.yaml
│       ├── experiment3.yaml
│       ├── experiment4.yaml
│       ├── experiment5.yaml
│       └── experiment6.yaml
├── src/
│   ├── env.py
│   ├── agent.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   ├── curriculum_experiment.py
│   ├── advanced_env.py
│   ├── factored_q_agent.py
│   └── advanced_curriculum_experiment.py
└── scripts/
    ├── visualize_curriculum_maps.py
    ├── visualize_advanced_curriculum_maps.py
    ├── edit_advanced_curriculum_maps.py
    └── edit_advanced_curriculum_maps_web.py
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use Conda or Miniforge, activate your environment first:

```bash
conda activate RL
```

## 1. Baseline

The baseline is a standard tabular Q-learning setup on a simple GridWorld.

### Files

- [src/env.py](/home/ouyang/others/project1/src/env.py): baseline environment
- [src/agent.py](/home/ouyang/others/project1/src/agent.py): baseline Q-learning agent
- [src/train.py](/home/ouyang/others/project1/src/train.py): training entrypoint
- [src/evaluate.py](/home/ouyang/others/project1/src/evaluate.py): evaluation entrypoint
- [configs/base.yaml](/home/ouyang/others/project1/configs/base.yaml): baseline config

### Run baseline training

```bash
python src/train.py --config configs/base.yaml
```

### Run baseline evaluation

```bash
python src/evaluate.py --config configs/base.yaml --episodes 200
```

### Baseline outputs

Typical outputs are written under `results/`:

- `results/models/q_table.npy`
- `results/logs/train_metrics.json`
- `results/logs/eval_summary.json`
- `results/figures/reward_curve.png`
- `results/figures/reward_moving_average.png`
- `results/figures/success_rate_curve.png`
- `results/figures/average_steps_curve.png`
- `results/figures/final_path.png`

## 2. Original Curriculum Experiment

The older curriculum experiment builds on the baseline environment and compares
direct training on a difficult map against progressive pretraining on easier maps.

### Files

- [src/curriculum_experiment.py](/home/ouyang/others/project1/src/curriculum_experiment.py)
- `configs/curriculum/experiment1.yaml`
- `configs/curriculum/experiment2.yaml`
- `scripts/visualize_curriculum_maps.py`

### Run

```bash
python src/curriculum_experiment.py --config configs/curriculum/experiment2.yaml
```

This track is still useful for baseline curriculum studies, but it does not use
the richer state/mechanism setting described below.

## 3. Advanced Curriculum Experiment

The advanced experiment is the main current research track.

It uses a richer grid environment while keeping the problem fully discrete so it
remains compatible with tabular Q-learning.

### Current recommended config

- [configs/curriculum/experiment6.yaml](/home/ouyang/others/project1/configs/curriculum/experiment6.yaml)

### Core files

- [src/advanced_env.py](/home/ouyang/others/project1/src/advanced_env.py)
- [src/factored_q_agent.py](/home/ouyang/others/project1/src/factored_q_agent.py)
- [src/advanced_curriculum_experiment.py](/home/ouyang/others/project1/src/advanced_curriculum_experiment.py)

### Run advanced experiment

```bash
PYTHONPATH=src python src/advanced_curriculum_experiment.py --config configs/curriculum/experiment6.yaml
```

If you use the project-specific Miniforge environment:

```bash
PYTHONPATH=src /home/ouyang/miniforge3/envs/RL/bin/python src/advanced_curriculum_experiment.py --config configs/curriculum/experiment6.yaml
```

### Advanced outputs

Outputs are written under:

- `results/advanced_curriculum/`

Typical contents:

- `results/advanced_curriculum/runs/<strategy>/seed_<seed>/train_log.json`
- `results/advanced_curriculum/runs/<strategy>/seed_<seed>/final_eval.json`
- `results/advanced_curriculum/summary.json`
- `results/advanced_curriculum/figures/l4_success_rate_compare.png`
- `results/advanced_curriculum/figures/l4_success_rate_vs_l4_training.png`
- `results/advanced_curriculum/figures/l4_bonus_collection_compare.png`
- `results/advanced_curriculum/figures/l4_key_collection_compare.png`
- `results/advanced_curriculum/figures/final_metrics_bar.png`

## 4. Advanced Environment Design

The advanced environment is defined in [src/advanced_env.py](/home/ouyang/others/project1/src/advanced_env.py).

This version intentionally keeps only a small number of mechanisms so the
research question stays focused and tabular Q-learning remains feasible.

### Actions

The robot has 4 actions:

- `up`
- `down`
- `left`
- `right`

The environment no longer uses:

- `wait`
- `interact`
- `recharge`
- explicit environment `time_limit` reward/penalty

### Map elements

Each level may contain:

- `S`: start cell
- `G`: goal cell
- `obstacles`: blocked cells
- `key_cells`: keys
- `door_cells`: locked door cells
- `hazard_cells`: timed hazard cells
- `bonus_cells`: one-time bonus reward cells

### Mechanisms

#### Key-door

- Entering a key cell automatically gives the agent a key
- Door cells cannot be traversed without a key
- Once the key is collected, door cells become traversable

This lets the curriculum teach "take a detour now to unlock a better route later".

#### Timed hazard

- The environment has a cyclic phase counter
- In `experiment6`, the cycle length is `4`
- Hazard is active in phases `[0, 1]`
- If the agent enters a hazard cell while hazard is active, the episode terminates immediately with a hazard penalty

This means a cell may be safe or unsafe depending on the current phase.

#### Bonus

- Bonus cells give a one-time positive reward the first time they are visited
- The same bonus cannot be collected twice in the same episode
- A level may contain multiple bonus cells

This creates a reward tradeoff between:

- shorter path to goal
- longer path with extra reward

## 5. Advanced State Representation

The advanced state is not just `(row, col)`.

It is:

```text
(row, col, has_key, phase, bonus_mask)
```

### Meaning of each component

- `row, col`: current position
- `has_key`: `0` or `1`
- `phase`: current hazard phase
- `bonus_mask`: bitmask describing which bonus cells have already been collected

This is important because the same position may correspond to very different
decision situations:

- before vs after collecting the key
- when hazard is active vs inactive
- before vs after collecting a bonus

### Why this matters for Q-learning

Tabular Q-learning learns `Q(state, action)`.

If these variables were not included in the state, the agent would mix together
incompatible situations, such as:

- a door that is passable only after key collection
- a hazard cell that is dangerous only in some phases
- a bonus that is valuable only before it has been collected

## 6. Reward Function

The current reward settings for `experiment6` are:

```yaml
step: -0.2
invalid: -0.75
hazard: -4.0
bonus: 2.0
goal: 20.0
```

### Interpretation

- `step = -0.2`: every move has a small cost, encouraging shorter paths
- `invalid = -0.75`: hitting walls or invalid moves is more costly than a normal step
- `hazard = -4.0`: stepping into an active hazard is a clear failure signal
- `bonus = +2.0`: bonus is attractive but not large enough to dominate the task
- `goal = +20.0`: reaching the goal remains the main objective

This reward design is intended to balance:

- success
- efficiency
- risk avoidance
- optional bonus collection

## 7. Curriculum Structure in Experiment 6

The current `experiment6` uses four levels.

### `level1`: key-door only

Purpose:

- teach the agent to understand how collecting a key changes which routes are available

### `level2`: hazard only

Purpose:

- teach the agent to avoid or route around timed hazard cells

### `level3`: bonus only

Purpose:

- teach the agent to weigh path efficiency against optional extra reward

### `level4`: combined task

Purpose:

- combine key-door, hazard, and bonus into a single final task
- test whether curriculum pretraining improves sample efficiency or final policy quality

## 8. Strategies Compared

The advanced experiment compares multiple training schedules:

### `direct_l4`

- train only on `level4`

### `curriculum_transfer`

- pretrain on `level1`
- then `level2`
- then `level3`
- finally continue training on `level4`

### `curriculum_transfer_reset_epsilon`

Same as `curriculum_transfer`, but exploration is partially reset at the start
of the `level4` phase.

This is useful for testing whether transfer is helped by additional exploration
when entering the final task.

## 9. Rollout Safety Cap

Although explicit `time_limit` reward/penalty was removed from the environment,
the experiment code still uses a rollout safety cap in
[src/advanced_curriculum_experiment.py](/home/ouyang/others/project1/src/advanced_curriculum_experiment.py).

Default:

```text
rollout_step_cap = rows * cols * 4
```

For an `8x8` map this is:

```text
256 steps
```

This is not an environment mechanism and does not add a time-limit penalty.
It is only an engineering safeguard to prevent infinite wandering episodes from
stalling training.

## 10. Map Visualization

To render the current advanced maps:

```bash
PYTHONPATH=src python scripts/visualize_advanced_curriculum_maps.py --config configs/curriculum/experiment6.yaml
```

Outputs are written under:

- `results/advanced_map_visualizations/experiment6/`

## 11. Map Editing Tools

This repository includes two editors for advanced curriculum maps.

### Local GUI / TUI editor

```bash
python scripts/edit_advanced_curriculum_maps.py --config configs/curriculum/experiment6.yaml
```

Behavior:

- GUI mode when a display is available
- automatic fallback to terminal mode when no `$DISPLAY` is present

### Browser-based editor

Recommended for headless servers:

```bash
python scripts/edit_advanced_curriculum_maps_web.py --config configs/curriculum/experiment6.yaml --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

If you are on a remote server, forward the port first.

### Editor capabilities

- switch between levels
- toggle walls
- add or remove key cells
- add or remove door cells
- add or remove hazard cells
- add or remove bonus cells
- save changes back to YAML

`key` and `bonus` both support multiple cells in the current editor.

## 12. Notes for Collaborators

### Which config should we use by default?

Use:

- [configs/curriculum/experiment6.yaml](/home/ouyang/others/project1/configs/curriculum/experiment6.yaml)

unless a discussion explicitly refers to an earlier ablation or archived setup.

### Which values are the source of truth?

Experiment behavior is determined by the YAML config file, not by the default
values in dataclasses inside the Python code.

For example:

- if `advanced_env.py` has a default `step = -0.5`
- but `experiment6.yaml` sets `step = -0.2`

then the experiment runs with `-0.2`.

### Why do some YAML coordinate lists look strange?

`PyYAML` may save nested coordinate lists in this form:

```yaml
- - 0
  - 1
```

This is valid YAML and equivalent to:

```yaml
- [0, 1]
```

The data is correct; only the formatting is less compact.

## 13. Suggested Workflow

1. Edit maps in `configs/curriculum/experiment6.yaml`
2. Render them with `scripts/visualize_advanced_curriculum_maps.py`
3. Run the advanced experiment
4. Compare:
   - `success_rate`
   - `avg_reward`
   - `avg_steps_success_only`
   - `key_collection_rate`
   - `door_usage_rate`
   - `bonus_collection_rate`
5. Iterate on level design and reward scale

## 14. Development Status

Current status:

- baseline is runnable
- original curriculum experiment is runnable
- advanced environment is runnable
- advanced map visualizer is runnable
- advanced GUI editors are runnable
- `experiment6` is the main current advanced setup

The advanced branch has already been simplified to focus on the mechanisms most
useful for tabular transfer experiments:

- keep: `key-door`, `timed hazard`, `bonus`
- remove: `wait`, `interact`, `recharge`, explicit environment `time_limit`
