# Generalization Work Log

This file records concrete project progress for the standalone delivery generalization codebase under `.`.

Going forward, each meaningful code change, experiment, or visualization update should be appended here.

## 2026-04-21

### Standalone Infrastructure Initialization

- Created a new self-contained project under `.`.
- Kept the new code independent from the old `RL_project_hw_advanced_sweep` codebase.
- Added the core problem-definition and environment files:
  - `src/instance.py`
  - `src/generator.py`
  - `src/oracle.py`
  - `src/env.py`
  - `src/encoders.py`
- Defined the core task as shortest-path delivery on randomly generated grid instances.
- Implemented exact oracle solving with BFS over the full delivery state space.
- Added random instance generation with solvability filtering and minimum optimal path length constraints.
- Added deterministic environment dynamics with delivery completion logic and action masking.
- Added multiple state encoders for later representation experiments:
  - absolute state encoding
  - feature-based encoding
  - local patch encoding

### Smoke Tests And Basic Validation

- Added `scripts/smoke_test_generalization.py` to validate:
  - random solvable instance generation
  - oracle shortest-path correctness
  - manual unit-style oracle sanity checks
- Added `scripts/smoke_test_delivery_env.py` to validate:
  - environment step dynamics
  - oracle rollout replay inside the environment
  - encoder outputs
- Ran the smoke tests successfully.
- Verified that the environment oracle rollout reaches the goal, completes all deliveries, and matches the oracle step count.

### PDF-Only Visualization Pipeline

- Standardized figure output to PDF only.
- Added shared plotting utilities in `src/plotting.py`.
- Refactored both smoke test scripts to use the shared plotting module instead of duplicating plotting logic.
- Ensured all current generated figures are written as PDF files inside:
  - `results/generalization_smoke/figures`
  - `results/delivery_env_smoke/figures`

### Visualization Styling Iteration

- Reworked the grid visualization from a raw debug plot into a more presentation-friendly figure style.
- Added:
  - unified color theme
  - rounded task blocks for pickup/dropoff cells
  - layered path rendering with stronger visual hierarchy
  - styled start/goal markers
  - title banner
  - compact legend panel
- Iterated on the start/goal appearance:
  - enlarged `S` and `G`
  - added layered badge-style rendering
  - later removed white borders on `S/G` based on visual feedback
- Removed the right-side map summary panel so the map itself occupies more space.
- Fixed the title layout so it no longer appears skewed by axis-relative placement.
- Changed the title and subtitle to centered alignment inside the top banner.

### Current Output Files

- Generalization smoke summary:
  - `results/generalization_smoke/smoke_summary.json`
- Delivery environment smoke summary:
  - `results/delivery_env_smoke/env_smoke_summary.json`
- Current figure outputs:
  - `results/generalization_smoke/figures/random_example_1.pdf`
  - `results/generalization_smoke/figures/random_example_2.pdf`
  - `results/generalization_smoke/figures/random_example_3.pdf`
  - `results/generalization_smoke/figures/manual_oracle_check.pdf`
  - `results/delivery_env_smoke/figures/delivery_env_oracle_rollout.pdf`

### Recommended Next Step

- Build the first training/evaluation layer for tabular Q-learning on the new standalone environment.
- Recommended immediate files to add next:
  - `src/train_tabular.py`
  - `src/evaluator.py`
  - `scripts/run_tabular_baseline.py`
- Suggested first experiment:
  - train a tabular Q-learning baseline on fixed random instances
  - evaluate both training-instance performance and held-out random-instance generalization
  - reuse the current PDF plotting style for rollout visualization and learning curves

## Logging Template

Use this structure for future entries:

### YYYY-MM-DD

#### What Changed

- ...

#### What Was Run

- Command:
  - `...`

#### Outputs

- ...

#### Observations

- ...

#### Next Step

- ...

## 2026-04-21

### First Tabular Baseline Stack

- Added `src/train_tabular.py`:
  - reusable tabular Q-learning trainer
  - epsilon-greedy exploration
  - hashable canonicalization for structured encoder outputs
  - serialized per-episode training summaries
- Added `src/evaluator.py`:
  - greedy rollout execution for learned tabular policies
  - per-instance rollout logging
  - split-level aggregate evaluation summaries
- Added `scripts/run_tabular_baseline.py`:
  - trains tabular Q-learning on one fixed random delivery instance
  - evaluates on the training instance and on held-out random instances
  - compares two state encodings:
    - `absolute`
    - `feature`
  - writes a JSON summary and PDF figures
- Extended `src/plotting.py` with:
  - training-curve plotting
  - evaluation-summary plotting

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_tabular_baseline.py`

### Outputs

- Summary JSON:
  - `results/tabular_baseline/baseline_summary.json`
- Figures:
  - `results/tabular_baseline/figures/training_curves_absolute.pdf`
  - `results/tabular_baseline/figures/training_curves_feature.pdf`
  - `results/tabular_baseline/figures/rollout_absolute.pdf`
  - `results/tabular_baseline/figures/rollout_feature.pdf`
  - `results/tabular_baseline/figures/evaluation_summary.pdf`

### Observations

- Both `absolute` and `feature` tabular encodings solved the fixed training instance:
  - train success rate = `1.0`
  - train mean return = `6.2`
  - train mean steps = `9.0`
- Both encodings failed completely on held-out random instances:
  - held-out success rate = `0.0`
  - held-out mean steps = `144.0`
- This is a useful baseline result because it clearly demonstrates the current lack of generalization.
- In this first version, the `feature` tabular encoding still does not improve cross-instance transfer.

### Next Step

- Build a stronger generalization experiment around random-instance training rather than single-instance memorization.
- Recommended immediate follow-up:
  - train on a batch/distribution of random instances
  - compare `absolute` versus `feature` encoding again
  - measure whether distributional training improves held-out success at all

### Visualization Readability Update

- Increased pickup/dropoff text size inside map cells.
- Raised pickup/dropoff text above path lines so labels are no longer hidden by rollout/oracle paths.
- Added a light text background box for better readability.
- Regenerated smoke-test and tabular-baseline PDF figures.

### Visualization Label Style Follow-Up

- Removed the white background boxes behind pickup/dropoff labels after visual review.
- Increased route step-number font size.
- Changed route step-number color from dark red to a warmer amber/brown tone.
- Regenerated smoke-test and tabular-baseline PDF figures again.

### Route Annotation Redesign

- Removed numeric step annotations from the route itself.
- Replaced route numbering with on-path direction arrows.
- Added a right-side route statistics card showing:
  - shortest-path steps
  - agent rollout total steps
- Updated smoke-test figures and tabular-baseline rollout figures to use the new route annotation format.

### Obstacle And Stats Card Refinement

- Replaced the soft raster-style obstacle rendering with explicit full-cell solid obstacle blocks.
- Reworked the right-side route statistics display into two separate cards:
  - `Shortest Path`
  - `Agent Rollout`
- Increased the prominence of the displayed step counts inside those cards.
- Fixed the previous overlap issue between the right-side statistics display and the legend.

### Full Figure Refresh

- Re-ran all current figure-producing scripts so every existing PDF reflects the latest shared plotting style.
- Refreshed:
  - smoke-test figures
  - environment rollout figure
  - tabular baseline figures
  - distributional tabular generalization figures

### Full Distributional Rollout Export

- Extended the distributional training script so it exports rollout PDFs for every evaluated instance, not just one representative example.
- Added full rollout figure sets for both learned Q-tables:
  - all training-distribution instances
  - all held-out-distribution instances

### Feature V2 Representation Experiment

- Added `encode_generalized_feature_state(...)` in `src/encoders.py`.
- Designed the stronger hand-crafted encoder to:
  - remove absolute row/column coordinates
  - emphasize local wall structure
  - include corridor / dead-end / junction indicators
  - include diagonal blocking signals
  - include clipped target-relative geometry
- Added `feature_v2` into the distributional tabular generalization experiment.

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_distributional_feature_v2 PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_distributional_tabular_generalization.py`

### Observations

- `feature_v2` matched the existing `feature` representation on the training distribution:
  - train success rate = `1.0`
  - train mean return = `5.2`
  - train mean steps = `14.0`
- But `feature_v2` still failed completely on held-out instances:
  - held-out success rate = `0.0`
  - held-out mean steps = `144.0`
- So richer hand-crafted local structure features alone were still not enough to unlock held-out generalization.

### Next Step

- The next likely bottleneck is no longer just feature choice inside a pure tabular setup.
- Recommended immediate follow-up:
  - move toward approximate Q-learning or another function-approximation method
  - use the patch-style representation rather than a purely hand-crafted small feature dictionary

### Stats Card Layout Polish

- Shifted the right-side step cards slightly further right.
- Reduced card width slightly to improve spacing relative to the map and legend.

### Distributional Training Experiment

- Extended `src/train_tabular.py` with cross-instance tabular training support via `train_q_learning_on_instances(...)`.
- Added `scripts/run_distributional_tabular_generalization.py`.
- Built the first mainline generalization experiment:
  - train tabular Q-learning across a batch of random training instances
  - evaluate on both the training distribution and a held-out random distribution
  - compare `absolute` and `feature` encodings
  - output JSON summaries and PDF figures

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/matplotlib PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_distributional_tabular_generalization.py`

### Outputs

- Summary JSON:
  - `results/distributional_tabular_generalization/distributional_summary.json`
- Figures:
  - `results/distributional_tabular_generalization/figures/training_curves_absolute.pdf`
  - `results/distributional_tabular_generalization/figures/training_curves_feature.pdf`
  - `results/distributional_tabular_generalization/figures/heldout_rollout_absolute.pdf`
  - `results/distributional_tabular_generalization/figures/heldout_rollout_feature.pdf`
  - `results/distributional_tabular_generalization/figures/evaluation_summary.pdf`

### Observations

- `absolute` encoding did not generalize and did not even stably solve the small training distribution:
  - train success rate = `0.0`
  - held-out success rate = `0.0`
- `feature` encoding improved substantially on the training distribution:
  - train success rate = `1.0`
  - train mean return = `5.3`
  - train mean steps = `13.5`
- But `feature` still failed completely on held-out instances:
  - held-out success rate = `0.0`
  - held-out mean steps = `144.0`
- This suggests two things:
  - training on an instance distribution is better than training on a single fixed instance
  - but the current tabular feature representation is still not enough for true held-out generalization

### Next Step

- Push the representation further rather than only tuning tabular training.
- Recommended immediate follow-up:
  - add richer relational features
  - add local obstacle-structure features
  - or move to approximate / function-approximation Q-learning using the patch-style representation

## 2026-04-21

### Dual Success-Rate Evaluation Upgrade

- Upgraded the standalone evaluation stack so every experiment now reports two separate success metrics:
  - `loose_success_rate`
  - `optimal_success_rate`
- Kept the old `success_rate` field as a compatibility alias for loose success.
- Extended per-rollout records in `src/evaluator.py` with:
  - `optimal_success`
  - `optimal_steps`
  - `step_excess`
- Defined optimal success as:
  - the agent must satisfy the normal task success condition
  - and the rollout length must exactly match the oracle shortest-path step count

### Plot Summary Refresh

- Updated `src/plotting.py` so evaluation summary PDFs now show three panels:
  - loose success rate
  - shortest-path success rate
  - mean return
- Updated both experiment entry scripts to export the new metric fields into their summary plots and JSON outputs:
  - `scripts/run_tabular_baseline.py`
  - `scripts/run_distributional_tabular_generalization.py`

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_tabular_dual_metric PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_tabular_baseline.py`
- Command:
  - `MPLCONFIGDIR=/tmp/mpl_dist_dual_metric PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_distributional_tabular_generalization.py`

### Outputs

- Baseline summary JSON:
  - `results/tabular_baseline/baseline_summary.json`
- Baseline evaluation PDF:
  - `results/tabular_baseline/figures/evaluation_summary.pdf`
- Distributional summary JSON:
  - `results/distributional_tabular_generalization/distributional_summary.json`
- Distributional evaluation PDF:
  - `results/distributional_tabular_generalization/figures/evaluation_summary.pdf`

### Observations

- Fixed-instance baseline:
  - `absolute` and `feature` both reached `1.0` loose success and `1.0` optimal success on the training map
  - both remained at `0.0` loose success and `0.0` optimal success on held-out maps
- Distributional training:
  - `absolute` stayed at `0.0` loose success and `0.0` optimal success on both train and held-out distributions
  - `feature` reached `1.0` loose success and `1.0` optimal success on the training distribution
  - `feature_v2` reached `1.0` loose success but only `0.75` optimal success on the training distribution
  - all three encoders remained at `0.0` loose success and `0.0` optimal success on held-out maps
- The new metric split exposed an important distinction:
  - `feature_v2` can finish all deliveries and reach the goal on the training distribution
  - but it is not consistently doing so along the shortest path

### Next Step

- Move beyond purely tabular memorization-style methods and test approximate Q-learning or another function-approximation approach on randomly generated maps.

## 2026-04-21

### Alternative State-Design Sweep: Path-Distance Abstraction

- Added a new encoder in `src/encoders.py`:
  - `encode_path_distance_state(...)`
- This state design explores a different direction from the earlier encoders:
  - it does not rely primarily on absolute coordinates
  - it does not focus mainly on local wall-shape descriptors
  - instead it encodes how each action changes shortest-path distance to the current target on the current grid
- Included features such as:
  - current shortest-path distance to target
  - current shortest-path distance to goal
  - sign of target / goal displacement
  - whether each action is blocked
  - whether each action is one of the best moves toward the target
  - shortest-path distance delta induced by each action

### New Experiment Script

- Added a separate comparison script:
  - `scripts/run_distance_state_experiment.py`
- Kept this experiment isolated from the previous summary files so older results remain reproducible.
- Compared three encoders under the same distributional training setup:
  - `feature`
  - `feature_v2`
  - `distance_state`

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_distance_state PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_distance_state_experiment.py`

### Outputs

- Summary JSON:
  - `results/distance_state_experiment/distance_state_summary.json`
- Evaluation PDF:
  - `results/distance_state_experiment/figures/evaluation_summary.pdf`
- Representative held-out rollout PDFs:
  - `results/distance_state_experiment/figures/heldout_rollout_feature.pdf`
  - `results/distance_state_experiment/figures/heldout_rollout_feature_v2.pdf`
  - `results/distance_state_experiment/figures/heldout_rollout_distance_state.pdf`

### Observations

- `distance_state` matched `feature` on the training distribution:
  - train loose success rate = `1.0`
  - train optimal success rate = `1.0`
- `distance_state` was more stable than `feature_v2` under the optimal-path metric:
  - `feature_v2` train optimal success = `0.75`
  - `distance_state` train optimal success = `1.0`
- But `distance_state` still failed completely on held-out instances:
  - held-out loose success rate = `0.0`
  - held-out optimal success rate = `0.0`
- So this new direction improved the quality of within-distribution policy execution, but it still did not solve out-of-distribution generalization.

### Next Step

- Since multiple hand-crafted tabular state designs now reach the same wall, the next experiment should likely change the learner class rather than only the state key:
  - approximate Q-learning
  - linear function approximation over patch / relational features
  - or a small neural Q-network over local map observations

## 2026-04-21

### Alternative State-Design Sweep: 3x3 Local Window

- Added a new encoder in `src/encoders.py`:
  - `encode_patch3x3_state(...)`
- This encoder uses the agent-centered `3x3` local observation window as the main state component.
- Kept the extra metadata intentionally small so this remains a real local-window experiment:
  - target row/column direction signs
  - target mode
  - active task / delivered mask / task count

### New Experiment Script

- Added:
  - `scripts/run_patch3x3_experiment.py`
- Compared three state designs under the same distributional training setup:
  - `feature`
  - `distance_state`
  - `patch3x3`

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_patch3x3 PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_patch3x3_experiment.py`

### Outputs

- Summary JSON:
  - `results/patch3x3_experiment/patch3x3_summary.json`
- Evaluation PDF:
  - `results/patch3x3_experiment/figures/evaluation_summary.pdf`
- Representative held-out rollout PDFs:
  - `results/patch3x3_experiment/figures/heldout_rollout_feature.pdf`
  - `results/patch3x3_experiment/figures/heldout_rollout_distance_state.pdf`
  - `results/patch3x3_experiment/figures/heldout_rollout_patch3x3.pdf`

### Observations

- `patch3x3` solved the training distribution in the loose sense:
  - train loose success rate = `1.0`
- But it was weaker than `feature` and `distance_state` on optimal-path execution:
  - `patch3x3` train optimal success rate = `0.75`
  - `feature` train optimal success rate = `1.0`
  - `distance_state` train optimal success rate = `1.0`
- `patch3x3` still failed completely on held-out maps:
  - held-out loose success rate = `0.0`
  - held-out optimal success rate = `0.0`
- So the raw local `3x3` observation is enough for within-distribution task completion, but by itself it still does not provide the structure needed for held-out generalization.

### Next Step

- A stronger next experiment would be to combine the local window with a global progress signal instead of testing either one alone:
  - `patch3x3 + distance_state`
  - or move to approximate / neural Q-learning on top of the patch observation

## 2026-04-21

### Alternative State-Design Sweep: Enhanced 3x3 Local Window

- Added a stronger local-window encoder in `src/encoders.py`:
  - `encode_patch3x3_plus_state(...)`
- This encoder keeps the raw `3x3` patch and augments it with:
  - patch occupancy summary counts
  - local topology indicators
  - shortest-path progress signals
  - per-action target-distance hints

### New Experiment Script

- Added:
  - `scripts/run_patch3x3_plus_experiment.py`
- Compared:
  - `feature`
  - `distance_state`
  - `patch3x3`
  - `patch3x3_plus`

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_patch3x3_plus PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_patch3x3_plus_experiment.py`

### Outputs

- Summary JSON:
  - `results/patch3x3_plus_experiment/patch3x3_plus_summary.json`
- Evaluation PDF:
  - `results/patch3x3_plus_experiment/figures/evaluation_summary.pdf`
- Representative held-out rollout PDFs:
  - `results/patch3x3_plus_experiment/figures/heldout_rollout_patch3x3.pdf`
  - `results/patch3x3_plus_experiment/figures/heldout_rollout_patch3x3_plus.pdf`

### Observations

- `patch3x3_plus` did not improve over `patch3x3` in this setup.
- Both achieved:
  - train loose success rate = `1.0`
  - train optimal success rate = `0.75`
  - held-out loose success rate = `0.0`
  - held-out optimal success rate = `0.0`
- The rollout-level behavior matched as well:
  - both overran one training instance by 2 steps
  - both failed all held-out instances at the time limit
- This suggests the current bottleneck is no longer solved by adding more hand-crafted signals around a tabular local-window key.

### Next Step

- Move the local-window idea into a learner that can share statistical strength across similar observations:
  - linear / approximate Q-learning over patch features
  - or a small neural Q-network over the patch observation

## 2026-04-21

### Approximate Q-Learning Baseline: Linear Q Over Vectorized Patch Features

- Added a new training module:
  - `src/train_linear.py`
- Implemented:
  - `LinearQLearningConfig`
  - `LinearQPolicy`
  - `train_linear_q_learning_on_instances(...)`
- Added vectorized local-window encoding for function approximation:
  - `encode_patch3x3_plus_vector(...)` in `src/encoders.py`
- Generalized the evaluator to accept any policy object with an `act(...)` method so both tabular and linear agents can reuse the same rollout / metric pipeline.

### Stability Follow-Up

- The first linear-Q run showed numeric instability and policy collapse late in training.
- Added stabilization into `src/train_linear.py`:
  - TD-error clipping
  - per-action weight-norm clipping
- Lowered the linear learning rate and increased training episodes in the experiment script.

### New Experiment Script

- Added:
  - `scripts/run_linear_patch_q_experiment.py`
- Compared:
  - `distance_state_tabular`
  - `patch3x3_plus_tabular`
  - `patch3x3_plus_linear`

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_linear_patch_q_v2 PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_linear_patch_q_experiment.py`

### Outputs

- Summary JSON:
  - `results/linear_patch_q_experiment/linear_patch_q_summary.json`
- Evaluation PDF:
  - `results/linear_patch_q_experiment/figures/evaluation_summary.pdf`
- Linear training-curve PDF:
  - `results/linear_patch_q_experiment/figures/training_curves_patch3x3_plus_linear.pdf`
- Linear held-out rollout PDF:
  - `results/linear_patch_q_experiment/figures/heldout_rollout_patch3x3_plus_linear.pdf`

### Observations

- This is the first experiment in the new standalone codebase that achieved non-zero held-out generalization.
- `patch3x3_plus_linear` reached:
  - train loose success rate = `0.5`
  - train optimal success rate = `0.5`
  - held-out loose success rate = `0.25`
  - held-out optimal success rate = `0.25`
- The held-out success was not just loose success:
  - one held-out map was solved exactly at the oracle shortest path length
- By contrast, the tabular reference agents still had:
  - held-out loose success rate = `0.0`
  - held-out optimal success rate = `0.0`
- So parameter sharing across similar observations appears to be the first ingredient that genuinely improves generalization in this project.
- The linear model is still unstable / incomplete as a solution:
  - it only solved 2 of 4 training maps
  - and 1 of 4 held-out maps
  - but this is still a meaningful qualitative step beyond the tabular baselines

### Next Step

- Push the approximate-Q direction further:
  - tune linear-Q hyperparameters
  - compare multiple vector encodings
  - or move to a small neural Q-network over the same patch representation

## 2026-04-21

### Linear Approximate Q-Learning Hyperparameter Sweep

- Added a dedicated sweep script:
  - `scripts/run_linear_patch_q_sweep.py`
- Scanned the linear approximate Q-learning setup over:
  - `alpha`
  - `episodes`
  - `epsilon_decay_fraction`
  - `max_td_error`
  - `max_weight_norm`
  - `seed`
- Total runs:
  - `96`
- The sweep reused the same train / held-out map splits as the previous linear-Q experiment so results are directly comparable.

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_linear_patch_q_sweep PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_linear_patch_q_sweep.py`

### Outputs

- Sweep summary JSON:
  - `results/linear_patch_q_sweep/linear_patch_q_sweep_summary.json`
- Held-out top-run summary PDF:
  - `results/linear_patch_q_sweep/figures/heldout_sweep_summary.pdf`
- Best-run training-curve PDF:
  - `results/linear_patch_q_sweep/figures/best_training_curves.pdf`

### Observations

- Best configuration:
  - `episodes=1200`
  - `alpha=0.01`
  - `epsilon_decay_fraction=0.75`
  - `max_td_error=3.0`
  - `max_weight_norm=80.0`
  - `seed=23`
- Best run performance:
  - train loose success rate = `1.0`
  - train optimal success rate = `1.0`
  - held-out loose success rate = `1.0`
  - held-out optimal success rate = `1.0`
- A second closely related configuration also reached perfect held-out generalization:
  - same settings except `episodes=1600`
- Sweep-wide pattern summary:
  - `alpha=0.01` was the strongest setting on average and produced the only perfect runs
  - `alpha=0.02` failed completely across the sweep
  - tighter stabilization worked better:
    - `max_td_error=3.0` beat `5.0`
    - `max_weight_norm=80.0` beat `120.0`
  - faster exploration decay helped:
    - `epsilon_decay_fraction=0.75` beat `0.85`
- Distribution of held-out optimal success across all `96` runs:
  - `0.0`: `65` runs
  - `0.25`: `23` runs
  - `0.5`: `6` runs
  - `1.0`: `2` runs

### Next Step

- The immediate follow-up should be validation rather than more blind tuning:
  - rerun the best configuration on a larger held-out set
  - test more random seeds
  - verify whether the perfect held-out result is robust or specific to this small evaluation set

### Sweep Visualization Follow-Up

- The original held-out sweep bar chart was too dense to read because run names were long and hyperparameter differences were hidden in the labels.
- Added a clearer table-style PDF ranking view:
  - `scripts/render_linear_patch_q_sweep_table.py`
- New readable output:
  - `results/linear_patch_q_sweep/figures/heldout_sweep_rankings_table.pdf`

## 2026-04-21

### Best Linear-Q Validation On 100 Random Held-Out Maps

- Added a dedicated validation script:
  - `scripts/run_best_linear_q_100_heldout.py`
- Retrained the best sweep configuration:
  - `episodes=1200`
  - `alpha=0.01`
  - `epsilon_decay_fraction=0.75`
  - `max_td_error=3.0`
  - `max_weight_norm=80.0`
  - `seed=23`
- Evaluated the retrained model on `100` newly generated held-out instances with seeds `610..709`.

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_best_linear_q_100 PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_best_linear_q_100_heldout.py`

### Outputs

- Validation summary JSON:
  - `results/best_linear_q_100_heldout/best_linear_q_100_heldout_summary.json`
- Training-curve PDF:
  - `results/best_linear_q_100_heldout/figures/training_curves_best_linear_q.pdf`
- Selected held-out rollout PDFs:
  - `results/best_linear_q_100_heldout/figures/best_linear_q_100_heldout/heldout_distribution_100_examples`

### Observations

- On the `100` random held-out maps, the best linear-Q configuration achieved:
  - held-out loose success rate = `0.94`
  - held-out optimal success rate = `0.90`
- In counts:
  - `94 / 100` loose successes
  - `90 / 100` optimal-path successes
  - `6 / 100` outright failures
- This confirms that the perfect `4 / 4` held-out result from the small sweep was not a fluke.
- The broader validation still shows very strong generalization.
- Among the non-optimal held-out cases:
  - some reached the goal but were not shortest-path optimal
  - some completed deliveries without finishing the full task
  - and a small number failed entirely

### Next Step

- The next meaningful follow-up is error analysis on the remaining `10` non-optimal held-out cases:
  - cluster them by map structure
  - compare against oracle shortest paths
  - identify whether failures are caused by local dead-end ambiguity, longer corridors, or specific obstacle patterns

## 2026-04-21

### Training-Distribution Expansion Test: 10 Training Maps, 100 Held-Out Maps

- Added a follow-up validation script:
  - `scripts/run_best_linear_q_train10_eval100.py`
- Kept the previously best hyperparameters fixed.
- Expanded the training distribution from `4` maps to `10` maps.
- Kept the held-out validation set at `100` random maps so the result is directly comparable with the previous `4-train / 100-heldout` experiment.

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_best_linear_q_train10_eval100 PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_best_linear_q_train10_eval100.py`

### Outputs

- Summary JSON:
  - `results/best_linear_q_train10_eval100/best_linear_q_train10_eval100_summary.json`
- Training-curve PDF:
  - `results/best_linear_q_train10_eval100/figures/training_curves_best_linear_q_train10.pdf`
- Selected held-out rollout PDFs:
  - `results/best_linear_q_train10_eval100/figures/best_linear_q_train10_eval100/heldout_distribution_100_examples`

### Observations

- With `10` training maps and the old best hyperparameters unchanged, performance dropped substantially:
  - train loose success rate = `0.50`
  - train optimal success rate = `0.40`
  - held-out loose success rate = `0.57`
  - held-out optimal success rate = `0.45`
- Compared with the earlier `4-train / 100-heldout` validation:
  - loose success fell from `0.94` to `0.57`
  - optimal success fell from `0.90` to `0.45`
- This indicates the previously best hyperparameters do not transfer automatically when the training distribution becomes broader.
- The likely interpretation is:
  - `10` training maps make the optimization problem harder
  - but the learner budget and schedule were not adjusted accordingly
  - so the fixed configuration now underfits the larger training distribution

### Next Step

- Retune the linear-Q hyperparameters specifically for the `10`-map training setup before drawing any conclusion about whether more training diversity helps or hurts.

## 2026-04-21

### Focused Hyperparameter Sweep For 10 Training Maps

- Added a dedicated focused sweep script:
  - `scripts/run_linear_patch_q_train10_sweep.py`
- Search goal:
  - recover performance after expanding the training distribution from `4` maps to `10` maps
- Focused search choices:
  - higher training budgets
  - smaller learning rates
  - fixed `max_td_error=3.0`
  - smaller weight norms
- Total runs:
  - `72`

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_linear_patch_q_train10_sweep PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_linear_patch_q_train10_sweep.py`

### Outputs

- Sweep summary JSON:
  - `results/linear_patch_q_train10_sweep/linear_patch_q_train10_sweep_summary.json`

### Observations

- Best configuration for the `10`-map training setup:
  - `episodes=2400`
  - `alpha=0.005`
  - `epsilon_decay_fraction=0.75`
  - `max_td_error=3.0`
  - `max_weight_norm=60.0`
  - `seed=23`
- Best run performance on `100` held-out maps:
  - held-out loose success rate = `0.95`
  - held-out optimal success rate = `0.93`
- This largely recovered and slightly exceeded the earlier `4-train / 100-heldout` baseline:
  - `4-train` best validation:
    - loose = `0.94`
    - optimal = `0.90`
  - `10-train` old hyperparameters:
    - loose = `0.57`
    - optimal = `0.45`
  - `10-train` retuned:
    - loose = `0.95`
    - optimal = `0.93`
- Interpretation:
  - adding more training maps did not hurt by itself
  - the previous drop came from using hyperparameters tuned for the smaller training distribution
  - once retuned, the broader training distribution is at least competitive and slightly stronger

### Next Step

- The next useful experiment is robustness, not more tuning:
  - rerun the new `10`-map best configuration on another fresh `100`-map held-out sample
  - or evaluate several seeds for the same best hyperparameters

## 2026-04-21

### First Neural-Q Baseline

- Added a lightweight neural-network Q-learning module:
  - `src/train_dqn.py`
- Implemented:
  - `DQNConfig`
  - `DQNPolicy`
  - `train_dqn_on_instances(...)`
- Used a small two-hidden-layer MLP over the existing `patch3x3_plus` vector input.
- Added an experiment script:
  - `scripts/run_dqn_patch_q_experiment.py`

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_dqn_patch_q PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_dqn_patch_q_experiment.py`

### Outputs

- Summary JSON:
  - `results/dqn_patch_q_experiment/dqn_patch_q_summary.json`
- Evaluation PDF:
  - `results/dqn_patch_q_experiment/figures/evaluation_summary.pdf`

### Observations

- The first neural-Q baseline successfully trained and generalized, but it did not beat the tuned linear baseline.
- Comparison on `10` training maps and `100` held-out maps:
  - tuned linear Q:
    - held-out loose success rate = `0.95`
    - held-out optimal success rate = `0.93`
  - first neural-Q baseline:
    - held-out loose success rate = `0.87`
    - held-out optimal success rate = `0.87`
- So the neural model is promising, but the first simple online-DQN style setup is not yet stronger than the tuned linear approximation baseline.

### Next Step

- Improve the neural-Q setup rather than abandoning it:
  - add replay
  - add a target network
  - or sweep hidden size / learning rate / training budget

## 2026-04-21

### Replay-Buffer DQN Upgrade

- Upgraded `src/train_dqn.py` from a one-step online neural-Q learner into a more standard DQN implementation.
- Added:
  - fixed-size replay buffer
  - mini-batch replay updates
  - target network
  - masked invalid-action bootstrapping
  - Double-DQN target action selection
  - replay size and gradient-update counts in per-episode logs
- Kept the existing `DQNConfig`, `DQNPolicy`, and `train_dqn_on_instances(...)` interface compatible with the previous experiment scripts.
- Added a dedicated comparison script:
  - `scripts/run_dqn_replay_patch_q_experiment.py`

### Current Neural Architecture

- Input:
  - `patch3x3_plus` vector from `encode_patch3x3_plus_vector(...)`
  - current dimension = `48`
- Network:
  - `Linear(48, hidden_dim)`
  - `ReLU`
  - `Linear(hidden_dim, hidden_dim)`
  - `ReLU`
  - `Linear(hidden_dim, 4)`
- The replay experiment used:
  - `hidden_dim=96`
  - output dimension `4`, one Q-value per movement action

### What Was Run

- Command:
  - `MPLCONFIGDIR=/tmp/mpl_dqn_replay PYTHONDONTWRITEBYTECODE=1 python scripts/run_dqn_replay_patch_q_experiment.py`

### Outputs

- Summary JSON:
  - `results/dqn_replay_patch_q_experiment/dqn_replay_patch_q_summary.json`
- Evaluation PDF:
  - `results/dqn_replay_patch_q_experiment/figures/evaluation_summary.pdf`
- DQN training curve PDF:
  - `results/dqn_replay_patch_q_experiment/figures/training_curves_patch3x3_plus_dqn_replay.pdf`
- Tuned linear-Q training curve PDF:
  - `results/dqn_replay_patch_q_experiment/figures/training_curves_patch3x3_plus_linear_tuned.pdf`
- Selected DQN held-out rollout PDFs:
  - `results/dqn_replay_patch_q_experiment/figures/patch3x3_plus_dqn_replay/heldout_distribution_100_examples`
- Selected tuned linear-Q held-out rollout PDFs:
  - `results/dqn_replay_patch_q_experiment/figures/patch3x3_plus_linear_tuned/heldout_distribution_100_examples`

### Observations

- Same setup as the current best linear baseline:
  - `10` training maps, seeds `410..419`
  - `100` held-out maps, seeds `610..709`
  - same `patch3x3_plus` vector state representation
- Tuned linear-Q baseline:
  - train loose success rate = `1.00`
  - train optimal success rate = `1.00`
  - held-out loose success rate = `0.95`
  - held-out optimal success rate = `0.93`
- Replay DQN:
  - train loose success rate = `1.00`
  - train optimal success rate = `1.00`
  - held-out loose success rate = `1.00`
  - held-out optimal success rate = `1.00`
- This is the first neural setup that clearly beats the tuned linear approximation baseline on the current held-out set.
- The main likely reason is not just the MLP itself, but the more stable DQN training recipe:
  - replay improves sample reuse
  - target network stabilizes bootstrapping
  - Double-DQN masking reduces invalid-action and overestimation artifacts

### Next Step

- Treat this as a strong first replay-DQN result, but verify robustness before making a final claim:
  - rerun on fresh `100`-map held-out seeds
  - repeat with several random training seeds
  - run a small sweep over `learning_rate`, `hidden_dim`, and `target_update_interval`

## 2026-04-21

### Experiment Report Draft

- Added a reader-facing Chinese report summarizing the full experiment trajectory:
  - `EXPERIMENT_REPORT.md`
- The report explains:
  - research goal
  - current random delivery task
  - loose versus shortest-path success metrics
  - pre-generalization background experiments
  - standalone `generalization` infrastructure
  - all major state-representation experiments
  - tabular, linear-Q, online-DQN, and replay-DQN results
  - key result paths
  - current conclusions and recommended next experiments

### Outputs

- Report:
  - `EXPERIMENT_REPORT.md`

### Observations

- The report compresses the full experiment history into a readable narrative rather than listing every sweep run.
- The main reported conclusion is:
  - tabular Q-learning mostly memorizes maps
  - function approximation is the turning point for generalization
  - tuned linear-Q is a strong baseline
  - replay-buffer DQN is currently the best method on the tested `100` held-out maps
