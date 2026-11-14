# BMSTGN

BMSTGN uses a task graph and the latest 10 observation frames to predict time gaps (TG) on directed edges over the next 10 frames. Frames are 30 minutes apart, giving a five-hour forecast horizon. The repository includes code, weights, and data for the main model, eight baselines, and seven ablations.

## Quick start

Run the following commands from the repository root. The code was tested with Python 3.9.19, PyTorch 2.3.1, and NumPy 1.26.4. If using Conda, run `conda create -n bmstgn python=3.9 -y` and `conda activate bmstgn` first.

```powershell
python -m pip install -r requirements.txt
python run.py check
python run.py predict --device cpu
python run.py evaluate --model bmstgn --split test --output results/bmstgn.json
```

`check` verifies the release files and runs all models on one validation window. Predictions have shape `[batch, 10, edges]` and are expressed in hours. The terminal displays predictions for the first edge in seconds. Use `predict --output results/prediction.npz` to save predictions for all edges together with their indices. The device is selected automatically by default. Use `--device cpu` or `--device cuda` to specify it.

[Training, evaluation, and ablation commands](COMMANDS.md) · [Dataset fields and their mapping to Supplementary Table 1](data/README.md)

## Data splits

The table shows the default partition. The six additional partitions in `data/partitions.json` each assign 36 plans to training, 12 to validation, and 12 to testing. All partitions use the same case files.

| Split | Cases | Forecast windows |
|---|---:|---:|
| Train | 36 | 14,595 |
| Validation | 12 | 5,481 |
| Test | 12 | 5,424 |

Each case represents a complete production plan. Together, the plans contain 6,518 tasks and 13,511 dependency edges. All windows from a plan belong to the same split within a partition. Supplementary Table 1 groups the inputs into task process and material information, planning information, static manufacturing system information, task execution information, and dynamic manufacturing system information. Historical TG and candidate and confirmed bottleneck task (BT) membership are aligned to edges. Inputs use only frames available by the forecast origin. Supervision and scoring fields are stored separately.

## Repository contents

| File | Purpose |
|---|---|
| `bmstgn.py` | BMSTGN and its seven ablations |
| `baselines.py` | GRU, STGCN, Graph WaveNet, BSTAN, and BTGCN |
| `rollout.py` | GCN–GRU with rollout and recursive plan scheduling |
| `data.py` | Loads model inputs by case and forecast window |
| `metrics.py` | BT identity and transition metrics with 95% confidence intervals |
| `path_metrics.py` | Critical path identity and transition metrics with 95% confidence intervals |
| `analysis.py` | Analysis by predecessor count and forecast lead, plus summaries across partitions |
| `run.py` | Training, prediction, evaluation, and the Persistence baseline |
| `weights/` | 14 sets of model weights for prediction and evaluation, plus training-set normalization parameters for GCN–GRU |
| `data/` | Case tensors, targets, and split manifests |

Use `train --partition split_01` through `split_06` to train BMSTGN, Graph WaveNet, and GCN–GRU on the additional partitions. Training uses seed `20261022`, and weights are selected on the validation split. GCN–GRU fits its normalization parameters on the training split of each partition. See the [training and evaluation commands](COMMANDS.md#six-additional-partitions).

## Evaluation outputs

`mae_s` and `rmse_s` average the case-level metrics equally and are reported in seconds. `change_mae_s` evaluates valid points where the reference TG changes by more than 60 seconds between adjacent frames. `final_tg_mae_s` evaluates TG at the frame when it becomes fixed. Metrics prefixed with `micro_` pool all valid edge-level observations.

`window_accuracy` is the percentage of eligible windows with a BT change in which the predicted identities are correct at all ten steps. `final_accuracy` is the percentage of correct predictions of the final BT identity, issued ten steps before the last change. `process_f1_d0` and `final_f1_d0` apply exact frame matching to process transitions and the final transition, respectively. The `d1` metrics allow a one-step difference. Accuracy, precision, recall, and F1 are reported as percentages. The full JSON output includes counts.

`confidence_intervals` contains the point estimate (`estimate`), the lower and upper 95% bounds (`lower` and `upper`), and the unit (`unit`). With model weights fixed, intervals use the 2.5th and 97.5th percentiles from 20,000 bootstrap resamples of complete plans. All models use the same resamples. TG metrics give each sampled plan equal weight. Accuracy and F1 are computed from the pooled counts in each resample.

## Critical path evaluation

`python run.py evaluate-paths --split test` evaluates nonterminal tasks whose final paths contain at least 20 tasks and whose reference paths changed. Paths are compared as complete ordered sequences of nodes.

`global_change_accuracy` measures the percentage of target paths predicted correctly at all ten steps in windows with a BT change anywhere in the graph. `path_change_accuracy` restricts this evaluation to target paths that changed. Both measures require complete reference paths at all ten future steps. `final_path_accuracy` evaluates the final path identity from a forecast issued ten steps before the last path transition. `final_path_f1_d0` and `final_path_f1_d1` score the last path transition in complete windows with exact frame matching and one-step tolerance, respectively. After the last reference transition, the reference path must equal the final path and remain stable. The complete paths on both sides of a predicted transition must match.

`cohort` reports counts of target tasks and target-window pairs. `by_case` reports counts by plan. Accuracy and F1 are percentages. The 95% confidence intervals use bootstrap resampling of complete plans.

## Analysis by predecessor count and forecast lead

`python run.py analyze --model bmstgn --by all --split test --output results/analysis_bmstgn.json` runs both analyses. Direct predecessor counts are grouped into 2, 3, and at least 4. Forecast leads range from 1 to 10 steps at 30 minutes per step. The analyses use the same identity and transition metrics and report point estimates, 95% confidence intervals, and counts by plan. TG errors give each plan equal weight. Accuracy and F1 use pooled counts.
