# Commands

Run these commands from the repository root. By default, `evaluate` loads the included trained weights and reports accuracy and F1 in percent, TG errors in seconds, and 95% confidence intervals. `train` starts from scratch and writes its outputs to a new directory. The default training seed for BMSTGN and its ablations is `20261022`. Override it with `--seed`. The default batch size is 32, and the training limit is 40 epochs.

## BMSTGN

```powershell
python run.py list
python run.py predict --model bmstgn --split test --output results/prediction.npz
python run.py evaluate --model bmstgn --split validation --output results/bmstgn_val.json
python run.py evaluate --model bmstgn --split test --output results/bmstgn_test.json
python run.py train --model bmstgn --output runs/bmstgn
python run.py evaluate --model bmstgn --split test --weights runs/bmstgn/model.pt
```

## Baselines

| Model | Command name |
|---|---|
| GRU | `gru` |
| STGCN | `stgcn` |
| Graph WaveNet | `graph_wavenet` |
| BSTAN | `bstan` |
| BTGCN | `btgcn` |
| GCN–GRU + rollout | `gcn_gru` |
| Persistence | `persistence` |
| Plan–CPM | `plan_cpm` |

Evaluate all baselines in PowerShell.

```powershell
foreach ($model in 'gru','stgcn','graph_wavenet','bstan','btgcn','gcn_gru','persistence','plan_cpm') {
    python run.py evaluate --model $model --split test --output "results/$model.json"
    if ($LASTEXITCODE -ne 0) { break }
}
```

Train the six learned baselines separately.

```powershell
foreach ($model in 'gru','stgcn','graph_wavenet','bstan','btgcn','gcn_gru') {
    python run.py train --model $model --output "runs/$model"
    if ($LASTEXITCODE -ne 0) { break }
}
```

To evaluate the weights from a specific training run, use `--weights`.

```powershell
python run.py evaluate --model gcn_gru --weights runs/gcn_gru/model.pt --split test
```

## Ablations

### -PM

```powershell
python run.py evaluate --ablation=-PM --split test
python run.py train --ablation=-PM --output runs/no_pm
```

### -DP

```powershell
python run.py evaluate --ablation=-DP --split test
python run.py train --ablation=-DP --output runs/no_dp
```

### BSF-GCN

```powershell
python run.py evaluate --ablation=BSF-GCN --split test
python run.py train --ablation=BSF-GCN --output runs/bsf_gcn
```

### -PDS

```powershell
python run.py evaluate --ablation=-PDS --split test
python run.py train --ablation=-PDS --output runs/no_pds
```

### -DGR

```powershell
python run.py evaluate --ablation=-DGR --split test
python run.py train --ablation=-DGR --output runs/no_dgr
```

### -Bridge

```powershell
python run.py evaluate --ablation=-Bridge --split test
python run.py train --ablation=-Bridge --output runs/no_bridge
```

### -CM

```powershell
python run.py evaluate --ablation=-CM --split validation --output results/cm_val.json
python run.py evaluate --ablation=-CM --split test --output results/cm_test.json
python run.py train --ablation=-CM --seed 20261022 --output runs/no_cm
```

## Critical paths

```powershell
python run.py evaluate-paths --split test --cohort-only
python run.py evaluate-paths --model bmstgn --split test --output results/paths_bmstgn.json
python run.py evaluate-paths --model graph_wavenet --split test --output results/paths_graph_wavenet.json
python run.py evaluate-paths --model gcn_gru --split test --output results/paths_gcn_gru.json
python run.py evaluate-paths --model persistence --split test --output results/paths_persistence.json
```

`--cohort-only` counts evaluation targets and windows. A full evaluation accepts the same model, ablation, and `--weights` arguments and reports accuracy and F1 in percent with 95% confidence intervals.

## Six additional partitions

`data/partitions.json` lists the plans for `split_01` through `split_06`. Each partition reads the existing case files and uses 36 training plans, 12 validation plans, and 12 test plans. Each of the three models is trained from scratch for 40 epochs with seed `20261022` and batch size 32.

Validation runs at epochs 1, 5, 10, 20, 30, and 40. Weights are selected by the highest validation F1 for process transitions with one-step tolerance, then the lowest TG MAE, then the earliest epoch. GCN–GRU fits normalization parameters on the training split of each partition and saves them with the weights. For testing, load the weights from the corresponding partition.

Run all 18 training and test runs in PowerShell.

```powershell
foreach ($partition in 'split_01','split_02','split_03','split_04','split_05','split_06') {
    foreach ($model in 'bmstgn','graph_wavenet','gcn_gru') {
        $run = "runs/cross_split/$partition/$model"
        python run.py train --model $model --partition $partition --seed 20261022 --batch-size 32 --epochs 40 --output $run
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $partition / $model" }
        python run.py evaluate --model $model --partition $partition --split test --weights "$run/model.pt" --output "$run/test.json"
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $partition / $model" }
    }
}
python run.py summarize-splits --input runs/cross_split --output results/cross_split_summary.json
```

To run a single partition, use the following commands.

```powershell
python run.py train --model bmstgn --partition split_01 --seed 20261022 --output runs/split_01_bmstgn
python run.py evaluate --model bmstgn --partition split_01 --weights runs/split_01_bmstgn/model.pt --split test --output results/split_01_bmstgn.json
```

The summary reports the mean (`mean`) and sample standard deviation (`sd`) across the six partitions. In each test report, `confidence_intervals` contains 95% confidence intervals from resampling complete plans. Specify a new directory for each training run.

## Analysis by predecessor count

```powershell
python run.py analyze --model bmstgn --by degree --split test --output results/degree_bmstgn.json
python run.py analyze --model graph_wavenet --by degree --split test --output results/degree_graph_wavenet.json
python run.py analyze --model gcn_gru --by degree --split test --output results/degree_gcn_gru.json
```

Group tasks by direct predecessor count (2, 3, or at least 4). The report includes whole-window identity accuracy, final identity accuracy, and F1 for process and final transitions with exact frame matching and one-step tolerance.

## Analysis by forecast lead

```powershell
python run.py analyze --model bmstgn --by lead --split test --output results/lead_bmstgn.json
python run.py analyze --model graph_wavenet --by lead --split test --output results/lead_graph_wavenet.json
python run.py analyze --model gcn_gru --by lead --split test --output results/lead_gcn_gru.json
```

For each lead from 1 to 10 steps (0.5 to 5 hours), the report includes TG errors, identity accuracy at that step within windows with changes, identity accuracy at process and final transitions, and F1 for transitions matched on the same frame. True positives are assigned to the matched step, false positives to the predicted step, and false negatives to the reference event step.

Use `--by all` to run both analyses together. `analyze` also accepts `--ablation`, `--partition`, and `--weights`. In `records`, TG errors are in seconds and accuracy and F1 are in percent. The `lower` and `upper` fields give the 95% confidence interval bounds from 20,000 bootstrap resamples of complete plans. Outputs are written to the specified local directory.
