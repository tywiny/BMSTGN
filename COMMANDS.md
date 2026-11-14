# 运行命令

以下命令在仓库根目录执行。`evaluate` 默认加载随包的已训练权重，输出准确率/F1（%）、TG 误差（秒）及 95% 置信区间。`train` 从头训练并将结果写入新目录；BMSTGN 和各消融的默认训练种子为 `20261022`，可用 `--seed` 指定。默认批次为 32，训练上限为 40 轮。

## BMSTGN

```powershell
python run.py list
python run.py predict --model bmstgn --split test --output results/prediction.npz
python run.py evaluate --model bmstgn --split validation --output results/bmstgn_val.json
python run.py evaluate --model bmstgn --split test --output results/bmstgn_test.json
python run.py train --model bmstgn --output runs/bmstgn
python run.py evaluate --model bmstgn --split test --weights runs/bmstgn/model.pt
```

## 对比模型

| 模型 | 命令名称 |
|---|---|
| GRU | `gru` |
| STGCN | `stgcn` |
| Graph WaveNet | `graph_wavenet` |
| BSTAN | `bstan` |
| BTGCN | `btgcn` |
| GCN–GRU + rollout | `gcn_gru` |
| Persistence | `persistence` |
| Plan–CPM | `plan_cpm` |

评估全部对比模型（PowerShell）：

```powershell
foreach ($model in 'gru','stgcn','graph_wavenet','bstan','btgcn','gcn_gru','persistence','plan_cpm') {
    python run.py evaluate --model $model --split test --output "results/$model.json"
    if ($LASTEXITCODE -ne 0) { break }
}
```

分别训练六种学习型对比模型：

```powershell
foreach ($model in 'gru','stgcn','graph_wavenet','bstan','btgcn','gcn_gru') {
    python run.py train --model $model --output "runs/$model"
    if ($LASTEXITCODE -ne 0) { break }
}
```

单独评估某次训练，例如：

```powershell
python run.py evaluate --model gcn_gru --weights runs/gcn_gru/model.pt --split test
```

## 消融

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

## 关键路径

```powershell
python run.py evaluate-paths --split test --cohort-only
python run.py evaluate-paths --model bmstgn --split test --output results/paths_bmstgn.json
python run.py evaluate-paths --model graph_wavenet --split test --output results/paths_graph_wavenet.json
python run.py evaluate-paths --model gcn_gru --split test --output results/paths_gcn_gru.json
python run.py evaluate-paths --model persistence --split test --output results/paths_persistence.json
```

`--cohort-only` 仅统计评价目标和窗口。完整评价支持同一组模型、消融及 `--weights` 参数，输出准确率、F1（%）和 95% 置信区间。
