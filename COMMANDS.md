# 运行命令

以下命令在仓库根目录执行。`evaluate` 默认加载随包的已训练权重。`train` 按固定随机种子初始化模型，从头训练并将结果写入新目录。可用 `--seed 42` 指定其他种子。默认批次为 32，训练上限为 40 轮。

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

### -DGR-node

```powershell
python run.py evaluate --ablation=-DGR-node --split test
python run.py train --ablation=-DGR-node --output runs/no_dgr_node
```

### -Bridge

```powershell
python run.py evaluate --ablation=-Bridge --split test
python run.py train --ablation=-Bridge --output runs/no_bridge
```
