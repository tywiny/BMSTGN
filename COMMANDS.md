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

## 六组额外划分

`data/partitions.json` 保存 `split_01` 至 `split_06` 的计划编号。每组使用 36 个训练计划、12 个验证计划和 12 个测试计划，直接读取现有数据文件。三个模型均从头训练，种子为 `20261022`，批次为 32，训练 40 轮。

在第 1、5、10、20、30、40 轮验证，依次按验证集过程迁移 F1（容差一步）最高、TG MAE 最低、轮次最早选择权重。GCN–GRU 在各组训练集上分别拟合归一化参数，并随权重保存。测试命令须加载对应划分的权重。

运行全部 18 次训练和测试（PowerShell）：

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

单独运行一组：

```powershell
python run.py train --model bmstgn --partition split_01 --seed 20261022 --output runs/split_01_bmstgn
python run.py evaluate --model bmstgn --partition split_01 --weights runs/split_01_bmstgn/model.pt --split test --output results/split_01_bmstgn.json
```

汇总输出六组结果的均值 `mean` 和样本标准差 `sd`。每次测试的 `confidence_intervals` 为计划级重采样得到的 95% 置信区间。训练输出目录须为新目录。

## 前序数分组

```powershell
python run.py analyze --model bmstgn --by degree --split test --output results/degree_bmstgn.json
python run.py analyze --model graph_wavenet --by degree --split test --output results/degree_graph_wavenet.json
python run.py analyze --model gcn_gru --by degree --split test --output results/degree_gcn_gru.json
```

按直接前序数 2、3、≥4 分组，输出整窗口身份准确率、最终身份准确率，以及过程和末次迁移 F1（同帧、容差一步）。

## 预测步长

```powershell
python run.py analyze --model bmstgn --by lead --split test --output results/lead_bmstgn.json
python run.py analyze --model graph_wavenet --by lead --split test --output results/lead_graph_wavenet.json
python run.py analyze --model gcn_gru --by lead --split test --output results/lead_gcn_gru.json
```

按 1–10 步（0.5–5 小时）统计 TG 误差、变化窗口内该步身份准确率、过程和末次迁移时的身份准确率，以及同帧迁移 F1。F1 中的 TP 归入匹配步、FP 归入预测步、FN 归入参考事件步。

`--by all` 一次完成两类分析。`analyze` 同样支持 `--ablation`、`--partition` 和 `--weights`。`records` 中 TG 误差单位为秒，准确率和 F1 单位为 %；`lower`、`upper` 为以完整计划重采样 20,000 次所得的 95% 置信区间。输出文件仅写入指定的本地目录。
