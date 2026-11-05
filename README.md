# BMSTGN

BMSTGN 根据任务图和最近 10 帧观测，预测未来 10 帧各有向边的时间间隔（TG）。每帧间隔 30 分钟，单次预测覆盖 5 小时。仓库包含主模型、八种对比模型和六组消融的代码、参数及数据。

## 快速运行

在仓库根目录执行。已验证的环境为 Python 3.9.19、PyTorch 2.3.1、NumPy 1.26.4。如果使用 Conda，可先运行 `conda create -n bmstgn python=3.9 -y` 和 `conda activate bmstgn`。

```powershell
python -m pip install -r requirements.txt
python run.py check
python run.py predict --case CP005 --origin 9 --device cpu
python run.py evaluate --model bmstgn --split test --output results/bmstgn.json
```

`check` 校验发布文件，并用一个验证窗口运行所有模型。预测结果形状为 `[批次, 10, 边数]`，单位为小时。终端示例以秒显示第一条边的预测。通过 `predict --output results/prediction.npz` 可保存全部边的预测及对应边索引。计算设备默认自动选择，也可指定 `--device cpu` 或 `--device cuda`。

[全部训练、评估与消融命令](COMMANDS.md) · [正文附表 1 与发布字段对照（含 CP002 示例）](data/README.md)

## 数据划分

| 集合 | 案例数 | 预测窗口数 |
|---|---:|---:|
| 训练 | 36 | 14,595 |
| 验证 | 12 | 5,481 |
| 测试 | 12 | 5,424 |

每个案例对应一个完整生产计划，合计 6,518 个任务和 13,511 条依赖边。同一计划的全部窗口位于同一集合。输入按正文附表 1 分为任务工艺与材料信息、任务计划信息、制造系统静态信息、任务执行信息、制造系统动态信息；历史 TG 与候选、确认 BT 另按边对齐。输入只取预测起点及此前可用的帧。监督和评分字段独立存放。

## 文件

| 文件 | 用途 |
|---|---|
| `bmstgn.py` | BMSTGN 与六组消融 |
| `baselines.py` | GRU、STGCN、Graph WaveNet、BSTAN、BTGCN |
| `rollout.py` | GCN–GRU + rollout 与计划递推 |
| `data.py` | 按案例和窗口读取模型输入 |
| `metrics.py` | 瓶颈任务（BT）身份与迁移评分 |
| `run.py` | 训练、预测、评估与 Persistence |
| `weights/` | 13 份用于预测和评估的模型权重，以及 GCN–GRU 的训练集归一化参数 |
| `data/` | 案例张量、目标和划分清单 |

## 评估输出

`mae_s`、`rmse_s` 为各案例指标的等权平均，单位为秒。`change_mae_s` 评价相邻参考 TG 变化超过 60 秒的有效位置，`final_tg_mae_s` 评价 TG 确定时刻。带 `micro_` 的指标按全部有效边点汇总。

`window_accuracy` 统计发生 BT 变化的有效窗口中十步身份全部正确的比例。`final_accuracy` 统计提前十步对末次变化后最终 BT 身份的命中率。`process_f1_d0`、`final_f1_d0` 分别对应过程迁移和末次迁移的同帧 F1；`d1` 允许偏差一步。准确率与 F1 以 0–1 表示，完整 JSON 保留计数。
