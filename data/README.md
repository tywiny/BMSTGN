# 数据集与算法输入

数据集含 60 个生产计划，共 6,518 个任务节点、13,511 条有向依赖边。单个计划含 48–160 个任务、94–330 条边。每个计划只属于一个集合。

| 集合 | 计划数 | 预测窗口数 |
|---|---:|---:|
| 训练 | 36 | 14,595 |
| 验证 | 12 | 5,481 |
| 测试 | 12 | 5,424 |

与正文附表 1 一致，`N` 为任务数，`M` 为依赖边数，`R`、`W` 为设备数和仓库数，`T` 为输入观测帧数。每帧间隔 30 分钟。预测起点为 `t` 时，模型读取 `t−9 … t` 的 10 帧，预测 `t+1 … t+10` 的 10 帧 TG。批量输出形状为 `[B,10,M]`，单位为小时。

## 附表 1 与发布字段的对应关系

附表 1 给出业务字段及其编码前的形状。发布文件按模型读取方式存放数值通道、类别索引和有效性标志。下表中的 `shared`、`bmstgn`、`comparison` 分别指同名 NPZ 文件。

| 附表 1 输入组 | 附表字段与形状 | 发布文件中的对应表示 |
|---|---|---|
| 任务工艺与材料信息 | `task [N,16]`、`material_required [N]`、`input_mass_g [N]` | `bmstgn` 的 `craft [N,39]`、`group [N]`、`material [N]` 表示组焊口总数等工艺数值及工艺、材料类别，`demand [N,2]` 表示物料需求件数和输入质量。对比模型读取 `comparison.static` 的相应通道。 |
| 任务计划信息 | `plan_times_s [N,4]`；`phase [N,4]`、`warehouse [N,2]`；`pool_share [N,2]`、`pool_index [2]`；`edge_index [2,M]`、`relation [M,2]`、`matrices [8,N,N]` | `shared.plan_h [N,5]` 的前四列依次为计划开始、完成、历时、加工时长，第五列为释放时间，单位均为小时。`bmstgn` 保留 `phase`、`warehouse_index`、`pool_share`、`pool_index`；依赖边由 `shared.edge_index` 和 `bmstgn.edge_type` 表示。八类关系矩阵在对比模型中体现为节点关系特征、逐边关系特征及稀疏邻接，发布包不另存稠密矩阵。 |
| 制造系统静态信息 | `equipment [R,5]`、`warehouses [W,3]`、`global_values [5]` | `bmstgn` 的 `equipment_kind`、`equipment_group`、`resource_numeric [R,6]` 表示设备类别、分组和属性；`warehouse_kind`、`warehouse_group`、`warehouse_capacity [W,1]` 表示仓库类别、分组和货位容量；`site [5]` 为系统公共属性。对比模型使用 `comparison.site_static`。 |
| 任务执行信息 | `stage [T,N,4,17]`、`delivered [T,N]`、`request_ms [T,N]` | 四个阶段依次为装料、加工、卸料和接收。`bmstgn.stage [T,N,4,62]` 存放已完成组焊口数、累计夹紧次数等阶段数值及掩码，`status`、`operation` 分别保存状态和工序索引。已交付量与运输请求的时间信息位于 `bmstgn.logistics`，对比模型从 `comparison.local` 读取。已发生的任务开始和完成时间保存在 `shared.start_ms`、`shared.finish_ms`；未发生时为 NaN。 |
| 制造系统动态信息 | `equipment [T,R,5]`、`warehouses [T,W,3]`、`transport_counts [T,2]`、`times_s [T]` | `bmstgn.equipment [T,R,9]` 表示设备运行、阻塞、队列和运输等待等状态，`bmstgn.warehouse [T,W,6]` 表示货位占用、预留及相应的有效性标记，`logistics` 含运输计数，`calendar [T,3]` 表示日历位置。对比模型使用 `comparison.common`。第 `k` 帧对应 `1800 × k` 秒。 |

## 历史 TG 与 BT

附表 1 表后的 `history [T,M,3]` 包含历史 TG、固定标记和原始 TG。发布包在 `shared.observed_tg`、`shared.fixed` 中保留前两项；`bmstgn.edge_extra [T,M,5]` 依次存放原始 TG 的数值编码、候选 BT 归属、确认 BT 归属、TG 历史差分和原始 TG 历史差分。附表中候选及确认 BT 按任务保存为 `[T,N]`，这里的归属按边表示，即该边前序是否为相应 BT。对比模型从 `comparison.edge_dynamic` 读取历史边特征。

`edge_index[0]` 为前序任务，`edge_index[1]` 为后继任务。节点编号从 0 开始，各模型使用相同边序。历史差分使用该帧及其前一帧；预测只取起点及此前可用的信息。

## 文件与目标

`splits.json` 给出各集合的计划编号。每个计划的文件位于 `集合/计划编号/`。

| 文件 | 内容 |
|---|---|
| `shared.npz` | 合法窗口起点 `origins`、边索引、计划时间、历史 TG 及起点状态所需的时间边界。 |
| `bmstgn.npz` | BMSTGN 及消融模型读取的任务、资源和边特征。 |
| `comparison.npz` | 对比模型共享的节点特征 `static [N,183]`、`local [T,N,356]`，边特征 `edge_static [M,18]`、`edge_dynamic [T,M,14]`，系统特征 `site_static [550]`、`common [T,609]`，以及传播邻接 `spatial_edge_index`、`spatial_edge_weight`。 |
| `targets.npz` | `tg_s` 为逐帧逐边的目标 TG（秒），`active` 为有效监督位置。`bt_start`、`bt_eligible`、`final_parent`、`freeze_at` 用于评分。`date_h` 为逐帧参考完成时刻（小时），仅供 GCN–GRU 训练使用。 |
