# Dataset and model inputs

The dataset contains 60 production plans with 6,518 tasks and 13,511 directed dependency edges. Individual plans contain 48–160 tasks and 94–330 edges. Each plan belongs to one split within a partition. The default partition is shown below.

| Split | Plans | Forecast windows |
|---|---:|---:|
| Train | 36 | 14,595 |
| Validation | 12 | 5,481 |
| Test | 12 | 5,424 |

As in Supplementary Table 1, `N` is the number of tasks, `M` the number of dependency edges, `R` the number of equipment units, `W` the number of warehouses, and `T` the number of observation frames. Frames are 30 minutes apart. At forecast origin `t`, models read the 10 frames from `t-9` through `t` and predict TG for `t+1` through `t+10`. The output has shape `[B,10,M]`, with TG in hours.

## Mapping to Supplementary Table 1

Supplementary Table 1 lists the input fields and their shapes before encoding. The released arrays store numerical channels, categorical indices, and validity flags in the format consumed by the models. In the table, `shared`, `bmstgn`, and `comparison` refer to the corresponding NPZ files.

| Input group | Fields and shapes in Supplementary Table 1 | Released representation |
|---|---|---|
| Task process and material information | `task [N,16]`, `material_required [N]`, `input_mass_g [N]` | In `bmstgn`, `craft [N,39]` contains process quantities such as the weld joint count, while `group [N]` and `material [N]` encode process and material categories. `demand [N,2]` stores the required item count and input mass. Baseline models read the corresponding channels in `comparison.static`. |
| Task planning information | `plan_times_s [N,4]`, `phase [N,4]`, `warehouse [N,2]`, `pool_share [N,2]`, `pool_index [2]`, `edge_index [2,M]`, `relation [M,2]`, `matrices [8,N,N]` | The columns of `shared.plan_h [N,5]` are planned start, planned finish, planned duration, processing time, and release time, all in hours. `bmstgn` retains `phase`, `warehouse_index`, `pool_share`, and `pool_index`. Dependencies are represented by `shared.edge_index` and `bmstgn.edge_type`. Baseline models encode the eight relation types through node features, edge features, and sparse adjacency. |
| Static manufacturing system information | `equipment [R,5]`, `warehouses [W,3]`, `global_values [5]` | `equipment_kind`, `equipment_group`, and `resource_numeric [R,6]` encode equipment categories, groups, and attributes. `warehouse_kind`, `warehouse_group`, and `warehouse_capacity [W,1]` encode warehouse categories, groups, and storage slot capacity. `site [5]` stores system-wide attributes. Baselines use `comparison.site_static`. |
| Task execution information | `stage [T,N,4,17]`, `delivered [T,N]`, `request_ms [T,N]` | The four stages are loading, processing, unloading, and receipt. `bmstgn.stage [T,N,4,62]` contains stage quantities such as completed weld joints and cumulative clamping counts, together with validity masks. `status` and `operation` contain state and operation indices. Delivered quantities and transport request times are stored in `bmstgn.logistics` and available to baselines through `comparison.local`. `shared.start_ms` and `shared.finish_ms` store task start and finish times when available and NaN otherwise. |
| Dynamic manufacturing system information | `equipment [T,R,5]`, `warehouses [T,W,3]`, `transport_counts [T,2]`, `times_s [T]` | `bmstgn.equipment [T,R,9]` describes equipment operation, blocking, queues, and transport waiting. `bmstgn.warehouse [T,W,6]` contains occupied and reserved storage slots and validity flags. `logistics` includes transport counts, and `calendar [T,3]` represents calendar position. Baselines use `comparison.common`. Frame `k` corresponds to `1800 × k` seconds. |

## TG and BT history

The `history [T,M,3]` input listed after Supplementary Table 1 contains historical TG, fixed flags, and raw TG. The first two channels are stored in `shared.observed_tg` and `shared.fixed`. The five channels of `bmstgn.edge_extra [T,M,5]` encode raw TG, candidate bottleneck task (BT) membership, confirmed BT membership, historical TG differences, and historical raw TG differences. Supplementary Table 1 lists candidate and confirmed BT by task as `[T,N]`. The released arrays encode whether each edge's predecessor is the corresponding BT. Baselines read historical edge features from `comparison.edge_dynamic`.

`edge_index[0]` identifies predecessor tasks and `edge_index[1]` identifies successor tasks. Node indices start at zero, and all models use the same edge order. Historical differences use the current and previous frames. Forecast inputs use only information available by the forecast origin.

## Files and targets

`splits.json` lists the plan IDs in the default partition. `partitions.json` lists six additional partitions and the training seed. Each plan is stored at `split/plan_id/` under its default split. The loader reads the corresponding files when a different partition is selected.

| File | Contents |
|---|---|
| `shared.npz` | Valid forecast origins (`origins`), edge indices, plan times, historical TG, and the time boundaries needed to construct the origin state. |
| `bmstgn.npz` | Task, resource, and edge features used by BMSTGN and its ablations. |
| `comparison.npz` | Shared baseline features: `static [N,183]`, `local [T,N,356]`, `edge_static [M,18]`, `edge_dynamic [T,M,14]`, `site_static [550]`, `common [T,609]`, and sparse propagation arrays `spatial_edge_index` and `spatial_edge_weight`. |
| `targets.npz` | `tg_s` contains reference TG by frame and edge in seconds. `active` marks valid positions for supervision. `bt_start`, `bt_eligible`, `final_parent`, and `freeze_at` support evaluation. `date_h` contains reference completion times by frame in hours for GCN–GRU training across partitions. |
