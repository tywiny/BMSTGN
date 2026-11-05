import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from baselines import LAYOUT


def topology(edge, nodes):
    source, target = edge
    parents = [np.unique(source[target == j]) for j in range(nodes)]
    remaining = set(range(nodes))
    levels = []
    while remaining:
        ready = [j for j in sorted(remaining) if not set(parents[j]) & remaining]
        if not ready:
            raise ValueError("Dependency graph contains a cycle")
        levels.append(ready)
        remaining.difference_update(ready)
    return parents, levels


def state(case, origins, device="cpu"):
    origins = np.asarray(origins, dtype=np.int64)
    now = origins[:, None] * 0.5
    actual_start = case.start_ms[origins] / 3600000.0
    actual_finish = case.finish_ms[origins] / 3600000.0
    started = np.isfinite(actual_start)
    done = np.isfinite(actual_finish)
    if (actual_start > now).any() or (actual_finish > now).any():
        raise ValueError("Invalid observation boundary")
    gate = np.maximum(np.maximum(now, case.release[None]), case.plan[None, :, 0])

    gate = np.where(started, now, gate)
    baseline = np.broadcast_to(case.plan[None, :, 2], (len(origins), case.n)).copy()
    baseline = np.where(
        started, np.maximum(case.plan[None, :, 2] - (now - np.nan_to_num(actual_start)), 0.0), baseline
    )
    baseline = np.where(done, 0.0, baseline)
    tensor = lambda a, dtype=torch.float32: torch.as_tensor(
        np.array(a, copy=True), dtype=dtype, device=device
    )
    result = dict(
        now=tensor(now),
        started=tensor(started, torch.bool),
        done=tensor(done, torch.bool),
        actual=tensor(np.where(done, actual_finish, 0.0)),
        base_gate=tensor(gate),
        duration=tensor(baseline),
        source=tensor(case.edge[0], torch.long),
        target=tensor(case.edge[1], torch.long),
        fixed=tensor(case.fixed[origins], torch.bool),
        last_tg=tensor(case.tg[origins]),
        support=tensor(case.support),
        levels=[],
        parents=case.parents,
    )
    for nodes in case.levels:
        width = max(1, max(len(case.parents[j]) for j in nodes))
        ix = np.zeros((len(nodes), width), np.int64)
        mask = np.zeros_like(ix, bool)
        for k, j in enumerate(nodes):
            p = case.parents[j]
            ix[k, : len(p)] = p
            mask[k, : len(p)] = True
        result["levels"].append(
            (tensor(nodes, torch.long), tensor(ix, torch.long), tensor(mask, torch.bool))
        )
    return result


def schedule(duration, s):

    if duration.shape != s["done"].shape:
        raise ValueError("Expected batch x tasks durations")
    finish = torch.zeros_like(duration)
    gates = s["base_gate"].clone()
    for nodes, parents, mask in s["levels"]:
        bound = finish[:, parents].masked_fill(~mask[None], -torch.inf).amax(-1)
        gate = torch.maximum(s["base_gate"][:, nodes], bound)
        gate = torch.where(s["started"][:, nodes], s["now"], gate)
        value = torch.where(
            s["done"][:, nodes], s["actual"][:, nodes], gate + duration[:, nodes].clamp_min(0)
        )
        finish = finish.index_copy(1, nodes, value)
        gates = gates.index_copy(1, nodes, gate)
    edge_dates = finish[:, s["source"]]
    maximum = finish.new_full(finish.shape, -torch.inf).scatter_reduce(
        1, s["target"][None].expand_as(edge_dates), edge_dates, reduce="amax", include_self=True
    )
    tg = maximum[:, s["target"]] - edge_dates
    tg = torch.where(s["fixed"], s["last_tg"], tg.clamp_min(0))
    return finish, gates, tg


def inverse(finish, s):

    values = finish[:, s["source"]]
    bound = finish.new_full(finish.shape, -torch.inf).scatter_reduce(
        1, s["target"][None].expand_as(values), values, reduce="amax", include_self=True
    )
    gate = torch.maximum(s["base_gate"], bound)
    gate = torch.where(s["started"], s["now"], gate)
    raw = finish - gate
    mask = ~s["done"]
    if bool((raw[mask] < -1e-3).any()):
        raise ValueError("Negative effective duration; target date/precedence contract inconsistent")
    return torch.where(mask, raw.clamp_min(0), torch.zeros_like(raw))


def feedback(duration, finish, gate, s, lead):
    clock = s["now"] + 0.5 * lead
    relative = finish - clock
    neighbor = torch.einsum("nm,bm->bn", s["support"], relative)
    return torch.stack(
        (
            torch.log1p(duration),
            relative.asinh(),
            (gate - clock).asinh(),
            (finish <= clock).float(),
            (gate <= clock).float(),
            neighbor.asinh(),
            (finish - s["now"]).asinh(),
            torch.full_like(duration, lead / 10.0),
        ),
        dim=-1,
    )


class RolloutInputs:
    def __init__(self, case, with_features=True):
        self.origins = case.origins
        shared = case.shared
        self.plan = shared["plan_h"][:, :4]
        self.release = shared["plan_h"][:, 4]
        self.edge = shared["edge_index"]
        self.tg = shared["observed_tg"]
        self.fixed = shared["fixed"]
        self.start_ms = shared["start_ms"]
        self.finish_ms = shared["finish_ms"]
        self.n = len(self.plan)
        self.parents, self.levels = topology(self.edge, self.n)
        self.source, self.target = self.edge
        adjacency = np.eye(self.n, dtype=np.float32)
        adjacency[self.target, self.source] = 1
        adjacency[self.source, self.target] = 1
        degree = adjacency.sum(-1)
        self.support = adjacency / np.sqrt(degree[:, None] * degree[None, :])
        if with_features:
            self._encode(case.comparison)

    def _encode(self, a):
        self.a = a
        l = LAYOUT
        off = l["association_offset"]
        end = off + l["association_width"]
        r = l["equipment_count"]
        w = l["warehouse_count"]
        self.task_static = np.concatenate((a["static"][:, :off], a["static"][:, end:]), -1)
        mapping = a["static"][:, off:end]
        self.phase = mapping[:, : 4 * r].reshape(self.n, 4, r)
        self.warehouse = mapping[:, 4 * r :].reshape(self.n, 2, w)
        es, ws, ed, wd = [
            l[k]
            for k in ("equipment_static", "warehouse_static", "equipment_dynamic", "warehouse_dynamic")
        ]
        site = a["site_static"]
        eend = r * es
        wend = eend + w * ws
        send = l["site_static_width"]
        eq_static = site[:eend].reshape(r, es)
        wh_static = site[eend:wend].reshape(w, ws)
        pools = site[wend + l["global_static"] : send].reshape(2, r)
        present = np.concatenate(
            (self.phase.sum(-1), self.warehouse.sum(-1), np.broadcast_to(pools.sum(-1), (self.n, 2))), -1
        )
        local_static = np.concatenate(
            (
                (self.phase @ eq_static).reshape(self.n, -1),
                (self.warehouse @ wh_static).reshape(self.n, -1),
                np.broadcast_to((pools @ eq_static).ravel(), (self.n, 2 * es)),
                present,
            ),
            -1,
        )
        self.node_static = np.concatenate((self.task_static, local_static), -1)
        self.eq = a["common"][:, : r * ed].reshape(-1, r, ed)
        self.wh = a["common"][:, r * ed : r * ed + w * wd].reshape(-1, w, wd)
        self.pool_dynamic = np.einsum("kr,trf->tkf", pools, self.eq, optimize=True).reshape(
            len(self.eq), -1
        )
        self.eq_id = self.phase.argmax(-1)
        self.eq_present = self.phase.any(-1)
        self.wh_id = self.warehouse.argmax(-1)
        self.wh_present = self.warehouse.any(-1)
        self.bg_static = np.concatenate(
            (
                eq_static.mean(0),
                eq_static.max(0),
                wh_static.mean(0),
                wh_static.max(0),
                site[wend : wend + l["global_static"]],
            )
        )
        self.bg_dynamic = np.concatenate(
            (
                self.eq.mean(1),
                self.eq.max(1),
                self.wh.mean(1),
                self.wh.max(1),
                a["common"][:, r * ed + w * wd :],
            ),
            -1,
        )

        statics, dynamics = [], []
        for ids in (self.target, self.source):
            static = np.zeros((self.n, 2 * a["edge_static"].shape[-1] + 1), np.float32)
            dynamic = np.zeros((len(a["local"]), self.n, 2 * a["edge_dynamic"].shape[-1]), np.float32)
            for node in range(self.n):
                edges = np.flatnonzero(ids == node)
                static[node, -1] = np.arcsinh(len(edges))
                if len(edges):
                    static[node, :-1] = np.concatenate(
                        (a["edge_static"][edges].mean(0), a["edge_static"][edges].max(0))
                    )
                    dynamic[:, node] = np.concatenate(
                        (a["edge_dynamic"][:, edges].mean(1), a["edge_dynamic"][:, edges].max(1)), -1
                    )
            statics.append(static)
            dynamics.append(dynamic)
        self.task_input_static = np.concatenate(
            (self.node_static, *statics, np.broadcast_to(self.bg_static, (self.n, len(self.bg_static)))),
            -1,
        ).astype(np.float32)
        self.incident_dynamic = np.concatenate(dynamics, -1)

    def batch(self, origins, device, norm):
        ix = np.asarray(origins)[:, None] + np.arange(-9, 1)
        a = self.a
        t, n = ix.size, self.n
        eq = self.eq[ix][:, :, self.eq_id] * self.eq_present[None, None, :, :, None]
        wh = self.wh[ix][:, :, self.wh_id] * self.wh_present[None, None, :, :, None]
        shape = (*ix.shape, n)
        x = np.concatenate(
            (
                a["local"][ix],
                eq.reshape(*shape, -1),
                wh.reshape(*shape, -1),
                self.incident_dynamic[ix],
                np.broadcast_to(self.pool_dynamic[ix][:, :, None], (*shape, self.pool_dynamic.shape[-1])),
                np.broadcast_to(self.bg_dynamic[ix][:, :, None], (*shape, self.bg_dynamic.shape[-1])),
            ),
            -1,
        )
        x = torch.as_tensor(x, device=device)
        stat = torch.as_tensor(self.task_input_static, device=device)
        return dict(
            dynamic=(x - norm["dynamic_mean"]) / norm["dynamic_scale"],
            static=(stat - norm["static_mean"]) / norm["static_scale"],
            state=state(self, origins, device),
        )


class DurationGCNGRU(nn.Module):
    def __init__(self, static_width, dynamic_width, hidden=192, dropout=0.1):
        super().__init__()
        self.spatial_input = nn.Linear(static_width + dynamic_width, 61)
        self.gcn = nn.ModuleList(nn.Linear(61, 61) for _ in range(3))
        self.temporal_input = nn.Linear(dynamic_width, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(hidden + 61 + static_width + dynamic_width, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.roll_cell = nn.GRUCell(hidden + 8, hidden)
        self.duration_head = nn.Sequential(nn.Linear(hidden, 96), nn.ReLU(), nn.Linear(96, 1))
        nn.init.zeros_(self.duration_head[-1].weight)
        nn.init.zeros_(self.duration_head[-1].bias)

    def forward(self, batch, return_trace=False):
        x = batch["dynamic"]
        static = batch["static"]
        s = batch["state"]
        b, t, n, _ = x.shape
        current = torch.cat((static[None].expand(b, -1, -1), x[:, -1]), -1)
        spatial = F.relu(self.spatial_input(current))
        for layer in self.gcn:
            spatial = F.relu(layer(torch.einsum("nm,bmf->bnf", s["support"], spatial)))
        temporal = F.relu(self.temporal_input(x)).permute(0, 2, 1, 3).reshape(b * n, t, -1)
        _, hidden = self.gru(temporal)
        temporal = hidden[-1].reshape(b, n, -1)
        context = self.fusion(torch.cat((spatial, temporal, current), -1))
        hidden = context.reshape(b * n, -1)
        duration = s["duration"]
        finish, gate, tg = schedule(duration, s)
        durations = []
        dates = []
        predictions = []
        features = []
        for lead in range(1, 11):
            prior = feedback(duration, finish, gate, s, lead - 1)
            hidden = self.roll_cell(torch.cat((context, prior), -1).reshape(b * n, -1), hidden)
            change = self.duration_head(hidden).reshape(b, n)
            duration = torch.expm1((torch.log1p(duration) + change).clamp(0, 12))
            duration = torch.where(s["done"], torch.zeros_like(duration), duration)
            finish, gate, tg = schedule(duration, s)
            durations.append(duration)
            dates.append(finish)
            predictions.append(tg)
            features.append(prior)
        result = torch.stack(predictions, 1)
        if not return_trace:
            return result
        return result, dict(
            duration=torch.stack(durations, 1),
            finish=torch.stack(dates, 1),
            feedback=torch.stack(features, 1),
        )
