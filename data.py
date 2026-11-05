from dataclasses import dataclass, field
from pathlib import Path
from functools import cached_property
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
HISTORY = HORIZON = 10


def read_arrays(path, keys=None):
    with np.load(path, allow_pickle=False) as records:
        return {key: records[key] for key in (records.files if keys is None else keys)}


def tensor(array, device):
    return torch.as_tensor(array, device=device)


@dataclass
class Graph:
    index: torch.Tensor
    nodes: int
    up_query: torch.Tensor
    up_key: torch.Tensor
    peer_query: torch.Tensor
    peer_key: torch.Tensor
    _statistics: dict = field(default_factory=dict)

    @classmethod
    def build(cls, index, nodes, device):
        source, target = index
        incoming = [np.flatnonzero(target == n) for n in range(nodes)]

        def pairs(upstream):
            pairs = [
                (i, j)
                for i, (s, t) in enumerate(index.T)
                for j in incoming[s if upstream else t]
                if upstream or i != j
            ]
            packed = np.asarray(pairs, dtype=np.int64).reshape(-1, 2).T
            return tuple(tensor(x, device) for x in packed)

        return cls(tensor(index, device), nodes, *pairs(True), *pairs(False))


@dataclass
class Boundary:
    last_tg: torch.Tensor
    fixed: torch.Tensor
    finish_h: torch.Tensor
    finish_valid: torch.Tensor


class NumericFields(dict):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def __missing__(self, key):
        with np.load(self.path, allow_pickle=False) as data:
            value = data[key]
        self[key] = value
        return value


class Case:
    def __init__(self, root, split, case_id):
        self.folder = Path(root) / split / case_id
        self.case_id = case_id
        self.split = split
        self.origins = self.shared["origins"]
        if (
            self.origins.ndim != 1
            or self.origins.size == 0
            or self.origins[0] < HISTORY - 1
            or (np.diff(self.origins) <= 0).any()
        ):
            raise ValueError("Invalid window origins")
        self._graphs = {}

    @cached_property
    def shared(self):
        return NumericFields(self.folder / "shared.npz")

    @cached_property
    def bm(self):
        return NumericFields(self.folder / "bmstgn.npz")

    @cached_property
    def comparison(self):
        return NumericFields(self.folder / "comparison.npz")

    @cached_property
    def labels(self):
        labels = NumericFields(self.folder / "targets.npz")
        labels["tg_h"] = labels["tg_s"].astype(np.float32) / 3600.0
        return labels

    @cached_property
    def changed(self):
        y = self.labels["tg_s"]
        mask = np.zeros(y.shape, dtype=bool)
        mask[1:] = np.abs(y[1:] - y[:-1]) > 60.0
        return mask

    def windows(self, origins):
        origins = np.asarray(origins)
        if (
            origins.ndim != 1
            or origins.dtype.kind not in "iu"
            or not origins.size
            or not np.isin(origins, self.origins).all()
        ):
            raise ValueError("Expected registered integer forecast origins")
        return origins[:, None] + np.arange(1 - HISTORY, 1)

    def graph(self, device):
        if device not in self._graphs:
            self._graphs[device] = Graph.build(
                self.shared["edge_index"], self.shared["finish_ms"].shape[1], device
            )
        return self._graphs[device]

    def bm_batch(self, origins, device, variant="full"):
        ix = self.windows(origins)
        g = self.graph(device)
        s = self.shared
        a = self.bm

        def const(k):
            return tensor(a[k], device)

        def frames(k):
            return tensor(a[k][ix], device)

        batch = {"graph": g}
        b, t = ix.shape

        def repeated(x):
            return x[None, None].expand(b, t, *x.shape)

        tg = tensor(s["observed_tg"][ix], device)
        fixed = tensor(s["fixed"][ix], device)
        extras = frames("edge_extra")
        diff_valid = tensor(ix > 0, device)[:, :, None, None].expand(b, t, tg.shape[-1], 1)
        batch["edge"] = torch.cat(
            (
                tg.asinh()[..., None],
                fixed[..., None],
                extras[..., :3],
                repeated(const("edge_type")),
                extras[..., 3:],
                diff_valid,
            ),
            -1,
        )
        finish = tensor(s["finish_ms"][origins], device)
        now_ms = tensor(np.asarray(origins), device).double()[:, None] * 1800000.0
        finish_valid = finish.isfinite()
        hours = torch.where(finish_valid, (finish - now_ms.double()) / 3600000, 0.0)
        batch["boundary"] = Boundary(tg[:, -1], fixed[:, -1], hours, finish_valid)
        if variant == "no-pm":
            return batch
        batch.update(
            {
                k: const(k)
                for k in (
                    "craft",
                    "group",
                    "material",
                    "equipment_kind",
                    "equipment_group",
                    "warehouse_kind",
                    "warehouse_group",
                    "phase",
                    "warehouse_index",
                    "pool_index",
                    "pool_share",
                    "resource_numeric",
                    "warehouse_capacity",
                    "demand",
                    "site",
                )
            }
        )
        plan = tensor(s["plan_h"], device)
        batch["plan_h"] = plan
        if variant == "zero-pds":
            return batch
        for k in ("status", "operation"):
            batch[k] = frames(k).long()
        batch["stage"] = frames("stage")
        clock = tensor(ix, device).double()[:, :, None, None].expand(b, t, g.nodes, 1) * 0.5
        ph = repeated(plan)
        batch["plan"] = (
            torch.cat((ph, clock, ph[..., :2] - clock, ph[..., 4:5] - clock), -1).float().asinh()
        )
        batch["equipment"] = torch.cat((repeated(batch["resource_numeric"]), frames("equipment")), -1)
        batch["warehouse"] = torch.cat((frames("warehouse"), repeated(batch["warehouse_capacity"])), -1)
        logistics = frames("logistics")
        site = batch["site"].expand(g.nodes, -1)
        batch["logistics"] = torch.cat(
            (
                repeated(batch["demand"]),
                logistics[..., :4],
                repeated(batch["pool_share"]),
                logistics[..., 4:],
                repeated(site),
                frames("calendar")[:, :, None].expand(-1, -1, g.nodes, -1),
            ),
            -1,
        )
        return batch

    def comparison_batch(self, origins, device, method=None):
        ix = self.windows(origins)
        a = self.comparison

        def assemble(static, history):
            dyn = tensor(a[history][ix], device)
            sta = tensor(a[static], device)
            return torch.cat((sta.expand(*ix.shape, *sta.shape), dyn), -1)

        batch = dict(
            x=assemble("static", "local"),
            edge_x=assemble("edge_static", "edge_dynamic"),
            site_x=assemble("site_static", "common"),
            node_static=tensor(a["static"], device),
            edge_index=tensor(self.shared["edge_index"], device),
            last_tg=tensor(self.shared["observed_tg"][origins], device),
        )
        if method in (None, "stgcn", "graph_wavenet"):
            batch["spatial_edge_index"] = tensor(a["spatial_edge_index"], device)
            batch["spatial_edge_weight"] = tensor(a["spatial_edge_weight"], device)
        return batch

    def target(self, origins, device):
        ix = np.asarray(origins)[:, None] + np.arange(1, HORIZON + 1)
        return tensor(self.labels["tg_h"][ix], device), tensor(self.labels["active"][ix], device)


def splits(root=DATA):
    groups = json.loads((Path(root) / "splits.json").read_text(encoding="utf8"))
    ids = [i for group in groups.values() for i in group]
    if len(set(ids)) != len(ids):
        raise ValueError("Overlapping case splits")
    return groups
