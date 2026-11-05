import math
from numbers import Integral
import torch
from torch import nn
from torch.nn import functional as F

LAYOUT = {
    "equipment_count": 20,
    "warehouse_count": 10,
    "equipment_static": 18,
    "warehouse_static": 14,
    "global_static": 10,
    "equipment_dynamic": 24,
    "warehouse_dynamic": 12,
    "association_offset": 59,
    "association_width": 100,
    "node_width": 539,
    "edge_width": 32,
    "site_static_width": 550,
    "common_width": 609,
}

PROFILES = {
    "gru": {"native_args": {"hidden_dim": 128, "num_layers": 2, "dropout": 0.1}, "parameters": 1177098},
    "graph_wavenet": {
        "native_args": {
            "hidden_dim": 112,
            "adaptive_dim": 24,
            "num_blocks": 3,
            "layers_per_block": 2,
            "dropout": 0.1,
        },
        "parameters": 1129098,
    },
    "stgcn": {
        "native_args": {"hidden_dim": 160, "num_blocks": 2, "chebyshev_order": 3, "dropout": 0.1},
        "parameters": 1198698,
    },
    "bstan": {"native_args": {"hidden_dim": 224, "heads": 3, "gat_layers": 2}, "parameters": 1110730},
    "btgcn": {"native_args": {"hidden_dim": 256}, "parameters": 1151307},
}


def require_positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_model_dimensions(input_dim, hidden_dim, horizon, edge_input_dim, target_scale):
    for name, value in (
        ("input_dim", input_dim),
        ("hidden_dim", hidden_dim),
        ("horizon", horizon),
        ("edge_input_dim", edge_input_dim),
    ):
        if value is None:
            raise ValueError(f"{name} must be supplied explicitly")
        require_positive_integer(name, value)
    if not math.isfinite(float(target_scale)) or float(target_scale) <= 0:
        raise ValueError("target_scale must be finite and positive")


def checked_edges(edge_index, num_nodes, device, edge_weight=None, dtype=None):

    require_positive_integer("num_nodes", num_nodes)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("edge_index must contain integer indices")
    indices = edge_index.to(device=device, dtype=torch.long)
    if indices.numel() and (indices.min() < 0 or indices.max() >= num_nodes):
        raise ValueError("edge_index contains a node outside [0,num_nodes)")
    weights = None
    if edge_weight is not None:
        if edge_weight.ndim != 1 or edge_weight.shape[0] != indices.shape[1]:
            raise ValueError("graph_edge_weight must have shape [E]")
        weights = edge_weight.to(device=device, dtype=dtype)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("graph_edge_weight must be finite and nonnegative")
    return indices, weights


def edge_to_nodes(values: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:

    result = []
    for indices in edge_index:
        out = values.new_zeros(*values.shape[:-2], num_nodes, values.shape[-1])
        out.index_add_(-2, indices, values)
        degree = torch.bincount(indices, minlength=num_nodes).to(values).clamp_min(1)
        shape = [1] * (values.ndim - 2) + [num_nodes, 1]
        result.append(out / degree.view(shape))
    return torch.cat(result, dim=-1)


def random_walk_supports(edge_index: torch.Tensor, num_nodes: int, like: torch.Tensor, edge_weight=None):

    adjacency = like.new_zeros(num_nodes, num_nodes)
    adjacency[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight.to(like)
    floor = 1 if edge_weight is None else 1e-12
    return tuple(a / a.sum(-1, keepdim=True).clamp_min(floor) for a in (adjacency, adjacency.T))


def scaled_symmetric_laplacian(
    edge_index: torch.Tensor, num_nodes: int, like: torch.Tensor, edge_weight=None
):

    adjacency = like.new_zeros(num_nodes, num_nodes)
    adjacency[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight.to(like)
    adjacency = torch.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(-1).clamp_min(1 if edge_weight is None else 1e-12).rsqrt()
    identity = torch.eye(num_nodes, device=like.device, dtype=like.dtype)
    laplacian = identity - degree[:, None] * adjacency * degree[None, :]

    largest = torch.linalg.eigvalsh(laplacian).amax().clamp_min(1e-6)
    return 2 * laplacian / largest - identity


def resource_tables(batch, l):
    site = batch["site_x"]
    b, t = site.shape[:2]
    r, w = l["equipment_count"], l["warehouse_count"]
    es, ws, ed, wd = (
        l[k] for k in ("equipment_static", "warehouse_static", "equipment_dynamic", "warehouse_dynamic")
    )
    eend = r * es
    wend = eend + w * ws
    send = l["site_static_width"]
    common = site[..., send:]
    eq = torch.cat(
        (site[..., :eend].reshape(b, t, r, es), common[..., : r * ed].reshape(b, t, r, ed)), -1
    )
    wh = torch.cat(
        (
            site[..., eend:wend].reshape(b, t, w, ws),
            common[..., r * ed : r * ed + w * wd].reshape(b, t, w, wd),
        ),
        -1,
    )
    offset = l["association_offset"]
    mapping = batch["node_static"][:, offset : offset + l["association_width"]]
    phase = mapping[:, : 4 * r].reshape(-1, 4, r)
    warehouse = mapping[:, 4 * r :].reshape(-1, 2, w)
    pools = site[0, 0, wend + l["global_static"] : send].reshape(2, r)
    rest = torch.cat((site[..., wend : wend + l["global_static"]], common[..., r * ed + w * wd :]), -1)
    return eq, wh, phase, warehouse, pools, rest


def select_resources(eq, wh, phase, warehouse, pools):
    b, t = eq.shape[:2]
    n = len(phase)
    a = torch.einsum("nkr,btrc->btnkc", phase.to(eq), eq)
    c = torch.einsum("nkw,btwc->btnkc", warehouse.to(wh), wh)
    p = torch.einsum("kr,btrc->btkc", pools.to(eq), eq)[:, :, None].expand(-1, -1, n, -1, -1)
    present = torch.cat((phase.sum(-1), warehouse.sum(-1), pools.sum(-1)[None].expand(n, -1)), -1)
    local = torch.cat(
        (a.flatten(-2), c.flatten(-2), p.flatten(-2), present[None, None].expand(b, t, -1, -1)), -1
    )
    return local, dict(equipment=eq, warehouses=wh, phase=a, warehouse=c, pools=p)


def task_without_associations(batch, l):
    offset = l["association_offset"]
    end = offset + l["association_width"]
    return torch.cat((batch["x"][..., :offset], batch["x"][..., end:]), -1)


class ResourceGraphInput(nn.Module):

    def __init__(self, layout, hidden=112, resource_dim=32, pairs=False):
        super().__init__()
        self.layout = dict(layout)
        self.pairs = pairs
        l = layout
        self.equipment = nn.Sequential(
            nn.Linear(l["equipment_static"] + l["equipment_dynamic"], resource_dim),
            nn.GELU(),
            nn.LayerNorm(resource_dim),
        )
        self.warehouse = nn.Sequential(
            nn.Linear(l["warehouse_static"] + l["warehouse_dynamic"], resource_dim),
            nn.GELU(),
            nn.LayerNorm(resource_dim),
        )
        self.node = nn.Linear(l["node_width"] - l["association_width"], hidden)
        self.local = nn.Linear(8 * resource_dim + 8, hidden)
        tail = (
            l["common_width"]
            - l["equipment_count"] * l["equipment_dynamic"]
            - l["warehouse_count"] * l["warehouse_dynamic"]
        )
        self.background = nn.Linear(4 * resource_dim + l["global_static"] + tail, hidden)
        self.edge = nn.Sequential(nn.Linear(l["edge_width"], hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.incident = nn.Linear(2 * hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def route(self, batch):
        eq, wh, phase, warehouse, pools, rest = resource_tables(batch, self.layout)
        eq = self.equipment(eq)
        wh = self.warehouse(wh)
        local, details = select_resources(eq, wh, phase, warehouse, pools)
        background = torch.cat((eq.mean(-2), eq.amax(-2), wh.mean(-2), wh.amax(-2), rest), -1)
        return local, background, details

    def encode(self, batch):
        local, background, _ = self.route(batch)
        task = task_without_associations(batch, self.layout)
        edge = self.edge(batch["edge_x"])
        nodes = self.node(task) + self.local(local) + self.background(background)[:, :, None]
        nodes = nodes + self.incident(edge_to_nodes(edge, batch["edge_index"], nodes.shape[2]))
        return self.norm(nodes), edge

    def forward(self, batch):
        node, edge = self.encode(batch)
        return (node, edge) if self.pairs else node


def flat_parts(batch, l):

    task = task_without_associations(batch, l)
    eq, wh, phase, warehouse, pools, rest = resource_tables(batch, l)
    local, _ = select_resources(eq, wh, phase, warehouse, pools)
    background = torch.cat((eq.mean(-2), eq.amax(-2), wh.mean(-2), wh.amax(-2), rest), -1)
    node = torch.cat((task, local), -1)
    src, dst = batch["edge_index"]
    e = len(src)
    expand = lambda x: x[:, :, None].expand(-1, -1, e, -1)
    return (
        node[:, :, src],
        node[:, :, dst],
        batch["edge_x"],
        expand(background),
        expand(task.mean(2)),
        expand(task.amax(2)),
        expand(batch["edge_x"].mean(2)),
        expand(batch["edge_x"].amax(2)),
    )


def sequence_width(l):
    e = l["equipment_static"] + l["equipment_dynamic"]
    w = l["warehouse_static"] + l["warehouse_dynamic"]
    task = l["node_width"] - l["association_width"]
    local = 6 * e + 2 * w + 8
    rest = (
        l["global_static"]
        + l["common_width"]
        - l["equipment_count"] * l["equipment_dynamic"]
        - l["warehouse_count"] * l["warehouse_dynamic"]
    )
    background = 2 * e + 2 * w + rest
    return 2 * (task + local) + background + 2 * task + 3 * l["edge_width"]


class FactEdgeReadout(nn.Module):

    def __init__(self, hidden_dim, horizon, target_scale=1.0):
        super().__init__()
        self.horizon, self.target_scale = horizon, target_scale
        self.network = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon)
        )

    def forward(self, nodes, batch, edge_state=None):
        if edge_state is None:
            raise ValueError("The native readout requires the already encoded edge state")
        src, dst = batch["edge_index"]
        edge = edge_state[:, -1]
        if nodes.ndim == 4:
            edge = edge[:, None].expand(-1, nodes.shape[1], -1, -1)
        delta = (
            self.network(torch.cat((nodes[..., src, :], nodes[..., dst, :], edge), -1))
            * self.target_scale
        )
        if nodes.ndim == 4:
            return batch["last_tg"][:, None] + delta.squeeze(-1)
        return batch["last_tg"][:, None] + delta.transpose(1, 2)


class FlatGRU(nn.Module):
    def __init__(self, layout, hidden, num_layers=2, dropout=0.1):
        super().__init__()
        self.layout = layout
        self.encoder = nn.GRU(
            sequence_width(layout),
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, batch):
        history = torch.cat(flat_parts(batch, self.layout), -1)
        b, t, e, f = history.shape
        _, state = self.encoder(history.permute(0, 2, 1, 3).reshape(b * e, t, f))
        return batch["last_tg"][:, None] + self.readout(state[-1]).reshape(b, e, 10).transpose(1, 2)


class TemporalGLU(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("temporal kernel_size must be positive")
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(channels, 2 * channels, (1, kernel_size))

    def forward(self, x):

        value, gate = self.conv(F.pad(x, (self.kernel_size - 1, 0, 0, 0))).chunk(2, dim=1)
        return (value + x) * gate.sigmoid()


class ChebyshevConv(nn.Module):
    def __init__(self, channels, order=3):
        super().__init__()
        if not isinstance(order, int) or order < 1:
            raise ValueError("Chebyshev order counts basis terms and must be a positive integer")
        self.order = order
        self.project = nn.Conv2d(channels * order, channels, 1)

    def forward(self, x, support):
        terms = [x]
        if self.order > 1:
            terms.append(torch.einsum("nm,bcmt->bcnt", support, x))
        for _ in range(2, self.order):
            terms.append(2 * torch.einsum("nm,bcmt->bcnt", support, terms[-1]) - terms[-2])
        return self.project(torch.cat(terms, dim=1))


class STConvBlock(nn.Module):
    def __init__(self, channels, order=3, dropout=0.1):
        super().__init__()
        self.first = TemporalGLU(channels)
        self.graph = ChebyshevConv(channels, order)
        self.second = TemporalGLU(channels)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, support):
        x = self.first(x)
        x = F.relu(self.graph(x, support) + x)
        x = self.second(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class STGCNForecast(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=96,
        horizon=10,
        edge_input_dim=None,
        num_blocks=2,
        chebyshev_order=3,
        dropout=0.1,
        target_scale=1.0,
        graph_channels=9,
        graph_dim=24,
        **kwargs,
    ):
        super().__init__()
        if hidden_dim < 1 or horizon < 1 or num_blocks < 1:
            raise ValueError("hidden_dim, horizon, and num_blocks must be positive")
        if not math.isfinite(float(target_scale)) or float(target_scale) <= 0:
            raise ValueError("target_scale must be finite and positive")
        self.input = ResourceGraphInput(LAYOUT, hidden_dim)
        self.blocks = nn.ModuleList(
            STConvBlock(hidden_dim, chebyshev_order, dropout) for _ in range(num_blocks)
        )
        self.final_temporal = TemporalGLU(hidden_dim)
        self.readout = FactEdgeReadout(hidden_dim, horizon)
        self.temporal_receptive_field = 1 + (2 * num_blocks + 1) * 2
        self._support_cache = None

    def _scaled_support(self, edge_index, nodes, like, edge_weight=None):

        if edge_weight is not None:
            return scaled_symmetric_laplacian(edge_index, nodes, like, edge_weight).detach()
        key = (nodes, like.device, like.dtype)
        cached = self._support_cache
        if cached is not None and cached[0] == key and torch.equal(cached[1], edge_index):
            return cached[2]
        support = scaled_symmetric_laplacian(edge_index, nodes, like).detach()
        self._support_cache = (key, edge_index.detach().clone(), support)
        return support

    def forward(self, batch):
        encoded, edge_state = self.input.encode(batch)
        x = encoded.permute(0, 3, 2, 1)
        support = self._scaled_support(
            batch.get("spatial_edge_index", batch["edge_index"]),
            x.shape[2],
            x,
            batch.get("spatial_edge_weight"),
        )
        for block in self.blocks:
            x = block(x, support)
        node_state = self.final_temporal(x)[..., -1].transpose(1, 2)
        return self.readout(node_state, batch, edge_state)


class GraphConv(nn.Module):
    def __init__(self, channels, order=2, dropout=0.1):
        super().__init__()
        if not isinstance(order, int) or order < 0:
            raise ValueError("graph diffusion order must be a nonnegative integer")
        self.order = order
        self.project = nn.Conv2d(channels * (1 + 3 * order), channels, 1)
        self.dropout = dropout

    def forward(self, x, supports):
        terms = [x]
        for support in supports:
            propagated = x
            for _ in range(self.order):
                if support.ndim == 2:
                    propagated = torch.einsum("mn,bcmt->bcnt", support, propagated)
                else:
                    propagated = torch.einsum("bmn,bcmt->bcnt", support, propagated)
                terms.append(propagated)
        return F.dropout(self.project(torch.cat(terms, dim=1)), p=self.dropout, training=self.training)


class WaveNetLayer(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.dilation = dilation
        self.filter = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation))
        self.gate = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation))
        self.skip = nn.Conv2d(channels, channels, 1)
        self.graph = GraphConv(channels, dropout=dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, supports):
        padded = F.pad(x, (self.dilation, 0, 0, 0))
        gated = self.filter(padded).tanh() * self.gate(padded).sigmoid()
        skip = self.skip(gated)
        x = self.graph(gated, supports) + x
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x, skip


class GraphWaveNetForecast(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=88,
        horizon=10,
        edge_input_dim=None,
        adaptive_dim=24,
        num_blocks=3,
        layers_per_block=2,
        dropout=0.1,
        target_scale=1.0,
        graph_channels=9,
        graph_dim=24,
        **kwargs,
    ):
        super().__init__()
        if hidden_dim < 1 or horizon < 1 or adaptive_dim < 1 or num_blocks < 1 or layers_per_block < 1:
            raise ValueError(
                "hidden_dim, horizon, adaptive_dim, num_blocks, and layers_per_block must be positive"
            )
        if not math.isfinite(float(target_scale)) or float(target_scale) <= 0:
            raise ValueError("target_scale must be finite and positive")
        self.input = ResourceGraphInput(LAYOUT, hidden_dim)

        self.source_embedding = nn.Linear(hidden_dim, adaptive_dim, bias=False)
        self.target_embedding = nn.Linear(hidden_dim, adaptive_dim, bias=False)
        self.adaptive_dim = adaptive_dim
        self.layers = nn.ModuleList(
            WaveNetLayer(hidden_dim, 2**layer, dropout)
            for _ in range(num_blocks)
            for layer in range(layers_per_block)
        )
        self.temporal_receptive_field = 1 + num_blocks * sum(
            2**layer for layer in range(layers_per_block)
        )
        self.end = nn.Sequential(
            nn.Conv2d(hidden_dim, 2 * hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(2 * hidden_dim, hidden_dim, 1),
            nn.ReLU(),
        )
        self.readout = FactEdgeReadout(hidden_dim, horizon)

    def forward(self, batch):
        encoded, edge_state = self.input.encode(batch)

        observed = encoded.mean(dim=1)
        source = self.source_embedding(observed)
        target = self.target_embedding(observed)
        adaptive = torch.softmax(
            F.relu(source @ target.transpose(1, 2)) / math.sqrt(self.adaptive_dim), dim=-1
        )
        supports = (
            *random_walk_supports(
                batch.get("spatial_edge_index", batch["edge_index"]),
                encoded.shape[2],
                encoded,
                batch.get("spatial_edge_weight"),
            ),
            adaptive,
        )
        x = encoded.permute(0, 3, 2, 1)
        skips = []
        for layer in self.layers:
            x, skip = layer(x, supports)
            skips.append(skip)

        node_state = self.end(F.relu(torch.stack(skips).sum(0) + x))[..., -1].transpose(1, 2)
        return self.readout(node_state, batch, edge_state)


def _support(edge_index, num_nodes, dtype, device, edge_weight=None):

    indices, edge_weight = checked_edges(edge_index, num_nodes, device, edge_weight, dtype)
    src, dst = indices

    nonself = src != dst
    src, dst = src[nonself], dst[nonself]
    loops = torch.arange(num_nodes, device=device)
    flat = torch.cat((src * num_nodes + dst, dst * num_nodes + src, loops * num_nodes + loops))
    base = (
        torch.ones(src.numel(), dtype=dtype, device=device)
        if edge_weight is None
        else edge_weight[nonself]
    )
    values = torch.cat((base, base, torch.ones(num_nodes, dtype=dtype, device=device)))
    unique, inverse = flat.unique(sorted=True, return_inverse=True)
    weights = torch.zeros(unique.numel(), dtype=dtype, device=device)
    weights.scatter_reduce_(0, inverse, values, reduce="amax", include_self=True)
    return unique.div(num_nodes, rounding_mode="floor"), unique.remainder(num_nodes), weights


class WeightedMultiHeadGAT(nn.Module):

    def __init__(self, hidden_dim, heads=3):
        super().__init__()
        require_positive_integer("hidden_dim", hidden_dim)
        require_positive_integer("heads", heads)
        self.hidden_dim, self.heads = hidden_dim, heads
        self.projection = nn.Linear(hidden_dim, heads * hidden_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, hidden_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, hidden_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x, support):

        src, dst, weight = support
        k, n, _ = x.shape
        z = self.projection(x).reshape(k, n, self.heads, self.hidden_dim)

        source_score = (z * self.attn_src).sum(-1)
        target_score = (z * self.attn_dst).sum(-1)
        logits = F.leaky_relu(source_score[:, src] + target_score[:, dst], negative_slope=0.2)
        grouped_index = dst[None, :, None].expand(k, -1, self.heads)
        maximum = logits.new_full((k, n, self.heads), -torch.inf)
        maximum.scatter_reduce_(1, grouped_index, logits, reduce="amax", include_self=True)
        unnormalized = (logits - maximum[:, dst]).exp()
        denominator = torch.zeros_like(maximum)
        denominator.scatter_add_(1, grouped_index, unnormalized)
        attention = unnormalized / denominator[:, dst].clamp_min(1e-12)
        weighted = attention * weight[None, :, None]
        matrix = weighted.new_zeros(k, self.heads, n, n)
        matrix[:, :, dst, src] = weighted.permute(0, 2, 1)
        return torch.matmul(matrix, z.permute(0, 2, 1, 3)).mean(dim=1)


class BSTANCell(nn.Module):

    def __init__(self, hidden_dim, xavier_initial=False):
        super().__init__()
        require_positive_integer("hidden_dim", hidden_dim)
        self.update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.reset = nn.Linear(2 * hidden_dim, hidden_dim)
        self.candidate = nn.Linear(2 * hidden_dim, hidden_dim)
        initial = torch.zeros(1, hidden_dim)
        if xavier_initial:
            nn.init.xavier_uniform_(initial)
        self.register_buffer("initial", initial)

    def forward(self, sequence):
        state = self.initial.expand(sequence.shape[0], -1)
        for value in sequence.unbind(dim=1):
            joined = torch.cat((state, value), dim=-1)
            z = self.update(joined).sigmoid()
            r = self.reset(joined).sigmoid()
            candidate = self.candidate(torch.cat((r * state, value), dim=-1)).tanh()
            state = (1 - z) * state + z * candidate
        return state


class BSTAN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        horizon=10,
        edge_input_dim=None,
        heads=3,
        gat_layers=2,
        target_scale=1.0,
        graph_channels=9,
        graph_dim=24,
        **kwargs,
    ):
        super().__init__()
        validate_model_dimensions(input_dim, hidden_dim, horizon, edge_input_dim, target_scale)
        require_positive_integer("heads", heads)
        require_positive_integer("gat_layers", gat_layers)
        self.horizon = horizon
        self.target_scale = float(target_scale)
        self.fields = ResourceGraphInput(LAYOUT, hidden_dim, pairs=True)
        self.spatial = nn.ModuleList(WeightedMultiHeadGAT(hidden_dim, heads) for _ in range(gat_layers))
        self.temporal = BSTANCell(hidden_dim, xavier_initial=True)

        self.readout = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon)
        )

    def forward(self, batch):
        node, edge = self.fields(batch)
        b, t, n, h = node.shape
        src, dst = batch["edge_index"].to(node.device, torch.long)
        support = _support(
            batch.get("spatial_edge_index", batch["edge_index"]),
            n,
            node.dtype,
            node.device,
            batch.get("spatial_edge_weight", batch.get("graph_edge_weight")),
        )
        spatial = node.reshape(b * t, n, h)
        for layer in self.spatial:
            spatial = layer(spatial, support)
        spatial = spatial.reshape(b, t, n, h)

        context = self.temporal(spatial.amax(dim=2))
        context = context[:, None].expand(-1, src.numel(), -1)
        local = torch.cat((context, spatial[:, -1, src], spatial[:, -1, dst], edge[:, -1]), -1)
        delta = self.readout(local).transpose(1, 2) * self.target_scale
        return batch["last_tg"][:, None, :] + delta


def _normalized_support(edge_index, n, dtype, device, edge_weight=None):
    if edge_weight is not None:

        src, dst, weight = _support(edge_index, n, dtype, device, edge_weight)
        degree = torch.zeros(n, dtype=dtype, device=device).index_add_(0, dst, weight).clamp_min(1e-12)
        return src, dst, weight * (degree[src] * degree[dst]).rsqrt()
    indices, _ = checked_edges(edge_index, n, device)
    src, dst = indices
    loops = torch.arange(n, device=device)
    flat = torch.cat((src * n + dst, dst * n + src, loops * n + loops)).unique()
    src, dst = flat.div(n, rounding_mode="floor"), flat.remainder(n)
    degree = torch.bincount(dst, minlength=n).to(dtype).clamp_min(1)
    weight = (degree[src] * degree[dst]).rsqrt()
    return src, dst, weight


def _propagate(x, support):
    src, dst, weight = support

    adjacency = x.new_zeros(x.shape[1], x.shape[1])
    adjacency[dst, src] = weight.to(x)
    return torch.matmul(adjacency, x)


class BTGCNCell(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        require_positive_integer("hidden_dim", hidden_dim)
        self.hidden_dim = hidden_dim
        self.update = nn.Linear(2 * hidden_dim, hidden_dim)
        self.reset = nn.Linear(2 * hidden_dim, hidden_dim)
        self.candidate = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, sequence):
        state = sequence.new_zeros(sequence.shape[0], self.hidden_dim)
        for value in sequence.unbind(dim=1):
            joined = torch.cat((state, value), -1)
            z = self.update(joined).sigmoid()
            r = self.reset(joined).sigmoid()
            candidate = self.candidate(torch.cat((r * state, value), -1)).tanh()
            state = (1 - z) * state + z * candidate
        return state


class BTGCN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        horizon=10,
        edge_input_dim=None,
        target_scale=1.0,
        graph_channels=9,
        graph_dim=24,
        **kwargs,
    ):
        super().__init__()
        validate_model_dimensions(input_dim, hidden_dim, horizon, edge_input_dim, target_scale)
        self.horizon = horizon
        self.target_scale = float(target_scale)
        self.fields = ResourceGraphInput(LAYOUT, hidden_dim, pairs=True)
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prelu = nn.PReLU()
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temporal = BTGCNCell(hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon)
        )

    def forward(self, batch):
        node, edge = self.fields(batch)
        b, t, n, h = node.shape
        src, dst = batch["edge_index"].to(node.device, torch.long)
        support = _normalized_support(
            batch.get("spatial_edge_index", batch["edge_index"]),
            n,
            node.dtype,
            node.device,
            batch.get("spatial_edge_weight"),
        )
        spatial = node.reshape(b * t, n, h)

        spatial = self.prelu(self.gcn1(_propagate(spatial, support)))
        spatial = self.gcn2(_propagate(spatial, support)).sigmoid()
        spatial = spatial.reshape(b, t, n, h)
        context = self.temporal(spatial.amax(dim=2))
        context = context[:, None].expand(-1, src.numel(), -1)
        local = torch.cat((context, spatial[:, -1, src], spatial[:, -1, dst], edge[:, -1]), -1)
        delta = self.readout(local).transpose(1, 2) * self.target_scale
        return batch["last_tg"][:, None, :] + delta


class ComparisonModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.direct_graph = method in ("bstan", "btgcn")
        args = PROFILES[method]["native_args"]
        if method == "gru":
            self.network = FlatGRU(LAYOUT, args["hidden_dim"], args["num_layers"], args["dropout"])
        else:
            cls = {
                "graph_wavenet": GraphWaveNetForecast,
                "stgcn": STGCNForecast,
                "bstan": BSTAN,
                "btgcn": BTGCN,
            }[method]
            self.network = cls(
                input_dim=LAYOUT["node_width"],
                edge_input_dim=LAYOUT["edge_width"],
                graph_channels=8,
                **args,
            )

    def forward(self, batch):
        if self.direct_graph:
            edge = batch["edge_index"]
            n = batch["x"].shape[2]
            ids = (edge[0] * n + edge[1])[edge[0] != edge[1]].unique(sorted=True)
            direct = torch.stack((ids.div(n, rounding_mode="floor"), ids.remainder(n)))
            batch = dict(
                batch,
                spatial_edge_index=direct,
                spatial_edge_weight=torch.ones_like(ids, dtype=batch["x"].dtype),
            )
        prediction = self.network(batch)
        target = batch["edge_index"][1][None, None].expand_as(prediction)
        minima = prediction.new_full((*prediction.shape[:2], batch["x"].shape[2]), torch.inf)
        minima.scatter_reduce_(2, target, prediction, reduce="amin", include_self=True)
        return (prediction - minima.gather(2, target)).clamp_min(0)
