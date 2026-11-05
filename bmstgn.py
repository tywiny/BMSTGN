import math
import torch
from torch import nn
from torch.nn import functional as F


class LatentNorm(nn.LayerNorm):
    def forward(self, value):

        return super().forward(value).to(value.dtype)


def mlp(a, hidden, out, norm=False):
    modules = [nn.Linear(a, hidden), nn.GELU(), nn.Linear(hidden, out)]
    if norm:
        modules.append(LatentNorm(out))
    return nn.Sequential(*modules)


def group_sum(value, index, groups):
    dtype = value.dtype
    if dtype in (torch.float16, torch.bfloat16):
        value = value.float()
    result = value.new_zeros(*value.shape[:-2], groups, value.shape[-1])
    return result.index_add(-2, index, value).to(dtype)


def group_mean(value, index, groups, count=None):
    total = group_sum(value, index, groups)
    if count is None:
        count = torch.bincount(index, minlength=groups).to(value).clamp_min(1)
    return total / count[:, None]


def group_softmax(score, index, groups):
    dtype = score.dtype
    score = score.float()
    ix = index.reshape(*([1] * (score.ndim - 2)), -1, 1).expand_as(score)
    maximum = score.new_full((*score.shape[:-2], groups, score.shape[-1]), -torch.inf)
    maximum = maximum.scatter_reduce(-2, ix, score, reduce="amax", include_self=True)
    exp = (score - maximum.index_select(-2, index)).exp()
    denominator = group_sum(exp, index, groups).index_select(-2, index)
    return (exp / denominator.clamp_min(1e-12)).to(dtype)


def group_min(value, index, groups):
    ix = index.reshape(*([1] * (value.ndim - 1)), -1).expand_as(value)
    out = value.new_full((*value.shape[:-1], groups), torch.inf)
    return out.scatter_reduce(-1, ix, value, reduce="amin", include_self=True)


def stats(graph, like):
    store = getattr(graph, "_statistics", None)
    key = (like.dtype, like.device)
    if store is not None and key in store:
        return store[key]
    src, dst = graph.index
    n = graph.nodes
    e = len(src)
    incoming = torch.bincount(dst, minlength=n)
    outgoing = torch.bincount(src, minlength=n)
    value = dict(
        in_count=incoming.to(like).clamp_min(1),
        out_count=outgoing.to(like).clamp_min(1),
        incidence=torch.stack((incoming > 0, outgoing > 0), -1).to(like),
        parent=(incoming > 0).to(like)[src],
        flags=torch.stack(
            (
                torch.bincount(graph.up_query, minlength=e) > 0,
                torch.bincount(graph.peer_query, minlength=e) > 0,
            ),
            -1,
        ).to(like),
    )
    if store is not None:
        store[key] = value
    return value


class GroupEncoders(nn.Module):

    widths = (32, 16, 48, 24, 24)

    def __init__(self, ablation):
        super().__init__()
        self.ablation = ablation
        self.group = nn.Embedding(4, 4)
        self.material = nn.Embedding(3, 4)
        self.equipment_kind = nn.Embedding(4, 4)
        self.equipment_group = nn.Embedding(5, 4)
        self.warehouse_kind = nn.Embedding(4, 4)
        self.warehouse_group = nn.Embedding(5, 4)
        self.craft = mlp(47, 64, 32, True)
        self.edge_fact = mlp(10, 32, 16, True)
        self.edge_init = mlp(16, 64, 48, True)
        self.static_plan = mlp(5, 32, 16, True)
        self.static_equipment = mlp(14, 32, 24, True)
        self.static_equipment_fusion = mlp(96, 48, 24, True)
        self.static_warehouse = mlp(9, 32, 16, True)
        self.static_logistics = mlp(65, 48, 24, True)
        self.static_gate = mlp(96, 32, 4)
        self.static_fusion = mlp(96, 224, 112, True)
        if ablation != "zero-pds":
            self.status = nn.Embedding(4, 4)
            self.operation = nn.Embedding(17, 4)
            self.stage_id = nn.Embedding(4, 4)
            self.plan = mlp(9, 32, 16, True)
            self.stage = mlp(74, 48, 16, True)
            self.stage_fusion = mlp(64, 64, 48, True)
            self.equipment = mlp(23, 48, 24, True)
            self.equipment_fusion = mlp(96, 48, 24, True)
            self.warehouse = mlp(15, 32, 16, True)
            self.logistics = mlp(78, 48, 24, True)
            self.group_gate = mlp(144, 32, 5)
            self.group_projection = nn.Linear(144, 112)
            self.craft_to_exec = nn.Linear(32, 48)
            self.exec_to_equipment = nn.Linear(48, 24)
            self.cross_craft_exec = nn.Linear(48, 112, bias=False)
            self.cross_exec_equipment = nn.Linear(24, 112, bias=False)
            self.cross_equipment_logistics = nn.Linear(24, 112, bias=False)
            self.node_norm = LatentNorm(112)
            self.pds_norm = LatentNorm(112)
        if ablation not in ("no-dp", "zero-pds"):
            self.edge_context = mlp(34, 48, 112)

    def encode_static(self, craft, f, s, graph, eq_categories, wh_categories):

        plan = self.static_plan(s["plan_h"].float().asinh())
        equipment = self.static_equipment(torch.cat((s["resource_numeric"], eq_categories), -1))
        local = self.static_equipment_fusion(equipment[s["phase"]].flatten(-2))
        warehouse = self.static_warehouse(torch.cat((s["warehouse_capacity"], wh_categories), -1))
        pool = torch.einsum("np,pc->nc", s["pool_share"], equipment[s["pool_index"]])
        logistics = self.static_logistics(
            torch.cat(
                (
                    s["demand"],
                    s["pool_share"],
                    s["site"].expand(len(craft), -1),
                    warehouse[s["warehouse_index"]].flatten(-2),
                    pool,
                ),
                -1,
            )
        )
        groups = (craft, plan, local, logistics)
        gate = self.static_gate(torch.cat(groups, -1)).sigmoid()
        return self.static_fusion(torch.cat([v * gate[:, i : i + 1] for i, v in enumerate(groups)], -1))

    def forward(self, f, s, graph):
        b, t = f["edge"].shape[:2]
        n = graph.nodes
        craft = self.craft(
            torch.cat((f["craft"], self.group(f["group"]), self.material(f["material"])), -1)
        )
        if self.ablation == "zero-pds":
            ec = torch.cat(
                (self.equipment_kind(f["equipment_kind"]), self.equipment_group(f["equipment_group"])), -1
            )
            wc = torch.cat(
                (self.warehouse_kind(f["warehouse_kind"]), self.warehouse_group(f["warehouse_group"])), -1
            )
            static = self.encode_static(craft, f, s, graph, ec, wc)
            edge = self.edge_init(self.edge_fact(f["edge"]))
            return static, edge.new_zeros(b, t, n, 112), edge
        static_craft = craft
        craft = craft[None, None].expand(b, t, n, -1)
        plan = self.plan(f["plan"])
        phase = self.stage_id.weight[None, None, None].expand(b, t, n, 4, 4)
        stage = self.stage(
            torch.cat((f["stage"], self.status(f["status"]), self.operation(f["operation"]), phase), -1)
        )
        stage = self.stage_fusion(stage.flatten(-2))
        eq_categories = torch.cat(
            (self.equipment_kind(f["equipment_kind"]), self.equipment_group(f["equipment_group"])), -1
        )
        equipment = self.equipment(
            torch.cat((f["equipment"], eq_categories[None, None].expand(b, t, -1, -1)), -1)
        )
        local_equipment = self.equipment_fusion(equipment[:, :, s["phase"]].flatten(-2))
        wh_categories = torch.cat(
            (self.warehouse_kind(f["warehouse_kind"]), self.warehouse_group(f["warehouse_group"])), -1
        )
        warehouse = self.warehouse(
            torch.cat((f["warehouse"], wh_categories[None, None].expand(b, t, -1, -1)), -1)
        )
        pool = torch.einsum("np,btpc->btnc", s["pool_share"].float(), equipment[:, :, s["pool_index"]])
        logistics = self.logistics(
            torch.cat((warehouse[:, :, s["warehouse_index"]].flatten(-2), f["logistics"], pool), -1)
        )
        groups = (craft, plan, stage, local_equipment, logistics)
        gate = self.group_gate(torch.cat(groups, -1)).sigmoid()
        fused = self.group_projection(
            torch.cat([z * gate[..., i : i + 1] for i, z in enumerate(groups)], -1)
        )
        fused = fused + self.cross_craft_exec(self.craft_to_exec(craft) * stage)
        fused = fused + self.cross_exec_equipment(self.exec_to_equipment(stage) * local_equipment)
        fused = fused + self.cross_equipment_logistics(local_equipment * logistics)
        nodes = self.node_norm(fused)
        static = self.encode_static(static_craft, f, s, graph, eq_categories, wh_categories)
        src, dst = graph.index
        edge_facts = self.edge_fact(f["edge"])
        if self.ablation == "no-dp":
            pds = self.pds_norm(nodes)
        else:
            counts = stats(graph, nodes)
            incidence = counts["incidence"][None, None].expand(b, t, n, 2)
            context = torch.cat(
                (
                    group_mean(edge_facts, dst, n, counts["in_count"]),
                    group_mean(edge_facts, src, n, counts["out_count"]),
                    incidence,
                ),
                -1,
            )
            pds = self.pds_norm(nodes + self.edge_context(context))
        edge = self.edge_init(edge_facts)
        return static, pds, edge


class ProductionMechanism(nn.Module):

    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1, bias=False))
        self.value = nn.Linear(112, 48)
        self.source = nn.Linear(112, 48)
        self.target = nn.Linear(112, 48)
        self.static_value = nn.Linear(112, 48, bias=False)
        self.static_source = nn.Linear(112, 48, bias=False)
        self.static_target = nn.Linear(112, 48, bias=False)
        self.node_fusion = mlp(144, 96, 48)
        self.edge_fusion = mlp(97, 96, 48)
        self.node_message = nn.Linear(48, 48)

    def project_static(self, static):
        return self.static_value(static), self.static_source(static), self.static_target(static)

    def predecessor_weights(self, edge, graph):
        return group_softmax(self.score(edge), graph.index[1], graph.nodes)

    def forward(self, static, nodes, edge, graph, static_terms=None):
        src, dst = graph.index
        alpha = self.predecessor_weights(edge, graph)
        sv, ss, st = self.project_static(static) if static_terms is None else static_terms
        value = self.value(nodes) + sv[None, None]
        predecessor = group_sum(alpha * value[:, :, src], dst, graph.nodes)
        source = self.source(nodes)[:, :, src] + ss[None, None, src]
        target = self.target(nodes)[:, :, dst] + st[None, None, dst]
        message = self.node_fusion(torch.cat((source, target, predecessor[:, :, src]), -1))
        flag = stats(graph, edge)["parent"][None, None, :, None].expand(*edge.shape[:-1], 1)
        pm_edge = edge + self.edge_fusion(torch.cat((edge, message, flag), -1))
        return pm_edge, self.node_message(message)


class EdgeScoreAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1, bias=False))
        self.value = nn.Linear(48, 48)

    def attention_weights(self, edge, query_index, key_index):
        return group_softmax(self.score(edge)[:, :, key_index], query_index, edge.shape[2])

    def forward(self, edge, query_index, key_index):
        weight = self.attention_weights(edge, query_index, key_index)
        return group_sum(weight * self.value(edge)[:, :, key_index], query_index, edge.shape[2])


class PackedEdgeAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(48, 48, bias=False)
        self.key = nn.Linear(48, 48, bias=False)
        self.value = nn.Linear(48, 48)

    def attention_weights(self, edge, query_index, key_index):
        b, t, e, _ = edge.shape
        q = self.query(edge).reshape(b, t, e, 4, 12)[:, :, query_index]
        k = self.key(edge).reshape(b, t, e, 4, 12)[:, :, key_index]
        return group_softmax((q * k).sum(-1) / math.sqrt(12), query_index, e)

    def forward(self, edge, query_index, key_index):
        b, t, e, _ = edge.shape
        weight = self.attention_weights(edge, query_index, key_index)
        value = self.value(edge).reshape(b, t, e, 4, 12)[:, :, key_index]
        return group_sum((weight[..., None] * value).flatten(-2), query_index, e)


class DualPropagation(nn.Module):
    def __init__(self):
        super().__init__()
        self.original_up = EdgeScoreAttention()
        self.original_peer = PackedEdgeAttention()
        self.pm_up = EdgeScoreAttention()
        self.pm_peer = PackedEdgeAttention()
        self.fusion = mlp(290, 96, 48)

    def forward(self, original, pm_edge, graph):
        flags = stats(graph, original)["flags"][None, None].expand(*original.shape[:-1], 2)
        pieces = (
            original,
            pm_edge,
            self.original_up(original, graph.up_query, graph.up_key),
            self.original_peer(original, graph.peer_query, graph.peer_key),
            self.pm_up(pm_edge, graph.up_query, graph.up_key),
            self.pm_peer(pm_edge, graph.peer_query, graph.peer_key),
            flags,
        )
        return self.fusion(torch.cat(pieces, -1)), flags


class GraphUpdate(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.gated = variant != "full"
        if variant != "no-node-update":
            self.node_gate = mlp(114 if variant == "no-dp" else 306, 160, 224 if self.gated else 112)
            self.node_norm = LatentNorm(112)
        self.edge_gate = mlp(98 if variant == "no-dp" else 146, 96, 96 if self.gated else 48)
        self.edge_norm = LatentNorm(48)

    def residual(self, value, update, norm):
        if self.gated:
            delta, gate = update.chunk(2, -1)
            update = gate.sigmoid() * delta
        return norm(value + update)

    def forward(self, nodes, edge, pm_edge, message, propagation, flags, graph):
        src, dst = graph.index
        counts = stats(graph, nodes)
        if self.variant != "no-node-update":
            incidence = counts["incidence"][None, None].expand(*nodes.shape[:-1], 2)
            if self.variant == "no-dp":
                joint = torch.cat((nodes, incidence), -1)
            else:
                joint = torch.cat(
                    (
                        nodes,
                        group_mean(message, dst, graph.nodes, counts["in_count"]),
                        group_mean(message, src, graph.nodes, counts["out_count"]),
                        group_mean(propagation, dst, graph.nodes, counts["in_count"]),
                        group_mean(propagation, src, graph.nodes, counts["out_count"]),
                        incidence,
                    ),
                    -1,
                )
            nodes = self.residual(nodes, self.node_gate(joint), self.node_norm)
        edge_items = (
            (edge, pm_edge - edge, flags)
            if self.variant == "no-dp"
            else (edge, pm_edge - edge, propagation, flags)
        )
        return nodes, self.residual(edge, self.edge_gate(torch.cat(edge_items, -1)), self.edge_norm)


class SpatialBlock(nn.Module):
    def __init__(self, ablation):
        super().__init__()
        self.pm = ProductionMechanism()
        if ablation in ("no-dp", "no-node-update"):
            self.pm.node_message = nn.Identity()
        self.dp = None if ablation == "no-dp" else DualPropagation()
        self.dgr = GraphUpdate(ablation)

    def project_static(self, static):
        return self.pm.project_static(static)

    def forward(self, static, nodes, edge, graph, static_terms=None):
        pm_edge, message = self.pm(static, nodes, edge, graph, static_terms)
        if self.dp is None:
            propagation = None
            flags = stats(graph, edge)["flags"][None, None].expand(*edge.shape[:-1], 2)
        else:
            propagation, flags = self.dp(edge, pm_edge, graph)
        return self.dgr(nodes, edge, pm_edge, message, propagation, flags, graph)


class NodeGCNSpatial(nn.Module):

    def __init__(self, dropout=0.1, hidden=196, endpoint_hidden=620):
        super().__init__()
        self.static_projection = nn.Linear(112, 112, bias=False)
        self.gcn_first = nn.Linear(112, hidden)
        self.gcn_second = nn.Linear(hidden, 112)
        self.node_norm = LatentNorm(112)
        self.dropout = nn.Dropout(dropout)
        self.endpoint_head = mlp(272, endpoint_hidden, 48)
        self.edge_norm = LatentNorm(48)

    def project_static(self, static):
        return self.static_projection(static)

    @staticmethod
    def normalized_adjacency(graph, like):
        store = getattr(graph, "_statistics", None)
        key = ("normalized_adjacency", like.device)
        if store is not None and key in store:
            return store[key].to(like)
        src, dst = graph.index
        adjacency = torch.eye(graph.nodes, device=like.device, dtype=torch.float32)
        adjacency[src, dst] = 1
        adjacency[dst, src] = 1
        inverse = adjacency.sum(-1).rsqrt()
        normalized = inverse[:, None] * adjacency * inverse[None, :]
        if store is not None:
            store[key] = normalized
        return normalized.to(like)

    def forward(self, static, nodes, edge, graph, static_terms=None):
        projected = self.project_static(static) if static_terms is None else static_terms
        value = nodes + projected[None, None]
        adjacency = self.normalized_adjacency(graph, value)
        value = self.dropout(F.gelu(self.gcn_first(torch.einsum("ij,btjc->btic", adjacency, value))))
        nodes = self.node_norm(F.gelu(self.gcn_second(torch.einsum("ij,btjc->btic", adjacency, value))))
        src, dst = graph.index
        edge = self.edge_norm(
            self.endpoint_head(torch.cat((nodes[:, :, src], nodes[:, :, dst], edge), -1))
        )
        return nodes, edge


class Temporal(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = LatentNorm(width)

    def forward(self, state):
        b, t, entities, width = state.shape
        sequence = state.permute(0, 2, 1, 3).reshape(b * entities, t, width)
        if sequence.is_cuda and sequence.dtype == torch.float16:
            with torch.backends.cudnn.flags(enabled=False):
                value, _ = self.gru(sequence)
        else:
            value, _ = self.gru(sequence)
        value = value.reshape(b, entities, t, width).permute(0, 2, 1, 3)
        return self.norm(state + self.dropout(value))


class FutureBridge(nn.Module):
    def __init__(self, width, hidden=None):
        super().__init__()
        self.time = mlp(10, 32, 10)
        self.lead = nn.Embedding(10, width)
        self.norm = LatentNorm(width)
        self.ffn = mlp(width, hidden or 4 * width, width)
        self.ffn_norm = LatentNorm(width)

    def forward(self, history):
        mixed = self.time(history.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        state = self.norm(mixed + history[:, -1:] + self.lead.weight.to(history)[None, :, None])
        return self.ffn_norm(state + self.ffn(state))


def project_tg(raw, boundary, graph):

    src, dst = graph.index
    observed = boundary.finish_valid[:, src][:, None]
    dates = boundary.finish_h[:, src][:, None]
    score = raw.double()
    count = group_sum(observed.float()[..., None], dst, graph.nodes)[..., 0]
    total = group_sum(torch.where(observed, score + dates, 0.0)[..., None], dst, graph.nodes)[..., 0]
    common = (total / count.clamp_min(1))[..., dst]
    factual = torch.where(observed, common - dates, score)
    result = (factual - group_min(factual, dst, graph.nodes)[..., dst]).clamp_min(0.0)
    return torch.where(boundary.fixed[:, None], boundary.last_tg[:, None], result.to(raw.dtype))


ABLATIONS = ("full", "no-pm", "no-dp", "gcn", "zero-pds", "no-node-update", "mean-bridge")


class WidePeerAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        assert width % 4 == 0
        self.width = width
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width)

    def forward(self, edge, query_index, key_index):
        b, t, e, _ = edge.shape
        depth = self.width // 4
        q = self.query(edge).reshape(b, t, e, 4, depth)[:, :, query_index]
        k = self.key(edge).reshape(b, t, e, 4, depth)[:, :, key_index]
        v = self.value(edge).reshape(b, t, e, 4, depth)[:, :, key_index]
        weight = group_softmax((q * k).sum(-1) / math.sqrt(depth), query_index, e)
        return group_sum((weight[..., None] * v).flatten(-2), query_index, e)


class WideUpstreamAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(width, width // 2), nn.GELU(), nn.Linear(width // 2, 1, bias=False)
        )
        self.value = nn.Linear(width, width)

    def forward(self, edge, query_index, key_index):
        weight = group_softmax(self.score(edge)[:, :, key_index], query_index, edge.shape[2])
        return group_sum(weight * self.value(edge)[:, :, key_index], query_index, edge.shape[2])


class EdgePropagation(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.original_up = WideUpstreamAttention(width)
        self.original_peer = WidePeerAttention(width)
        self.pm_up = WideUpstreamAttention(width)
        self.pm_peer = WidePeerAttention(width)
        self.fusion = mlp(6 * width + 2, 2 * width, width)

    def forward(self, edge, graph):
        flags = stats(graph, edge)["flags"][None, None].expand(*edge.shape[:-1], 2)

        pieces = (
            edge,
            edge,
            self.original_up(edge, graph.up_query, graph.up_key),
            self.original_peer(edge, graph.peer_query, graph.peer_key),
            self.pm_up(edge, graph.up_query, graph.up_key),
            self.pm_peer(edge, graph.peer_query, graph.peer_key),
            flags,
        )
        return self.fusion(torch.cat(pieces, -1)), None, flags


class EdgeResidualOnly(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.edge_gate = mlp(2 * width + 2, 2 * width, 2 * width)
        self.edge_norm = LatentNorm(width)
        with torch.no_grad():
            self.edge_gate[-1].bias[width:].fill_(-1.0)

    def forward(self, edge, dp, flags):
        delta, gate = self.edge_gate(torch.cat((edge, dp, flags), -1)).chunk(2, -1)
        return self.edge_norm(edge + gate.sigmoid() * delta)


class EdgeSpatial(nn.Module):
    def project_static(self, static):
        return None

    def __init__(self, width):
        super().__init__()
        self.pm = None
        self.dp = EdgePropagation(width)
        self.dgr = EdgeResidualOnly(width)

    def forward(self, static, nodes, edge, graph, static_terms=None):
        dp, _, flags = self.dp(edge, graph)
        return nodes, self.dgr(edge, dp, flags)


class EdgeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_fact = mlp(10, 32, 16, True)
        self.edge_init = mlp(16, 79, 104, True)

    def forward(self, batch, static, graph):
        edge = self.edge_init(self.edge_fact(batch["edge"]))
        return None, None, edge


class BMSTGN(nn.Module):
    def __init__(self, ablation="full", dropout=0.1):
        super().__init__()
        if ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation: {ablation}")
        self.ablation = ablation
        self.has_nodes = ablation != "no-pm"
        width = 48 if self.has_nodes else 104
        self.encoders = GroupEncoders(ablation) if self.has_nodes else EdgeEncoder()

        def spatial_block():
            if ablation == "no-pm":
                return EdgeSpatial(width)
            if ablation == "gcn":
                return NodeGCNSpatial(dropout, hidden=189, endpoint_hidden=547)
            return SpatialBlock(ablation)

        self.spatial = nn.ModuleList(spatial_block() for _ in range(2))
        if self.has_nodes:
            self.history_temporal = nn.ModuleList(Temporal(112, dropout) for _ in range(2))
            self.future_temporal = nn.ModuleList(Temporal(112, dropout) for _ in range(2))
        self.history_edge_temporal = nn.ModuleList(Temporal(width, dropout) for _ in range(2))
        self.future_edge_temporal = nn.ModuleList(Temporal(width, dropout) for _ in range(2))
        if ablation != "mean-bridge":
            if self.has_nodes:
                self.node_bridge = FutureBridge(112)
            self.edge_bridge = FutureBridge(width, None if self.has_nodes else 620)
        hidden = 48 if self.has_nodes else 112
        self.readout = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1, bias=False)
        )

    def forward(self, batch):
        b = batch
        graph = b["graph"]
        static, nodes, edge = self.encoders(b, b, graph)
        terms = [spatial.project_static(static) for spatial in self.spatial]
        for i, spatial in enumerate(self.spatial):
            if self.has_nodes:
                nodes = self.history_temporal[i](nodes)
            edge = self.history_edge_temporal[i](edge)
            nodes, edge = spatial(static, nodes, edge, graph, terms[i])
        if self.ablation == "mean-bridge":
            nodes = nodes.mean(1, keepdim=True).expand(-1, 10, -1, -1)
            edge = edge.mean(1, keepdim=True).expand(-1, 10, -1, -1)
        else:
            if self.has_nodes:
                nodes = self.node_bridge(nodes)
            edge = self.edge_bridge(edge)
        for i, spatial in enumerate(self.spatial):
            if self.has_nodes:
                nodes = self.future_temporal[i](nodes)
            edge = self.future_edge_temporal[i](edge)
            nodes, edge = spatial(static, nodes, edge, graph, terms[i])
        raw = b["boundary"].last_tg[:, None] + self.readout(edge)[..., 0]
        return project_tg(raw, b["boundary"], graph)
