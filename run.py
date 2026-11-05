import argparse
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from data import Case, DATA, ROOT, read_arrays, splits, tensor
from bmstgn import BMSTGN
from baselines import ComparisonModel, PROFILES
from rollout import DurationGCNGRU, RolloutInputs, inverse, schedule, state
from metrics import Accumulator, score_events_case

MASKS = {
    "full": "full",
    "-PM": "no-pm",
    "-DP": "no-dp",
    "BSF-GCN": "gcn",
    "-PDS": "zero-pds",
    "-DGR-node": "no-node-update",
    "-Bridge": "mean-bridge",
}
METHODS = (
    "bmstgn",
    "gru",
    "stgcn",
    "graph_wavenet",
    "bstan",
    "btgcn",
    "gcn_gru",
    "persistence",
    "plan_cpm",
)
RULES = ("persistence", "plan_cpm")
SEEDS = {k: (20261022 if k == "bmstgn" else 20261026 if k == "gcn_gru" else 20261023) for k in METHODS}
WEIGHTS = ROOT / "weights"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_save(obj, path):
    temp = path.with_suffix(".tmp")
    torch.save(obj, temp)
    temp.replace(path)


def read_state(path):
    return torch.load(path, map_location="cpu", weights_only=True)


class Predictor:
    def __init__(self, method, mask, device, training=False, weights=None):
        self.method, self.mask, self.device = method, mask, device
        self.rollouts = {}
        self.norm = None
        if method != "bmstgn" and mask != "full":
            raise ValueError("Ablations apply to BMSTGN")
        if method in RULES:
            if weights is not None or training:
                raise ValueError("This model has no weights")
            self.model = None
            return
        if method == "bmstgn":
            self.model = BMSTGN(MASKS[mask])
        elif method == "gcn_gru":
            self.model = DurationGCNGRU(375, 661)
            self.norm = {
                k: tensor(v, device) for k, v in read_arrays(WEIGHTS / "gcn_gru_norm.npz").items()
            }
        else:
            self.model = ComparisonModel(method)
        if weights is not None:
            checkpoint = read_state(weights)
            if "state_dict" in checkpoint:
                if checkpoint.get("method") != method or checkpoint.get("mask") != mask:
                    raise ValueError(
                        "Checkpoint model or ablation differs from the requested configuration"
                    )
                checkpoint = checkpoint["state_dict"]
            self.model.load_state_dict(checkpoint, strict=True)
        elif not training:
            stem = "bmstgn_" + MASKS[mask] if mask != "full" else method
            self.model.load_state_dict(read_state(WEIGHTS / (stem + ".pt")), strict=True)
        self.model.to(device).train(training)

    def rollout(self, case):
        if case.case_id not in self.rollouts:
            self.rollouts = {case.case_id: RolloutInputs(case, self.method == "gcn_gru")}
        return self.rollouts[case.case_id]

    def forward(self, case, origins, train=False):
        if self.method == "persistence":
            case.windows(origins)
            return (
                tensor(case.shared["observed_tg"][origins], self.device)[:, None].expand(-1, 10, -1),
                None,
            )
        if self.method in ("gcn_gru", "plan_cpm"):
            case.windows(origins)
            ri = self.rollout(case)
            if self.method == "plan_cpm":
                s = state(ri, origins, self.device)
                return schedule(s["duration"], s)[2][:, None].expand(-1, 10, -1), None
            batch = ri.batch(origins, self.device, self.norm)
            if train:
                p, trace = self.model(batch, return_trace=True)
                return p, (batch["state"], trace["duration"])
            return self.model(batch), None
        is_bm = self.method == "bmstgn"
        batch = (
            case.bm_batch(origins, self.device, MASKS[self.mask])
            if is_bm
            else case.comparison_batch(origins, self.device, self.method)
        )
        mixed = self.device.type == "cuda" and (is_bm or (train and torch.cuda.is_bf16_supported()))
        precision = (
            torch.autocast("cuda", dtype=torch.float16 if is_bm else torch.bfloat16)
            if mixed
            else nullcontext()
        )
        with precision:
            prediction = self.model(batch)
        return prediction.float(), batch


def validate_prediction(p, case, origins):
    if p.shape != (len(origins), 10, case.shared["edge_index"].shape[1]) or not torch.isfinite(p).all():
        raise ValueError(f"Invalid prediction in {case.case_id}")


@torch.inference_mode()
def evaluate(predictor, root, group, batch_size, events=True, paper=True):
    if predictor.model is not None:
        predictor.model.eval()
    total = Accumulator() if events else None
    cases = []
    squared = absolute = count = 0
    for cid in splits(root)[group]:
        case = Case(root, group, cid)
        pred = []
        local_abs = local_square = 0.0
        local_count = changed_count = fact_count = 0
        changed_sum = fact_sum = 0.0
        for lo in range(0, len(case.origins), batch_size):
            origins = case.origins[lo : lo + batch_size]
            values, _ = predictor.forward(case, origins)
            validate_prediction(values, case, origins)
            truth, mask = case.target(origins, predictor.device)
            error = (values - truth).abs()[mask].double()
            local_abs += error.sum().item() * 3600
            local_square += error.square().sum().item() * 3600**2
            if paper:
                fr = origins[:, None] + np.arange(1, 11)
                changed = tensor(case.changed[fr], predictor.device) & mask
                err = (values - truth).abs().double() * 3600.0
                changed_sum += err[changed].sum().item()
                changed_count += int(changed.sum())
                edge = case.shared["edge_index"][1]
                frozen = tensor(fr[:, :, None] == case.labels["freeze_at"][edge], predictor.device)
                fact_sum += err[frozen].sum().item()
                fact_count += int(frozen.sum())
            local_count += error.numel()
            if events:
                pred.append(values.cpu().numpy())
        if events:
            total.merge(score_events_case(case, case.origins, np.concatenate(pred), paper=paper))
        absolute += local_abs
        squared += local_square
        count += local_count
        cases.append(
            dict(
                case=cid,
                windows=len(case.origins),
                count=local_count,
                mae_s=local_abs / local_count if local_count else None,
                rmse_s=math.sqrt(local_square / local_count) if local_count else None,
                change_mae_s=changed_sum / changed_count if changed_count else None,
                final_tg_mae_s=fact_sum / fact_count if fact_count else None,
            )
        )
    means = [k["mae_s"] for k in cases if k["count"]]
    report = dict(
        split=group,
        cases=len(cases),
        windows=sum(k["windows"] for k in cases),
        mae_s=float(np.mean(means)),
        micro_mae_s=absolute / count,
        micro_rmse_s=math.sqrt(squared / count),
        by_case=cases,
    )
    for key in ("rmse_s", "change_mae_s", "final_tg_mae_s"):
        vals = [row[key] for row in cases if row[key] is not None]
        if vals:
            report[key] = float(np.mean(vals))
    if events:
        migration = total.finish()
        report.update(
            migration=migration,
            process_f1_d0=migration["full_process"]["with_optional_bonus"]["0"]["f1"],
            process_f1_d1=migration["full_process"]["with_optional_bonus"]["1"]["f1"],
            final_f1_d0=migration["final_migration"]["with_optional_bonus"]["0"]["f1"],
            final_f1_d1=migration["final_migration"]["with_optional_bonus"]["1"]["f1"],
        )
        for name in ("window", "final"):
            hit = total.identity[name + "_correct"]
            n = total.identity[name + "_count"]
            report[name + "_accuracy"] = hit / n if n else None
            report[name + "_correct"] = hit
            report[name + "_count"] = n
    return report


def forecast_loss(prediction, last_tg, graph, truth, mask):

    if prediction.shape != truth.shape or mask.shape != truth.shape:
        raise ValueError("Predictions, targets and loss masks must have identical shapes")
    if last_tg.shape != (prediction.shape[0], prediction.shape[2]):
        raise ValueError("Incorrect last-TG anchor shape")
    if not bool(torch.isfinite(prediction).all()):
        raise FloatingPointError("Nonfinite prediction")
    finite = torch.isfinite(truth)
    valid = mask.bool() & finite
    truth = torch.where(finite, truth.detach(), 0.0)

    def average(values, selected):
        return torch.where(selected, values, 0.0).sum() / selected.sum().clamp_min(1)

    level = (
        average(F.smooth_l1_loss(prediction * 3600, truth * 3600, beta=5.0, reduction="none"), valid)
        / 100
    )
    before_prediction = torch.cat((last_tg[:, None], prediction[:, :-1]), 1)
    before_truth = torch.cat((last_tg[:, None], truth[:, :-1]), 1)
    previous_valid = torch.cat((torch.isfinite(last_tg)[:, None], valid[:, :-1]), 1)
    delta = (
        average(
            F.smooth_l1_loss(
                (prediction - before_prediction) * 3600,
                (truth - before_truth) * 3600,
                beta=5.0,
                reduction="none",
            ),
            valid & previous_valid,
        )
        / 100
    )
    b, h, e = prediction.shape
    n = graph.nodes
    destination = graph.index[1]
    index = destination[None, None].expand(b, h, e)
    degree = torch.bincount(destination, minlength=n)
    count = prediction.new_zeros(b, h, n).scatter_add(2, index, valid.to(prediction))
    minimum = prediction.new_full((b, h, n), torch.inf).scatter_reduce(
        2, index, truth, reduce="amin", include_self=True
    )
    winners = truth <= minimum.gather(2, index) + 1e-8
    ties = prediction.new_zeros(b, h, n).scatter_add(2, index, winners.to(prediction))
    eligible = (count == degree[None, None]) & (degree[None, None] > 1) & (ties == 1)
    logits = -prediction * 3600 / 10.0
    maximum = prediction.new_full((b, h, n), -torch.inf).scatter_reduce(
        2, index, logits, reduce="amax", include_self=True
    )
    centered = logits - maximum.gather(2, index)
    denominator = prediction.new_zeros(b, h, n).scatter_add(2, index, centered.exp())
    log_probability = centered - denominator.gather(2, index).clamp_min(1e-30).log()
    chosen = winners & eligible.gather(2, index)
    ranking = -torch.where(chosen, log_probability, 0.0).sum() / eligible.sum().clamp_min(1)
    total = level + 0.1 * delta + 0.01 * ranking
    return total, dict(
        tg_seconds_scaled=level,
        tg_increment_scaled=delta,
        unique_bottleneck_ce=ranking,
        eligible_rank_groups=eligible.sum(),
    )


def comparison_terms(prediction, last_tg, observed_valid, edge, truth, mask, nodes):
    prediction, truth, last_tg = prediction.float(), truth.float(), last_tg.float()
    valid = mask.bool() & torch.isfinite(truth)
    truth = torch.where(valid, truth.detach(), 0.0)
    level = F.smooth_l1_loss(prediction * 3600, truth * 3600, beta=5.0, reduction="none") / 100.0
    previous_p = torch.cat((last_tg[:, None], prediction[:, :-1]), 1)
    previous_y = torch.cat((last_tg[:, None], truth[:, :-1]), 1)
    delta_valid = valid & torch.cat((observed_valid[:, None].bool(), valid[:, :-1]), 1)
    delta = (
        F.smooth_l1_loss(
            (prediction - previous_p) * 3600, (truth - previous_y) * 3600, beta=5.0, reduction="none"
        )
        / 100.0
    )
    destination = edge[1]
    index = destination[None, None].expand_as(prediction)
    shape = (*prediction.shape[:2], nodes)
    degree = torch.bincount(destination, minlength=nodes)
    count = prediction.new_zeros(shape).scatter_add(2, index, valid.float())
    minimum = prediction.new_full(shape, torch.inf).scatter_reduce(
        2, index, torch.where(valid, truth, torch.inf), reduce="amin", include_self=True
    )
    winners = valid & (truth <= minimum.gather(2, index) + 1e-8)
    ties = prediction.new_zeros(shape).scatter_add(2, index, winners.float())
    eligible = (count == degree[None, None]) & (degree[None, None] > 1) & (ties == 1)
    logits = -prediction * 3600 / 10.0
    maximum = prediction.new_full(shape, -torch.inf).scatter_reduce(
        2, index, logits, reduce="amax", include_self=True
    )
    centered = logits - maximum.gather(2, index)
    denominator = prediction.new_zeros(shape).scatter_add(2, index, centered.exp())
    log_p = centered - denominator.gather(2, index).clamp_min(1e-30).log()
    chosen = winners & eligible.gather(2, index)
    return dict(
        tg_sum=torch.where(valid, level, 0.0).sum(),
        tg_count=valid.sum(),
        delta_sum=torch.where(delta_valid, delta, 0.0).sum(),
        delta_count=delta_valid.sum(),
        ranking_sum=-torch.where(chosen, log_p, 0.0).sum(),
        ranking_count=eligible.sum(),
    )


def learning_rate(step, steps_per_epoch, epochs=40):
    warm = 2 * steps_per_epoch
    total = epochs * steps_per_epoch
    ratio = (
        step / max(1, warm)
        if step <= warm
        else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
    )
    return 1e-5 + (5e-4 - 1e-5) * ratio


def loss_denominators(case, batch_size):
    totals = dict(tg=0, delta=0, ranking=0)
    edge = tensor(case.shared["edge_index"], "cpu")
    for lo in range(0, len(case.origins), batch_size):
        origins = case.origins[lo : lo + batch_size]
        y, mask = case.target(origins, "cpu")
        last = tensor(case.shared["observed_tg"][origins], "cpu")
        values = comparison_terms(
            y, last, torch.isfinite(last), edge, y, mask, case.shared["plan_h"].shape[0]
        )
        for k in totals:
            totals[k] += int(values[k + "_count"])
    return totals


def training_schedule(cases, method, seed, epoch, batch_size, sequence_rng):
    if method == "bmstgn":
        rng = np.random.default_rng(seed + epoch)
        jobs = []
        for cid, case in cases.items():
            order = rng.permutation(len(case.origins))
            jobs.extend(
                (cid, case.origins[order[i : i + batch_size]]) for i in range(0, len(order), batch_size)
            )
        rng.shuffle(jobs)
        return jobs
    rng = sequence_rng if method == "gcn_gru" else np.random.default_rng(seed + epoch * 104729)
    ids = list(cases)
    rng.shuffle(ids)
    jobs = []
    for cid in ids:
        origins = cases[cid].origins.copy()
        rng.shuffle(origins)
        jobs.extend((cid, origins[i : i + batch_size]) for i in range(0, len(origins), batch_size))
    return jobs


def train(args, device):
    method = args.model
    seed = args.seed if args.seed is not None else SEEDS[method]
    if method in RULES:
        raise ValueError("Use evaluate for this parameter-free model")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    predictor = Predictor(method, args.ablation, device, training=True)
    cases = {cid: Case(args.data, "train", cid) for cid in splits(args.data)["train"]}
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    bm = method == "bmstgn"
    duration = method == "gcn_gru"
    optimizer = torch.optim.AdamW(
        predictor.model.parameters(),
        lr=3e-4 if duration else 5e-4,
        weight_decay=1e-5 if duration else 1e-4,
        fused=True if bm and device.type == "cuda" else None,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bm and device.type == "cuda", init_scale=128.0)
    counts = {}
    for cid, c in cases.items():
        if duration:
            counts[cid] = int(np.isnan(c.shared["finish_ms"][c.origins]).sum()) * 10
        elif not bm:
            counts[cid] = loss_denominators(c, args.batch_size)
    rng = np.random.default_rng(seed)
    best = None
    best_epoch = 0
    step = 0
    skipped = 0
    for epoch in range(1, args.epochs + 1):
        predictor.model.train()
        jobs = training_schedule(cases, method, seed, epoch, args.batch_size, rng)
        loss_sum = 0.0
        for cid, origins in jobs:
            step += 1
            case = cases[cid]
            optimizer.zero_grad(set_to_none=True)
            lr = (
                (3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (epoch - 1) / 40)))
                if duration
                else learning_rate(step, len(jobs))
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            p, batch = predictor.forward(case, origins, train=True)
            if duration:
                s, duration_hat = batch
                dates = tensor(case.labels["date_h"][origins[:, None] + np.arange(1, 11)], device)
                y = torch.stack([inverse(dates[:, h], s) for h in range(10)], 1)
                mask = (~s["done"])[:, None].expand_as(y)
                err = F.smooth_l1_loss(duration_hat.log1p(), y.log1p(), beta=0.1, reduction="none")
                loss = err[mask].sum() * len(jobs) / (len(cases) * counts[cid])
            else:
                y, mask = case.target(origins, device)
                if bm:
                    loss, _ = forecast_loss(p, batch["boundary"].last_tg, batch["graph"], y, mask)
                else:
                    last = batch["last_tg"]
                    parts = comparison_terms(
                        p, last, torch.isfinite(last), batch["edge_index"], y, mask, batch["x"].shape[2]
                    )
                    loss = (
                        sum(
                            w * parts[k + "_sum"] / max(1, counts[cid][k])
                            for k, w in [("tg", 1.0), ("delta", 0.1), ("ranking", 0.01)]
                        )
                        * len(jobs)
                        / len(cases)
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                predictor.model.parameters(),
                1.0 if bm else 2.0,
                error_if_nonfinite=not scaler.is_enabled(),
            )
            skipped = 0 if torch.isfinite(norm) else skipped + 1
            if skipped >= 5:
                raise FloatingPointError("Five consecutive nonfinite gradient norms")
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item()
        payload = dict(
            method=method,
            mask=args.ablation,
            epoch=epoch,
            seed=seed,
            state_dict=predictor.model.state_dict(),
        )
        if bm:
            atomic_save(payload, output / "model.pt")
            print(json.dumps(dict(epoch=epoch, loss=loss_sum / len(jobs))), flush=True)
        else:
            result = evaluate(
                predictor, args.data, "validation", args.batch_size, events=duration, paper=False
            )
            key = (
                (result["process_f1_d1"], result["process_f1_d0"], -result["mae_s"])
                if duration
                else (-result["mae_s"],)
            )
            if best is None or key > best:
                best = key
                best_epoch = epoch
                atomic_save(payload, output / "model.pt")
            print(
                json.dumps(
                    dict(
                        epoch=epoch,
                        loss=loss_sum / len(jobs),
                        validation_mae_s=result["mae_s"],
                        selected_epoch=best_epoch,
                    )
                ),
                flush=True,
            )
            if (duration and epoch >= 15 and epoch - best_epoch >= 10) or (
                not duration and epoch - best_epoch >= 8
            ):
                break


def resolve_device(name):
    device = torch.device(
        "cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    torch.set_num_threads(2)
    return device


def main():
    parser = argparse.ArgumentParser(description="BMSTGN")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    check = sub.add_parser("check")
    check.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    for name in ("predict", "evaluate", "train"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--model", choices=METHODS, default="bmstgn")
        cmd.add_argument("--ablation", choices=MASKS, default="full")
        cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        cmd.add_argument("--data", type=Path, default=DATA)
        if name != "predict":
            cmd.add_argument("--batch-size", type=int, default=32)
        if name == "train":
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--epochs", type=int, default=40)
            cmd.add_argument("--seed", type=int)
        else:
            cmd.add_argument("--weights", type=Path)
            cmd.add_argument("--split", choices=["train", "validation", "test"], default="test")
            cmd.add_argument("--output", type=Path)
        if name == "predict":
            cmd.add_argument("--case")
            cmd.add_argument("--origin", type=int)
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(METHODS))
        print("\n".join(list(MASKS)[1:]))
        return
    device = resolve_device(args.device)
    if args.command == "check":
        manifest = json.loads((ROOT / "checksums.json").read_text(encoding="utf8"))
        for rel, expected in manifest.items():
            if sha256(ROOT / rel) != expected:
                raise ValueError(f"Checksum mismatch: {rel}")
        c = Case(DATA, "validation", splits()["validation"][0])
        with torch.inference_mode():
            for method in METHODS:
                p = Predictor(method, "full", device)
                values, _ = p.forward(c, c.origins[:1])
                validate_prediction(values, c, c.origins[:1])
            for mask in list(MASKS)[1:]:
                p = Predictor("bmstgn", mask, device)
                p.model.eval()
                values, _ = p.forward(c, c.origins[:1])
                validate_prediction(values, c, c.origins[:1])
        print(f"OK: {len(manifest)} files; {len(METHODS)} models; {len(MASKS)-1} ablations")
        return
    if getattr(args, "batch_size", 1) < 1:
        parser.error("Batch size must be positive")
    if args.command == "train":
        if not 1 <= args.epochs <= 40:
            parser.error("Epochs must be 1..40")
        train(args, device)
        return
    predictor = Predictor(args.model, args.ablation, device, weights=args.weights)
    if args.command == "evaluate":
        result = evaluate(predictor, args.data, args.split, args.batch_size)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf8") as f:
                json.dump(result, f, indent=2, allow_nan=False)
        print(
            json.dumps({k: v for k, v in result.items() if k not in ("migration", "by_case")}, indent=2)
        )
    else:
        group = splits(args.data)[args.split]
        cid = args.case or group[0]
        if cid not in group:
            parser.error("Case is outside selected split")
        c = Case(args.data, args.split, cid)
        origins = np.array([args.origin if args.origin is not None else c.origins[0]], dtype=np.int64)
        with torch.inference_mode():
            p, _ = predictor.forward(c, origins)
            validate_prediction(p, c, origins)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("xb") as f:
                np.savez_compressed(
                    f, tg_h=p.cpu().numpy(), origins=origins, edge_index=c.shared["edge_index"]
                )
        print(
            json.dumps(
                dict(
                    case=cid,
                    origin=int(origins[0]),
                    shape=list(p.shape),
                    first_edge_seconds=(p[0, :, 0] * 3600).tolist(),
                )
            )
        )


if __name__ == "__main__":
    main()
