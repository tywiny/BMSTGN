import json
from pathlib import Path
import numpy as np
import torch
from data import Case, splits
from metrics import score_events_case


def summarize_splits(root):
    from data import PARTITIONS

    metrics = ("mae_s", "window_accuracy", "final_accuracy", "process_f1_d0", "process_f1_d1")
    records = []
    for model in ("bmstgn", "graph_wavenet", "gcn_gru"):
        reports = []
        for partition in PARTITIONS[1:]:
            path = Path(root) / partition / model / "test.json"
            report = json.loads(path.read_text(encoding="utf8"))
            if (report.get("partition"), report.get("model"), report.get("split")) != (
                partition, model, "test"
            ) or report.get("ablation") != "full":
                raise ValueError("Cross-split report configuration does not match its directory")
            if report.get("cases") != 12 or report.get("units", {}).get("accuracy") != "%":
                raise ValueError("Expected twelve test plans and percentage metrics")
            reports.append(report)
        for metric in metrics:
            values = np.array([report[metric] for report in reports], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Cross-split summary requires six finite values per metric")
            records.append(dict(
                model=model, metric=metric, mean=float(values.mean()),
                sd=float(values.std(ddof=1)), n=len(values), unit="s" if metric == "mae_s" else "%",
            ))
    return dict(
        partitions=list(PARTITIONS[1:]), runs=18,
        dispersion="sample standard deviation", records=records,
    )


class Breakdown:
    def __init__(self, case, prediction):
        self.case = case
        self.degree = np.bincount(
            case.shared["edge_index"][1], minlength=len(case.shared["plan_h"])
        )
        self.groups = {
            name: dict(
                window=[0, 0],
                final_identity=[0, 0],
                process={str(delta): [0, 0, 0] for delta in (0, 1)},
                final={str(delta): [0, 0, 0] for delta in (0, 1)},
            )
            for name in ("2", "3", ">=4")
        }
        self.leads = [
            dict(
                window_identity=[0, 0], process_identity=[0, 0], final_identity=[0, 0],
                process=[0, 0, 0], final=[0, 0, 0], final_event_windows=0,
            )
            for _ in range(10)
        ]
        frames = case.origins[:, None] + np.arange(1, 11)
        error = np.abs(prediction - case.labels["tg_h"][frames]).astype(np.float64) * 3600
        active = case.labels["active"][frames]
        target = case.shared["edge_index"][1]
        frozen = (frames[:, :, None] == case.labels["freeze_at"][target]) & (
            self.degree[target][None, None] > 1
        )
        for name, mask in (
            ("mae_s", active),
            ("change_mae_s", active & case.changed[frames]),
            ("final_tg_mae_s", frozen),
        ):
            for h, row in enumerate(self.leads):
                selected = error[:, h][mask[:, h]]
                row[name] = [float(selected.sum()), len(selected)]

    def group(self, task):
        return str(self.degree[task]) if self.degree[task] < 4 else ">=4"

    def identities(self, actual, predicted, timeline, origins):
        self.origins = origins
        frames = origins[:, None] + np.arange(1, 11)
        future = actual[frames]
        valid = timeline["eligible"]
        moving = valid[frames].all(1) & (
            (actual[frames - 1] != future) & valid[frames - 1]
        ).any(1)
        correct = predicted == future
        for name, row in self.groups.items():
            selection = self.degree >= 4 if name == ">=4" else self.degree == int(name)
            mask = moving & selection[None]
            row["window"] = [int((correct.all(1) & mask).sum()), int(mask.sum())]
        for h, row in enumerate(self.leads):
            row["window_identity"] = [int((correct[:, h] & moving).sum()), int(moving.sum())]
        event_mask = timeline["stable"] | timeline["optional"]
        self.final_times = np.where(event_mask, np.arange(len(actual))[:, None], -1).max(0)
        times, tasks = np.where(event_mask)
        final = times == self.final_times[tasks]
        lookup = {int(origin): row for row, origin in enumerate(origins)}
        for h, row in enumerate(self.leads, 1):
            available = np.array([int(at) - h in lookup for at in times], dtype=bool)
            at, task = times[available], tasks[available]
            rr = np.array([lookup[int(t) - h] for t in at], dtype=np.int64)
            hit = predicted[rr, h - 1, task] == actual[at, task]
            last = final[available]
            row["process_identity"] = [int(hit.sum()), len(hit)]
            row["final_identity"] = [int(hit[last].sum()), int(last.sum())]
            if h == 10:
                for name, cell in self.groups.items():
                    degree = self.degree[task]
                    selected = last & (degree >= 4 if name == ">=4" else degree == int(name))
                    cell["final_identity"] = [int(hit[selected].sum()), int(selected.sum())]

    def event(self, scope, delta, row, task, scored):
        family = "process" if scope == "full_process" else "final"
        counts = scored["with_optional_bonus"][:3]
        if delta in (0, 1):
            cell = self.groups[self.group(task)][family][str(delta)]
            for index, count in enumerate(counts):
                cell[index] += int(count)
        if delta != 0:
            return
        local = np.zeros((10, 3), dtype=np.int64)
        for predicted_time, reference_time in scored["matched_times"]:
            if predicted_time != reference_time:
                raise ValueError("Lead attribution requires same-frame matching")
            local[predicted_time, 0] += 1
        for at in scored["FP_times"]:
            local[at, 1] += 1
        for at in scored["FN_times"]:
            local[at, 2] += 1
        if local.sum(0).tolist() != list(counts):
            raise ValueError("Lead counts do not partition the event score")
        for h, cell in enumerate(self.leads):
            for index in range(3):
                cell[family][index] += int(local[h, index])
        if family == "final":
            lead = int(self.final_times[task] - self.origins[row] - 1)
            self.leads[lead]["final_event_windows"] += 1

    def finish(self):
        return dict(case=self.case.case_id, by_degree=self.groups, by_lead=self.leads)


def summarize_breakdown(rows, split, by, repetitions=20000, seed=20261026):
    if not rows or repetitions < 1 or seed < 0:
        raise ValueError("Invalid bootstrap settings")
    seed += {"validation": 0, "test": 1, "train": 2}[split]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(rows), size=(repetitions, len(rows)))
    weights = np.zeros((repetitions, len(rows)), dtype=np.int64)
    np.add.at(weights, (np.arange(repetitions)[:, None], index), 1)
    records = []

    def record(values, metric, kind="ratio", **labels):
        values = np.asarray(values, dtype=np.float64)
        details = {}
        if kind == "mean":
            valid = values[:, 1] > 0
            means = values[valid, 0] / values[valid, 1]
            denominator = weights[:, valid].sum(1)
            positive = denominator > 0
            draws = (weights[positive][:, valid] @ means) / denominator[positive]
            estimate = float(means.mean()) if len(means) else None
            details = dict(count=int(values[:, 1].sum()), contributing_plans=int(valid.sum()))
        else:
            total = values.sum(0)
            if kind == "f1":
                tp, fp, fn = total
                numerator = 2 * values[:, 0]
                denominator = 2 * values[:, 0] + values[:, 1] + values[:, 2]
                details = dict(tp=int(tp), fp=int(fp), fn=int(fn))
            else:
                numerator, denominator = values.T
                details = dict(correct=int(total[0]), count=int(total[1]))
            sampled_denominator = weights @ denominator
            positive = sampled_denominator > 0
            draws = 100 * (weights[positive] @ numerator) / sampled_denominator[positive]
            estimate = float(100 * numerator.sum() / denominator.sum()) if denominator.sum() else None
        lower, upper = np.quantile(draws, [0.025, 0.975]) if len(draws) else (None, None)
        records.append(dict(
            **labels, metric=metric, estimate=estimate,
            lower=float(lower) if lower is not None else None,
            upper=float(upper) if upper is not None else None,
            unit="s" if kind == "mean" else "%", valid_resamples=len(draws), **details,
        ))

    if by in ("degree", "all"):
        for group in ("2", "3", ">=4"):
            cells = [row["by_degree"][group] for row in rows]
            for name in ("window", "final_identity"):
                record([cell[name] for cell in cells], name + "_accuracy", group=group)
            for name in ("process", "final"):
                for delta in ("0", "1"):
                    record([cell[name][delta] for cell in cells], name + "_f1",
                           "f1", group=group, tolerance=int(delta))
    if by in ("lead", "all"):
        for h in range(10):
            cells = [row["by_lead"][h] for row in rows]
            labels = dict(lead=h + 1, forecast_hours=(h + 1) / 2)
            for name in ("mae_s", "change_mae_s", "final_tg_mae_s"):
                record([cell[name] for cell in cells], name, "mean", **labels)
            for name in ("window_identity", "process_identity", "final_identity"):
                record([cell[name] for cell in cells], name + "_accuracy", **labels)
            for name in ("process", "final"):
                record([cell[name] for cell in cells], name + "_f1", "f1", tolerance=0, **labels)
    return dict(
        records=records,
        confidence=dict(level=0.95, method="plan-cluster percentile bootstrap",
                        repetitions=repetitions, seed=seed, plans=len(rows)),
    )


@torch.inference_mode()
def evaluate_breakdown(
    predictor, root, split, batch_size, by="all", bootstrap_samples=20000, bootstrap_seed=20261026
):
    if batch_size < 1 or by not in ("degree", "lead", "all"):
        raise ValueError("Invalid analysis configuration")
    if predictor.model is not None:
        predictor.model.eval()
    rows = []
    for case_id in splits(root)[split]:
        case = Case(root, split, case_id)
        predictions = []
        for start in range(0, len(case.origins), batch_size):
            values, _ = predictor.forward(case, case.origins[start : start + batch_size])
            predictions.append(values.cpu().numpy())
        prediction = np.concatenate(predictions)
        breakdown = Breakdown(case, prediction)
        score_events_case(case, case.origins, prediction, analysis=breakdown)
        rows.append(breakdown.finish())
    return dict(
        split=split, model=predictor.method, ablation=predictor.mask, analysis=by,
        units=dict(tg_error="s", accuracy="%", f1="%"), by_case=rows,
        **summarize_breakdown(rows, split, by, bootstrap_samples, bootstrap_seed),
    )
