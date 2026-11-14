from collections import deque
import numpy as np
import torch
from data import Case, splits
from metrics import raw_bt


class PathEncoder:
    def __init__(self, incoming):
        self.incoming = incoming
        self.n = len(incoming)
        children = [[] for _ in incoming]
        degree = np.array([len(parents) for parents in incoming])
        remaining = degree.copy()
        for task, parents in enumerate(incoming):
            for parent in parents:
                children[parent].append(task)
        queue = deque(np.flatnonzero(remaining == 0).tolist())
        self.order = []
        self.competition = np.zeros(self.n, dtype=bool)
        while queue:
            task = queue.popleft()
            self.order.append(task)
            self.competition[task] = degree[task] >= 2 or any(
                self.competition[parent] for parent in incoming[task]
            )
            for child in children[task]:
                remaining[child] -= 1
                if not remaining[child]:
                    queue.append(child)
        if len(self.order) != self.n:
            raise ValueError("Dependency graph contains a cycle")
        self.terminal = np.array([not children[task] for task in range(self.n)])
        self.maps = [{} for _ in incoming]
        self.parts = [(None, None)]
        self.roots = {}
        for task in self.order:
            if not incoming[task]:
                self.roots[task] = len(self.parts)
                self.parts.append((task, 0))

    def encode(self, states):
        shape = states.shape
        flat = np.asarray(states).reshape(-1, self.n)
        result = np.zeros_like(flat, dtype=np.int32)
        rows = np.arange(len(flat))
        for task in self.order:
            parents = self.incoming[task]
            if not parents:
                result[:, task] = self.roots[task]
                continue
            if len(parents) == 1:
                parent_ids = result[:, parents[0]]
            else:
                parent = flat[:, task]
                valid = np.isin(parent, parents)
                parent_ids = np.zeros(len(flat), dtype=np.int32)
                parent_ids[valid] = result[rows[valid], parent[valid]]
            unique, inverse = np.unique(parent_ids, return_inverse=True)
            mapped = np.zeros(len(unique), dtype=np.int32)
            mapping = self.maps[task]
            for index, identity in enumerate(unique):
                if identity:
                    key = int(identity)
                    if key not in mapping:
                        mapping[key] = len(self.parts)
                        self.parts.append((task, key))
                    mapped[index] = mapping[key]
            result[:, task] = mapped[inverse]
        return result.reshape(shape)

    def path(self, identity):
        nodes = []
        while identity:
            task, identity = self.parts[int(identity)]
            nodes.append(task)
        return nodes


def path_changes(sequence):
    return (
        (sequence[:, 1:] > 0)
        & (sequence[:, :-1] > 0)
        & (sequence[:, 1:] != sequence[:, :-1])
    )


class PathEvaluation:
    def __init__(self, case):
        self.origins = case.origins
        self.source, self.target = case.shared["edge_index"]
        self.nodes = len(case.shared["plan_h"])
        incoming = [self.source[self.target == task].tolist() for task in range(self.nodes)]
        self.encoder = PathEncoder(incoming)
        actual, _ = raw_bt(case.labels["tg_h"], self.source, self.target, self.nodes)
        final_ids = self.encoder.encode(case.labels["final_parent"])
        reference = self.encoder.encode(actual)
        self.observed_bt = case.labels["bt_start"]
        self.observed_ids = self.encoder.encode(self.observed_bt)
        lengths = np.array([len(self.encoder.path(identity)) for identity in final_ids])
        candidates = self.encoder.competition & ~self.encoder.terminal & (lengths >= 20)
        migrations = (
            (reference[1:] > 0)
            & (reference[:-1] > 0)
            & (reference[1:] != reference[:-1])
        )
        self.targets = np.flatnonzero(candidates & migrations.any(0))
        frames = self.origins[:, None] + np.arange(11)
        self.reference = reference[frames]
        previous, future = actual[frames[:, :-1]], actual[frames[:, 1:]]
        global_change = (
            (previous >= 0) & (future >= 0) & (previous != future)
        ).any(axis=(1, 2))
        complete = (self.reference[:, 1:, self.targets] > 0).all(1)
        self.global_mask = complete & global_change[:, None]
        self.change_mask = self.global_mask & path_changes(
            self.reference[:, :, self.targets]
        ).any(1)
        lookup = {int(origin): row for row, origin in enumerate(self.origins)}
        self.events = []
        for task in self.targets:
            changes = np.flatnonzero(migrations[:, task]) + 1
            tau = int(changes[-1])
            if not np.all(reference[tau:, task] == final_ids[task]):
                continue
            origin = tau - 10
            if origin not in lookup:
                continue
            complete = (self.reference[:, :, task] > 0).all(1)
            if not complete[lookup[origin]]:
                continue
            rows = np.flatnonzero(
                (self.origins < tau) & (self.origins + 10 >= tau) & complete
            )
            self.events.append(
                dict(
                    task=int(task),
                    tau=tau,
                    before=int(reference[tau - 1, task]),
                    after=int(final_ids[task]),
                    rows=rows,
                    identity_row=lookup[origin],
                )
            )
        self.cohort = dict(
            candidate_targets=int(candidates.sum()),
            targets=len(self.targets),
            final_targets=len(self.events),
            global_change_windows=int(self.global_mask.sum()),
            path_change_windows=int(self.change_mask.sum()),
            migration_windows=sum(len(event["rows"]) for event in self.events),
        )

    def score(self, prediction):
        if prediction.shape != (len(self.origins), 10, len(self.source)):
            raise ValueError("Expected predictions for every registered origin in order")
        predicted_bt, _ = raw_bt(
            prediction, self.source, self.target, self.nodes, self.observed_bt
        )
        predicted = self.encoder.encode(predicted_bt)
        sequence = np.concatenate((self.observed_ids[:, None], predicted), axis=1)
        exact = (
            (predicted[:, :, self.targets] == self.reference[:, 1:, self.targets])
            & (predicted[:, :, self.targets] > 0)
        ).all(1)
        last = np.where(path_changes(sequence), np.arange(1, 11)[None, :, None], 0).max(1)
        migration = {
            str(delta): dict(tp=0, fp=0, fn=0, evaluation_windows=0) for delta in (0, 1)
        }
        identity_correct = 0
        for event in self.events:
            task, rows = event["task"], event["rows"]
            alarm = last[rows, task]
            has_alarm = alarm > 0
            before = sequence[rows, np.maximum(alarm - 1, 0), task]
            after = sequence[rows, alarm, task]
            identities = (before == event["before"]) & (after == event["after"])
            for delta in (0, 1):
                hit = (
                    has_alarm
                    & identities
                    & (np.abs(self.origins[rows] + alarm - event["tau"]) <= delta)
                )
                counts = migration[str(delta)]
                counts["tp"] += int(hit.sum())
                counts["fp"] += int((has_alarm & ~hit).sum())
                counts["fn"] += int((~hit).sum())
                counts["evaluation_windows"] += len(rows)
            identity_correct += int(
                sequence[event["identity_row"], 10, task] == event["after"]
            )
        return dict(
            global_change_correct=int((exact & self.global_mask).sum()),
            global_change_count=self.cohort["global_change_windows"],
            path_change_correct=int((exact & self.change_mask).sum()),
            path_change_count=self.cohort["path_change_windows"],
            final_path_correct=identity_correct,
            final_path_count=len(self.events),
            migration=migration,
        )


def summarize_paths(rows, split, repetitions, seed):
    seed += {"validation": 0, "test": 1, "train": 2}[split]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(repetitions, len(rows)))
    weights = np.zeros((repetitions, len(rows)), dtype=np.int64)
    np.add.at(weights, (np.arange(repetitions)[:, None], indices), 1)
    pairs = {}
    for name in ("global_change", "path_change", "final_path"):
        pairs[name + "_accuracy"] = np.array(
            [[row[name + "_correct"], row[name + "_count"]] for row in rows]
        )
    for delta in ("0", "1"):
        tp, fp, fn = np.array(
            [[row["migration"][delta][key] for key in ("tp", "fp", "fn")] for row in rows]
        ).T
        pairs["final_path_f1_d" + delta] = np.column_stack((2 * tp, 2 * tp + fp + fn))
    estimates, intervals = {}, {}
    for name, counts in pairs.items():
        numerator, denominator = counts.sum(0)
        estimates[name] = float(100 * numerator / denominator) if denominator else None
        resampled = weights @ counts
        valid = resampled[:, 1] > 0
        draws = 100 * resampled[valid, 0] / resampled[valid, 1]
        lower, upper = np.quantile(draws, [0.025, 0.975]) if len(draws) else (None, None)
        intervals[name] = dict(
            estimate=estimates[name],
            lower=float(lower) if lower is not None else None,
            upper=float(upper) if upper is not None else None,
            unit="%",
            valid_resamples=int(valid.sum()),
        )
    return dict(
        **estimates,
        confidence=dict(
            level=0.95,
            method="plan-cluster percentile bootstrap",
            repetitions=repetitions,
            seed=seed,
            plans=len(rows),
        ),
        confidence_intervals=intervals,
    )


@torch.inference_mode()
def evaluate_paths(
    predictor, root, split, batch_size, bootstrap_samples=20000, bootstrap_seed=20261026
):
    if batch_size < 1 or bootstrap_samples < 1 or bootstrap_seed < 0:
        raise ValueError("Invalid batch size or bootstrap settings")
    if predictor is not None and predictor.model is not None:
        predictor.model.eval()
    rows = []
    for case_id in splits(root)[split]:
        case = Case(root, split, case_id)
        context = PathEvaluation(case)
        row = dict(case=case_id, windows=len(case.origins), cohort=context.cohort)
        if predictor is not None:
            predictions = []
            for start in range(0, len(case.origins), batch_size):
                origins = case.origins[start : start + batch_size]
                values, _ = predictor.forward(case, origins)
                predictions.append(values.cpu().numpy())
            row.update(context.score(np.concatenate(predictions)))
        rows.append(row)
    report = dict(
        split=split,
        cases=len(rows),
        windows=sum(row["windows"] for row in rows),
        selection=dict(terminal=False, minimum_final_path_nodes=20, reference_path_changes=True),
        cohort={key: sum(row["cohort"][key] for row in rows) for key in rows[0]["cohort"]},
        by_case=rows,
    )
    if predictor is not None:
        report.update(
            model=predictor.method,
            ablation=predictor.mask,
            units=dict(accuracy="%", f1="%"),
            **summarize_paths(rows, split, bootstrap_samples, bootstrap_seed),
        )
    return report
