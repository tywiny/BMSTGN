from collections import Counter
import numpy as np

EPS_H = 1e-8
TOLERANCES = (0, 1, 2, 5)
SCOPES = ("full_process", "final_migration")
MODES = ("stable_local", "with_optional_bonus")
MARGIN_H = 5.0 / 60.0


def raw_bt(tg_h, sources, targets, node_count, previous=None):

    tg = np.asarray(tg_h, dtype=np.float64)
    squeeze = tg.ndim == 2
    if squeeze:
        tg = tg[None]
    if tg.ndim != 3 or not np.isfinite(tg).all():
        raise ValueError("TG must be finite [time,edges] or [batch,time,edges]")
    sources, targets = np.asarray(sources), np.asarray(targets)
    batch, length, _ = tg.shape
    out = np.full((batch, length, node_count), -1, dtype=np.int64)
    prior = (
        np.full((batch, node_count), -1, dtype=np.int64)
        if previous is None
        else np.asarray(previous, dtype=np.int64).reshape(batch, node_count).copy()
    )
    ties = Counter(tied_task_frames=0, unresolved_task_frames=0)
    for task in np.unique(targets):
        edges = np.flatnonzero(targets == task)
        src = sources[edges]
        values = tg[..., edges]
        at_min = values <= values.min(-1, keepdims=True) + EPS_H
        count = at_min.sum(-1)
        unique = count == 1
        unique_value = src[at_min.argmax(-1)]
        state = prior[:, task].copy()
        for t in range(length):
            retained = ((state[:, None] == src[None]) & at_min[:, t]).any(-1)
            state = np.where(unique[:, t], unique_value[:, t], np.where(retained, state, -1))
            out[:, t, task] = state
        if len(edges) >= 2:
            ties["tied_task_frames"] += int((~unique).sum())
            ties["unresolved_task_frames"] += int((out[:, :, task] < 0).sum())
    return (out[0] if squeeze else out), dict(ties)


def truth_timeline(bt, eligible, multi):

    bt = np.asarray(bt, dtype=np.int64)
    valid = np.asarray(eligible, dtype=bool) & (bt >= 0) & np.asarray(multi, dtype=bool)[None]
    if valid.shape != bt.shape:
        raise ValueError("truth/eligibility shape mismatch")
    masks = {
        key: np.zeros_like(valid)
        for key in ("raw", "stable", "optional", "fleeting", "returned", "censored")
    }
    for task in np.flatnonzero(multi):
        anchor = -1
        for t in range(len(bt)):
            if not valid[t, task]:
                anchor = -1
                continue
            if t == 0 or not valid[t - 1, task]:
                anchor = int(bt[t, task])
                continue
            if bt[t, task] == bt[t - 1, task]:
                continue
            masks["raw"][t, task] = True
            if t + 1 == len(bt) or not valid[t + 1, task]:
                masks["censored"][t, task] = True
                continue
            if bt[t + 1, task] != bt[t, task]:
                masks["fleeting"][t, task] = True
                masks["optional"][t, task] = True
            elif bt[t, task] == anchor:
                masks["returned"][t, task] = True
                masks["optional"][t, task] = True
            else:
                masks["stable"][t, task] = True
                anchor = int(bt[t, task])
    assert np.array_equal(masks["raw"], masks["stable"] | masks["optional"] | masks["censored"])
    assert not np.any(masks["stable"] & masks["optional"])
    masks["eligible"] = valid
    masks["event_eligible"] = valid & ~masks["censored"]
    return masks


def events(mask, values):
    return [(int(t), int(values[t])) for t in np.flatnonzero(mask)]


def empty_counts():

    return Counter(
        tp=0, fp=0, fn=0, matched=0, early=0, exact=0, late=0, signed_steps=0, absolute_steps=0
    )


def add_counts(counts, result):
    tp, fp, fn, offsets = result
    counts.update(
        tp=tp,
        fp=fp,
        fn=fn,
        matched=len(offsets),
        early=sum(v < 0 for v in offsets),
        exact=sum(v == 0 for v in offsets),
        late=sum(v > 0 for v in offsets),
        signed_steps=sum(offsets),
        absolute_steps=sum(abs(v) for v in offsets),
    )


def finish_counts(values):
    out = dict(values)
    tp, fp, fn = (out[k] for k in ("tp", "fp", "fn"))
    out.update(
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        precision_denominator=tp + fp,
        recall_denominator=tp + fn,
        mean_signed_step=out["signed_steps"] / max(tp, 1),
        mean_absolute_step=out["absolute_steps"] / max(tp, 1),
        has_scoring_events=bool(tp + fp + fn),
    )
    out["mean_signed_minutes"] = out["mean_signed_step"] * 30
    out["mean_absolute_minutes"] = out["mean_absolute_step"] * 30
    return out


def membership(values_h):
    values = np.asarray(values_h, np.float64)
    return values <= values.min(-1, keepdims=True) + MARGIN_H + EPS_H


def bidirectional_membership(true_h, predicted_h, actual, sources):

    predicted_h = np.asarray(predicted_h, np.float64)
    exact = np.asarray(actual)[..., None] == np.asarray(sources)
    cols = exact.argmax(-1)
    true_source_prediction = np.take_along_axis(predicted_h, cols[..., None], axis=-1)
    reverse_gap = np.abs(predicted_h - true_source_prediction)
    prediction_close = exact | (reverse_gap <= MARGIN_H + EPS_H)
    return membership(true_h) & prediction_close & exact.any(-1, keepdims=True)


def scoring_questions(questions, actual, predicted, eligible, allowed, joint, sources):

    cols = {int(s): j for j, s in enumerate(sources)}
    selected = np.array([cols.get(int(s), 0) for s in predicted])
    known = np.array([int(s) in cols for s in predicted])
    row = np.arange(len(predicted))
    violations = np.asarray(eligible, bool) & known & allowed[row, selected] & ~joint[row, selected]
    out = []
    for original in questions:
        q = dict(original, prediction_guard_required=False)
        if not q["required"] and violations[q["time"] : q["run_end"] + 1].any():
            q.update(
                required=True,
                prediction_guard_required=True,
                scoring_bound=q["run_end"],
                intervals={},
                old_side=[q["old_source"]],
            )
            for j, source in enumerate(sources):
                if int(source) == q["old_source"] or not allowed[q["time"] : q["run_end"] + 1, j].all():
                    continue
                left = q["time"]
                while (
                    left > 0
                    and eligible[left - 1]
                    and actual[left - 1] == q["old_source"]
                    and allowed[left - 1, j]
                ):
                    left -= 1
                q["intervals"][str(int(source))] = [int(left), int(q["run_end"])]
            assert str(q["new_source"]) in q["intervals"]
        out.append(q)
    return out, violations


def truth_questions(actual, previous_true, stable, allowed, origin_allowed, eligible, sources):

    actual, eligible = np.asarray(actual), np.asarray(eligible, bool)
    cols = {int(s): i for i, s in enumerate(sources)}
    questions = []
    h = len(actual)
    for at, new_source in stable:
        old = int(previous_true if at == 0 else actual[at - 1])
        old_col = cols[old]
        end = at
        while end + 1 < h and eligible[end + 1] and actual[end + 1] == new_source:
            end += 1
        clear = next((t for t in range(at, end + 1) if not allowed[t, old_col]), None)
        q = dict(
            time=int(at),
            old_source=old,
            new_source=int(new_source),
            run_end=int(end),
            clarity=None if clear is None else int(clear),
            required=clear is not None,
            intervals={},
            old_side=[],
        )
        if clear is not None:
            before_set = origin_allowed if at == 0 else allowed[at - 1]
            q["old_side"] = [
                int(s) for j, s in enumerate(sources) if before_set[j] and not allowed[clear, j]
            ]
            for col, source in enumerate(sources):
                if not bool(allowed[at : clear + 1, col].all()):
                    continue
                left = at
                while (
                    left > 0 and eligible[left - 1] and actual[left - 1] == old and allowed[left - 1, col]
                ):
                    left -= 1
                q["intervals"][str(int(source))] = [int(left), int(clear)]
            assert str(int(new_source)) in q["intervals"] and old in q["old_side"]
            assert str(old) not in q["intervals"]
        questions.append(q)
    return questions


def match_pairs(predicted, questions, compatible):

    choices = [
        [(ai, abs(pt - q["time"])) for ai, q in enumerate(questions) if compatible(pi, q)]
        for pi, (pt, ps) in enumerate(predicted)
    ]
    memo = {}

    def solve(pi, used):
        if (pi, used) in memo:
            return memo[pi, used]
        if pi == len(predicted):
            return 0, 0, ()
        sub = solve(pi + 1, used)
        answer = sub[0], sub[1], (-1,) + sub[2]
        for ai, distance in sorted(choices[pi], key=lambda v: (v[1], v[0])):
            if used & (1 << ai):
                continue
            sub = solve(pi + 1, used | (1 << ai))
            candidate = sub[0] + 1, sub[1] + distance, (ai,) + sub[2]
            if (candidate[0], -candidate[1]) > (answer[0], -answer[1]):
                answer = candidate
        memo[pi, used] = answer
        return answer

    return [(pi, ai) for pi, ai in enumerate(solve(0, 0)[2]) if ai >= 0]


def alarm_neutrality(predicted, previous, eligible, allowed, sources, alarms):

    cols = {int(s): i for i, s in enumerate(sources)}
    before = np.concatenate(([int(previous)], np.asarray(predicted)[:-1]))
    out = {}
    for t, source in alarms:
        old = int(before[t])
        end = t
        while end + 1 < len(predicted) and eligible[end + 1] and predicted[end + 1] == source:
            end += 1
        departure_ok = (
            t == 0
            or not eligible[t - 1]
            or (old in cols and (allowed[t - 1, cols[old]] or allowed[t, cols[old]]))
        )
        ok = departure_ok and source in cols and bool(allowed[t : end + 1, cols[source]].all())
        out[int(t)] = dict(neutral=bool(ok), old_source=old, run_end=int(end))
    return out


def missed_event_evidence(predicted, eligible, allowed, sources, questions):

    cols = {int(s): j for j, s in enumerate(sources)}
    good = np.array([s in cols and allowed[t, cols[int(s)]] for t, s in enumerate(predicted)])
    evidence = {}
    for q in questions:
        if not q["required"]:
            continue
        start = max(0, min(v[0] for v in q["intervals"].values()) - 1)
        end = q["run_end"]
        mask = np.asarray(eligible[start : end + 1], bool)
        accepted = good[start : end + 1][mask]
        evidence[q["time"]] = dict(
            neutral=bool(len(accepted) and accepted.all()),
            interval=[int(start), int(end)],
            eligible_frames=int(mask.sum()),
            outside_set_frames=int((~accepted).sum()),
        )
    return evidence


def duration_context(actual, predicted, eligible, allowed, joint, sources, stable):
    selected = np.asarray(predicted)[:, None] == np.asarray(sources)[None]
    valid = np.asarray(eligible, bool)
    return dict(
        actual=np.asarray(actual),
        valid=valid,
        true_bad=valid & ~(selected & allowed).any(-1),
        joint_bad=valid & ~(selected & joint).any(-1),
        stable_times=[int(t) for t, s in stable],
    )


def source_drift_evidence(pt, q, run_end, context):

    if context is None:
        return None
    start, end = max(pt, q["time"]), min(run_end, q["run_end"])
    bad = np.flatnonzero(context["true_bad"][start : end + 1]) + start
    if len(bad):
        return dict(
            reason="near_source_left_true_tolerance",
            interval=[int(start), int(end)],
            bad_times=bad.tolist(),
        )
    return None


def optional_duration_evidence(pt, q, run_end, delta, context):

    if context is None:
        return None
    at = q["time"]
    actual, valid = context["actual"], context["valid"]
    truth_end = at
    while (
        truth_end + 1 < len(actual) and valid[truth_end + 1] and actual[truth_end + 1] == q["new_source"]
    ):
        truth_end += 1

    start = max(pt, truth_end + 1 + delta)
    next_stable = next((t for t in context["stable_times"] if t > at), len(actual))
    end = min(run_end, next_stable - 1)
    for t in range(start, end):
        if context["joint_bad"][t] and context["joint_bad"][t + 1] and actual[t] == actual[t + 1]:
            return dict(
                reason="optional_excursion_held_after_return",
                truth_run_end=int(truth_end),
                interval=[int(start), int(end)],
                bad_times=[int(t), int(t + 1)],
                next_stable_time=int(next_stable),
            )
    return None


def same_source_phase(context, at):
    actual, valid = context["actual"], context["valid"]
    left = right = int(at)
    while left > 0 and valid[left - 1] and actual[left - 1] == actual[at]:
        left -= 1
    while right + 1 < len(actual) and valid[right + 1] and actual[right + 1] == actual[at]:
        right += 1
    return left, right


def related_interval(at, source, delta, neutrality, questions, context):
    candidates = []
    for q in questions:
        if not q["required"] or q.get("prediction_guard_required", False):
            continue
        interval = q["intervals"].get(str(source))
        if interval is None or neutrality[at]["old_source"] not in q["old_side"]:
            continue
        if interval[0] - delta <= at <= interval[1] + delta:
            candidates.append(q)
    if not candidates:
        return same_source_phase(context, at), None
    q = min(candidates, key=lambda q: (abs(at - q["time"]), q["time"]))
    if q["time"] == 0:
        left = 0
    else:
        left, _ = same_source_phase(context, q["time"] - 1)
    return (left, q["run_end"]), q["time"]


def transition_neutrality(alarms, questions, neutrality, delta, context):
    local = {at: dict(proof) for at, proof in neutrality.items()}
    for at, source in alarms:
        proof = local[at]
        if proof["neutral"]:
            (left, right), _ = related_interval(at, source, delta, local, questions, context)
            valid = context["valid"][left : right + 1]
            bad = context["joint_bad"][left : right + 1]
            proof["neutral"] = not bool((bad & valid).any())
    return local


def score_task(
    alarms,
    questions,
    optional,
    neutrality,
    delta,
    missed_evidence=None,
    violations=None,
    state_context=None,
    exact=True,
):
    if exact:
        neutrality = transition_neutrality(alarms, questions, neutrality, delta, state_context)
    required = [q for q in questions if q["required"] and not q.get("prediction_guard_required", False)]
    guarded = [q for q in questions if q.get("prediction_guard_required", False)]
    rejected = {}

    def reject(kind, pt, q, proof):
        if proof is not None:
            rejected[(kind, pt, q["time"])] = dict(
                kind=kind, predicted_time=int(pt), true_time=int(q["time"]), **proof
            )
            return True
        return False

    def compatible(pi, q):
        pt, source = alarms[pi]
        if exact and not q.get("prediction_guard_required", False) and abs(pt - q["time"]) > delta:
            return False
        interval = q["intervals"].get(str(source))
        if interval is None or neutrality[pt]["old_source"] not in q["old_side"]:
            return False
        if not interval[0] - delta <= pt <= interval[1] + delta:
            return False

        if source == q["new_source"] and abs(pt - q["time"]) <= delta:
            return True
        if not q.get("prediction_guard_required", False) and reject(
            "stable", pt, q, source_drift_evidence(pt, q, neutrality[pt]["run_end"], state_context)
        ):
            return False
        start = min(pt, q["time"])
        end = max(neutrality[pt]["run_end"], q["time"])
        return violations is None or not np.asarray(violations)[start : end + 1].any()

    pairs = match_pairs(alarms, required, compatible)
    used_p, used_q = {pi for pi, ai in pairs}, {ai for pi, ai in pairs}
    remaining = [i for i in range(len(alarms)) if i not in used_p]
    bonus_q = [dict(time=t, new_source=s) for t, s in optional]
    rest = [alarms[i] for i in remaining]

    def optional_compatible(pi, q):
        pt, source = rest[pi]
        if source != q["new_source"] or abs(pt - q["time"]) > delta:
            return False
        return not reject(
            "optional",
            pt,
            q,
            optional_duration_evidence(pt, q, neutrality[pt]["run_end"], delta, state_context),
        )

    bonus_pairs = match_pairs(rest, bonus_q, optional_compatible)
    used_p.update(remaining[pi] for pi, ai in bonus_pairs)

    guard_remaining = [i for i in range(len(alarms)) if i not in used_p]
    guard_pairs_local = match_pairs(
        [alarms[i] for i in guard_remaining], guarded, lambda pi, q: compatible(guard_remaining[pi], q)
    )
    guard_pairs = [(guard_remaining[pi], qi) for pi, qi in guard_pairs_local]
    used_p.update(pi for pi, qi in guard_pairs)
    used_guard = {qi for pi, qi in guard_pairs}
    unused = [i for i in range(len(alarms)) if i not in used_p]
    neutral = [i for i in unused if neutrality[alarms[i][0]]["neutral"]]
    fp = len(unused) - len(neutral)
    missed_evidence = missed_evidence or {}
    neutral_q = [
        ai
        for ai, q in enumerate(required)
        if ai not in used_q and missed_evidence.get(q["time"], {}).get("neutral", False)
    ]
    guard_fn = len(guarded) - len(used_guard)
    fn = len(required) - len(used_q) - len(neutral_q) + guard_fn
    stable_offsets = [alarms[pi][0] - required[ai]["time"] for pi, ai in pairs]
    bonus_offsets = [rest[pi][0] - bonus_q[ai]["time"] for pi, ai in bonus_pairs]
    stable_tp, optional_tp = len(pairs), len(bonus_pairs)
    assert stable_tp + fn + len(neutral_q) + len(guard_pairs) == len(required) + len(guarded)
    assert stable_tp + optional_tp + fp + len(neutral) + len(guard_pairs) == len(alarms)
    return dict(
        stable_local=(stable_tp, fp, fn, stable_offsets),
        with_optional_bonus=(stable_tp + optional_tp, fp, fn, stable_offsets + bonus_offsets),
        stable_tp=stable_tp,
        optional_tp=optional_tp,
        neutral_alarms=len(neutral),
        neutral_FN=len(neutral_q),
        required_stable=len(required) + len(guarded),
        ambiguous_stable=len(questions) - len(required) - len(guarded),
        prediction_guard_required=len(guarded),
        prediction_guard_matches=len(guard_pairs),
        prediction_guard_FN=guard_fn,
        optional_truth=len(optional),
        optional_missed=len(optional) - optional_tp,
        predicted=len(alarms),
        duration_guard_rejected_stable_candidates=sum(k[0] == "stable" for k in rejected),
        duration_guard_rejected_optional_candidates=sum(k[0] == "optional" for k in rejected),
        neutral_alarm_times=[alarms[i][0] for i in neutral],
        FP_times=[alarms[i][0] for i in unused if i not in neutral],
        FN_times=(
            [q["time"] for ai, q in enumerate(required) if ai not in used_q and ai not in neutral_q]
            + [q["time"] for ai, q in enumerate(guarded) if ai not in used_guard]
        ),
        neutral_FN_evidence=[
            dict(time=required[ai]["time"], **missed_evidence[required[ai]["time"]]) for ai in neutral_q
        ],
    )


class Accumulator:
    def __init__(self):
        self.windows = 0
        self.events = Counter()
        self.identity = Counter()
        self.counts = {s: {m: {t: empty_counts() for t in TOLERANCES} for m in MODES} for s in SCOPES}
        self.details = {s: {t: Counter() for t in TOLERANCES} for s in SCOPES}

    def merge(self, other):
        self.windows += other.windows
        self.events.update(other.events)
        self.identity.update(other.identity)
        for s in SCOPES:
            for t in TOLERANCES:
                self.details[s][t].update(other.details[s][t])
                for m in MODES:
                    self.counts[s][m][t].update(other.counts[s][m][t])

    def finish(self):
        for s in SCOPES:
            for t in TOLERANCES:
                d = self.details[s][t]
                for m in MODES:
                    c = self.counts[s][m][t]
                    assert (
                        c["fn"] + d["stable_tp"] + d["neutral_FN"] + d["prediction_guard_matches"]
                        == d["required_stable"]
                    )
                    assert c["tp"] == d["stable_tp"] + (
                        d["optional_tp"] if m == "with_optional_bonus" else 0
                    )
                    assert (
                        c["fp"]
                        + d["stable_tp"]
                        + d["optional_tp"]
                        + d["neutral_alarms"]
                        + d["prediction_guard_matches"]
                        == d["predicted"]
                    )
        identity = dict(self.identity)
        for k in ("raw_correct", "acceptable_correct"):
            identity[k.replace("correct", "accuracy")] = identity.get(k, 0) / max(
                identity.get("count", 0), 1
            )
        return {
            s: {m: {str(t): finish_counts(self.counts[s][m][t]) for t in TOLERANCES} for m in MODES}
            for s in SCOPES
        }


def score_events_case(case, origins, prediction, paper=True):
    labels = case.labels
    src, dst = case.shared["edge_index"]
    n = case.shared["plan_h"].shape[0]
    multi = np.bincount(dst, minlength=n) > 1
    ctx = dict(source=src, target=dst, multi=multi)
    true_tg = labels["tg_h"].astype(np.float64)
    actual, _ = raw_bt(true_tg, src, dst, n)
    timeline = truth_timeline(actual, labels["bt_eligible"], multi)
    observed = labels["bt_start"][np.searchsorted(case.origins, origins)]
    p, _ = raw_bt(prediction, src, dst, n, observed)
    frames = origins[:, None] + np.arange(1, p.shape[1] + 1)[None]
    a, tg = actual[frames], true_tg[frames]
    masks = {k: v[frames] for k, v in timeline.items()}
    before = np.concatenate((observed[:, None], p[:, :-1]), axis=1)
    pmask = masks["event_eligible"] & (before >= 0) & (p >= 0) & (before != p)
    final_events = {}
    if paper:
        for task in np.flatnonzero(multi):
            positions = np.flatnonzero(timeline["stable"][:, task] | timeline["optional"][:, task])
            if len(positions):
                final_events[task] = positions[-1]
    relevant = np.any(pmask | masks["stable"] | masks["optional"], axis=1)
    acc = Accumulator()
    acc.windows = len(origins)
    for kind in ("raw", "stable", "optional", "fleeting", "returned", "censored"):
        acc.events[kind] = int(masks[kind].sum())
    acc.events["predicted"] = int(pmask.sum())
    acc.events["ignored_censored_alarms"] = int(
        ((before >= 0) & (p >= 0) & (before != p) & masks["censored"]).sum()
    )

    for task in np.flatnonzero(ctx["multi"]):
        edges = np.flatnonzero(ctx["target"] == task)
        sources = ctx["source"][edges]
        allowed = membership(tg[:, :, edges])
        joint = bidirectional_membership(tg[:, :, edges], prediction[:, :, edges], a[:, :, task], sources)
        source_equal = p[:, :, task, None] == sources[None, None]
        true_good = (source_equal & allowed).any(-1)
        good = (source_equal & joint).any(-1)
        valid = masks["eligible"][:, :, task]
        acc.identity.update(
            count=int(valid.sum()),
            raw_correct=int(((p[:, :, task] == a[:, :, task]) & valid).sum()),
            acceptable_correct=int((good & valid).sum()),
            outside_set_frames=int((~good & valid).sum()),
            true_acceptable_correct=int((true_good & valid).sum()),
            outside_true_set_frames=int((~true_good & valid).sum()),
            prediction_gap_rejected_frames=int((true_good & ~good & valid).sum()),
        )
    for row, task in zip(*np.where(relevant)):
        edges = np.flatnonzero(ctx["target"] == task)
        sources = ctx["source"][edges]
        allowed = membership(tg[row][:, edges])
        origin_allowed = membership(true_tg[origins[row], edges])
        valid = masks["eligible"][row, :, task]
        alarms = events(pmask[row, :, task], p[row, :, task])
        stable = events(masks["stable"][row, :, task], a[row, :, task])
        optional = events(masks["optional"][row, :, task], a[row, :, task])
        questions = truth_questions(
            a[row, :, task], actual[origins[row], task], stable, allowed, origin_allowed, valid, sources
        )
        joint = bidirectional_membership(
            tg[row][:, edges], prediction[row][:, edges], a[row, :, task], sources
        )
        questions, violations = scoring_questions(
            questions, a[row, :, task], p[row, :, task], valid, allowed, joint, sources
        )
        neutrality = alarm_neutrality(p[row, :, task], observed[row, task], valid, joint, sources, alarms)
        missed_evidence = missed_event_evidence(p[row, :, task], valid, joint, sources, questions)
        state_context = duration_context(
            a[row, :, task], p[row, :, task], valid, allowed, joint, sources, stable
        )
        if paper:
            final_at = final_events.get(task)
            scopes = [("full_process", alarms, questions, optional)]
            if final_at is not None and origins[row] < final_at <= frames[row, -1]:
                step = final_at - origins[row] - 1
                scopes.append(
                    (
                        "final_migration",
                        alarms[-1:],
                        [q for q in questions if q["time"] == step],
                        [(t, s) for t, s in optional if t == step],
                    )
                )
        else:
            last_optional = (
                optional[-1:] if optional and (not stable or optional[-1][0] > stable[-1][0]) else []
            )
            scopes = [
                ("full_process", alarms, questions, optional),
                ("final_migration", alarms[-1:], questions[-1:], last_optional),
            ]
        for scope, aa, qq, oo in scopes:
            for delta in TOLERANCES:
                scored = score_task(
                    aa, qq, oo, neutrality, delta, missed_evidence, violations, state_context, exact=paper
                )
                for mode in MODES:
                    add_counts(acc.counts[scope][mode][delta], scored[mode])
                acc.details[scope][delta].update(
                    {
                        k: scored[k]
                        for k in (
                            "stable_tp",
                            "optional_tp",
                            "neutral_alarms",
                            "neutral_FN",
                            "required_stable",
                            "ambiguous_stable",
                            "prediction_guard_required",
                            "prediction_guard_matches",
                            "prediction_guard_FN",
                            "optional_truth",
                            "optional_missed",
                            "predicted",
                            "duration_guard_rejected_stable_candidates",
                            "duration_guard_rejected_optional_candidates",
                        )
                    }
                )

    previous_truth = np.concatenate((actual[origins, None], a[:, :-1]), axis=1)
    before_valid = np.concatenate(
        (timeline["eligible"][origins, None], masks["eligible"][:, :-1]), axis=1
    )
    moving = masks["eligible"].all(1) & ((previous_truth != a) & before_valid).any(1)
    acc.identity.update(
        window_count=int(moving.sum()), window_correct=int(((p == a).all(1) & moving).sum())
    )
    valid = timeline["eligible"]
    lookup = {int(o): i for i, o in enumerate(origins)}
    for task in np.flatnonzero(multi):
        changes = (
            np.flatnonzero(valid[1:, task] & valid[:-1, task] & (actual[1:, task] != actual[:-1, task]))
            + 1
        )
        if not len(changes):
            continue
        tau = int(changes[-1])
        final = int(labels["final_parent"][task])
        if final < 0 or tau - 10 not in lookup or tau + 1 >= len(actual) or not valid[tau + 1, task]:
            continue
        if actual[tau + 1, task] != final or ((actual[tau:, task] != final) & valid[tau:, task]).any():
            continue
        acc.identity.update(final_count=1, final_correct=int(p[lookup[tau - 10], 9, task] == final))
    return acc
