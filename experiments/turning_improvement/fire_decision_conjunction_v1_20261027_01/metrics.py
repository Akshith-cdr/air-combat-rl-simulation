"""Environment-free fixed-threshold monitoring and unchanged qualification screen."""

import math
import statistics

from .model import probability


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    def q(p):
        position = (len(values) - 1) * p
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (position - low) * (values[high] - values[low])
    return dict(zip(("min", "p10", "median", "p90", "max"), (q(p) for p in (0., .1, .5, .9, 1.))))


def evaluate(rows, logits):
    active = [r for r in rows if r["fit_mask"]]
    if not active or len(active) != len(logits) or any(not math.isfinite(z) for z in logits):
        raise ValueError("Finite score for every actionable row required")
    if any(type(r["label"]) is not int or r["label"] not in (0, 1) for r in active):
        raise ValueError("Binary labels required")
    probs = [probability(z) for z in logits]
    decisions = [p >= .5 for p in probs]
    predictions = {(r["episode_id"], r["step"]): d for r, d in zip(active, decisions)}
    scores = {(r["episode_id"], r["step"]): z for r, z in zip(active, logits)}
    if len(predictions) != len(active):
        raise ValueError("Duplicate row identity")
    positive = [z for r, z in zip(active, logits) if r["label"]]
    negative = [z for r, z in zip(active, logits) if not r["label"]]
    tp = sum(d and r["label"] == 1 for r, d in zip(active, decisions))
    fp = sum(d and r["label"] == 0 for r, d in zip(active, decisions))
    per_episode, windows = {}, []
    current, previous, false_active = None, None, False
    window = None
    def close_window():
        if window is not None:
            windows.append(window)
    for row in rows:
        episode = row["episode_id"]
        if episode != previous:
            close_window(); window = None; false_active = False
            current = per_episode.setdefault(episode, dict(eligible_windows=0, hit_windows=0, false_request_windows=0))
            previous = episode
        eligible = bool(row["label"] and row["fit_mask"])
        decision = predictions.get((episode, row["step"]), False)
        if eligible and window is None:
            window = dict(episode_id=episode, pair_id=row.get("pair_id"), start_step=row["step"],
                          end_step=row["step"], request=False, max_logit=scores[(episode, row["step"])])
            current["eligible_windows"] += 1
        if not eligible and window is not None:
            close_window(); window = None
        if eligible:
            window["end_step"] = row["step"]
            window["max_logit"] = max(window["max_logit"], scores[(episode, row["step"])])
            window["max_probability"] = probability(window["max_logit"])
            if decision and not window["request"]:
                window["request"] = True
                current["hit_windows"] += 1
        false = decision and not eligible
        if false and not false_active:
            current["false_request_windows"] += 1
        false_active = false
    close_window()
    def loss(z, y):
        return max(z, 0) + math.log1p(math.exp(-abs(z))) - y*z
    metrics = dict(actionable_rows=len(active), eligible_rows=len(positive), ineligible_rows=len(negative),
                   true_positive=tp, false_negative=len(positive)-tp, false_positive=fp, true_negative=len(negative)-fp,
                   precision=tp/(tp+fp) if tp+fp else None,
                   eligible_step_recall=tp/len(positive) if positive else None,
                   ineligible_step_false_positive_rate=fp/len(negative) if negative else None,
                   eligible_windows=len(windows), eligible_windows_with_request=sum(w["request"] for w in windows),
                   eligible_window_recall=sum(w["request"] for w in windows)/len(windows) if windows else None,
                   false_request_windows=sum(x["false_request_windows"] for x in per_episode.values()),
                   unweighted_bce=statistics.mean(loss(z, r["label"]) for r, z in zip(active, logits)),
                   eligible_bce=statistics.mean(loss(z, 1) for z in positive) if positive else None,
                   ineligible_bce=statistics.mean(loss(z, 0) for z in negative) if negative else None,
                   probability_quantiles=quantiles(probs),
                   eligible_probability_quantiles=quantiles([p for p, r in zip(probs, active) if r["label"]]),
                   ineligible_probability_quantiles=quantiles([p for p, r in zip(probs, active) if not r["label"]]),
                   per_episode=per_episode)
    metrics["score_separation"] = dict(eligible_logit=quantiles(positive), ineligible_logit=quantiles(negative),
        median_logit_gap=statistics.median(positive)-statistics.median(negative) if positive and negative else None,
        worst_logit_gap=min(positive)-max(negative) if positive and negative else None)
    return metrics, windows


def qualifies(record):
    return (record["eligible_windows_with_request"] >= 1
            and record["false_positive"] <= math.floor(.005 * record["ineligible_rows"])
            and record["false_request_windows"] <= 5)


def monitor(model, rows):
    logits = [model.logit(r["normalized_observation"]) for r in rows if r["fit_mask"]]
    record, windows = evaluate(rows, logits)
    record["audit_only_rows"] = sum(not r["fit_mask"] for r in rows)
    record["audit_only_requests"] = sum(model.request(r["normalized_observation"]) for r in rows if not r["fit_mask"])
    record["qualifies"] = qualifies(record)
    return record, windows
