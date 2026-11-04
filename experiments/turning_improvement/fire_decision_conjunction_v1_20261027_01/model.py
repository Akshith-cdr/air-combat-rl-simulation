"""Direct 19-observation eligibility model. No PPO/actor features or gate calls."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real


SCHEMA = "direct-observation-conjunction-v1"


def observation(values):
    if len(values) != 19 or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in values):
        raise ValueError("Exactly 19 finite existing normalized observations required")
    return [float(v) for v in values]


def probability(logit):
    if not math.isfinite(logit):
        raise ValueError("Finite decision logit required")
    return math.exp(-math.log1p(math.exp(-logit))) if logit >= 0 else math.exp(logit - math.log1p(math.exp(logit)))


@dataclass(frozen=True)
class Conjunction:
    range_upper: float
    alignment_lower: float
    range_scale: float
    alignment_scale: float

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()) or min(self.range_scale, self.alignment_scale) <= 0:
            raise ValueError("Finite cutpoints and positive scales required")

    def logit(self, values):
        x = observation(values)
        score = min((self.range_upper - x[10]) / self.range_scale,
                    (x[11] - self.alignment_lower) / self.alignment_scale)
        if not math.isfinite(score):
            raise ValueError("Decision-score overflow")
        return score

    def request(self, values):
        return probability(self.logit(values)) >= .5

    def to_dict(self):
        return dict(schema=SCHEMA, input_space="normalized_observation", input_dimension=19,
                    feature_indices=[10, 11], decision_threshold=.5, **asdict(self))

    @classmethod
    def from_dict(cls, data):
        expected = {"schema", "input_space", "input_dimension", "feature_indices", "decision_threshold",
                    "range_upper", "alignment_lower", "range_scale", "alignment_scale"}
        if (type(data) is not dict or set(data) != expected or data["schema"] != SCHEMA
                or data["input_space"] != "normalized_observation" or data["input_dimension"] != 19
                or data["feature_indices"] != [10, 11] or data["decision_threshold"] != .5
                or any(type(data[k]) not in (int, float) for k in ("range_upper", "alignment_lower", "range_scale", "alignment_scale"))):
            raise ValueError("Unsupported decision-model interface or threshold")
        return cls(**{key: data[key] for key in ("range_upper", "alignment_lower", "range_scale", "alignment_scale")})


def fit(training):
    """One deterministic supervised fit, using only actionable TRAINING rows.

    Minimum positive-containing rectangle: range <= max_positive(range),
    alignment >= min_positive(alignment). No production gate cutpoint is used.
    Duplicate rows do not move bounds/scales. No optimizer or search on validation.
    """
    if not training or any(r.get("split") != "training" for r in training):
        raise ValueError("Only a nonempty training split may fit this model")
    usable, audit = [], 0
    for row in training:
        if type(row.get("label")) is not int or row["label"] not in (0, 1) or type(row.get("fit_mask")) is not bool:
            raise ValueError("Invalid label/mask")
        x = observation(row["normalized_observation"])
        if row["fit_mask"]:
            usable.append((x, row["label"], row["episode_id"], row["step"]))
        else:
            if row["label"] != 0:
                raise ValueError("Audit-only rows cannot be positive")
            audit += 1
    positives = [x for x, y, _, _ in usable if y]
    if not positives or not any(not y for _, y, _, _ in usable):
        raise ValueError("Both actionable training classes required")
    def span(index):
        values = [x[index] for x, _, _, _ in usable]
        value = max(values) - min(values)
        return value if value > 0 else 1.
    model = Conjunction(max(x[10] for x in positives), min(x[11] for x in positives), span(10), span(11))
    contradictions = [(episode, step) for x, y, episode, step in usable if model.request(x) != bool(y)]
    if contradictions:
        raise ValueError("Observable conjunction cannot separate training labels; first conflicts: " + repr(contradictions[:5]))
    identities = [(episode, step, y) for _, y, episode, step in usable]
    return model, dict(actionable_rows=len(usable), positives=len(positives), negatives=len(usable)-len(positives),
                       audit_excluded=audit, fit_passes=1, optimizer_updates=0,
                       training_identities_sha256=hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest())


def replace_fire_action(action, values, model):
    """Explicit opt-in composition only; never calls or bypasses the launch gate."""
    if len(action) != 5 or any(not isinstance(v, Real) or not math.isfinite(v) for v in action):
        raise ValueError("Existing finite five-component aircraft action required")
    output = action.copy() if hasattr(action, "copy") else list(action)
    output[4] = float(model.request(values))
    return output
