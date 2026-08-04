"""Decode independent classifier outputs into combinations permitted by a KG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ConstrainedPrediction:
    """A KG-valid joint prediction, or an explicit open-set rejection."""

    labels: dict[str, Any] | None
    confidence: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class HierarchicalPrediction:
    """A constrained identity with independently rejectable open-set attributes."""

    labels: dict[str, Any] | None
    confidence: dict[str, float]
    status: str
    unknown_tasks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def decode_kg_constrained(
    probabilities: Mapping[str, Sequence[float]],
    vocabularies: Mapping[str, Sequence[Any]],
    valid_combinations: Sequence[Mapping[str, Any]],
    *,
    unknown_threshold: float = 0.5,
    min_task_probability: float = 0.0,
) -> ConstrainedPrediction:
    """Return the most likely complete KG tuple, otherwise ``unknown``.

    A candidate tuple is scored by the product of the corresponding independent
    task probabilities.  Confidence is that score normalized over all valid KG
    tuples.  The minimum per-task threshold prevents a single weak task from
    being hidden by a small candidate set.
    """
    if not 0.0 <= unknown_threshold <= 1.0 or not 0.0 <= min_task_probability <= 1.0:
        raise ValueError("probability thresholds must be between 0 and 1")
    tasks = tuple(probabilities)
    if not tasks:
        raise ValueError("at least one classification task is required")

    lookup: dict[str, dict[Any, float]] = {}
    for task in tasks:
        if task not in vocabularies:
            raise ValueError(f"missing vocabulary for task {task!r}")
        values = list(vocabularies[task])
        scores = list(probabilities[task])
        if len(values) != len(scores):
            raise ValueError(f"probability and vocabulary lengths differ for task {task!r}")
        if any(not math.isfinite(float(score)) or float(score) < 0.0 for score in scores):
            raise ValueError(f"probabilities for task {task!r} must be finite and non-negative")
        lookup[task] = dict(zip(values, map(float, scores)))

    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for combination in valid_combinations:
        if any(task not in combination for task in tasks):
            continue
        task_scores = [lookup[task].get(combination[task], 0.0) for task in tasks]
        joint_score = math.prod(task_scores)
        candidates.append((joint_score, min(task_scores), {task: combination[task] for task in tasks}))

    total_score = sum(candidate[0] for candidate in candidates)
    if not candidates or total_score <= 0.0:
        return ConstrainedPrediction(None, 0.0, "unknown")
    joint_score, weakest_score, labels = max(candidates, key=lambda candidate: candidate[0])
    confidence = joint_score / total_score
    if confidence < unknown_threshold or weakest_score < min_task_probability:
        return ConstrainedPrediction(None, confidence, "unknown")
    return ConstrainedPrediction(labels, confidence, "known")


def decode_kg_hierarchical(
    probabilities: Mapping[str, Sequence[float]],
    vocabularies: Mapping[str, Sequence[Any]],
    valid_combinations: Sequence[Mapping[str, Any]],
    *,
    identity_tasks: Sequence[str],
    open_set_tasks: Sequence[str],
    identity_unknown_threshold: float = 0.5,
    attribute_thresholds: Mapping[str, float] | None = None,
    novelty_scores: Mapping[str, float] | None = None,
    novelty_thresholds: Mapping[str, float] | None = None,
) -> HierarchicalPrediction:
    """Decode a KG identity while allowing novel conditional attributes.

    Identity fields (for example radar, aircraft, and operator) are selected as
    one KG-valid tuple. Open-set fields (for example radar mode) are then
    restricted to values linked to that identity, but can each be rejected to
    ``None``. ``novelty_scores`` must be supplied by a calibrated OOD detector;
    larger values mean more novel. A task's softmax threshold is only a fallback,
    not a sufficient novelty detector by itself.
    """
    identity_tasks = tuple(identity_tasks)
    open_set_tasks = tuple(open_set_tasks)
    if not identity_tasks or set(identity_tasks) & set(open_set_tasks):
        raise ValueError("identity_tasks must be non-empty and disjoint from open_set_tasks")
    missing_tasks = (set(identity_tasks) | set(open_set_tasks)) - set(probabilities)
    if missing_tasks:
        raise ValueError(f"missing probabilities for tasks: {sorted(missing_tasks)}")

    identity_combinations: list[dict[str, Any]] = []
    seen_identities: set[tuple[Any, ...]] = set()
    for combination in valid_combinations:
        if any(task not in combination for task in identity_tasks):
            continue
        identity_key = tuple(combination[task] for task in identity_tasks)
        if identity_key not in seen_identities:
            seen_identities.add(identity_key)
            identity_combinations.append(dict(zip(identity_tasks, identity_key)))
    identity = decode_kg_constrained(
        {task: probabilities[task] for task in identity_tasks},
        {task: vocabularies[task] for task in identity_tasks},
        identity_combinations,
        unknown_threshold=identity_unknown_threshold,
    )
    if identity.labels is None:
        return HierarchicalPrediction(None, {"identity": identity.confidence}, "unknown", identity_tasks)

    labels = dict(identity.labels)
    confidence = {"identity": identity.confidence}
    unknown_tasks: list[str] = []
    attribute_thresholds = attribute_thresholds or {}
    novelty_scores = novelty_scores or {}
    novelty_thresholds = novelty_thresholds or {}
    matching_rows = [
        combination
        for combination in valid_combinations
        if all(combination.get(task) == value for task, value in identity.labels.items())
    ]
    for task in open_set_tasks:
        values = list(vocabularies[task])
        scores = list(map(float, probabilities[task]))
        if len(values) != len(scores):
            raise ValueError(f"probability and vocabulary lengths differ for task {task!r}")
        allowed = {row[task] for row in matching_rows if task in row}
        allowed_scores = [(score, value) for value, score in zip(values, scores) if value in allowed]
        best_score, best_value = max(allowed_scores, default=(0.0, None))
        confidence[task] = best_score
        is_novel = novelty_scores.get(task, 0.0) >= novelty_thresholds.get(task, 1.0)
        if best_score < attribute_thresholds.get(task, 0.5) or is_novel:
            labels[task] = None
            unknown_tasks.append(task)
        else:
            labels[task] = best_value

    status = "partially_known" if unknown_tasks else "known"
    return HierarchicalPrediction(labels, confidence, status, tuple(unknown_tasks))
