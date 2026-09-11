"""Helpers for keeping evaluation labels out of inference payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


GROUND_TRUTH_KEYS = frozenset(
    {
        "ground_truth_label",
        "ground_truth_track_label",
        "ground_truth_mode_sequence",
        "synthetic_truth_value",
    }
)


def is_ground_truth_key(key: object) -> bool:
    """Return whether a mapping key identifies synthetic evaluation truth."""
    normalized_key = str(key).casefold()
    return "ground_truth" in normalized_key or normalized_key in GROUND_TRUTH_KEYS


def strip_ground_truth(value: Any) -> Any:
    """Return a recursively copied payload without ground-truth fields.

    Matching all keys containing ``ground_truth`` also protects inference from
    newly added truth metadata without requiring every producer and consumer to
    update an exact-key allowlist in lockstep. Scalar values are deliberately
    left untouched: report prose may legitimately contain those words.
    """
    if isinstance(value, Mapping):
        return {
            key: strip_ground_truth(child)
            for key, child in value.items()
            if not is_ground_truth_key(key)
        }
    if isinstance(value, list):
        return [strip_ground_truth(child) for child in value]
    if isinstance(value, tuple):
        return tuple(strip_ground_truth(child) for child in value)
    return value


def ground_truth_paths(
    value: Any, path: tuple[object, ...] = ()
) -> list[tuple[object, ...]]:
    """Find paths to ground-truth mapping keys in a nested payload."""
    paths: list[tuple[object, ...]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = (*path, key)
            if is_ground_truth_key(key):
                paths.append(child_path)
            paths.extend(ground_truth_paths(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(ground_truth_paths(child, (*path, index)))
    return paths
