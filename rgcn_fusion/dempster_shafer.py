"""Dempster-Shafer mass utilities.

Mass vectors use the following order for a frame with ``n`` singleton hypotheses:
all non-empty subsets encoded by bit masks ``1..(2**n - 1)``. For example,
with hypotheses ``["A", "B"]`` the vector is ``[{A}, {B}, {A,B}]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Interval:
    """Belief/plausibility interval for one singleton hypothesis."""

    hypothesis: str
    belief: float
    plausibility: float


def subset_masks(num_hypotheses: int) -> list[int]:
    """Return bit-mask encodings for all non-empty frame subsets."""
    if num_hypotheses < 1:
        raise ValueError("num_hypotheses must be positive")
    return list(range(1, 2**num_hypotheses))


def validate_masses(masses: np.ndarray, *, atol: float = 1e-6) -> np.ndarray:
    """Validate and normalize a one- or two-dimensional mass array."""
    arr = np.asarray(masses, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError("masses must be a 1D or 2D array")
    if np.any(arr < -atol):
        raise ValueError("mass values must be non-negative")
    arr = np.clip(arr, 0.0, None)
    totals = arr.sum(axis=-1, keepdims=True)
    if np.any(totals <= atol):
        raise ValueError("each mass vector must have positive total mass")
    return arr / totals


def belief_plausibility(
    masses: Iterable[float], hypotheses: list[str]
) -> list[Interval]:
    """Compute singleton belief/plausibility intervals from a mass vector."""
    mass = validate_masses(np.asarray(list(masses), dtype=np.float64))
    masks = subset_masks(len(hypotheses))
    if mass.shape[0] != len(masks):
        raise ValueError(
            f"expected {len(masks)} masses for {len(hypotheses)} hypotheses, "
            f"got {mass.shape[0]}"
        )

    intervals: list[Interval] = []
    for idx, hypothesis in enumerate(hypotheses):
        singleton = 1 << idx
        belief = sum(value for mask, value in zip(masks, mass) if mask == singleton)
        plausibility = sum(value for mask, value in zip(masks, mass) if mask & singleton)
        intervals.append(Interval(hypothesis, float(belief), float(plausibility)))
    return intervals


def combine_masses(left: Iterable[float], right: Iterable[float]) -> np.ndarray:
    """Combine two mass vectors with normalized Dempster's rule of combination."""
    left_arr = validate_masses(np.asarray(list(left), dtype=np.float64))
    right_arr = validate_masses(np.asarray(list(right), dtype=np.float64))
    if left_arr.shape != right_arr.shape:
        raise ValueError("mass vectors must have the same shape")

    num_subsets = left_arr.shape[0]
    num_hypotheses = int(np.log2(num_subsets + 1))
    if 2**num_hypotheses - 1 != num_subsets:
        raise ValueError("mass vector length must be 2**n - 1")

    masks = subset_masks(num_hypotheses)
    combined = np.zeros(num_subsets, dtype=np.float64)
    conflict = 0.0
    for i, mask_i in enumerate(masks):
        for j, mask_j in enumerate(masks):
            product = left_arr[i] * right_arr[j]
            intersection = mask_i & mask_j
            if intersection == 0:
                conflict += product
            else:
                combined[masks.index(intersection)] += product

    if conflict >= 1.0 - 1e-12:
        raise ValueError("total conflict: Dempster combination is undefined")
    return validate_masses(combined / (1.0 - conflict))
