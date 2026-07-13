"""Dempster-Shafer mass utilities.

Mass vectors use the following order for a frame with ``n`` singleton hypotheses:
all non-empty subsets encoded by bit masks ``1..(2**n - 1)`` when ``n`` is at
most 10. Larger frames use singleton masks ``1 << i`` plus one full-frame
uncertainty mask to avoid exponential growth. For example, with hypotheses
``["A", "B"]`` the full-subset vector is ``[{A}, {B}, {A,B}]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


MAX_FULL_SUBSET_HYPOTHESES = 10


@dataclass(frozen=True)
class Interval:
    """Belief/plausibility interval for one singleton hypothesis."""

    hypothesis: str
    belief: float
    plausibility: float


def subset_masks(num_hypotheses: int) -> list[int]:
    """Return bit-mask encodings for the supported mass vector frame.

    Frames with at most ``MAX_FULL_SUBSET_HYPOTHESES`` hypotheses use every
    non-empty subset. Larger frames fall back to singleton masses plus one
    residual full-frame uncertainty mass.
    """
    if num_hypotheses < 1:
        raise ValueError("num_hypotheses must be positive")
    if num_hypotheses <= MAX_FULL_SUBSET_HYPOTHESES:
        return list(range(1, 2**num_hypotheses))
    singleton_masks = [1 << idx for idx in range(num_hypotheses)]
    full_frame_mask = (1 << num_hypotheses) - 1
    return singleton_masks + [full_frame_mask]


def _masks_for_mass_length(num_masses: int) -> list[int]:
    """Infer supported masks from a mass-vector length."""
    full_subset_hypotheses = int(np.log2(num_masses + 1))
    if (
        full_subset_hypotheses <= MAX_FULL_SUBSET_HYPOTHESES
        and 2**full_subset_hypotheses - 1 == num_masses
    ):
        return subset_masks(full_subset_hypotheses)

    large_frame_hypotheses = num_masses - 1
    if large_frame_hypotheses > MAX_FULL_SUBSET_HYPOTHESES:
        return subset_masks(large_frame_hypotheses)

    raise ValueError(
        "mass vector length must match either a full-subset frame with at most "
        f"{MAX_FULL_SUBSET_HYPOTHESES} hypotheses or a singleton-plus-uncertainty "
        f"frame with more than {MAX_FULL_SUBSET_HYPOTHESES} hypotheses"
    )


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

    num_masses = left_arr.shape[0]
    masks = _masks_for_mass_length(num_masses)
    mask_indices = {mask: idx for idx, mask in enumerate(masks)}
    combined = np.zeros(num_masses, dtype=np.float64)
    conflict = 0.0
    for i, mask_i in enumerate(masks):
        for j, mask_j in enumerate(masks):
            product = left_arr[i] * right_arr[j]
            intersection = mask_i & mask_j
            if intersection == 0:
                conflict += product
            else:
                combined[mask_indices[intersection]] += product

    if conflict >= 1.0 - 1e-12:
        raise ValueError("total conflict: Dempster combination is undefined")
    return validate_masses(combined / (1.0 - conflict))
