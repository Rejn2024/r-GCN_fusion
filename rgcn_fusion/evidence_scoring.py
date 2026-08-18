"""Shared scoring primitives for evidence ETL pipelines."""

from __future__ import annotations


def ds_masses_from_score(score: float, ambiguity: float) -> list[float]:
    """Build a two-hypothesis DS mass vector: [non_match, match, uncertain]."""
    uncertainty = min(0.6, max(0.05, ambiguity))
    committed = 1.0 - uncertainty
    match = committed * max(0.0, min(1.0, score))
    non_match = committed - match
    return [round(non_match, 6), round(match, 6), round(uncertainty, 6)]
