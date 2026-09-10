"""Compatibility scoring for passive-ESM observations of non-radar RF equipment."""

from __future__ import annotations

from typing import Any, Iterable

RF_EMISSION_TYPES = ("data_link", "radio", "radar_altimeter")


def _normalise(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _range(props: dict[str, Any]) -> tuple[float, float] | None:
    """Return a frequency interval in MHz, accepting MHz or GHz fields."""
    if props.get("frequency_min_mhz") is not None and props.get("frequency_max_mhz") is not None:
        return float(props["frequency_min_mhz"]), float(props["frequency_max_mhz"])
    if props.get("frequency_min_ghz") is not None and props.get("frequency_max_ghz") is not None:
        return 1000.0 * float(props["frequency_min_ghz"]), 1000.0 * float(props["frequency_max_ghz"])
    return None


def _interval_score(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_min, left_max = sorted(left)
    right_min, right_max = sorted(right)
    overlap = max(0.0, min(left_max, right_max) - max(left_min, right_min))
    if overlap:
        return min(1.0, overlap / max(left_max - left_min, 1e-9))
    distance = min(abs(left_max - right_min), abs(right_max - left_min))
    scale = max(left_max - left_min, right_max - right_min, 1.0)
    return max(0.0, 1.0 - distance / scale)


def score_non_radar_rf(
    observation: dict[str, Any], equipment_rows: Iterable[dict[str, Any]]
) -> tuple[float, int, dict[str, float]]:
    """Return candidate compatibility, comparison count, and RF feature scores.

    No detections produce zero comparisons (absence of evidence), rather than a
    negative score. Detected data-link, radio, and altimeter emissions are matched
    only against corresponding equipment attached to the candidate aircraft.
    """
    emissions = observation.get("rf_emissions") or {}
    equipment_by_type = {
        _normalise(row.get("emission_type") or row.get("type") or row.get("label")): row
        for row in equipment_rows or ()
    }
    aliases = {"datalink": "data_link", "radaraltimeter": "radar_altimeter"}
    scores: list[float] = []
    features: dict[str, float] = {}
    for emission_type in RF_EMISSION_TYPES:
        measured = emissions.get(emission_type)
        if not isinstance(measured, dict):
            continue
        expected = equipment_by_type.get(emission_type)
        if expected is None:
            expected = next((row for key, row in equipment_by_type.items() if aliases.get(key) == emission_type), None)
        components: list[float] = []
        if expected is not None:
            if measured.get("equipment") is not None and expected.get("name") is not None:
                components.append(float(_normalise(measured["equipment"]) == _normalise(expected["name"])))
            measured_range, expected_range = _range(measured), _range(expected)
            if measured_range and expected_range:
                components.append(_interval_score(measured_range, expected_range))
            for field in ("frequency_band", "access_method", "modulation", "waveform", "encryption", "frequency_hopping"):
                if measured.get(field) is not None and expected.get(field) is not None:
                    components.append(float(_normalise(measured[field]) == _normalise(expected[field])))
        score = sum(components) / len(components) if components else 0.0
        features[f"rf_{emission_type}_compatibility"] = round(score, 6)
        scores.append(score)
    compatibility = sum(scores) / len(scores) if scores else 0.0
    features["non_radar_rf_compatibility_score"] = round(compatibility, 6)
    features["non_radar_rf_observed_count"] = float(len(scores))
    return compatibility, len(scores), features
