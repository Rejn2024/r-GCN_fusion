"""Synthetic intelligence-report generation and ingestion helpers.

The functions in this module keep intelligence reports as evidence/provenance
objects rather than mutating canonical aircraft/radar KG facts.  They support the
same leakage-safe pattern used by the ESM observation ETL: generated reports may
contain truth metadata for evaluation, but ingestion functions score and expose
only report claims, source credibility, recency, and optional external priors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from kg_generator import AIRCRAFT, RADARS, slug
from rgcn_fusion.observation_etl import ds_masses_from_score
from rgcn_fusion.dempster_shafer import combine_masses

CLAIM_TYPES = (
    "operator",
    "aircraft_variant",
    "aircraft_family",
    "radar_type",
    "radar_mode",
    "location",
    "relation",
    "last_observed_time",
    "expected_behavior",
)
MIN_REPORTS_PER_OBSERVATION = 10
MAX_REPORTS_PER_OBSERVATION = 12
DEFAULT_REPORT_RECENCY_HALF_LIFE_DAYS = 14.0
DEFAULT_INTELLIGENCE_WEIGHT = 0.15
ALTERNATIVE_REFUTATION_DISCOUNT = 0.15


@dataclass(frozen=True)
class PriorScore:
    """Named prior component used when scoring an extracted report claim."""

    name: str
    value: float


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def report_recency_score(
    report: dict[str, Any],
    *,
    reference_time: datetime | None = None,
    half_life_days: float = DEFAULT_REPORT_RECENCY_HALF_LIFE_DAYS,
) -> float:
    """Return exponential recency decay for a report relative to an observation.

    ``collected_at`` is preferred over ``published_at`` because it is closer to
    when the reported information was true.  Missing timestamps are neutral-low
    evidence rather than a hard rejection.
    """
    reference_time = reference_time or datetime.now(UTC)
    observed_at = _parse_utc(report.get("collected_at") or report.get("published_at"))
    if observed_at is None:
        return 0.25
    age_days = max(0.0, (reference_time - observed_at).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def _external_prior_score(
    context: dict[str, Any], prior_name: str, candidate_value: Any
) -> float:
    if candidate_value is None:
        return 0.5
    prior_maps = [
        context.get(f"{prior_name}_priors"),
        (
            (context.get("priors") or {}).get(prior_name)
            if isinstance(context.get("priors"), dict)
            else None
        ),
    ]
    for prior_map in prior_maps:
        if isinstance(prior_map, dict) and candidate_value in prior_map:
            return max(0.0, min(1.0, float(prior_map[candidate_value])))
    contextual_value = context.get(prior_name)
    if contextual_value is None:
        return 0.5
    if isinstance(contextual_value, (list, tuple, set)):
        return 1.0 if candidate_value in contextual_value else 0.0
    return 1.0 if contextual_value == candidate_value else 0.0


def report_claim_score(
    report: dict[str, Any],
    claim: dict[str, Any],
    *,
    observation_time: datetime | None = None,
) -> float:
    """Blend report/claim quality features into one bounded support score."""
    context = report.get("external_context") or {}
    claim_type = str(claim.get("claim_type", ""))
    value = claim.get("object_id") or claim.get("object_value")
    prior = (
        _external_prior_score(context, claim_type, value)
        if isinstance(context, dict)
        else 0.5
    )
    recency = report_recency_score(report, reference_time=observation_time)
    credibility = max(0.0, min(1.0, float(report.get("credibility_score", 0.5))))
    extraction = max(0.0, min(1.0, float(claim.get("extraction_confidence", 0.5))))
    confidence = max(0.0, min(1.0, float(claim.get("claim_confidence", 0.5))))
    specificity = max(0.0, min(1.0, float(claim.get("specificity_score", 0.7))))
    kg_consistency = max(0.0, min(1.0, float(claim.get("kg_consistency_score", 0.7))))
    score = (
        0.25 * confidence
        + 0.20 * credibility
        + 0.15 * recency
        + 0.15 * extraction
        + 0.10 * prior
        + 0.10 * kg_consistency
        + 0.05 * specificity
    )
    return round(max(0.0, min(1.0, score)), 6)


def final_signed_compatibility(
    claim: dict[str, Any], candidate: dict[str, Any]
) -> float:
    """Return the claim's final signed effect on a candidate in ``[-1, 1]``.

    This implements the *final signed compatibility* convention: positive values
    support the candidate and negative values refute it, so callers must not
    apply a second stance sign.  Exact identifiers receive full weight, while
    family and relation matches may be supplied on candidate rows as resolved KG
    context.  Refuting an incompatible alternative is deliberately weak positive
    evidence because eliminating one alternative does not establish this one.
    """
    claim_type = str(claim.get("claim_type") or "")
    object_id = claim.get("object_id") or claim.get("object_value")
    field_by_type = {
        "operator": "operator",
        "aircraft_variant": "aircraft_id",
        "aircraft_family": "aircraft_family_id",
        "radar_type": "radar_id",
        "radar_mode": "mode_id",
        "location": "location_id",
        "relation": "relation_id",
    }
    candidate_field = field_by_type.get(claim_type)
    if candidate_field is None or object_id is None:
        return 0.0
    candidate_value = candidate.get(candidate_field)
    if candidate_value is None:
        return 0.0

    alignment = 1.0 if str(candidate_value) == str(object_id) else -1.0
    stance = str(claim.get("stance") or "supports").lower()
    if stance in {"refutes", "refute", "denies", "denied"}:
        return -1.0 if alignment > 0 else ALTERNATIVE_REFUTATION_DISCOUNT
    if stance not in {"supports", "support", "asserts", "asserted"}:
        return 0.0
    return alignment


def claim_candidate_contribution(
    claim: dict[str, Any], candidate: dict[str, Any]
) -> float:
    """Return quality-weighted final compatibility without reapplying stance."""
    quality = max(
        0.0, min(1.0, float(claim.get("text_score", claim.get("claim_score", 0.0))))
    )
    return round(quality * final_signed_compatibility(claim, candidate), 6)


def _candidate_specific_claim_masses(contribution: float) -> list[float]:
    """Convert a signed contribution to candidate-specific DS masses."""
    strength = min(1.0, abs(float(contribution)))
    uncertainty = 1.0 - strength
    committed = strength
    if contribution >= 0.0:
        return [0.0, round(committed, 6), round(uncertainty, 6)]
    return [round(committed, 6), 0.0, round(uncertainty, 6)]


def aggregate_candidate_intelligence(
    candidate: dict[str, Any],
    claims: Iterable[dict[str, Any]],
    *,
    intelligence_weight: float = DEFAULT_INTELLIGENCE_WEIGHT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate independent claim evidence and fuse it with sensor evidence.

    Claims are deduplicated by source: only the largest absolute contribution
    from one source is retained.  The returned edges preserve every non-zero
    compatibility for graph learning, while scalar/DS features use the
    source-independent subset.
    """
    if not 0.0 <= intelligence_weight <= 1.0:
        raise ValueError("intelligence_weight must be between 0 and 1")
    all_edges: list[dict[str, Any]] = []
    strongest_by_source: dict[str, tuple[float, dict[str, Any]]] = {}
    for claim in claims:
        compatibility = final_signed_compatibility(claim, candidate)
        if compatibility == 0.0:
            continue
        contribution = claim_candidate_contribution(claim, candidate)
        edge = {
            "source": claim["id"],
            "target": candidate["id"],
            "compatibility": compatibility,
            "claim_score": float(claim.get("text_score", 0.0)),
            "contribution": contribution,
            "match_basis": str(claim.get("claim_type") or "unknown"),
        }
        all_edges.append(edge)
        source = str(claim.get("source_id") or claim.get("report_id") or claim["id"])
        previous = strongest_by_source.get(source)
        if previous is None or abs(contribution) > abs(previous[0]):
            strongest_by_source[source] = (contribution, claim)

    retained = list(strongest_by_source.values())
    positive = sum(value for value, _claim in retained if value > 0.0)
    negative = sum(-value for value, _claim in retained if value < 0.0)
    total_strength = positive + negative
    intel_score = 0.5 if total_strength == 0.0 else positive / total_strength
    conflict = (
        0.0 if total_strength == 0.0 else min(positive, negative) / total_strength
    )
    coverage = min(1.0, len(retained) / 3.0)
    effective_weight = intelligence_weight * coverage
    sensor_score = float(
        candidate.get("sensor_score", candidate.get("text_score", 0.0))
    )
    final_score = (
        1.0 - effective_weight
    ) * sensor_score + effective_weight * intel_score

    sensor_masses = candidate.get("sensor_ds_masses") or candidate.get("ds_masses")
    fused_masses = (
        list(sensor_masses)
        if sensor_masses
        else ds_masses_from_score(sensor_score, 0.2)
    )
    sensor_masses = list(fused_masses)
    for contribution, _claim in retained:
        try:
            fused_masses = (
                combine_masses(
                    fused_masses, _candidate_specific_claim_masses(contribution)
                )
                .round(6)
                .tolist()
            )
        except ValueError:
            # Total conflict is itself useful information; retain the previous
            # valid masses and expose conflict through ``intel_conflict``.
            conflict = 1.0
            break

    features = {
        "sensor_score": round(sensor_score, 6),
        "sensor_ds_masses": sensor_masses,
        "intel_support_score": round(positive, 6),
        "intel_refute_score": round(negative, 6),
        "intel_net_score": round(positive - negative, 6),
        "intel_score": round(intel_score, 6),
        "intel_conflict": round(conflict, 6),
        "intel_uncertainty": round(float(fused_masses[2]), 6),
        "intel_claim_count": float(len(retained)),
        "intel_source_count": float(len(retained)),
        "intel_effective_weight": round(effective_weight, 6),
        "final_score": round(final_score, 6),
        "text_score": round(final_score, 6),
        "ds_masses": fused_masses,
    }
    return features, all_edges


def build_candidate_intelligence_rows(
    report_rows: dict[str, list[dict[str, Any]]],
    candidates: Iterable[dict[str, Any]],
    *,
    intelligence_weight: float = DEFAULT_INTELLIGENCE_WEIGHT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build candidate updates and direct claim-to-candidate evidence edges."""
    candidates = list(candidates)
    claims_by_series: dict[str, list[dict[str, Any]]] = {}
    for claim in report_rows["claims"]:
        claims_by_series.setdefault(str(claim.get("series_id")), []).append(claim)
    updates: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("aircraft_id") and candidate.get("radar_id"):
            candidate.setdefault(
                "relation_id",
                f"relation:{candidate['aircraft_id']}:USES_RADAR:{candidate['radar_id']}",
            )
        claims = claims_by_series.get(str(candidate.get("series_id")), [])
        features, candidate_edges = aggregate_candidate_intelligence(
            candidate, claims, intelligence_weight=intelligence_weight
        )
        updates.append({"id": candidate["id"], **features})
        edges.extend(candidate_edges)
    updates_by_observation: dict[str, list[dict[str, Any]]] = {}
    observation_by_id = {
        candidate["id"]: candidate.get("observation_id") for candidate in candidates
    }
    for update in updates:
        updates_by_observation.setdefault(
            str(observation_by_id.get(update["id"])), []
        ).append(update)
    for observation_updates in updates_by_observation.values():
        observation_updates.sort(key=lambda row: row["final_score"], reverse=True)
        for rank, update in enumerate(observation_updates, start=1):
            update["intel_rank"] = rank
    return updates, edges


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _wrong_aircraft(rng: random.Random, truth: dict[str, Any]) -> Any:
    options = [
        a for a in AIRCRAFT if f"aircraft:{slug(a.variant)}" != truth.get("aircraft_id")
    ]
    return rng.choice(options)


def _claim(
    *,
    claim_type: str,
    subject_id: str,
    object_id: str,
    object_value: str,
    correct: bool,
    rng: random.Random,
    stance: str = "supports",
    kg_entity_id: str | None = None,
) -> dict[str, Any]:
    """Build one scored synthetic assertion without exposing truth to inference."""
    confidence = rng.uniform(0.68, 0.95) if correct else rng.uniform(0.35, 0.76)
    return {
        "claim_type": claim_type,
        "stance": stance,
        "subject_id": subject_id,
        "predicate": "SUPPORTS" if stance == "supports" else "REFUTES",
        "object_id": object_id,
        "object_value": object_value,
        "object_kind": claim_type,
        "claim_text": f"{claim_type.replace('_', ' ').title()}: {object_value}.",
        "claim_confidence": round(confidence, 6),
        "extraction_confidence": round(rng.uniform(0.72, 0.98), 6),
        "specificity_score": round(rng.uniform(0.65, 0.96), 6),
        "kg_consistency_score": round(
            rng.uniform(0.72, 0.98) if correct else rng.uniform(0.18, 0.66), 6
        ),
        "synthetic_truth_value": "correct" if correct else "contradictory",
        "kg_entity_id": kg_entity_id,
    }


def _aircraft_for_truth(truth: dict[str, Any]) -> Any:
    return next(
        aircraft
        for aircraft in AIRCRAFT
        if f"aircraft:{slug(aircraft.variant)}" == truth["aircraft_id"]
    )


def _reported_identity(
    rng: random.Random, truth: dict[str, Any], *, correct: bool
) -> tuple[Any, str]:
    aircraft = _aircraft_for_truth(truth) if correct else _wrong_aircraft(rng, truth)
    if correct:
        operator = truth["operator"]
    else:
        alternatives = [
            value for value in aircraft.operators if value != truth["operator"]
        ]
        operator = rng.choice(
            alternatives
            or [
                value
                for item in AIRCRAFT
                for value in item.operators
                if value != truth["operator"]
            ]
        )
    return aircraft, operator


def generate_intelligence_reports_for_series(
    series: dict[str, Any],
    *,
    seed: int | None = None,
    min_reports: int = MIN_REPORTS_PER_OBSERVATION,
    max_reports: int = MAX_REPORTS_PER_OBSERVATION,
) -> list[dict[str, Any]]:
    """Generate track-aware sighting and pattern-of-life intelligence reports."""
    if min_reports < 1 or max_reports < min_reports:
        raise ValueError("report count bounds must be positive and ordered")
    observations = series.get("observations") or []
    if not observations:
        return []
    first_observation, last_observation = observations[0], observations[-1]
    truth = (
        series.get("ground_truth_track_label")
        or first_observation.get("ground_truth_label")
        or {}
    )
    if not truth:
        raise ValueError("synthetic intelligence reports require ground_truth_label")
    rng = random.Random(
        seed if seed is not None else hash(series["series_id"]) & ((1 << 63) - 1)
    )
    track_start = _parse_utc(
        first_observation.get("timestamp_iso8601")
    ) or datetime.now(UTC)
    track_end = _parse_utc(last_observation.get("timestamp_iso8601")) or track_start
    report_count = rng.randint(min_reports, max_reports)
    reports: list[dict[str, Any]] = []
    sighting_count = max(1, round(report_count * 0.65))
    for idx in range(report_count):
        is_sighting = idx < sighting_count
        # Sightings are mostly accurate, with a deterministic erroneous example
        # whenever there is more than one report of that kind.
        correct = idx == 0 or (idx != 1 and rng.random() < 0.82)
        aircraft, operator = _reported_identity(rng, truth, correct=correct)
        radar = RADARS[aircraft.radar]
        location = last_observation["estimated_emitter_location"]
        if correct:
            reported_area = location["area"]
            reported_lat = float(location["estimated_latitude_deg"]) + rng.gauss(
                0, 0.025
            )
            reported_lon = float(location["estimated_longitude_deg"]) + rng.gauss(
                0, 0.025
            )
            last_seen = track_end + timedelta(seconds=rng.uniform(-90, 20))
        else:
            areas = [
                "North Sea",
                "Eastern Mediterranean",
                "Baltic Sea",
                "Arabian Gulf",
                "South China Sea",
                "Bay of Bengal",
                "Sea of Japan",
                "Western Pacific",
            ]
            reported_area = rng.choice(
                [area for area in areas if area != location["area"]]
            )
            reported_lat = float(location["estimated_latitude_deg"]) + rng.uniform(
                1.0, 5.0
            )
            reported_lon = float(location["estimated_longitude_deg"]) + rng.uniform(
                1.0, 5.0
            )
            last_seen = track_end + timedelta(hours=rng.uniform(-12, -2))
        collected_at = last_seen + timedelta(seconds=rng.uniform(15, 240))
        published_at = collected_at + timedelta(seconds=rng.uniform(30.0, 900.0))
        credibility = rng.uniform(0.68, 0.95) if correct else rng.uniform(0.32, 0.72)
        aircraft_id = f"aircraft:{slug(aircraft.variant)}"
        radar_id = f"radar:{slug(radar.name)}"
        common_claims = [
            _claim(
                claim_type="aircraft_variant",
                subject_id=series["series_id"],
                object_id=aircraft_id,
                object_value=aircraft.variant,
                correct=correct,
                rng=rng,
                kg_entity_id=aircraft_id,
            ),
            _claim(
                claim_type="operator",
                subject_id=series["series_id"],
                object_id=operator,
                object_value=operator,
                correct=correct,
                rng=rng,
                kg_entity_id=f"operator:{slug(operator)}",
            ),
            _claim(
                claim_type="radar_type",
                subject_id=series["series_id"],
                object_id=radar_id,
                object_value=radar.name,
                correct=correct,
                rng=rng,
                kg_entity_id=radar_id,
            ),
        ]
        if is_sighting:
            claims = common_claims + [
                _claim(
                    claim_type="location",
                    subject_id=series["series_id"],
                    object_id=f"area:{slug(reported_area)}",
                    object_value=reported_area,
                    correct=correct,
                    rng=rng,
                ),
                _claim(
                    claim_type="last_observed_time",
                    subject_id=series["series_id"],
                    object_id=f"time:{_iso(last_seen)}",
                    object_value=_iso(last_seen),
                    correct=correct,
                    rng=rng,
                ),
            ]
            report_type = "sighting_report"
            detail = {
                "aircraft_variant": aircraft.variant,
                "aircraft_family": aircraft.family,
                "operator": operator,
                "radar": radar.name,
                "last_observed_at": _iso(last_seen),
                "location": {
                    "area": reported_area,
                    "latitude_deg": round(reported_lat, 6),
                    "longitude_deg": round(reported_lon, 6),
                },
                "track_time_window": {
                    "start": _iso(track_start),
                    "end": _iso(track_end),
                },
            }
        else:
            typical_modes = [mode.name for mode in radar.modes[:3]]
            behavior = f"{aircraft.role} operations by {operator}; typical {radar.name} modes: {', '.join(typical_modes)}"
            claims = common_claims + [
                _claim(
                    claim_type="aircraft_family",
                    subject_id=series["series_id"],
                    object_id=f"aircraft_family:{slug(aircraft.family)}",
                    object_value=aircraft.family,
                    correct=correct,
                    rng=rng,
                    kg_entity_id=f"aircraft_family:{slug(aircraft.family)}",
                ),
                _claim(
                    claim_type="expected_behavior",
                    subject_id=series["series_id"],
                    object_id=f"behavior:{slug(aircraft.variant)}:{slug(aircraft.role)}",
                    object_value=behavior,
                    correct=correct,
                    rng=rng,
                ),
            ]
            report_type = "pattern_of_life_report"
            detail = {
                "aircraft_family": aircraft.family,
                "aircraft_variant": aircraft.variant,
                "operator": operator,
                "radar": radar.name,
                "role": aircraft.role,
                "expected_radar_modes": typical_modes,
                "expected_operating_area": reported_area,
                "expected_altitude_ceiling_m": aircraft.service_ceiling_m,
                "expected_speed_ceiling_mach": aircraft.max_speed_mach,
            }
        report = {
            "report_id": f"intel_report:{series['series_id']}:{idx + 1:02d}",
            "series_id": series["series_id"],
            "source_id": f"source:{rng.choice(['visual_observer_a', 'air_defence_b', 'liaison_c', 'pattern_analyst_d'])}",
            "source_type": "observer_network" if is_sighting else "pattern_analysis",
            "report_type": report_type,
            "published_at": _iso(published_at),
            "collected_at": _iso(collected_at),
            "ingested_at": _iso(
                published_at + timedelta(seconds=rng.uniform(5.0, 120.0))
            ),
            "credibility_score": round(credibility, 6),
            "external_context": {
                "operator_priors": {truth["operator"]: 0.75},
                "aircraft_family_priors": {
                    f"aircraft_family:{slug(truth['aircraft_family'])}": 0.70
                },
                "radar_type_priors": {truth["radar_id"]: 0.70},
                "radar_mode_priors": {truth["mode_id"]: 0.65},
            },
            "claims": claims,
            "sighting" if is_sighting else "pattern_of_life": detail,
        }
        reports.append(report)
    return reports


def generate_intelligence_reports_for_observation(
    observation: dict[str, Any], **kwargs: Any
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for callers that have a single observation."""
    return generate_intelligence_reports_for_series(
        {
            "series_id": observation.get("series_id", observation["observation_id"]),
            "ground_truth_track_label": observation.get("ground_truth_label"),
            "observations": [observation],
        },
        **kwargs,
    )


def add_intelligence_reports_to_series(
    data: dict[str, Any],
    *,
    seed: int = 7,
    min_reports: int = MIN_REPORTS_PER_OBSERVATION,
    max_reports: int = MAX_REPORTS_PER_OBSERVATION,
) -> dict[str, Any]:
    """Attach one shared intelligence-report set to every observation series."""
    if min_reports < 1 or max_reports < min_reports:
        raise ValueError("report count bounds must be positive and ordered")

    rng = random.Random(seed)
    enriched = json.loads(json.dumps(data))
    for series in enriched.get("observation_series", []):
        observations = series.get("observations", [])
        for obs in observations:
            obs.pop("intelligence_reports", None)
        if not observations:
            series["intelligence_reports"] = []
            continue
        reports = generate_intelligence_reports_for_series(
            series,
            seed=rng.getrandbits(64),
            min_reports=min_reports,
            max_reports=max_reports,
        )
        observation_ids = [obs["observation_id"] for obs in observations]
        for report_index, report in enumerate(reports, start=1):
            report["report_id"] = (
                f"intel_report:{series['series_id']}:{report_index:02d}"
            )
            report.pop("observation_id", None)
            report["series_id"] = series["series_id"]
            report["valid_for_observation_ids"] = observation_ids
            for claim_index, claim in enumerate(report.get("claims") or [], start=1):
                claim["claim_id"] = (
                    f"intel_claim:{series['series_id']}:{report_index:02d}:"
                    f"{claim_index:02d}"
                )
                claim["subject_id"] = series["series_id"]
        series["intelligence_reports"] = reports
    meta = enriched.setdefault("metadata", {})
    meta["intelligence_reports_per_series"] = [min_reports, max_reports]
    meta["intelligence_report_types"] = ["sighting_report", "pattern_of_life_report"]
    meta["intelligence_claim_types"] = list(CLAIM_TYPES)
    return enriched


def flatten_reports_from_series(data: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for series in data.get("observation_series", []):
        reports.extend(series.get("intelligence_reports") or [])
    return reports


def _great_circle_distance_km(
    latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float
) -> float:
    """Return the great-circle distance between two WGS84 positions."""
    lat_a, lat_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_lat = lat_b - lat_a
    delta_lon = math.radians(longitude_b - longitude_a)
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_a) * math.cos(lat_b) * math.sin(delta_lon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * math.asin(min(1.0, math.sqrt(haversine)))


def report_observation_proximity(
    report: dict[str, Any],
    observation: dict[str, Any],
    *,
    max_sighting_distance_km: float = 200.0,
    max_sighting_time_delta_s: float = 1800.0,
    max_pattern_time_delta_s: float = 86400.0,
) -> dict[str, float | str] | None:
    """Describe a report/observation proximity match, or return ``None``.

    Sighting reports must be close in both time and geographic position. Pattern
    reports use their expected operating area and a wider temporal window. This
    prevents a series-level container from making every report applicable to
    every observation merely because their ``series_id`` values match.
    """
    observation_time = _parse_utc(observation.get("timestamp_iso8601"))
    observation_location = observation.get("estimated_emitter_location") or {}
    if report.get("report_type") == "sighting_report":
        sighting = report.get("sighting") or {}
        reported_location = sighting.get("location") or {}
        reported_time = _parse_utc(sighting.get("last_observed_at"))
        required = (
            observation_time,
            reported_time,
            observation_location.get("estimated_latitude_deg"),
            observation_location.get("estimated_longitude_deg"),
            reported_location.get("latitude_deg"),
            reported_location.get("longitude_deg"),
        )
        if any(value is None for value in required):
            return None
        time_delta_s = abs((observation_time - reported_time).total_seconds())
        distance_km = _great_circle_distance_km(
            float(observation_location["estimated_latitude_deg"]),
            float(observation_location["estimated_longitude_deg"]),
            float(reported_location["latitude_deg"]),
            float(reported_location["longitude_deg"]),
        )
        if (
            time_delta_s > max_sighting_time_delta_s
            or distance_km > max_sighting_distance_km
        ):
            return None
        return {
            "match_basis": "time_and_geography",
            "time_delta_s": round(time_delta_s, 3),
            "distance_km": round(distance_km, 3),
        }

    pattern = report.get("pattern_of_life") or {}
    expected_area = pattern.get("expected_operating_area")
    observed_area = observation_location.get("area")
    report_time = _parse_utc(report.get("collected_at") or report.get("published_at"))
    if (
        not expected_area
        or expected_area != observed_area
        or not observation_time
        or not report_time
    ):
        return None
    time_delta_s = abs((observation_time - report_time).total_seconds())
    if time_delta_s > max_pattern_time_delta_s:
        return None
    return {
        "match_basis": "time_and_operating_area",
        "time_delta_s": round(time_delta_s, 3),
        "distance_km": 0.0,
    }


def build_report_evidence_rows(
    series_records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build unique report/claim nodes and link claims to all series observations."""
    report_rows, claim_rows, contains_edges, support_edges = [], [], [], []
    track_rows, track_observation_edges = [], []
    report_proximity_edges, report_track_edges, kg_entity_edges = [], [], []
    contradiction_edges: list[dict[str, Any]] = []
    for series in series_records:
        observations = series.get("observations", [])
        reports = series.get("intelligence_reports") or []
        track_id = f"evidence:track:{series.get('series_id')}"
        track_rows.append({"id": track_id, "series_id": series.get("series_id")})
        track_observation_edges.extend(
            {
                "source": track_id,
                "target": f"evidence:observation:{observation['observation_id']}",
            }
            for observation in observations
        )
        claims_for_series: list[dict[str, Any]] = []
        reference_time = (
            _parse_utc(observations[0].get("timestamp_iso8601"))
            if observations
            else None
        )
        for report in reports:
            recency = report_recency_score(report, reference_time=reference_time)
            report_id = f"evidence:report:{report['report_id']}"
            report_rows.append(
                {
                    "id": report_id,
                    "report_id": report["report_id"],
                    "series_id": series.get("series_id"),
                    "source_id": report.get("source_id"),
                    "report_type": report.get("report_type"),
                    "source_type": report.get("source_type"),
                    "published_at": report.get("published_at"),
                    "collected_at": report.get("collected_at"),
                    "credibility_score": float(report.get("credibility_score", 0.5)),
                    "recency_score": round(recency, 6),
                    "degree_score": float(len(report.get("claims") or [])),
                    "text_score": float(report.get("credibility_score", 0.5)),
                    "ds_masses": ds_masses_from_score(
                        float(report.get("credibility_score", 0.5)) * recency, 0.25
                    ),
                }
            )
            applicable_observations = []
            for observation in observations:
                proximity = report_observation_proximity(report, observation)
                if proximity is None:
                    continue
                observation_id = f"evidence:observation:{observation['observation_id']}"
                applicable_observations.append(observation_id)
                report_proximity_edges.append(
                    {"source": report_id, "target": observation_id, **proximity}
                )
            if applicable_observations:
                report_track_edges.append({"source": report_id, "target": track_id})
            for claim in report.get("claims") or []:
                score = report_claim_score(
                    report, claim, observation_time=reference_time
                )
                claim_id = f"evidence:claim:{claim['claim_id']}"
                claim_row = {
                    "id": claim_id,
                    "claim_id": claim["claim_id"],
                    "report_id": report["report_id"],
                    "series_id": series.get("series_id"),
                    "claim_type": claim.get("claim_type"),
                    "stance": claim.get("stance", "supports"),
                    "object_id": claim.get("object_id"),
                    "object_value": claim.get("object_value"),
                    "kg_entity_id": claim.get("kg_entity_id"),
                    "source_id": report.get("source_id"),
                    "credibility_score": float(report.get("credibility_score", 0.5)),
                    "recency_score": round(recency, 6),
                    "claim_confidence": float(claim.get("claim_confidence", 0.5)),
                    "extraction_confidence": float(
                        claim.get("extraction_confidence", 0.5)
                    ),
                    "degree_score": 1.0,
                    "text_score": score,
                    "ds_masses": ds_masses_from_score(
                        score, 0.2 if claim.get("stance") == "supports" else 0.35
                    ),
                }
                claim_rows.append(claim_row)
                claims_for_series.append(claim_row)
                contains_edges.append({"source": report_id, "target": claim_id})
                if claim.get("kg_entity_id"):
                    kg_entity_edges.append(
                        {
                            "source": claim_id,
                            "target": claim["kg_entity_id"],
                            "claim_type": claim.get("claim_type"),
                            "stance": claim.get("stance", "supports"),
                            "score": score,
                        }
                    )
                for observation_id in applicable_observations:
                    support_edges.append(
                        {
                            "source": claim_id,
                            "target": observation_id,
                            "score": score,
                            "stance": claim.get("stance", "supports"),
                        }
                    )
        for left_idx, left in enumerate(claims_for_series):
            for right in claims_for_series[left_idx + 1 :]:
                if left.get("claim_type") == right.get("claim_type") and left.get(
                    "object_id"
                ) != right.get("object_id"):
                    contradiction_edges.append(
                        {
                            "source": (
                                left["id"]
                                if left["text_score"] >= right["text_score"]
                                else right["id"]
                            ),
                            "target": (
                                right["id"]
                                if left["text_score"] >= right["text_score"]
                                else left["id"]
                            ),
                            "reason": str(left.get("claim_type")),
                            "score_delta": round(
                                abs(
                                    float(left["text_score"])
                                    - float(right["text_score"])
                                ),
                                6,
                            ),
                        }
                    )
    return {
        "reports": report_rows,
        "claims": claim_rows,
        "contains_edges": contains_edges,
        "support_edges": support_edges,
        "contradiction_edges": contradiction_edges,
        "tracks": track_rows,
        "track_observation_edges": track_observation_edges,
        "report_proximity_edges": report_proximity_edges,
        "report_track_edges": report_track_edges,
        "kg_entity_edges": kg_entity_edges,
    }


class ReportNeo4jETL:
    """Minimal Neo4j writer for intelligence-report evidence rows."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def ingest(self, series_records: list[dict[str, Any]]) -> dict[str, int]:
        rows = build_report_evidence_rows(series_records)
        with self.driver.session(database=self.database) as session:
            candidates = []
            for record in session.run("""
                MATCH (c:CandidateEvidence)
                OPTIONAL MATCH (aircraft:AircraftVariant {id: c.aircraft_id})
                    -[:VARIANT_OF]->(family:AircraftFamily)
                RETURN properties(c) AS candidate, family.id AS aircraft_family_id
                """):
                candidate = dict(record["candidate"])
                candidate["aircraft_family_id"] = record["aircraft_family_id"]
                candidates.append(candidate)
            candidate_updates, candidate_edges = build_candidate_intelligence_rows(
                rows, candidates
            )
            rows["candidate_updates"] = candidate_updates
            rows["candidate_edges"] = candidate_edges
            session.execute_write(_write_report_evidence_rows, rows)
        return {name: len(values) for name, values in rows.items()}


def _write_report_evidence_rows(tx, rows: dict[str, list[dict[str, Any]]]) -> None:
    tx.run(
        """
    UNWIND $rows AS row
    MERGE (n:EvidenceEntity:Track {id: row.id})
    SET n += row
    """,
        rows=rows["tracks"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MERGE (n:EvidenceEntity:IntelligenceReport {id: row.id})
    SET n += row
    """,
        rows=rows["reports"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MERGE (n:EvidenceEntity:ReportClaim {id: row.id})
    SET n += row
    """,
        rows=rows["claims"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:Track {id: row.source})
    MATCH (t:Observation {id: row.target})
    MERGE (s)-[:TRACK_HAS_OBSERVATION]->(t)
    """,
        rows=rows["track_observation_edges"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:IntelligenceReport {id: row.source})
    MATCH (t:Observation {id: row.target})
    MERGE (s)-[r:REPORT_NEAR_OBSERVATION]->(t)
    SET r.match_basis = row.match_basis,
        r.time_delta_s = row.time_delta_s,
        r.distance_km = row.distance_km
    """,
        rows=rows["report_proximity_edges"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:IntelligenceReport {id: row.source})
    MATCH (t:Track {id: row.target})
    MERGE (s)-[:REPORT_APPLIES_TO_TRACK]->(t)
    """,
        rows=rows["report_track_edges"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:ReportClaim {id: row.source})
    MATCH (t {id: row.target})
    WHERE t:AircraftVariant OR t:AircraftFamily OR t:Radar OR t:Operator OR t:RadarMode
    MERGE (s)-[r:CLAIM_ASSERTS_KG_ENTITY]->(t)
    SET r.claim_type = row.claim_type,
        r.stance = row.stance,
        r.score = row.score
    """,
        rows=rows["kg_entity_edges"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:EvidenceEntity {id: row.source})
    MATCH (t:EvidenceEntity {id: row.target})
    MERGE (s)-[:REPORT_CONTAINS_CLAIM]->(t)
    """,
        rows=rows["contains_edges"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:EvidenceEntity {id: row.source})
    MATCH (t:EvidenceEntity {id: row.target})
    MERGE (s)-[r:CLAIM_SUPPORTS_OBSERVATION]->(t)
    SET r.score = row.score, r.stance = row.stance
    """,
        rows=rows["support_edges"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:EvidenceEntity {id: row.source})
    MATCH (t:EvidenceEntity {id: row.target})
    MERGE (s)-[r:CONTRADICTS_CLAIM]->(t)
    SET r.score_delta = row.score_delta, r.reason = row.reason
    """,
        rows=rows["contradiction_edges"],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (c:CandidateEvidence {id: row.id})
    SET c += row
    WITH c, row
    OPTIONAL MATCH (:Observation)-[r:HAS_CANDIDATE]->(c)
    SET r.sensor_score = row.sensor_score,
        r.score = row.final_score,
        r.rank = row.intel_rank
    """,
        rows=rows.get("candidate_updates", []),
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:ReportClaim {id: row.source})
    MATCH (t:CandidateEvidence {id: row.target})
    MERGE (s)-[r:CLAIM_SUPPORTS_CANDIDATE]->(t)
    SET r.compatibility = row.compatibility,
        r.claim_score = row.claim_score,
        r.contribution = row.contribution,
        r.match_basis = row.match_basis
    """,
        rows=[
            row for row in rows.get("candidate_edges", []) if row["contribution"] > 0
        ],
    )
    tx.run(
        """
    UNWIND $rows AS row
    MATCH (s:ReportClaim {id: row.source})
    MATCH (t:CandidateEvidence {id: row.target})
    MERGE (s)-[r:CLAIM_REFUTES_CANDIDATE]->(t)
    SET r.compatibility = row.compatibility,
        r.claim_score = row.claim_score,
        r.contribution = row.contribution,
        r.match_basis = row.match_basis
    """,
        rows=[
            row for row in rows.get("candidate_edges", []) if row["contribution"] < 0
        ],
    )


def load_reports_json(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "observation_series" in data:
        return flatten_reports_from_series(data)
    if isinstance(data, dict) and "reports" in data:
        return data["reports"]
    if isinstance(data, list):
        return data
    raise ValueError(
        "report JSON must be a series dataset, {'reports': [...]}, or a list"
    )


def series_from_series_json(path: str | Path) -> list[dict[str, Any]]:
    """Load series records with their shared intelligence reports."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(
        data.get("observation_series"), list
    ):
        raise ValueError("series JSON must contain an 'observation_series' list")
    return data["observation_series"]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load synthetic intelligence reports into Neo4j as evidence nodes."
    )
    parser.add_argument(
        "--series",
        type=Path,
        default=Path("generated/demo_esm_observation_series_with_intel.json"),
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--neo4j-database", default="neo4j")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    series_records = series_from_series_json(args.series)
    etl = ReportNeo4jETL(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database
    )
    try:
        result = etl.ingest(series_records)
    finally:
        etl.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
