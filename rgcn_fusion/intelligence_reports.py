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
)
MIN_REPORTS_PER_OBSERVATION = 10
MAX_REPORTS_PER_OBSERVATION = 12
MIN_ORDERS_OF_BATTLE_PER_OBSERVATION = 2
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


def _external_prior_score(context: dict[str, Any], prior_name: str, candidate_value: Any) -> float:
    if candidate_value is None:
        return 0.5
    prior_maps = [
        context.get(f"{prior_name}_priors"),
        (context.get("priors") or {}).get(prior_name) if isinstance(context.get("priors"), dict) else None,
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
    prior = _external_prior_score(context, claim_type, value) if isinstance(context, dict) else 0.5
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


def final_signed_compatibility(claim: dict[str, Any], candidate: dict[str, Any]) -> float:
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


def claim_candidate_contribution(claim: dict[str, Any], candidate: dict[str, Any]) -> float:
    """Return quality-weighted final compatibility without reapplying stance."""
    quality = max(0.0, min(1.0, float(claim.get("text_score", claim.get("claim_score", 0.0)))))
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
    conflict = 0.0 if total_strength == 0.0 else min(positive, negative) / total_strength
    coverage = min(1.0, len(retained) / 3.0)
    effective_weight = intelligence_weight * coverage
    sensor_score = float(candidate.get("sensor_score", candidate.get("text_score", 0.0)))
    final_score = (1.0 - effective_weight) * sensor_score + effective_weight * intel_score

    sensor_masses = candidate.get("sensor_ds_masses") or candidate.get("ds_masses")
    fused_masses = list(sensor_masses) if sensor_masses else ds_masses_from_score(sensor_score, 0.2)
    sensor_masses = list(fused_masses)
    for contribution, _claim in retained:
        try:
            fused_masses = combine_masses(
                fused_masses, _candidate_specific_claim_masses(contribution)
            ).round(6).tolist()
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
    observation_by_id = {candidate["id"]: candidate.get("observation_id") for candidate in candidates}
    for update in updates:
        updates_by_observation.setdefault(str(observation_by_id.get(update["id"])), []).append(update)
    for observation_updates in updates_by_observation.values():
        observation_updates.sort(key=lambda row: row["final_score"], reverse=True)
        for rank, update in enumerate(observation_updates, start=1):
            update["intel_rank"] = rank
    return updates, edges


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _wrong_aircraft(rng: random.Random, truth: dict[str, Any]) -> Any:
    options = [a for a in AIRCRAFT if f"aircraft:{slug(a.variant)}" != truth.get("aircraft_id")]
    return rng.choice(options)


def _wrong_radar_and_mode(rng: random.Random, truth: dict[str, Any]) -> tuple[Any, Any]:
    radar_options = [r for r in RADARS.values() if f"radar:{slug(r.name)}" != truth.get("radar_id")]
    radar = rng.choice(radar_options)
    return radar, rng.choice(list(radar.modes))


def _claim_value_for_type(
    rng: random.Random,
    claim_type: str,
    truth: dict[str, Any],
    observation: dict[str, Any],
    *,
    correct: bool,
) -> tuple[str, str, str]:
    if claim_type == "operator":
        if correct:
            value = truth["operator"]
        else:
            operators = sorted({op for a in AIRCRAFT for op in a.operators if op != truth["operator"]})
            value = rng.choice(operators)
        return value, value, "operator"
    if claim_type == "aircraft_variant":
        aircraft = None if correct else _wrong_aircraft(rng, truth)
        value = truth["aircraft_id"] if correct else f"aircraft:{slug(aircraft.variant)}"
        text = truth["aircraft_variant"] if correct else aircraft.variant
        return value, text, "aircraft_variant"
    if claim_type == "aircraft_family":
        if correct:
            value = f"aircraft_family:{slug(truth['aircraft_family'])}"
            text = truth["aircraft_family"]
        else:
            aircraft = _wrong_aircraft(rng, truth)
            value = f"aircraft_family:{slug(aircraft.family)}"
            text = aircraft.family
        return value, text, "aircraft_family"
    if claim_type == "radar_type":
        if correct:
            return truth["radar_id"], truth["radar"], "radar"
        radar, _ = _wrong_radar_and_mode(rng, truth)
        return f"radar:{slug(radar.name)}", radar.name, "radar"
    if claim_type == "radar_mode":
        if correct:
            return truth["mode_id"], truth["mode"], "radar_mode"
        radar, mode = _wrong_radar_and_mode(rng, truth)
        return f"radar_mode:{slug(radar.name)}:{slug(mode.name)}", mode.name, "radar_mode"
    if claim_type == "location":
        loc = observation.get("estimated_emitter_location", {})
        if correct:
            area = loc.get("area", "unknown area")
        else:
            areas = ["North Sea", "Eastern Mediterranean", "Baltic Sea", "Arabian Gulf", "South China Sea", "Bay of Bengal", "Sea of Japan", "Western Pacific"]
            area = rng.choice([a for a in areas if a != loc.get("area")])
        return f"area:{slug(area)}", area, "area"
    if claim_type == "relation":
        if correct:
            value = f"relation:{truth['aircraft_id']}:USES_RADAR:{truth['radar_id']}"
            text = f"{truth['aircraft_variant']} uses {truth['radar']}"
        else:
            aircraft = _wrong_aircraft(rng, truth)
            radar, _ = _wrong_radar_and_mode(rng, truth)
            value = f"relation:aircraft:{slug(aircraft.variant)}:USES_RADAR:radar:{slug(radar.name)}"
            text = f"{aircraft.variant} uses {radar.name}"
        return value, text, "relation_hypothesis"
    raise ValueError(f"unsupported claim type: {claim_type}")


def generate_intelligence_reports_for_observation(
    observation: dict[str, Any],
    *,
    seed: int | None = None,
    min_reports: int = MIN_REPORTS_PER_OBSERVATION,
    max_reports: int = MAX_REPORTS_PER_OBSERVATION,
) -> list[dict[str, Any]]:
    """Generate 10--12 synthetic intelligence reports for one ESM observation.

    Reports intentionally mix correct, incorrect, and explicitly refuting claims
    so downstream demos can exercise corroboration and contradiction handling.
    """
    if min_reports < 1 or max_reports < min_reports:
        raise ValueError("report count bounds must be positive and ordered")
    truth = observation.get("ground_truth_label") or {}
    if not truth:
        raise ValueError("synthetic intelligence reports require ground_truth_label")
    rng = random.Random(seed if seed is not None else hash(observation["observation_id"]) & ((1 << 63) - 1))
    obs_time = _parse_utc(observation.get("timestamp_iso8601")) or datetime.now(UTC)
    report_count = rng.randint(min_reports, max_reports)
    claim_cycle = list(CLAIM_TYPES)
    reports: list[dict[str, Any]] = []
    for idx in range(report_count):
        is_order_of_battle = idx < MIN_ORDERS_OF_BATTLE_PER_OBSERVATION
        claim_type = "operator" if is_order_of_battle else claim_cycle[(idx - 2) % len(claim_cycle)]
        correct = rng.random() < 0.72
        # The first order of battle always identifies the true operator country;
        # the second provides a competing assessment for fusion experiments.
        if idx == 0:
            correct = True
        elif idx in (1, 5):
            correct = False
        stance = "supports" if idx == 0 or rng.random() > 0.12 else "refutes"
        value, text_value, value_kind = _claim_value_for_type(rng, claim_type, truth, observation, correct=correct)
        offset_s = rng.uniform(-1800.0, 900.0)
        collected_at = obs_time + timedelta(seconds=offset_s)
        published_at = collected_at + timedelta(seconds=rng.uniform(30.0, 900.0))
        credibility = rng.uniform(0.62, 0.95) if correct else rng.uniform(0.25, 0.78)
        confidence = rng.uniform(0.60, 0.92) if correct else rng.uniform(0.35, 0.82)
        claim = {
            "claim_id": f"intel_claim:{observation['observation_id']}:{idx + 1:02d}",
            "claim_type": claim_type,
            "stance": stance,
            "subject_id": observation["observation_id"],
            "predicate": "SUPPORTS" if stance == "supports" else "REFUTES",
            "object_id": value,
            "object_value": text_value,
            "object_kind": value_kind,
            "claim_text": f"{stance.title()} {claim_type.replace('_', ' ')} assessment: {text_value}.",
            "claim_confidence": round(confidence, 6),
            "extraction_confidence": round(rng.uniform(0.70, 0.98), 6),
            "specificity_score": round(rng.uniform(0.55, 0.95), 6),
            "kg_consistency_score": round(rng.uniform(0.70, 0.98) if correct else rng.uniform(0.15, 0.70), 6),
            "synthetic_truth_value": "correct" if correct else "contradictory",
        }
        report = {
            "report_id": f"intel_report:{observation['observation_id']}:{idx + 1:02d}",
            "observation_id": observation["observation_id"],
            "series_id": observation.get("series_id"),
            "source_id": f"source:{rng.choice(['sigint_a', 'osint_b', 'liaison_c', 'analyst_d'])}",
            "source_type": rng.choice(["sigint", "osint", "liaison", "analyst_assessment"]),
            "report_type": "order_of_battle" if is_order_of_battle else "automated_synthetic_intelligence",
            "published_at": _iso(published_at),
            "collected_at": _iso(collected_at),
            "ingested_at": _iso(published_at + timedelta(seconds=rng.uniform(5.0, 120.0))),
            "credibility_score": round(credibility, 6),
            "external_context": {
                "operator_priors": {truth["operator"]: 0.75},
                "aircraft_family_priors": {f"aircraft_family:{slug(truth['aircraft_family'])}": 0.70},
                "radar_type_priors": {truth["radar_id"]: 0.70},
                "radar_mode_priors": {truth["mode_id"]: 0.65},
            },
            "claims": [claim],
        }
        if is_order_of_battle:
            report["order_of_battle"] = {
                "operator_country": text_value,
                "assessed_aircraft_variant": truth["aircraft_variant"],
                "assessment_scope": observation.get("estimated_emitter_location", {}).get("area", "unknown area"),
            }
        reports.append(report)
    return reports


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
        reports = generate_intelligence_reports_for_observation(
            observations[0], seed=rng.getrandbits(64),
            min_reports=min_reports, max_reports=max_reports,
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
    meta["intelligence_claim_types"] = list(CLAIM_TYPES)
    return enriched


def flatten_reports_from_series(data: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for series in data.get("observation_series", []):
        reports.extend(series.get("intelligence_reports") or [])
    return reports


def build_report_evidence_rows(
    series_records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build unique report/claim nodes and link claims to all series observations."""
    report_rows, claim_rows, contains_edges, support_edges = [], [], [], []
    contradiction_edges: list[dict[str, Any]] = []
    for series in series_records:
        observations = series.get("observations", [])
        reports = series.get("intelligence_reports") or []
        claims_for_series: list[dict[str, Any]] = []
        reference_time = _parse_utc(observations[0].get("timestamp_iso8601")) if observations else None
        for report in reports:
            recency = report_recency_score(report, reference_time=reference_time)
            report_id = f"evidence:report:{report['report_id']}"
            report_rows.append({
                "id": report_id, "report_id": report["report_id"],
                "series_id": series.get("series_id"), "source_id": report.get("source_id"),
                "source_type": report.get("source_type"), "published_at": report.get("published_at"),
                "collected_at": report.get("collected_at"),
                "credibility_score": float(report.get("credibility_score", 0.5)),
                "recency_score": round(recency, 6),
                "degree_score": float(len(report.get("claims") or [])),
                "text_score": float(report.get("credibility_score", 0.5)),
                "ds_masses": ds_masses_from_score(float(report.get("credibility_score", 0.5)) * recency, 0.25),
            })
            for claim in report.get("claims") or []:
                score = report_claim_score(report, claim, observation_time=reference_time)
                claim_id = f"evidence:claim:{claim['claim_id']}"
                claim_row = {
                    "id": claim_id, "claim_id": claim["claim_id"], "report_id": report["report_id"],
                    "series_id": series.get("series_id"), "claim_type": claim.get("claim_type"),
                    "stance": claim.get("stance", "supports"), "object_id": claim.get("object_id"),
                    "object_value": claim.get("object_value"),
                    "source_id": report.get("source_id"),
                    "credibility_score": float(report.get("credibility_score", 0.5)),
                    "recency_score": round(recency, 6),
                    "claim_confidence": float(claim.get("claim_confidence", 0.5)),
                    "extraction_confidence": float(claim.get("extraction_confidence", 0.5)),
                    "degree_score": 1.0, "text_score": score,
                    "ds_masses": ds_masses_from_score(score, 0.2 if claim.get("stance") == "supports" else 0.35),
                }
                claim_rows.append(claim_row)
                claims_for_series.append(claim_row)
                contains_edges.append({"source": report_id, "target": claim_id})
                for obs in observations:
                    support_edges.append({"source": claim_id, "target": f"evidence:observation:{obs['observation_id']}", "score": score, "stance": claim.get("stance", "supports")})
        for left_idx, left in enumerate(claims_for_series):
            for right in claims_for_series[left_idx + 1:]:
                if left.get("claim_type") == right.get("claim_type") and left.get("object_id") != right.get("object_id"):
                    contradiction_edges.append({
                        "source": left["id"] if left["text_score"] >= right["text_score"] else right["id"],
                        "target": right["id"] if left["text_score"] >= right["text_score"] else left["id"],
                        "reason": str(left.get("claim_type")),
                        "score_delta": round(abs(float(left["text_score"]) - float(right["text_score"])), 6),
                    })
    return {"reports": report_rows, "claims": claim_rows, "contains_edges": contains_edges, "support_edges": support_edges, "contradiction_edges": contradiction_edges}


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
            candidate_updates, candidate_edges = build_candidate_intelligence_rows(rows, candidates)
            rows["candidate_updates"] = candidate_updates
            rows["candidate_edges"] = candidate_edges
            session.execute_write(_write_report_evidence_rows, rows)
        return {name: len(values) for name, values in rows.items()}


def _write_report_evidence_rows(tx, rows: dict[str, list[dict[str, Any]]]) -> None:
    tx.run("""
    UNWIND $rows AS row
    MERGE (n:EvidenceEntity:IntelligenceReport {id: row.id})
    SET n += row
    """, rows=rows["reports"])
    tx.run("""
    UNWIND $rows AS row
    MERGE (n:EvidenceEntity:ReportClaim {id: row.id})
    SET n += row
    """, rows=rows["claims"])
    tx.run("""
    UNWIND $rows AS row
    MATCH (s:EvidenceEntity {id: row.source})
    MATCH (t:EvidenceEntity {id: row.target})
    MERGE (s)-[:REPORT_CONTAINS_CLAIM]->(t)
    """, rows=rows["contains_edges"])
    tx.run("""
    UNWIND $rows AS row
    MATCH (s:EvidenceEntity {id: row.source})
    MATCH (t:EvidenceEntity {id: row.target})
    MERGE (s)-[r:CLAIM_SUPPORTS_OBSERVATION]->(t)
    SET r.score = row.score, r.stance = row.stance
    """, rows=rows["support_edges"])
    tx.run("""
    UNWIND $rows AS row
    MATCH (s:EvidenceEntity {id: row.source})
    MATCH (t:EvidenceEntity {id: row.target})
    MERGE (s)-[r:CONTRADICTS_CLAIM]->(t)
    SET r.score_delta = row.score_delta, r.reason = row.reason
    """, rows=rows["contradiction_edges"])
    tx.run("""
    UNWIND $rows AS row
    MATCH (c:CandidateEvidence {id: row.id})
    SET c += row
    WITH c, row
    OPTIONAL MATCH (:Observation)-[r:HAS_CANDIDATE]->(c)
    SET r.sensor_score = row.sensor_score,
        r.score = row.final_score,
        r.rank = row.intel_rank
    """, rows=rows.get("candidate_updates", []))
    tx.run("""
    UNWIND $rows AS row
    MATCH (s:ReportClaim {id: row.source})
    MATCH (t:CandidateEvidence {id: row.target})
    MERGE (s)-[r:CLAIM_SUPPORTS_CANDIDATE]->(t)
    SET r.compatibility = row.compatibility,
        r.claim_score = row.claim_score,
        r.contribution = row.contribution,
        r.match_basis = row.match_basis
    """, rows=[row for row in rows.get("candidate_edges", []) if row["contribution"] > 0])
    tx.run("""
    UNWIND $rows AS row
    MATCH (s:ReportClaim {id: row.source})
    MATCH (t:CandidateEvidence {id: row.target})
    MERGE (s)-[r:CLAIM_REFUTES_CANDIDATE]->(t)
    SET r.compatibility = row.compatibility,
        r.claim_score = row.claim_score,
        r.contribution = row.contribution,
        r.match_basis = row.match_basis
    """, rows=[row for row in rows.get("candidate_edges", []) if row["contribution"] < 0])


def load_reports_json(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "observation_series" in data:
        return flatten_reports_from_series(data)
    if isinstance(data, dict) and "reports" in data:
        return data["reports"]
    if isinstance(data, list):
        return data
    raise ValueError("report JSON must be a series dataset, {'reports': [...]}, or a list")


def series_from_series_json(path: str | Path) -> list[dict[str, Any]]:
    """Load series records with their shared intelligence reports."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("observation_series"), list):
        raise ValueError("series JSON must contain an 'observation_series' list")
    return data["observation_series"]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load synthetic intelligence reports into Neo4j as evidence nodes.")
    parser.add_argument("--series", type=Path, default=Path("generated/demo_esm_observation_series_with_intel.json"))
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--neo4j-database", default="neo4j")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    series_records = series_from_series_json(args.series)
    etl = ReportNeo4jETL(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
    try:
        result = etl.ingest(series_records)
    finally:
        etl.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
