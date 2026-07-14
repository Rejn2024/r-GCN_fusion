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
DEFAULT_REPORT_RECENCY_HALF_LIFE_DAYS = 14.0


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
        claim_type = claim_cycle[idx % len(claim_cycle)]
        correct = rng.random() < 0.72
        if idx in (1, 5):
            correct = False
        stance = "supports" if rng.random() > 0.12 else "refutes"
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
        reports.append(report)
    return reports


def add_intelligence_reports_to_series(
    data: dict[str, Any], *, seed: int = 7, min_reports: int = MIN_REPORTS_PER_OBSERVATION, max_reports: int = MAX_REPORTS_PER_OBSERVATION
) -> dict[str, Any]:
    """Attach synthetic intelligence reports to every observation in a series dataset."""
    rng = random.Random(seed)
    enriched = json.loads(json.dumps(data))
    for series in enriched.get("observation_series", []):
        for obs in series.get("observations", []):
            obs["intelligence_reports"] = generate_intelligence_reports_for_observation(
                obs,
                seed=rng.getrandbits(64),
                min_reports=min_reports,
                max_reports=max_reports,
            )
    meta = enriched.setdefault("metadata", {})
    meta["intelligence_reports_per_observation"] = [min_reports, max_reports]
    meta["intelligence_claim_types"] = list(CLAIM_TYPES)
    return enriched


def flatten_reports_from_series(data: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for series in data.get("observation_series", []):
        for obs in series.get("observations", []):
            reports.extend(obs.get("intelligence_reports") or [])
    return reports


def build_report_evidence_rows(observations: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Transform generated reports into ingestion-ready node and edge rows."""
    report_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    contains_edges: list[dict[str, Any]] = []
    support_edges: list[dict[str, Any]] = []
    contradiction_edges: list[dict[str, Any]] = []
    claims_by_observation: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        obs_time = _parse_utc(obs.get("timestamp_iso8601"))
        for report in obs.get("intelligence_reports") or []:
            recency = report_recency_score(report, reference_time=obs_time)
            report_rows.append({
                "id": f"evidence:report:{report['report_id']}",
                "report_id": report["report_id"],
                "observation_id": obs["observation_id"],
                "series_id": obs.get("series_id"),
                "source_id": report.get("source_id"),
                "source_type": report.get("source_type"),
                "published_at": report.get("published_at"),
                "collected_at": report.get("collected_at"),
                "credibility_score": float(report.get("credibility_score", 0.5)),
                "recency_score": round(recency, 6),
                "degree_score": float(len(report.get("claims") or [])),
                "text_score": float(report.get("credibility_score", 0.5)),
                "ds_masses": ds_masses_from_score(float(report.get("credibility_score", 0.5)) * recency, 0.25),
            })
            for claim in report.get("claims") or []:
                score = report_claim_score(report, claim, observation_time=obs_time)
                claim_id = f"evidence:claim:{claim['claim_id']}"
                claim_row = {
                    "id": claim_id,
                    "claim_id": claim["claim_id"],
                    "report_id": report["report_id"],
                    "observation_id": obs["observation_id"],
                    "series_id": obs.get("series_id"),
                    "claim_type": claim.get("claim_type"),
                    "stance": claim.get("stance", "supports"),
                    "object_id": claim.get("object_id"),
                    "object_value": claim.get("object_value"),
                    "credibility_score": float(report.get("credibility_score", 0.5)),
                    "recency_score": round(recency, 6),
                    "claim_confidence": float(claim.get("claim_confidence", 0.5)),
                    "extraction_confidence": float(claim.get("extraction_confidence", 0.5)),
                    "degree_score": 1.0,
                    "text_score": score,
                    "ds_masses": ds_masses_from_score(score, 0.2 if claim.get("stance") == "supports" else 0.35),
                }
                claim_rows.append(claim_row)
                contains_edges.append({"source": f"evidence:report:{report['report_id']}", "target": claim_id})
                support_edges.append({"source": claim_id, "target": f"evidence:observation:{obs['observation_id']}", "score": score, "stance": claim.get("stance", "supports")})
                claims_by_observation.setdefault(obs["observation_id"], []).append(claim_row)
    for claim_rows_for_obs in claims_by_observation.values():
        for left_idx, left in enumerate(claim_rows_for_obs):
            for right in claim_rows_for_obs[left_idx + 1:]:
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

    def ingest(self, observations: list[dict[str, Any]]) -> dict[str, int]:
        rows = build_report_evidence_rows(observations)
        with self.driver.session(database=self.database) as session:
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


def load_reports_json(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "observation_series" in data:
        return flatten_reports_from_series(data)
    if isinstance(data, dict) and "reports" in data:
        return data["reports"]
    if isinstance(data, list):
        return data
    raise ValueError("report JSON must be a series dataset, {'reports': [...]}, or a list")


def observations_from_series_json(path: str | Path) -> list[dict[str, Any]]:
    """Load observations, including nested intelligence reports, from a series JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("observation_series"), list):
        raise ValueError("series JSON must contain an 'observation_series' list")
    return [obs for series in data["observation_series"] for obs in series.get("observations", [])]


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
    observations = observations_from_series_json(args.series)
    etl = ReportNeo4jETL(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
    try:
        result = etl.ingest(observations)
    finally:
        etl.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
