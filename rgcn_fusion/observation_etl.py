"""ETL utilities for turning ESM observations into Neo4j r-GCN evidence nodes.

The ETL compares each observation's measured radar parameters with RadarMode
nodes already loaded in Neo4j, writes observation/candidate nodes labelled
``EvidenceEntity``, and connects them with typed candidate relationships,
including contradictory candidate relationships, that can be consumed by
:mod:`rgcn_fusion.train`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from neo4j import GraphDatabase

MEASURED_TO_KG_FIELDS = {
    "measured_centre_frequency_ghz": ("centre_frequency_min_ghz", "centre_frequency_max_ghz"),
    "measured_bandwidth_mhz": ("bandwidth_min_mhz", "bandwidth_max_mhz"),
    "measured_prf_hz": ("prf_min_hz", "prf_max_hz"),
    "measured_pulse_width_us": ("pulse_width_min_us", "pulse_width_max_us"),
    "measured_duty_cycle": ("duty_cycle_min", "duty_cycle_max"),
    "measured_coherent_processing_interval_ms": (
        "coherent_processing_interval_min_ms",
        "coherent_processing_interval_max_ms",
    ),
    "measured_dwell_time_ms": ("dwell_time_min_ms", "dwell_time_max_ms"),
}
RESIDUAL_FEATURES_BY_MEASUREMENT = {
    "measured_centre_frequency_ghz": "center_frequency_residual",
    "measured_prf_hz": "prf_residual",
    "measured_bandwidth_mhz": "bandwidth_residual",
    "measured_pulse_width_us": "pulse_width_residual",
    "measured_duty_cycle": "duty_cycle_residual",
    "measured_coherent_processing_interval_ms": "coherent_processing_interval_residual",
    "measured_dwell_time_ms": "dwell_time_residual",
}

DEFAULT_FEATURE_PROPERTIES = ("degree_score", "text_score", "recency_score")
DEFAULT_LABEL_PROPERTY = "ds_masses"
DEFAULT_MAX_CANDIDATES = 5
MIN_CONTRADICTION_SCORE_DELTA = 0.05


@dataclass(frozen=True)
class CandidateScore:
    """A scored KG candidate for one ESM observation."""

    mode_id: str
    radar_id: str | None
    aircraft_id: str | None
    operator: str | None
    mode_score: float
    aircraft_score: float
    total_score: float
    matched_fields: int
    compared_fields: int
    feature_scores: dict[str, float] | None = None


def load_observations(path: str | Path) -> list[dict[str, Any]]:
    """Load observations from the JSON schema emitted by ``esm_observation_generator.py``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    observations = data.get("observations", data if isinstance(data, list) else None)
    if not isinstance(observations, list):
        raise ValueError("observation file must contain an 'observations' list or be a list of observations")
    return observations


def _interval_overlap_score(obs_min: float, obs_max: float, kg_min: float, kg_max: float) -> float:
    if obs_max < obs_min:
        obs_min, obs_max = obs_max, obs_min
    if kg_max < kg_min:
        kg_min, kg_max = kg_max, kg_min
    obs_width = max(obs_max - obs_min, 1e-12)
    overlap = max(0.0, min(obs_max, kg_max) - max(obs_min, kg_min))
    if overlap > 0.0:
        return min(1.0, overlap / obs_width)
    obs_center = (obs_min + obs_max) / 2.0
    kg_center = (kg_min + kg_max) / 2.0
    scale = max(abs(kg_center), abs(obs_center), kg_max - kg_min, 1.0)
    distance = min(abs(obs_center - kg_min), abs(obs_center - kg_max), abs(obs_center - kg_center))
    return max(0.0, 1.0 - distance / scale)


def _measurement_interval(measurement: dict[str, Any]) -> tuple[float, float] | None:
    if not isinstance(measurement, dict):
        return None
    if "min" in measurement and "max" in measurement:
        return float(measurement["min"]), float(measurement["max"])
    if "value" in measurement:
        value = float(measurement["value"])
        error = float(measurement.get("error", 0.0))
        return value - error, value + error
    return None


def _mode_feature_scores(
    observation: dict[str, Any], mode_props: dict[str, Any]
) -> tuple[float, int, int, dict[str, float]]:
    esm = observation.get("esm_radar_parameters", {})
    scores: list[float] = []
    residuals: dict[str, float] = {}
    matched_fields = 0
    missing_fields = 0
    for obs_field, (kg_min_field, kg_max_field) in MEASURED_TO_KG_FIELDS.items():
        interval = _measurement_interval(esm.get(obs_field))
        if interval is None or mode_props.get(kg_min_field) is None or mode_props.get(kg_max_field) is None:
            missing_fields += 1
            continue
        kg_min = float(mode_props[kg_min_field])
        kg_max = float(mode_props[kg_max_field])
        score = _interval_overlap_score(interval[0], interval[1], kg_min, kg_max)
        scores.append(score)
        matched_fields += int(score >= 0.5)

        obs_center = (interval[0] + interval[1]) / 2.0
        kg_center = (kg_min + kg_max) / 2.0
        kg_width = max(kg_max - kg_min, abs(kg_center), 1.0)
        residual_name = RESIDUAL_FEATURES_BY_MEASUREMENT.get(obs_field)
        if residual_name:
            residuals[residual_name] = round(abs(obs_center - kg_center) / kg_width, 6)

    for obs_field, kg_field in (("observed_waveform", "waveform"), ("observed_scan_type", "scan_type")):
        if obs_field in esm and kg_field in mode_props:
            score = 1.0 if esm[obs_field] == mode_props[kg_field] else 0.0
            scores.append(score)
            matched_fields += int(score >= 0.5)
        else:
            missing_fields += 1

    residuals["radar_interval_overlap_score"] = round(sum(scores) / len(scores), 6) if scores else 0.0
    residuals["waveform_match_score"] = 1.0 if esm.get("observed_waveform") == mode_props.get("waveform") else 0.0
    residuals["scan_type_match_score"] = 1.0 if esm.get("observed_scan_type") == mode_props.get("scan_type") else 0.0
    residuals["missing_feature_count"] = float(missing_fields)
    if not scores:
        return 0.0, 0, 0, residuals
    return sum(scores) / len(scores), matched_fields, len(scores), residuals


def _mode_score(observation: dict[str, Any], mode_props: dict[str, Any]) -> tuple[float, int, int]:
    score, matched_fields, compared_fields, _features = _mode_feature_scores(observation, mode_props)
    return score, matched_fields, compared_fields


def _speed_consistency_score(observation: dict[str, Any], aircraft_props: dict[str, Any] | None) -> float:
    if not aircraft_props:
        return 0.5
    kin = observation.get("approximate_kinematics", {})
    speed_max = float(kin.get("ground_speed_max_kph", kin.get("ground_speed_kph", 0.0)))
    aircraft_speed = float(aircraft_props.get("max_speed_mach", 0.0)) * 1060.0
    if not aircraft_speed:
        return 1.0
    return 1.0 if speed_max <= aircraft_speed * 1.05 else max(0.0, aircraft_speed / speed_max)


def _altitude_consistency_score(observation: dict[str, Any], aircraft_props: dict[str, Any] | None) -> float:
    if not aircraft_props:
        return 0.5
    kin = observation.get("approximate_kinematics", {})
    altitude_max = float(kin.get("altitude_max_m", kin.get("altitude_m", 0.0)))
    aircraft_ceiling = float(aircraft_props.get("service_ceiling_m", 0.0))
    if not aircraft_ceiling:
        return 1.0
    return 1.0 if altitude_max <= aircraft_ceiling * 1.05 else max(0.0, aircraft_ceiling / altitude_max)


def _kinematic_score(observation: dict[str, Any], aircraft_props: dict[str, Any] | None) -> float:
    if not aircraft_props:
        return 0.5
    speed_ok = _speed_consistency_score(observation, aircraft_props)
    altitude_ok = _altitude_consistency_score(observation, aircraft_props)
    return (speed_ok + altitude_ok) / 2.0


def _aircraft_radar_score(row: dict[str, Any]) -> float:
    """Score whether KG context links the candidate aircraft to the candidate radar."""
    if not row.get("aircraft_id"):
        return 0.5
    if row.get("aircraft_uses_radar") is not None:
        return 1.0 if bool(row["aircraft_uses_radar"]) else 0.0
    if row.get("radar_id") or row.get("radar_props"):
        # Rows fetched by ObservationNeo4jETL come from
        # (aircraft:AircraftVariant)-[:USES_RADAR]->(radar), so a populated
        # aircraft/radar pair is KG evidence that this aircraft can carry the
        # candidate radar.
        return 1.0
    return 0.5


def _aircraft_score(observation: dict[str, Any], row: dict[str, Any]) -> float:
    """Blend observed kinematics with KG aircraft-to-radar compatibility."""
    kinematic_score = _kinematic_score(observation, row.get("aircraft_props"))
    radar_score = _aircraft_radar_score(row)
    return 0.8 * kinematic_score + 0.2 * radar_score


def _external_prior_score(observation: dict[str, Any], prior_name: str, candidate_value: Any) -> float:
    """Return a neutral-or-context prior without reading supervised truth labels.

    Priors are optional deployment context, not labels.  Accepted shapes are:
    ``external_context.{prior_name}_priors[value]``,
    ``external_context.priors.{prior_name}[value]``, or a single
    ``external_context.{prior_name}`` value that matches the candidate.
    """
    if candidate_value is None:
        return 0.5
    context = observation.get("external_context") or {}
    if not isinstance(context, dict):
        return 0.5

    prior_maps = [
        context.get(f"{prior_name}_priors"),
        (context.get("priors") or {}).get(prior_name)
        if isinstance(context.get("priors"), dict)
        else None,
    ]
    for prior_map in prior_maps:
        if isinstance(prior_map, dict) and candidate_value in prior_map:
            return max(0.0, min(1.0, float(prior_map[candidate_value])))

    contextual_value = context.get(prior_name)
    if contextual_value is None:
        return 0.5
    if isinstance(contextual_value, (list, tuple, set)):
        return 1.0 if candidate_value in contextual_value else 0.0
    return 1.0 if candidate_value == contextual_value else 0.0


def _recency_score(observation: dict[str, Any], *, now: datetime | None = None, half_life_days: float = 30.0) -> float:
    now = now or datetime.now(UTC)
    timestamp = observation.get("timestamp_iso8601")
    if not timestamp:
        return 0.0
    observed_at = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(UTC)
    age_days = max(0.0, (now - observed_at).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def ds_masses_from_score(score: float, ambiguity: float) -> list[float]:
    """Build a two-hypothesis DS mass vector: [non_match, match, uncertain]."""
    uncertainty = min(0.6, max(0.05, ambiguity))
    committed = 1.0 - uncertainty
    match = committed * max(0.0, min(1.0, score))
    non_match = committed - match
    return [round(non_match, 6), round(match, 6), round(uncertainty, 6)]


def score_candidates(
    observation: dict[str, Any],
    kg_rows: Iterable[dict[str, Any]],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[CandidateScore]:
    """Score KG rows against one observation without using supervised truth labels.

    Radar-mode scores come from measured ESM fields, aircraft scores blend
    approximate kinematics with KG aircraft-to-radar compatibility, and any
    operator prior must be supplied through ``external_context`` rather than
    ``ground_truth_label``.
    """
    scored: list[CandidateScore] = []
    for row in kg_rows:
        mode_score, matched_fields, compared_fields, feature_scores = _mode_feature_scores(observation, row["mode_props"])
        aircraft_score = _aircraft_score(observation, row)
        operator_score = _external_prior_score(observation, "operator", row.get("operator"))
        total = 0.75 * mode_score + 0.15 * aircraft_score + 0.10 * operator_score
        scored.append(CandidateScore(
            mode_id=row["mode_id"],
            radar_id=row.get("radar_id"),
            aircraft_id=row.get("aircraft_id"),
            operator=row.get("operator"),
            mode_score=round(mode_score, 6),
            aircraft_score=round(aircraft_score, 6),
            total_score=round(total, 6),
            matched_fields=matched_fields,
            compared_fields=compared_fields,
            feature_scores=feature_scores,
        ))

    return sorted(scored, key=lambda item: item.total_score, reverse=True)[:max_candidates]


def contradiction_edges_for_candidates(
    candidate_ids: list[str],
    candidates: list[CandidateScore],
    *,
    min_score_delta: float = MIN_CONTRADICTION_SCORE_DELTA,
) -> list[dict[str, Any]]:
    """Return directed edges from stronger candidates to incompatible alternatives.

    These edges make counter-evidence explicit for the r-GCN: when one candidate
    hypothesis is scored higher than another candidate for the same observation
    and the candidates point at different mode/radar/aircraft hypotheses, the
    stronger candidate contradicts the weaker alternative.
    """
    contradiction_edges: list[dict[str, Any]] = []
    for left_idx, (left_id, left) in enumerate(zip(candidate_ids, candidates, strict=True)):
        for right_id, right in zip(candidate_ids[left_idx + 1 :], candidates[left_idx + 1 :], strict=True):
            score_delta = round(left.total_score - right.total_score, 6)
            if score_delta < min_score_delta:
                continue
            reasons = [
                field
                for field in ("mode_id", "radar_id", "aircraft_id", "operator")
                if getattr(left, field) != getattr(right, field)
            ]
            if not reasons:
                continue
            contradiction_edges.append({
                "source": left_id,
                "target": right_id,
                "score_delta": score_delta,
                "reason": ",".join(reasons),
            })
    return contradiction_edges


class ObservationNeo4jETL:
    """Insert observations and scored candidate evidence into Neo4j."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def fetch_kg_candidate_rows(self) -> list[dict[str, Any]]:
        """Fetch RadarMode rows with associated Radar, AircraftVariant, and Operator context."""
        query = """
        MATCH (radar:Radar)-[:HAS_MODE]->(mode:RadarMode)
        OPTIONAL MATCH (aircraft:AircraftVariant)-[:USES_RADAR]->(radar)
        OPTIONAL MATCH (operator:Operator)-[:OPERATES]->(aircraft)
        RETURN mode.id AS mode_id,
               properties(mode) AS mode_props,
               radar.id AS radar_id,
               properties(radar) AS radar_props,
               aircraft.id AS aircraft_id,
               properties(aircraft) AS aircraft_props,
               aircraft IS NOT NULL AS aircraft_uses_radar,
               operator.name AS operator
        """
        with self.driver.session(database=self.database) as session:
            return [dict(record) for record in session.run(query)]

    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT evidence_entity_id IF NOT EXISTS FOR (n:EvidenceEntity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT observation_id IF NOT EXISTS FOR (n:Observation) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT candidate_evidence_id IF NOT EXISTS FOR (n:CandidateEvidence) REQUIRE n.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    def ingest(
        self,
        observations: list[dict[str, Any]],
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        include_ground_truth_labels: bool = False,
    ) -> dict[str, int]:
        """Ingest scored observations without leaking truth labels by default.

        ``ground_truth_label`` may be present on input observations for offline
        evaluation. Unless ``include_ground_truth_labels`` is explicitly enabled,
        those labels are not copied onto EvidenceEntity nodes where future
        classification models might consume them as features or targets.
        """
        self.ensure_constraints()
        kg_rows = self.fetch_kg_candidate_rows()
        obs_rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        candidate_edges: list[dict[str, Any]] = []
        contradiction_edges: list[dict[str, Any]] = []
        truth_edges: list[dict[str, Any]] = []
        similarity_edges: list[dict[str, Any]] = []
        by_best_mode: dict[str, list[str]] = {}

        now = datetime.now(UTC)
        for observation in observations:
            candidates = score_candidates(observation, kg_rows, max_candidates=max_candidates)
            if not candidates:
                continue
            best = candidates[0]
            second_score = candidates[1].total_score if len(candidates) > 1 else 0.0
            ambiguity = max(0.05, 1.0 - (best.total_score - second_score)) if len(candidates) > 1 else 0.1
            obs_id = observation["observation_id"]
            label = observation.get("ground_truth_label", {})
            obs_node_id = f"evidence:observation:{obs_id}"
            recency = _recency_score(observation, now=now)
            obs_row = {
                "id": obs_node_id,
                "observation_id": obs_id,
                "timestamp_iso8601": observation.get("timestamp_iso8601"),
                "degree_score": round(len(candidates) / max_candidates, 6),
                "text_score": best.total_score,
                "recency_score": round(recency, 6),
                DEFAULT_LABEL_PROPERTY: ds_masses_from_score(best.total_score, ambiguity),
                "radar_id": label.get("radar_id") if include_ground_truth_labels else None,
                "mode_id": label.get("mode_id") if include_ground_truth_labels else None,
                "aircraft_id": label.get("aircraft_id") if include_ground_truth_labels else None,
                "operator": label.get("operator") if include_ground_truth_labels else None,
                "best_candidate_mode_id": best.mode_id,
                "best_candidate_aircraft_id": best.aircraft_id,
                "best_candidate_score": best.total_score,
            }
            obs_rows.append(obs_row)
            by_best_mode.setdefault(best.mode_id, []).append(obs_node_id)

            candidate_ids: list[str] = []
            for rank, candidate in enumerate(candidates, start=1):
                candidate_id = f"evidence:candidate:{obs_id}:{rank}"
                candidate_ids.append(candidate_id)
                candidate_aircraft_props = next(
                    (row.get("aircraft_props") for row in kg_rows if row.get("aircraft_id") == candidate.aircraft_id),
                    None,
                )
                candidate_rows.append({
                    "id": candidate_id,
                    "observation_id": obs_id,
                    "rank": rank,
                    "degree_score": round(candidate.matched_fields / max(candidate.compared_fields, 1), 6),
                    "text_score": candidate.total_score,
                    "recency_score": round(recency, 6),
                    DEFAULT_LABEL_PROPERTY: ds_masses_from_score(candidate.total_score, 0.2 if rank == 1 else 0.35),
                    "radar_id": candidate.radar_id,
                    "mode_id": candidate.mode_id,
                    "aircraft_id": candidate.aircraft_id,
                    "operator": candidate.operator,
                    "mode_score": candidate.mode_score,
                    "aircraft_score": candidate.aircraft_score,
                    "speed_consistency_score": _speed_consistency_score(observation, candidate_aircraft_props),
                    "altitude_consistency_score": _altitude_consistency_score(observation, candidate_aircraft_props),
                    "heading_consistency_score": 1.0,
                    "observation_uncertainty_width": round(
                        sum(
                            abs(float(v.get("max", 0.0)) - float(v.get("min", 0.0)))
                            for v in observation.get("esm_radar_parameters", {}).values()
                            if isinstance(v, dict) and "min" in v and "max" in v
                        ),
                        6,
                    ),
                    "candidate_ambiguity_count": float(len(candidates)),
                    **(candidate.feature_scores or {}),
                })
                candidate_edges.append({"source": obs_node_id, "target": candidate_id, "score": candidate.total_score, "rank": rank})
                if (
                    include_ground_truth_labels
                    and candidate.mode_id == label.get("mode_id")
                    and candidate.aircraft_id == label.get("aircraft_id")
                ):
                    truth_edges.append({"source": obs_node_id, "target": candidate_id})

            contradiction_edges.extend(contradiction_edges_for_candidates(candidate_ids, candidates))

        for ids in by_best_mode.values():
            for left, right in zip(ids, ids[1:]):
                similarity_edges.append({"source": left, "target": right})
                similarity_edges.append({"source": right, "target": left})

        with self.driver.session(database=self.database) as session:
            session.execute_write(
                _write_evidence_rows,
                obs_rows,
                candidate_rows,
                candidate_edges,
                contradiction_edges,
                truth_edges,
                similarity_edges,
            )
        return {
            "observations": len(obs_rows),
            "candidates": len(candidate_rows),
            "candidate_edges": len(candidate_edges),
            "contradiction_edges": len(contradiction_edges),
            "truth_edges": len(truth_edges),
            "similarity_edges": len(similarity_edges),
        }


def _write_evidence_rows(
    tx,
    obs_rows,
    candidate_rows,
    candidate_edges,
    contradiction_edges,
    truth_edges,
    similarity_edges,
):
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:EvidenceEntity:Observation {id: row.id})
        SET n += row
        """,
        rows=obs_rows,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MERGE (n:EvidenceEntity:CandidateEvidence {id: row.id})
        SET n += row
        """,
        rows=candidate_rows,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (source:EvidenceEntity {id: row.source})
        MATCH (target:EvidenceEntity {id: row.target})
        MERGE (source)-[r:HAS_CANDIDATE]->(target)
        SET r.score = row.score, r.rank = row.rank
        """,
        rows=candidate_edges,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (source:EvidenceEntity {id: row.source})
        MATCH (target:EvidenceEntity {id: row.target})
        MERGE (source)-[r:CONTRADICTS_CANDIDATE]->(target)
        SET r.score_delta = row.score_delta, r.reason = row.reason
        """,
        rows=contradiction_edges,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (source:EvidenceEntity {id: row.source})
        MATCH (target:EvidenceEntity {id: row.target})
        MERGE (source)-[:GROUND_TRUTH_CANDIDATE]->(target)
        """,
        rows=truth_edges,
    )
    tx.run(
        """
        UNWIND $rows AS row
        MATCH (source:EvidenceEntity {id: row.source})
        MATCH (target:EvidenceEntity {id: row.target})
        MERGE (source)-[:SHARES_BEST_MODE]->(target)
        """,
        rows=similarity_edges,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load ESM observations into Neo4j as r-GCN evidence nodes.")
    parser.add_argument("--observations", type=Path, default=Path("generated/esm_observations.json"))
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument(
        "--include-ground-truth-labels",
        action="store_true",
        help="Copy supervised truth labels and truth edges into Neo4j for offline evaluation only",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    observations = load_observations(args.observations)
    etl = ObservationNeo4jETL(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.neo4j_database)
    try:
        result = etl.ingest(
            observations,
            max_candidates=args.max_candidates,
            include_ground_truth_labels=args.include_ground_truth_labels,
        )
    finally:
        etl.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
