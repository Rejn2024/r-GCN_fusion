"""Parallel, deterministic candidate scoring for notebook graph construction."""

from __future__ import annotations

from array import array
from datetime import UTC, datetime
from typing import Any

from .intelligence_reports import (
    aggregate_candidate_intelligence,
    report_claim_score,
    report_observation_proximity,
    report_recency_score,
)
from .observation_etl import ds_masses_from_score, score_candidates

_WORKER_CONTEXT: dict[str, Any] | None = None


def initialise_scoring_worker(context: dict[str, Any]) -> None:
    """Install read-only scoring data once in each process-pool worker."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def score_series_in_worker(
    task: tuple[int, dict[str, Any]],
) -> tuple[int, dict[str, Any], dict[str, dict[str, dict[str, float | str]]]]:
    """Process-pool entry point; results retain their source-series position."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("scoring worker has not been initialised")
    return score_series_observations(task, _WORKER_CONTEXT)


def build_series_fragment_in_worker(
    task: tuple[int, dict[str, Any]],
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Score and materialise one series using the process worker's context."""
    if _WORKER_CONTEXT is None:
        raise RuntimeError("scoring worker has not been initialised")
    return build_series_fragment(task, _WORKER_CONTEXT)


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten_numeric(f"{prefix}.{key}" if prefix else key, nested, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def _segment_indices(observations: list[dict[str, Any]], threshold: float) -> list[int]:
    def frequency(observation: dict[str, Any]) -> float | None:
        value = observation.get("esm_radar_parameters", {}).get(
            "measured_centre_frequency_ghz"
        )
        if isinstance(value, dict):
            value = value.get("value")
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    frequencies = [frequency(observation) for observation in observations]
    segments, current = [], 0
    for index, value in enumerate(frequencies):
        if (
            index
            and value is not None
            and frequencies[index - 1] is not None
            and abs(value - frequencies[index - 1]) > threshold
        ):
            current += 1
        segments.append(current)
    return segments


def build_series_fragment(
    task: tuple[int, dict[str, Any]], context: dict[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Build an index-local graph fragment which can be merged deterministically.

    Global node indices are deliberately not assigned in workers.  This avoids shared
    mutable state and lets the parent concatenate fragments in source-series order.
    """
    position, scored, report_proximities = score_series_observations(task, context)
    series = task[1]
    observations = sorted(series["observations"], key=lambda obs: obs["sequence_index"])
    rows: list[dict[str, float]] = []
    metadata: list[dict[str, Any]] = []
    observation_offsets: list[int] = []
    candidates: list[tuple[int, int, array]] = []
    claim_offsets: list[int] = []
    report_kg_edges: list[tuple[int, str]] = []
    include_reports = context.get("include_intel_report_nodes", True)
    include_candidates = context.get("include_candidate_nodes", True)
    n = max(len(observations), 1)

    for obs, segment in zip(
        observations,
        _segment_indices(observations, context["segment_frequency_shift_ghz"]),
    ):
        observation_offset = len(rows)
        observation_offsets.append(observation_offset)
        features = {
            "node_kind_observation": 1.0,
            "node_kind_candidate": 0.0,
            "elapsed_time_s": float(obs.get("elapsed_time_s", 0.0)),
            "sequence_fraction": float(obs.get("sequence_index", 0)) / max(n - 1, 1),
            "segment_index": float(segment),
            "duration_s": float(series.get("duration_s", 0.0)),
            "observation_count": float(series.get("observation_count", n)),
        }
        _flatten_numeric("esm", obs.get("esm_radar_parameters", {}), features)
        _flatten_numeric("kin", obs.get("approximate_kinematics", {}), features)
        _flatten_numeric("loc", obs.get("estimated_emitter_location", {}), features)
        rows.append(features)
        timestamp = obs.get("timestamp_iso8601")
        observation_time = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(UTC)
            if timestamp
            else None
        )
        metadata.append(
            {
                "node_kind": "observation",
                "observation_id": obs["observation_id"],
                "series_id": obs["series_id"],
                "sequence_index": obs["sequence_index"],
                "timestamp_iso8601": timestamp,
            }
        )

        if include_reports and obs is observations[0]:
            for report_rank, report in enumerate(
                series.get("intelligence_reports") or [], start=1
            ):
                report_offset = len(rows)
                rows.append(
                    {
                        "node_kind_observation": 0.0,
                        "node_kind_candidate": 0.0,
                        "node_kind_report": 1.0,
                        "node_kind_claim": 0.0,
                        "report_rank": float(report_rank),
                        "report_credibility_score": float(
                            report.get("credibility_score", 0.5)
                        ),
                        "report_recency_score": float(
                            report_recency_score(
                                report, reference_time=observation_time
                            )
                        ),
                        "report_claim_count": float(len(report.get("claims") or [])),
                    }
                )
                metadata.append(
                    {
                        "node_kind": "intelligence_report",
                        "report_id": report["report_id"],
                        "report_type": report.get("report_type"),
                        "series_id": obs["series_id"],
                    }
                )
                for claim_rank, claim in enumerate(report.get("claims") or [], start=1):
                    claim_offset = len(rows)
                    claim_score = report_claim_score(
                        report, claim, observation_time=observation_time
                    )
                    rows.append(
                        {
                            "node_kind_observation": 0.0,
                            "node_kind_candidate": 0.0,
                            "node_kind_report": 0.0,
                            "node_kind_claim": 1.0,
                            "claim_rank": float(claim_rank),
                            "claim_confidence": float(
                                claim.get("claim_confidence", 0.5)
                            ),
                            "claim_extraction_confidence": float(
                                claim.get("extraction_confidence", 0.5)
                            ),
                            "claim_specificity_score": float(
                                claim.get("specificity_score", 0.5)
                            ),
                            "claim_kg_consistency_score": float(
                                claim.get("kg_consistency_score", 0.5)
                            ),
                            "claim_text_score": float(claim_score),
                            "claim_supports": (
                                1.0
                                if claim.get("stance", "supports") == "supports"
                                else 0.0
                            ),
                            "claim_refutes": (
                                1.0 if claim.get("stance") == "refutes" else 0.0
                            ),
                        }
                    )
                    claim_offsets.append(claim_offset)
                    metadata.append(
                        {
                            "node_kind": "report_claim",
                            "id": f"evidence:claim:{claim['claim_id']}",
                            "claim_id": claim["claim_id"],
                            "claim_type": claim.get("claim_type"),
                            "object_id": claim.get("object_id"),
                            "kg_entity_id": claim.get("kg_entity_id"),
                            "stance": claim.get("stance", "supports"),
                            "text_score": claim_score,
                            "source_id": report.get("source_id"),
                            "report_id": report.get("report_id"),
                            "report_node_offset": report_offset,
                            "observation_node_indices": [],
                            "series_id": obs["series_id"],
                        }
                    )
                    if claim.get("kg_entity_id") is not None:
                        report_kg_edges.append((claim_offset, claim["kg_entity_id"]))

        if include_candidates:
            enriched = scored[obs["observation_id"]]
            for rank, (
                _final_score,
                sensor_rank,
                score,
                _operator,
                _candidate,
                intelligence,
                direct_edges,
            ) in enumerate(enriched, start=1):
                candidate_offset = len(rows)
                fused_non_match, fused_match, fused_uncertain = intelligence[
                    "ds_masses"
                ]
                intel_features = {
                    f"candidate_{name}": float(intelligence[name])
                    for name in (
                        "sensor_score",
                        "intel_support_score",
                        "intel_refute_score",
                        "intel_net_score",
                        "intel_score",
                        "intel_conflict",
                        "intel_uncertainty",
                        "intel_claim_count",
                        "intel_source_count",
                        "intel_effective_weight",
                        "final_score",
                    )
                }
                rows.append(
                    {
                        "node_kind_observation": 0.0,
                        "node_kind_candidate": 1.0,
                        "candidate_rank": float(rank),
                        "candidate_sensor_rank": float(sensor_rank),
                        "candidate_rank_fraction": float(rank - 1)
                        / max(len(enriched) - 1, 1),
                        "candidate_count": float(len(enriched)),
                        "candidate_mode_score": float(score.mode_score),
                        "candidate_aircraft_score": float(score.aircraft_score),
                        "candidate_total_score": float(intelligence["final_score"]),
                        "candidate_fused_non_match_mass": float(fused_non_match),
                        "candidate_fused_match_mass": float(fused_match),
                        "candidate_fused_uncertain_mass": float(fused_uncertain),
                        "candidate_matched_fields": float(score.matched_fields),
                        "candidate_compared_fields": float(score.compared_fields),
                        **intel_features,
                        **{
                            f"candidate_{name}": float(value)
                            for name, value in (
                                getattr(score, "feature_scores", None) or {}
                            ).items()
                        },
                    }
                )
                metadata.append(
                    {
                        "node_kind": "candidate",
                        "series_id": obs["series_id"],
                        "rank": rank,
                    }
                )
                candidates.append((observation_offset, candidate_offset, direct_edges))

    return (
        position,
        {
            "feature_rows": rows,
            "node_meta": metadata,
            "observation_offsets": observation_offsets,
            "candidate_links": candidates,
            "claim_offsets": claim_offsets,
            "report_kg_edges": report_kg_edges,
        },
        report_proximities,
    )


def score_series_observations(
    task: tuple[int, dict[str, Any]], context: dict[str, Any]
) -> tuple[int, dict[str, Any], dict[str, dict[str, dict[str, float | str]]]]:
    """Score every observation in one series without assigning global node indices."""
    series_position, series = task
    results: dict[str, Any] = {}
    report_proximities: dict[str, dict[str, dict[str, float | str]]] = {}
    observations = sorted(series["observations"], key=lambda obs: obs["sequence_index"])
    claim_ordinal_by_evidence_id = {
        f"evidence:claim:{claim['claim_id']}": claim_ordinal
        for claim_ordinal, claim in enumerate(
            claim
            for report in series.get("intelligence_reports") or []
            for claim in report.get("claims") or []
        )
    }
    for obs in observations:
        timestamp = obs.get("timestamp_iso8601")
        observation_time = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(UTC)
            if timestamp
            else None
        )
        applicable_claims = []
        observation_report_proximities = {}
        for report in series.get("intelligence_reports") or []:
            proximity = report_observation_proximity(report, obs)
            if proximity is None:
                continue
            observation_report_proximities[report["report_id"]] = proximity
            for claim in report.get("claims") or []:
                applicable_claims.append(
                    {
                        **claim,
                        "id": f"evidence:claim:{claim['claim_id']}",
                        "report_id": report.get("report_id"),
                        "source_id": report.get("source_id"),
                        "series_id": obs["series_id"],
                        "text_score": report_claim_score(
                            report, claim, observation_time=observation_time
                        ),
                    }
                )

        report_proximities[obs["observation_id"]] = observation_report_proximities

        enriched_candidates = []
        for sensor_rank, (sensor_score, score, operator) in enumerate(
            _candidate_scores(obs, context), start=1
        ):
            candidate = {
                "id": f"candidate:{obs['observation_id']}:{sensor_rank}",
                "observation_id": obs["observation_id"],
                "series_id": obs["series_id"],
                "mode_id": score.mode_id,
                "radar_id": score.radar_id,
                "aircraft_id": score.aircraft_id,
                "aircraft_family_id": context["aircraft_family_by_aircraft"].get(
                    score.aircraft_id
                ),
                "operator": operator,
                "relation_id": (
                    f"relation:{score.aircraft_id}:USES_RADAR:{score.radar_id}"
                    if score.aircraft_id is not None and score.radar_id is not None
                    else None
                ),
                "sensor_score": sensor_score,
                "sensor_ds_masses": ds_masses_from_score(
                    sensor_score, 0.2 if sensor_rank == 1 else 0.35
                ),
            }
            intelligence, direct_edges = aggregate_candidate_intelligence(
                candidate, applicable_claims
            )
            # The notebook needs only claim identity and contribution polarity.
            # Pack both into one signed int32 instead of returning an edge dictionary
            # for every claim/candidate pair through the process-pool pickle channel.
            compact_direct_edges = array("i")
            for edge in direct_edges:
                contribution = edge["contribution"]
                claim_ordinal = claim_ordinal_by_evidence_id.get(edge["source"])
                if claim_ordinal is None or abs(contribution) < context.get(
                    "claim_candidate_min_abs_contribution", 0.0
                ):
                    continue
                encoded_ordinal = claim_ordinal + 1
                compact_direct_edges.append(
                    encoded_ordinal if contribution > 0.0 else -encoded_ordinal
                )
            enriched_candidates.append(
                (
                    intelligence["final_score"],
                    sensor_rank,
                    score,
                    operator,
                    candidate,
                    intelligence,
                    compact_direct_edges,
                )
            )
        enriched_candidates.sort(key=lambda item: item[0], reverse=True)
        results[obs["observation_id"]] = enriched_candidates[
            : context["max_kg_candidates"]
        ]
    return series_position, results, report_proximities


def _candidate_scores(
    observation: dict[str, Any], context: dict[str, Any]
) -> list[tuple[float, Any, Any]]:
    """Score templates once, then expand their operator variants."""
    templates = context["candidate_templates"]
    variants = context["candidate_variants"]
    template_scores = score_candidates(
        observation, templates, max_candidates=len(templates)
    )
    expanded = []
    external = observation.get("external_context") or {}
    priors = external.get("priors") if isinstance(external.get("priors"), dict) else {}
    operator_priors = (
        external.get("operator_priors", priors.get("operator", {}))
        if isinstance(external, dict)
        else {}
    )
    contextual_operator = (
        external.get("operator") if isinstance(external, dict) else None
    )
    for score in template_scores:
        for row in variants[(score.mode_id, score.radar_id, score.aircraft_id)]:
            operator = row["operator"]
            if isinstance(operator_priors, dict) and operator in operator_priors:
                operator_score = max(0.0, min(1.0, float(operator_priors[operator])))
            elif isinstance(contextual_operator, (list, tuple, set)):
                operator_score = 1.0 if operator in contextual_operator else 0.0
            elif contextual_operator is None:
                operator_score = 0.5
            else:
                operator_score = 1.0 if operator == contextual_operator else 0.0
            final_score = round(
                0.75 * score.mode_score
                + 0.15 * score.aircraft_score
                + 0.10 * operator_score,
                6,
            )
            expanded.append((final_score, score, operator))
    # Keep retrieval broad enough for intelligence to rescue a candidate that is
    # plausible but not in the final sensor-only shortlist.  Graph size is bounded
    # separately after intelligence-aware reranking in score_series_observations.
    return sorted(expanded, key=lambda item: item[0], reverse=True)[
        : context["max_kg_retrieval_candidates"]
    ]
