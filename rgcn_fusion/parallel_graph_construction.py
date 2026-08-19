"""Parallel, deterministic candidate scoring for notebook graph construction."""

from __future__ import annotations

from array import array
from datetime import UTC, datetime
from typing import Any

from .intelligence_reports import (
    aggregate_candidate_intelligence,
    report_claim_score,
    report_observation_proximity,
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
                if (
                    claim_ordinal is None
                    or abs(contribution)
                    < context.get("claim_candidate_min_abs_contribution", 0.0)
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
    contextual_operator = external.get("operator") if isinstance(external, dict) else None
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
