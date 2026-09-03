"""Parallel, deterministic candidate scoring for notebook graph construction."""

from __future__ import annotations

from array import array
from datetime import UTC, datetime
import heapq
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
) -> tuple[int, dict[str, Any]]:
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
) -> tuple[int, dict[str, Any]]:
    """Build an index-local graph fragment which can be merged deterministically.

    Global node indices are deliberately not assigned in workers.  This avoids shared
    mutable state and lets the parent concatenate fragments in source-series order.
    """
    series = task[1]
    observations = sorted(series["observations"], key=lambda obs: obs["sequence_index"])
    prepared_reports = _prepare_reports(series)
    position, scored, applicable_report_ordinals, first_observation_claim_scores = (
        _score_series_observations(task, context, prepared_reports)
    )
    rows: list[dict[str, float]] = []
    metadata: list[dict[str, Any]] = []
    observation_offsets: list[int] = []
    candidates: list[tuple[int, int, array]] = []
    claim_offsets: list[int] = []
    report_kg_edges: list[tuple[int, str]] = []
    report_offsets: list[int] = []
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
                report_offsets.append(report_offset)
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
                    claim_score = first_observation_claim_scores[len(claim_offsets)]
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

    report_links = [
        (report_offsets[report_ordinal], observation_offsets[observation_ordinal])
        for observation_ordinal, report_ordinals in enumerate(
            applicable_report_ordinals
        )
        for report_ordinal in report_ordinals
    ]
    return position, {
        "feature_rows": rows,
        "node_meta": metadata,
        "observation_offsets": observation_offsets,
        "candidate_links": candidates,
        "claim_offsets": claim_offsets,
        "report_kg_edges": report_kg_edges,
        "report_links": report_links,
    }


def score_series_observations(
    task: tuple[int, dict[str, Any]], context: dict[str, Any]
) -> tuple[int, dict[str, Any], dict[str, dict[str, dict[str, float | str]]]]:
    """Score every observation in one series without assigning global node indices."""
    position, results, applicable_reports, _claim_scores = _score_series_observations(
        task, context, _prepare_reports(task[1])
    )
    series = task[1]
    reports = series.get("intelligence_reports") or []
    observations = sorted(series["observations"], key=lambda obs: obs["sequence_index"])
    # Preserve the public diagnostic payload while fragment construction uses compact
    # integer offsets and never sends these dictionaries through the process pool.
    report_proximities = {
        observation["observation_id"]: {
            reports[report_ordinal]["report_id"]: report_observation_proximity(
                reports[report_ordinal], observation
            )
            for report_ordinal in applicable_reports[observation_ordinal]
        }
        for observation_ordinal, observation in enumerate(observations)
    }
    return position, results, report_proximities


def _prepare_reports(
    series: dict[str, Any],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Attach invariant claim provenance once per track rather than per observation."""
    return [
        (
            report,
            [
                {
                    **claim,
                    "id": f"evidence:claim:{claim['claim_id']}",
                    "report_id": report.get("report_id"),
                    "source_id": report.get("source_id"),
                    "series_id": series["series_id"],
                }
                for claim in report.get("claims") or []
            ],
        )
        for report in series.get("intelligence_reports") or []
    ]


def _score_series_observations(
    task: tuple[int, dict[str, Any]],
    context: dict[str, Any],
    prepared_reports: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[int, dict[str, Any], list[list[int]], list[float]]:
    """Score a track and return compact report applicability for fragment assembly."""
    series_position, series = task
    results: dict[str, Any] = {}
    applicable_report_ordinals: list[list[int]] = []
    observations = sorted(series["observations"], key=lambda obs: obs["sequence_index"])
    first_timestamp = observations[0].get("timestamp_iso8601") if observations else None
    first_observation_time = (
        datetime.fromisoformat(first_timestamp.replace("Z", "+00:00")).astimezone(UTC)
        if first_timestamp
        else None
    )
    first_observation_claim_scores = [
        report_claim_score(report, claim, observation_time=first_observation_time)
        for report, prepared_claims in prepared_reports
        for claim in prepared_claims
    ]
    claim_ordinal_by_evidence_id = {
        f"evidence:claim:{claim['claim_id']}": claim_ordinal
        for claim_ordinal, claim in enumerate(
            claim
            for report in series.get("intelligence_reports") or []
            for claim in report.get("claims") or []
        )
    }
    for observation_ordinal, obs in enumerate(observations):
        timestamp = obs.get("timestamp_iso8601")
        observation_time = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(UTC)
            if timestamp
            else None
        )
        applicable_claims = []
        applicable_reports = []
        for report_ordinal, (report, prepared_claims) in enumerate(prepared_reports):
            proximity = report_observation_proximity(report, obs)
            if proximity is None:
                continue
            applicable_reports.append(report_ordinal)
            for prepared_claim in prepared_claims:
                claim_score = (
                    first_observation_claim_scores[
                        claim_ordinal_by_evidence_id[prepared_claim["id"]]
                    ]
                    if observation_ordinal == 0
                    else report_claim_score(
                        report, prepared_claim, observation_time=observation_time
                    )
                )
                applicable_claims.append({**prepared_claim, "text_score": claim_score})
        applicable_report_ordinals.append(applicable_reports)

        def enriched_candidates():
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
                yield (
                    intelligence["final_score"],
                    sensor_rank,
                    score,
                    operator,
                    candidate,
                    intelligence,
                    compact_direct_edges,
                )

        results[obs["observation_id"]] = _stable_nlargest(
            enriched_candidates(),
            context["max_kg_candidates"],
            key=lambda item: item[0],
        )
    return (
        series_position,
        results,
        applicable_report_ordinals,
        first_observation_claim_scores,
    )


def _stable_nlargest(items: Any, limit: int, *, key: Any) -> list[Any]:
    """Return a deterministic top-k without sorting or retaining the complete input."""
    selected = heapq.nlargest(
        limit,
        enumerate(items),
        key=lambda indexed: (key(indexed[1]), -indexed[0]),
    )
    selected.sort(key=lambda indexed: (-key(indexed[1]), indexed[0]))
    return [item for _index, item in selected]


def _candidate_scores(
    observation: dict[str, Any], context: dict[str, Any]
) -> list[tuple[float, Any, Any]]:
    """Score templates once, then expand their operator variants."""
    templates = context["candidate_templates"]
    variants = context["candidate_variants"]
    template_scores = score_candidates(
        observation, templates, max_candidates=len(templates)
    )
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

    def expanded():
        for score in template_scores:
            for row in variants[(score.mode_id, score.radar_id, score.aircraft_id)]:
                operator = row["operator"]
                if isinstance(operator_priors, dict) and operator in operator_priors:
                    operator_score = max(
                        0.0, min(1.0, float(operator_priors[operator]))
                    )
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
                yield final_score, score, operator

    # Keep retrieval broad enough for intelligence to rescue a candidate that is
    # plausible but not in the final sensor-only shortlist.  Graph size is bounded
    # separately after intelligence-aware reranking in score_series_observations.
    return _stable_nlargest(
        expanded(), context["max_kg_retrieval_candidates"], key=lambda item: item[0]
    )
