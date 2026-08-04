from datetime import UTC, datetime

from esm_observation_series_generator import generate_observation_series_with_intelligence_reports
from rgcn_fusion.intelligence_reports import (
    CLAIM_TYPES,
    aggregate_candidate_intelligence,
    build_candidate_intelligence_rows,
    build_report_evidence_rows,
    claim_candidate_contribution,
    final_signed_compatibility,
    flatten_reports_from_series,
    report_claim_score,
)


def test_series_generator_keeps_measurements_per_observation_and_reports_per_series():
    data = generate_observation_series_with_intelligence_reports(
        count=2, seed=101, intelligence_seed=202,
        start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 2, tzinfo=UTC),
        workers=1,
    )
    assert data["metadata"]["intelligence_reports_per_series"] == [10, 12]
    for series in data["observation_series"]:
        reports = series["intelligence_reports"]
        observation_ids = [obs["observation_id"] for obs in series["observations"]]
        assert 10 <= len(reports) <= 12
        assert all(report["valid_for_observation_ids"] == observation_ids for report in reports)
        assert all(report["series_id"] == series["series_id"] for report in reports)
        for obs in series["observations"]:
            assert "esm_radar_parameters" in obs
            assert "approximate_kinematics" in obs
            assert "intelligence_reports" not in obs
    assert data["metadata"]["intelligence_claim_types"] == list(CLAIM_TYPES)


def test_flatten_reports_does_not_duplicate_shared_series_reports():
    data = generate_observation_series_with_intelligence_reports(count=2, seed=303, intelligence_seed=404, workers=1)
    reports = flatten_reports_from_series(data)
    assert len(reports) == sum(len(series["intelligence_reports"]) for series in data["observation_series"])
    assert len({report["report_id"] for report in reports}) == len(reports)


def test_report_evidence_links_each_shared_claim_to_every_observation():
    data = generate_observation_series_with_intelligence_reports(count=1, seed=505, intelligence_seed=606, workers=1)
    series = data["observation_series"][0]
    rows = build_report_evidence_rows(data["observation_series"])
    claim_count = sum(len(report["claims"]) for report in series["intelligence_reports"])
    assert len(rows["reports"]) == len(series["intelligence_reports"])
    assert len(rows["claims"]) == claim_count
    assert len(rows["support_edges"]) == claim_count * len(series["observations"])
    assert rows["contradiction_edges"]
    assert all(abs(sum(row["ds_masses"]) - 1.0) < 1e-6 for row in rows["claims"])

def test_report_claim_score_uses_optional_external_priors():
    report = {
        "published_at": "2025-01-01T00:00:00Z",
        "collected_at": "2025-01-01T00:00:00Z",
        "credibility_score": 0.8,
        "external_context": {"operator_priors": {"Favoured": 1.0, "Disfavoured": 0.0}},
    }
    base_claim = {
        "claim_type": "operator",
        "claim_confidence": 0.8,
        "extraction_confidence": 0.8,
        "specificity_score": 0.8,
        "kg_consistency_score": 0.8,
    }

    favoured = report_claim_score(report, {**base_claim, "object_id": "Favoured"}, observation_time=datetime(2025, 1, 1, tzinfo=UTC))
    disfavoured = report_claim_score(report, {**base_claim, "object_id": "Disfavoured"}, observation_time=datetime(2025, 1, 1, tzinfo=UTC))

    assert favoured > disfavoured


def _candidate(**overrides):
    return {
        "id": "evidence:candidate:obs-1:1",
        "observation_id": "obs-1",
        "series_id": "series-1",
        "mode_id": "mode:a",
        "radar_id": "radar:a",
        "aircraft_id": "aircraft:a",
        "aircraft_family_id": "family:a",
        "operator": "Operator A",
        "text_score": 0.8,
        "ds_masses": [0.1, 0.7, 0.2],
        **overrides,
    }


def _claim(**overrides):
    return {
        "id": "evidence:claim:1",
        "series_id": "series-1",
        "report_id": "report-1",
        "source_id": "source-1",
        "claim_type": "radar_type",
        "object_id": "radar:a",
        "stance": "supports",
        "text_score": 0.8,
        **overrides,
    }


def test_final_signed_compatibility_applies_stance_exactly_once():
    candidate = _candidate()
    assert final_signed_compatibility(_claim(), candidate) == 1.0
    assert final_signed_compatibility(_claim(stance="refutes"), candidate) == -1.0
    assert final_signed_compatibility(_claim(object_id="radar:b"), candidate) == -1.0
    assert 0.0 < final_signed_compatibility(
        _claim(object_id="radar:b", stance="refutes"), candidate
    ) < 1.0
    assert claim_candidate_contribution(_claim(stance="refutes"), candidate) == -0.8


def test_family_claim_uses_resolved_kg_context_and_unknown_claim_is_neutral():
    candidate = _candidate()
    assert final_signed_compatibility(
        _claim(claim_type="aircraft_family", object_id="family:a"), candidate
    ) == 1.0
    assert final_signed_compatibility(
        _claim(claim_type="location", object_id="somewhere"), candidate
    ) == 0.0


def test_candidate_intelligence_deduplicates_sources_and_fuses_masses():
    candidate = _candidate()
    claims = [
        _claim(id="claim-1", source_id="shared", text_score=0.4),
        _claim(id="claim-2", source_id="shared", text_score=0.9),
        _claim(id="claim-3", source_id="independent", stance="refutes", text_score=0.3),
    ]
    features, edges = aggregate_candidate_intelligence(candidate, claims)

    assert len(edges) == 3  # graph provenance is retained for every claim
    assert features["intel_source_count"] == 2.0
    assert features["intel_support_score"] == 0.9
    assert features["intel_refute_score"] == 0.3
    assert features["final_score"] != features["sensor_score"]
    assert abs(sum(features["ds_masses"]) - 1.0) < 1e-6
    assert 0.0 <= features["intel_uncertainty"] <= 1.0

    repeated, _edges = aggregate_candidate_intelligence(
        {**candidate, **features}, claims
    )
    assert repeated["sensor_score"] == features["sensor_score"]
    assert repeated["sensor_ds_masses"] == features["sensor_ds_masses"]
    assert repeated["ds_masses"] == features["ds_masses"]


def test_candidate_intelligence_builds_direct_edges_and_reranks_shortlist():
    rows = {"claims": [
        _claim(text_score=1.0),
        _claim(
            id="relation-claim",
            claim_type="relation",
            object_id="relation:aircraft:a:USES_RADAR:radar:a",
            text_score=0.7,
            source_id="relation-source",
        ),
    ]}
    candidates = [
        _candidate(id="candidate-a", text_score=0.6, radar_id="radar:a"),
        _candidate(id="candidate-b", text_score=0.65, radar_id="radar:b"),
    ]
    updates, edges = build_candidate_intelligence_rows(rows, candidates, intelligence_weight=0.5)
    updates_by_id = {row["id"]: row for row in updates}

    assert updates_by_id["candidate-a"]["intel_rank"] == 1
    assert updates_by_id["candidate-b"]["intel_rank"] == 2
    assert {edge["target"] for edge in edges} == {"candidate-a", "candidate-b"}
    assert next(edge for edge in edges if edge["target"] == "candidate-a")["contribution"] > 0
    assert next(edge for edge in edges if edge["target"] == "candidate-b")["contribution"] < 0
    relation_edge = next(edge for edge in edges if edge["source"] == "relation-claim")
    assert relation_edge["target"] == "candidate-a"
    assert relation_edge["compatibility"] == 1.0
