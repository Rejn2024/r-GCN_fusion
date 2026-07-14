from datetime import UTC, datetime

from esm_observation_series_generator import generate_observation_series_with_intelligence_reports
from rgcn_fusion.intelligence_reports import (
    CLAIM_TYPES,
    build_report_evidence_rows,
    flatten_reports_from_series,
    report_claim_score,
)


def test_series_generator_adds_10_to_12_intelligence_reports_per_observation():
    data = generate_observation_series_with_intelligence_reports(
        count=2,
        seed=101,
        intelligence_seed=202,
        start=datetime(2025, 1, 1, tzinfo=UTC),
        end=datetime(2025, 1, 2, tzinfo=UTC),
        workers=1,
    )

    observations = [obs for series in data["observation_series"] for obs in series["observations"]]
    assert observations
    for obs in observations:
        reports = obs["intelligence_reports"]
        assert 10 <= len(reports) <= 12
        claim_types = {report["claims"][0]["claim_type"] for report in reports}
        assert {"operator", "aircraft_variant", "radar_type", "radar_mode"}.issubset(claim_types)
        assert all(report["published_at"].endswith("Z") for report in reports)
        assert all(report["collected_at"].endswith("Z") for report in reports)
    assert data["metadata"]["intelligence_claim_types"] == list(CLAIM_TYPES)


def test_generated_reports_include_correct_and_contradictory_claims():
    data = generate_observation_series_with_intelligence_reports(count=1, seed=303, intelligence_seed=404, workers=1)
    reports = flatten_reports_from_series(data)
    truth_values = {report["claims"][0]["synthetic_truth_value"] for report in reports}

    assert "correct" in truth_values
    assert "contradictory" in truth_values


def test_report_evidence_rows_include_contradictions_and_ds_masses():
    data = generate_observation_series_with_intelligence_reports(count=1, seed=505, intelligence_seed=606, workers=1)
    observations = [obs for series in data["observation_series"] for obs in series["observations"]]

    rows = build_report_evidence_rows(observations)

    assert rows["reports"]
    assert rows["claims"]
    assert rows["contains_edges"]
    assert rows["support_edges"]
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
