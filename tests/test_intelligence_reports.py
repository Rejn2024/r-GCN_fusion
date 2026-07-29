from datetime import UTC, datetime

from esm_observation_series_generator import generate_observation_series_with_intelligence_reports
from rgcn_fusion.intelligence_reports import CLAIM_TYPES, build_report_evidence_rows, flatten_reports_from_series, report_claim_score


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
