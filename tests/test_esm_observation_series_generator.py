from datetime import UTC, datetime

from esm_observation_series_generator import generate_observation_series


def test_series_defaults_and_schema_fields():
    data = generate_observation_series(count=3, seed=11, start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 2, tzinfo=UTC))
    assert data["metadata"]["series_count"] == 3
    assert data["metadata"]["default_count"] == 2500
    assert len(data["observation_series"]) == 3
    for entry in data["observation_series"]:
        assert entry["emitter_type"] == "aircraft"
        assert entry["observation_count"] == len(entry["observations"])
        assert entry["observation_count"] >= 3
        for obs in entry["observations"]:
            assert obs["series_id"] == entry["series_id"]
            assert obs["timestamp_iso8601"].endswith("Z")
            assert obs["estimated_emitter_location"]
            assert obs["approximate_kinematics"]
            assert obs["esm_radar_parameters"]
            assert obs["ground_truth_label"]["aircraft_id"].startswith("aircraft:")
            assert obs["candidate_labels_from_shared_kg_features"]


def test_series_length_options_and_nominal_spacing():
    data = generate_observation_series(count=20, seed=19, min_duration_s=1.0, max_duration_s=2.0, sample_interval_s=0.5)
    lengths = {entry["observation_count"] for entry in data["observation_series"]}
    assert min(lengths) >= 3
    assert max(lengths) <= 5
    assert len(lengths) > 1
    for entry in data["observation_series"]:
        elapsed = [obs["elapsed_time_s"] for obs in entry["observations"]]
        assert elapsed == sorted(elapsed)
        intervals = [later - earlier for earlier, later in zip(elapsed, elapsed[1:])]
        assert all(0.42 <= interval <= 0.58 for interval in intervals)
