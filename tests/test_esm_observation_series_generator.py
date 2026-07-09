from datetime import UTC, datetime

from esm_observation_series_generator import generate_observation_series, observations_without_ground_truth


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


def test_series_can_switch_modes_multiple_times_and_records_truth_sequence():
    data = generate_observation_series(
        count=8,
        seed=23,
        min_duration_s=4.0,
        max_duration_s=4.0,
        sample_interval_s=0.5,
        mode_switch_probability=1.0,
    )

    multi_shift_entries = [
        entry for entry in data["observation_series"] if len(entry["mode_shift_sequence_indices"]) > 1
    ]

    assert multi_shift_entries
    for entry in multi_shift_entries:
        observed_modes = [obs["ground_truth_label"]["mode"] for obs in entry["observations"]]
        sequence_modes = [truth["mode"] for truth in entry["ground_truth_mode_sequence"]]
        derived_shift_indices = [
            idx for idx, (before, after) in enumerate(zip(observed_modes, observed_modes[1:]), start=1)
            if before != after
        ]
        assert sequence_modes == observed_modes
        assert entry["mode_shift_sequence_indices"] == derived_shift_indices
        assert entry["ground_truth_track_label"]["mode"] == "multiple"


def test_observations_without_ground_truth_strips_only_truth_labels():
    data = generate_observation_series(count=1, seed=31, mode_switch_probability=1.0)
    entry = data["observation_series"][0]

    inference_rows = observations_without_ground_truth(entry)

    assert len(inference_rows) == len(entry["observations"])
    assert all("ground_truth_label" not in row for row in inference_rows)
    assert all("esm_radar_parameters" in row for row in inference_rows)
    assert all("candidate_labels_from_shared_kg_features" in row for row in inference_rows)
    assert "ground_truth_label" in entry["observations"][0]
