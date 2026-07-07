from datetime import UTC, datetime

from esm_observation_generator import generate_observations
from kg_generator import generate_graph


def test_observations_have_labels_and_time_formats():
    data = generate_observations(8, seed=3, start=datetime(2025, 1, 1, tzinfo=UTC), end=datetime(2025, 1, 2, tzinfo=UTC))
    assert data["metadata"]["count"] == 8
    assert len(data["observations"]) == 8
    for obs in data["observations"]:
        assert obs["timestamp_unix"] > 0
        assert obs["timestamp_iso8601"].endswith("Z")
        assert obs["ground_truth_label"]["aircraft_id"].startswith("aircraft:")
        assert obs["ground_truth_label"]["mode_id"].startswith("radar_mode:")
        assert obs["candidate_labels_from_shared_kg_features"]


def test_observations_reference_existing_kg_nodes_and_plausible_ranges():
    graph = generate_graph()
    node_ids = {node["id"] for node in graph["nodes"]}
    data = generate_observations(20, seed=17)
    for obs in data["observations"]:
        label = obs["ground_truth_label"]
        assert label["aircraft_id"] in node_ids
        assert label["radar_id"] in node_ids
        assert label["mode_id"] in node_ids
        loc = obs["estimated_emitter_location"]["error_box"]
        assert loc["min_latitude_deg"] <= obs["estimated_emitter_location"]["estimated_latitude_deg"] <= loc["max_latitude_deg"]
        assert loc["min_longitude_deg"] <= obs["estimated_emitter_location"]["estimated_longitude_deg"] <= loc["max_longitude_deg"]
        kin = obs["approximate_kinematics"]
        assert kin["ground_speed_min_kph"] <= kin["ground_speed_kph"] <= kin["ground_speed_max_kph"]
        assert kin["altitude_min_m"] <= kin["altitude_m"] <= kin["altitude_max_m"]
        esm = obs["esm_radar_parameters"]
        freq = esm["measured_centre_frequency_ghz"]
        assert freq["min"] <= freq["value"] <= freq["max"]
