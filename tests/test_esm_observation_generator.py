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
        assert "candidate_labels_from_shared_kg_features" not in obs


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


def test_esm_radar_parameters_only_contains_measured_or_observed_values():
    data = generate_observations(20, seed=23)
    expected_keys = {
        "observed_waveform",
        "observed_scan_type",
        "observed_pri_modulation",
        "observed_intrapulse_modulation",
        "observed_frequency_pattern",
        "observed_polarization",
        "measured_centre_frequency_ghz",
        "measured_bandwidth_mhz",
        "measured_prf_hz",
        "measured_pulse_repetition_interval_us",
        "measured_pulse_width_us",
        "measured_duty_cycle",
        "measured_coherent_processing_interval_ms",
        "measured_dwell_time_ms",
        "measured_frequency_agility_mhz",
        "measured_scan_period_s",
    }
    for obs in data["observations"]:
        esm = obs["esm_radar_parameters"]
        assert set(esm) == expected_keys
        assert not any(key.endswith(("_min", "_max")) or "_min_" in key or "_max_" in key for key in esm)
        for key, measurement in esm.items():
            if key.startswith("measured_"):
                assert measurement["min"] <= measurement["value"] <= measurement["max"]


def test_canonical_radar_modes_define_enriched_esm_parameters():
    graph = generate_graph()
    radar_nodes = [node for node in graph["nodes"] if node["label"] == "Radar"]
    mode_nodes = [node for node in graph["nodes"] if node["label"] == "RadarMode"]

    assert radar_nodes and mode_nodes
    for node in radar_nodes:
        props = node["properties"]
        assert props["polarization"]
        assert props["supported_pri_modulations"]
        assert props["supported_intrapulse_modulations"]
        assert props["supported_frequency_patterns"]
        assert 0 <= props["frequency_agility_min_mhz"] <= props["frequency_agility_max_mhz"]
        assert 0 < props["scan_period_min_s"] <= props["scan_period_max_s"]
    for node in mode_nodes:
        props = node["properties"]
        assert props["pri_modulation"]
        assert props["intrapulse_modulation"]
        assert props["frequency_pattern"]
        assert props["polarization"]
        assert 0 <= props["frequency_agility_min_mhz"] <= props["frequency_agility_max_mhz"]
        assert 0 < props["scan_period_min_s"] <= props["scan_period_max_s"]
