from concurrent.futures import ProcessPoolExecutor

from rgcn_fusion.parallel_graph_construction import (
    initialise_scoring_worker,
    score_series_in_worker,
    score_series_observations,
)


def _context():
    row = {
        "mode_id": "radar_mode:test",
        "mode_props": {
            "waveform": "pulse_doppler",
            "scan_type": "sector",
            "centre_frequency_min_ghz": 9.4,
            "centre_frequency_max_ghz": 9.8,
        },
        "radar_id": "radar:test",
        "radar_props": {"name": "Test Radar"},
        "aircraft_id": "aircraft:test",
        "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15_000},
        "aircraft_uses_radar": True,
        "operator": "Testland",
    }
    key = (row["mode_id"], row["radar_id"], row["aircraft_id"])
    return {
        "candidate_templates": [{**row, "operator": None}],
        "candidate_variants": {key: [row]},
        "aircraft_family_by_aircraft": {"aircraft:test": "family:test"},
        "max_kg_candidates": 5,
    }


def _task():
    observation = {
        "observation_id": "obs-1",
        "series_id": "series-1",
        "sequence_index": 0,
        "timestamp_iso8601": "2026-01-01T00:00:00Z",
        "esm_radar_parameters": {
            "waveform": "pulse_doppler",
            "scan_type": "sector",
            "measured_centre_frequency_ghz": 9.6,
        },
        "approximate_kinematics": {"speed_mach": 1.2, "altitude_m": 10_000},
    }
    return 0, {"series_id": "series-1", "observations": [observation]}


def test_process_worker_matches_serial_scoring():
    context = _context()
    expected = score_series_observations(_task(), context)

    with ProcessPoolExecutor(
        max_workers=1,
        initializer=initialise_scoring_worker,
        initargs=(context,),
    ) as executor:
        actual = list(executor.map(score_series_in_worker, [_task()]))[0]

    assert actual == expected
    _, observations = actual
    candidate = observations["obs-1"][0][4]
    assert candidate["aircraft_family_id"] == "family:test"
    assert candidate["operator"] == "Testland"
