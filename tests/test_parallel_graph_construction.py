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
        "estimated_emitter_location": {
            "estimated_latitude_deg": 51.5,
            "estimated_longitude_deg": -0.1,
        },
    }
    report = {
        "report_id": "report-1",
        "report_type": "sighting_report",
        "sighting": {
            "last_observed_at": "2026-01-01T00:05:00Z",
            "location": {"latitude_deg": 51.5, "longitude_deg": -0.1},
        },
        "claims": [
            {
                "claim_id": "claim-1",
                "claim_type": "radar_type",
                "object_id": "radar:test",
                "stance": "supports",
                "claim_confidence": 1.0,
            }
        ],
    }
    return 0, {
        "series_id": "series-1",
        "observations": [observation],
        "intelligence_reports": [report],
    }


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
    _, observations, report_proximities = actual
    candidate = observations["obs-1"][0][4]
    assert candidate["aircraft_family_id"] == "family:test"
    assert candidate["operator"] == "Testland"
    compact_edges = observations["obs-1"][0][6]
    assert compact_edges.typecode == "i"
    assert compact_edges.tolist() == [1]
    assert report_proximities == {
        "obs-1": {
            "report-1": {
                "match_basis": "time_and_geography",
                "time_delta_s": 300.0,
                "distance_km": 0.0,
            }
        }
    }


def test_radar_only_candidate_does_not_invent_none_relation_id():
    context = _context()
    row = context["candidate_templates"][0]
    row.update(
        aircraft_id=None,
        aircraft_props=None,
        aircraft_uses_radar=False,
        operator=None,
    )
    context["candidate_variants"] = {
        (row["mode_id"], row["radar_id"], None): [row]
    }

    candidate = score_series_observations(_task(), context)[1]["obs-1"][0][4]

    assert candidate["aircraft_id"] is None
    assert candidate["relation_id"] is None
