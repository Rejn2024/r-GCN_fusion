from concurrent.futures import ProcessPoolExecutor

from rgcn_fusion.parallel_graph_construction import (
    build_series_fragment,
    build_series_fragment_in_worker,
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
        "max_kg_retrieval_candidates": 100,
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


def test_process_worker_builds_deterministic_index_local_fragment():
    context = {
        **_context(),
        "include_candidate_nodes": True,
        "include_intel_report_nodes": True,
        "segment_frequency_shift_ghz": 0.75,
    }
    expected = build_series_fragment(_task(), context)

    with ProcessPoolExecutor(
        max_workers=1,
        initializer=initialise_scoring_worker,
        initargs=(context,),
    ) as executor:
        actual = list(executor.map(build_series_fragment_in_worker, [_task()]))[0]

    assert actual == expected
    _, fragment, _ = actual
    assert [meta["node_kind"] for meta in fragment["node_meta"]] == [
        "observation",
        "intelligence_report",
        "report_claim",
        "candidate",
    ]
    assert fragment["observation_offsets"] == [0]
    assert fragment["claim_offsets"] == [2]
    assert fragment["candidate_links"][0][:2] == (0, 3)


def test_radar_only_candidate_does_not_invent_none_relation_id():
    context = _context()
    row = context["candidate_templates"][0]
    row.update(
        aircraft_id=None,
        aircraft_props=None,
        aircraft_uses_radar=False,
        operator=None,
    )
    context["candidate_variants"] = {(row["mode_id"], row["radar_id"], None): [row]}

    candidate = score_series_observations(_task(), context)[1]["obs-1"][0][4]

    assert candidate["aircraft_id"] is None
    assert candidate["relation_id"] is None


def test_intelligence_reranks_broad_retrieval_pool_before_final_limit():
    context = _context()
    sensor_favourite = context["candidate_templates"][0]
    intel_favourite = {
        **sensor_favourite,
        "mode_id": "radar_mode:intel",
        "radar_id": "radar:intel",
        "mode_props": {
            **sensor_favourite["mode_props"],
            "waveform": "other",
            "scan_type": "other",
        },
    }
    context["candidate_templates"].append({**intel_favourite, "operator": None})
    context["candidate_variants"][
        (
            intel_favourite["mode_id"],
            intel_favourite["radar_id"],
            intel_favourite["aircraft_id"],
        )
    ] = [intel_favourite]
    context["max_kg_retrieval_candidates"] = 2
    context["max_kg_candidates"] = 1
    task_position, series = _task()
    series["intelligence_reports"][0]["claims"] = [
        {
            "claim_id": f"claim-{index}",
            "claim_type": "radar_type",
            "object_id": "radar:intel",
            "stance": "supports",
            "claim_confidence": 1.0,
        }
        for index in range(3)
    ]

    candidates = score_series_observations((task_position, series), context)[1]["obs-1"]

    assert len(candidates) == 1
    assert candidates[0][4]["radar_id"] == "radar:intel"
    assert candidates[0][1] == 2
