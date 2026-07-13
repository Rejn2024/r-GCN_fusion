from rgcn_fusion.observation_etl import contradiction_edges_for_candidates, ds_masses_from_score, score_candidates


def _observation():
    return {
        "observation_id": "esm_obs_00001",
        "timestamp_iso8601": "2025-01-01T00:00:00Z",
        "esm_radar_parameters": {
            "observed_waveform": "pulse_doppler",
            "observed_scan_type": "sector",
            "measured_centre_frequency_ghz": {"min": 9.5, "max": 9.7},
            "measured_bandwidth_mhz": {"min": 100, "max": 140},
            "measured_prf_hz": {"min": 6000, "max": 18000},
            "measured_pulse_width_us": {"min": 1.5, "max": 2.1},
            "measured_duty_cycle": {"min": 0.02, "max": 0.04},
            "measured_coherent_processing_interval_ms": {"min": 20, "max": 28},
            "measured_dwell_time_ms": {"min": 100, "max": 140},
        },
        "approximate_kinematics": {"ground_speed_max_kph": 900, "altitude_max_m": 9000},
        "ground_truth_label": {"operator": "Testland"},
    }


def test_score_candidates_prefers_overlapping_radar_mode():
    rows = [
        {
            "mode_id": "radar_mode:good:track_while_scan",
            "mode_props": {
                "waveform": "pulse_doppler",
                "scan_type": "sector",
                "centre_frequency_min_ghz": 9.4,
                "centre_frequency_max_ghz": 9.8,
                "bandwidth_min_mhz": 90,
                "bandwidth_max_mhz": 150,
                "prf_min_hz": 5000,
                "prf_max_hz": 19000,
                "pulse_width_min_us": 1.0,
                "pulse_width_max_us": 2.5,
                "duty_cycle_min": 0.01,
                "duty_cycle_max": 0.05,
                "coherent_processing_interval_min_ms": 16,
                "coherent_processing_interval_max_ms": 32,
                "dwell_time_min_ms": 80,
                "dwell_time_max_ms": 160,
            },
            "radar_id": "radar:good",
            "aircraft_id": "aircraft:good",
            "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
            "operator": "Testland",
        },
        {
            "mode_id": "radar_mode:bad:mapping",
            "mode_props": {
                "waveform": "synthetic_aperture_or_real_beam",
                "scan_type": "ground",
                "centre_frequency_min_ghz": 14.0,
                "centre_frequency_max_ghz": 15.0,
                "bandwidth_min_mhz": 250,
                "bandwidth_max_mhz": 300,
                "prf_min_hz": 1000,
                "prf_max_hz": 2000,
                "pulse_width_min_us": 10.0,
                "pulse_width_max_us": 20.0,
                "duty_cycle_min": 0.2,
                "duty_cycle_max": 0.3,
                "coherent_processing_interval_min_ms": 80,
                "coherent_processing_interval_max_ms": 100,
                "dwell_time_min_ms": 300,
                "dwell_time_max_ms": 400,
            },
            "radar_id": "radar:bad",
            "aircraft_id": "aircraft:bad",
            "aircraft_props": {"max_speed_mach": 1.0, "service_ceiling_m": 5000},
            "operator": "Other",
        },
    ]

    candidates = score_candidates(_observation(), rows, max_candidates=2)

    assert candidates[0].mode_id == "radar_mode:good:track_while_scan"
    assert candidates[0].total_score > candidates[1].total_score


def test_ds_masses_are_normalized():
    masses = ds_masses_from_score(0.8, 0.25)

    assert len(masses) == 3
    assert sum(masses) == 1.0
    assert masses[1] > masses[0]


def test_score_candidates_ignores_ground_truth_labels_for_scoring():
    rows = [
        {
            "mode_id": "radar_mode:a",
            "mode_props": {
                "waveform": "pulse_doppler",
                "scan_type": "sector",
                "centre_frequency_min_ghz": 9.4,
                "centre_frequency_max_ghz": 9.8,
            },
            "radar_id": "radar:a",
            "aircraft_id": "aircraft:a",
            "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
            "operator": "Testland",
        },
        {
            "mode_id": "radar_mode:b",
            "mode_props": {
                "waveform": "pulse_doppler",
                "scan_type": "sector",
                "centre_frequency_min_ghz": 9.4,
                "centre_frequency_max_ghz": 9.8,
            },
            "radar_id": "radar:b",
            "aircraft_id": "aircraft:b",
            "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
            "operator": "Other",
        },
    ]
    obs = _observation()
    leaked_obs = {
        **obs,
        "ground_truth_label": {
            "operator": "Other",
            "radar_id": "radar:b",
            "mode_id": "radar_mode:b",
            "aircraft_id": "aircraft:b",
        },
    }

    baseline = score_candidates(obs, rows, max_candidates=2)
    with_conflicting_truth = score_candidates(leaked_obs, rows, max_candidates=2)

    assert [(c.mode_id, c.total_score) for c in baseline] == [
        (c.mode_id, c.total_score) for c in with_conflicting_truth
    ]


def test_score_candidates_uses_external_operator_context_for_prior():
    rows = [
        {
            "mode_id": "radar_mode:a",
            "mode_props": {
                "waveform": "pulse_doppler",
                "scan_type": "sector",
                "centre_frequency_min_ghz": 9.4,
                "centre_frequency_max_ghz": 9.8,
            },
            "radar_id": "radar:a",
            "aircraft_id": "aircraft:a",
            "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
            "operator": "Testland",
        },
        {
            "mode_id": "radar_mode:b",
            "mode_props": {
                "waveform": "pulse_doppler",
                "scan_type": "sector",
                "centre_frequency_min_ghz": 9.4,
                "centre_frequency_max_ghz": 9.8,
            },
            "radar_id": "radar:b",
            "aircraft_id": "aircraft:b",
            "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
            "operator": "Other",
        },
    ]
    obs = {
        **_observation(),
        "external_context": {"operator_priors": {"Other": 1.0, "Testland": 0.0}},
    }

    candidates = score_candidates(obs, rows, max_candidates=2)

    assert candidates[0].operator == "Other"
    assert candidates[0].total_score > candidates[1].total_score


def test_score_candidates_uses_kg_aircraft_radar_compatibility():
    rows = [
        {
            "mode_id": "radar_mode:linked",
            "mode_props": {
                "waveform": "pulse_doppler",
                "scan_type": "sector",
                "centre_frequency_min_ghz": 9.4,
                "centre_frequency_max_ghz": 9.8,
            },
            "radar_id": "radar:linked",
            "radar_props": {"name": "Linked Radar"},
            "aircraft_id": "aircraft:linked",
            "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
            "aircraft_uses_radar": True,
            "operator": "Testland",
        },
        {
            "mode_id": "radar_mode:unlinked",
            "mode_props": {
                "waveform": "pulse_doppler",
                "scan_type": "sector",
                "centre_frequency_min_ghz": 9.4,
                "centre_frequency_max_ghz": 9.8,
            },
            "radar_id": "radar:unlinked",
            "radar_props": {"name": "Unlinked Radar"},
            "aircraft_id": "aircraft:unlinked",
            "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
            "aircraft_uses_radar": False,
            "operator": "Testland",
        },
    ]

    candidates = score_candidates(_observation(), rows, max_candidates=2)

    assert candidates[0].aircraft_id == "aircraft:linked"
    assert candidates[0].aircraft_score > candidates[1].aircraft_score
    assert candidates[0].total_score > candidates[1].total_score


def test_contradiction_edges_mark_stronger_incompatible_candidates():
    candidates = score_candidates(
        _observation(),
        [
            {
                "mode_id": "radar_mode:good:track_while_scan",
                "mode_props": {
                    "waveform": "pulse_doppler",
                    "scan_type": "sector",
                    "centre_frequency_min_ghz": 9.4,
                    "centre_frequency_max_ghz": 9.8,
                },
                "radar_id": "radar:good",
                "aircraft_id": "aircraft:good",
                "aircraft_props": {"max_speed_mach": 2.0, "service_ceiling_m": 15000},
                "operator": "Testland",
            },
            {
                "mode_id": "radar_mode:bad:mapping",
                "mode_props": {
                    "waveform": "synthetic_aperture_or_real_beam",
                    "scan_type": "ground",
                    "centre_frequency_min_ghz": 14.0,
                    "centre_frequency_max_ghz": 15.0,
                },
                "radar_id": "radar:bad",
                "aircraft_id": "aircraft:bad",
                "aircraft_props": {"max_speed_mach": 1.0, "service_ceiling_m": 5000},
                "operator": "Other",
            },
        ],
        max_candidates=2,
    )

    edges = contradiction_edges_for_candidates(["candidate:1", "candidate:2"], candidates)

    assert edges == [
        {
            "source": "candidate:1",
            "target": "candidate:2",
            "score_delta": candidates[0].total_score - candidates[1].total_score,
            "reason": "mode_id,radar_id,aircraft_id,operator",
        }
    ]
