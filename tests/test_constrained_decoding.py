import pytest

from rgcn_fusion.constrained_decoding import decode_kg_constrained, decode_kg_hierarchical


VOCABULARIES = {
    "radar_type": ["Zhuk-N110", "Captor-E"],
    "aircraft_variant": ["MiG-35", "Eurofighter Typhoon"],
    "operator_country": ["Russia", "Japan", "United Kingdom"],
}
VALID = [
    {"radar_type": "Zhuk-N110", "aircraft_variant": "MiG-35", "operator_country": "Russia"},
    {
        "radar_type": "Captor-E",
        "aircraft_variant": "Eurofighter Typhoon",
        "operator_country": "United Kingdom",
    },
]


def test_decoder_cannot_compose_individually_likely_but_impossible_labels():
    prediction = decode_kg_constrained(
        {
            "radar_type": [0.8, 0.2],
            "aircraft_variant": [0.1, 0.9],
            "operator_country": [0.1, 0.8, 0.1],
        },
        VOCABULARIES,
        VALID,
        unknown_threshold=0.0,
    )

    assert prediction.status == "known"
    assert prediction.labels in VALID
    assert prediction.labels != {
        "radar_type": "Zhuk-N110",
        "aircraft_variant": "Eurofighter Typhoon",
        "operator_country": "Japan",
    }


def test_decoder_rejects_weak_joint_evidence_as_unknown():
    prediction = decode_kg_constrained(
        {
            "radar_type": [0.51, 0.49],
            "aircraft_variant": [0.51, 0.49],
            "operator_country": [0.34, 0.33, 0.33],
        },
        VOCABULARIES,
        VALID,
        min_task_probability=0.6,
    )

    assert prediction.status == "unknown"
    assert prediction.labels is None


def test_decoder_rejects_invalid_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        decode_kg_constrained({}, {}, [], unknown_threshold=1.1)


def test_hierarchical_decoder_keeps_known_identity_when_radar_mode_is_novel():
    vocabularies = {**VOCABULARIES, "radar_mode": ["air_search", "track_while_scan"]}
    valid = [
        {**VALID[0], "radar_mode": "air_search"},
        {**VALID[0], "radar_mode": "track_while_scan"},
        {**VALID[1], "radar_mode": "air_search"},
    ]
    prediction = decode_kg_hierarchical(
        {
            "radar_type": [0.95, 0.05],
            "aircraft_variant": [0.95, 0.05],
            "operator_country": [0.95, 0.01, 0.04],
            # A closed-set head can be confidently wrong for an unseen mode.
            "radar_mode": [0.1, 0.9],
        },
        vocabularies,
        valid,
        identity_tasks=("radar_type", "aircraft_variant", "operator_country"),
        open_set_tasks=("radar_mode",),
        novelty_scores={"radar_mode": 0.92},
        novelty_thresholds={"radar_mode": 0.7},
    )

    assert prediction.status == "partially_known"
    assert prediction.labels == {**VALID[0], "radar_mode": None}
    assert prediction.unknown_tasks == ("radar_mode",)


def test_hierarchical_decoder_restricts_known_mode_to_selected_radar_identity():
    vocabularies = {**VOCABULARIES, "radar_mode": ["zhuk_mode", "captor_mode"]}
    valid = [
        {**VALID[0], "radar_mode": "zhuk_mode"},
        {**VALID[1], "radar_mode": "captor_mode"},
    ]
    prediction = decode_kg_hierarchical(
        {
            "radar_type": [0.9, 0.1],
            "aircraft_variant": [0.9, 0.1],
            "operator_country": [0.9, 0.05, 0.05],
            "radar_mode": [0.6, 0.99],
        },
        vocabularies,
        valid,
        identity_tasks=("radar_type", "aircraft_variant", "operator_country"),
        open_set_tasks=("radar_mode",),
    )

    assert prediction.status == "known"
    assert prediction.labels == {**VALID[0], "radar_mode": "zhuk_mode"}
