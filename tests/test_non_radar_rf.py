import ast
import json
from pathlib import Path

from rgcn_fusion.non_radar_rf import score_non_radar_rf


def test_non_radar_rf_scores_matching_aircraft_equipment_without_truth():
    observation = {
        "ground_truth_label": {"aircraft_variant": "must not be read"},
        "rf_emissions": {
            "data_link": {
                "equipment": "Tactical Link A",
                "frequency_min_mhz": 960,
                "frequency_max_mhz": 1215,
                "access_method": "TDMA",
            }
        },
    }
    matching = [{
        "emission_type": "data_link",
        "name": "Tactical Link A",
        "frequency_min_mhz": 960,
        "frequency_max_mhz": 1215,
        "access_method": "TDMA",
    }]
    incompatible = [{
        "emission_type": "data_link",
        "name": "Tactical Link C",
        "frequency_min_mhz": 2200,
        "frequency_max_mhz": 2500,
        "access_method": "directional",
    }]

    match_score, count, features = score_non_radar_rf(observation, matching)
    mismatch_score, _, _ = score_non_radar_rf(observation, incompatible)

    assert count == 1
    assert match_score == 1.0
    assert mismatch_score < 0.5
    assert features["rf_data_link_compatibility"] == 1.0


def test_no_rf_detection_is_absence_of_evidence():
    score, count, features = score_non_radar_rf({}, [])
    assert (score, count) == (0.0, 0)
    assert features["non_radar_rf_observed_count"] == 0.0


def test_distinguishing_notebook_has_rf_etl_scores_and_relations():
    path = Path("notebooks/Track_identification_non_radar_rf.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell {index}")

    assert 'obs.get("rf_emissions", {})' in Path(
        "rgcn_fusion/parallel_graph_construction.py"
    ).read_text(encoding="utf-8")
    assert '"rf_equipment": rf_equipment_by_aircraft.get(aircraft_id, [])' in source
    assert "candidate_non_radar_rf_score" in source
    assert "has_compatible_rf_emission" in source
    assert "rf_emission_for_candidate" in source
    assert "contradicts_rf_emission" in source
    assert "NON_RADAR_RF_COMPATIBILITY_THRESHOLD = 0.50" in source
