import json
from pathlib import Path

from rgcn_fusion.data_privacy import ground_truth_paths, strip_ground_truth


def test_strip_ground_truth_removes_known_and_future_truth_fields_recursively():
    payload = {
        "ground_truth_label": {"aircraft": "secret"},
        "reports": [
            {
                "GROUND_TRUTH_CONFIDENCE": 1.0,
                "text": "Analyst says ground_truth is unavailable.",
            }
        ],
        "synthetic_truth_value": "secret",
    }

    inference_payload = strip_ground_truth(payload)

    assert inference_payload == {
        "reports": [{"text": "Analyst says ground_truth is unavailable."}]
    }
    assert ground_truth_paths(inference_payload) == []


def test_ground_truth_paths_reports_structural_leaks_not_scalar_text():
    payload = {
        "text": "The phrase ground_truth in prose is not an inference label.",
        "nested": [{"new_ground_truth_annotation": True}],
    }

    assert ground_truth_paths(payload) == [
        ("nested", 0, "new_ground_truth_annotation")
    ]


def test_track_notebooks_validate_ground_truth_keys_instead_of_serialized_values():
    for notebook_path in (
        Path("notebooks/Track_identification.ipynb"),
        Path("notebooks/Track_identification_non_radar_rf.ipynb"),
    ):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        assert "from rgcn_fusion.data_privacy import" in source
        assert "inference_view = strip_ground_truth(loaded_series)" in source
        assert "leaked_truth_paths = ground_truth_paths(inference_records)" in source
        assert '"ground_truth" not in serialized_features' not in source
