import json
from pathlib import Path


NOTEBOOK = Path("notebooks/observation_series_and_intel_rgcn_classification_advanced_network_llm.ipynb")


def _code_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_llm_notebook_builds_joint_frame_from_kg_and_projects_attributes():
    source = _code_source()

    assert "from rgcn_fusion import attribute_assessments, decode_kg_constrained" in source
    assert '"aircraft_family": row["aircraft_props"].get("family")' in source
    assert "def joint_focal_mass(node_idx: int)" in source
    assert "attribute_assessments(" in source
    assert 'rec["kg_structured_prediction"]' in source
    assert '"kg_joint_worlds.json"' in source


def test_llm_notebook_treats_structured_result_as_authoritative():
    source = _code_source()

    assert '"kg_structured_prediction": structured_predictions_by_node[node_idx]' in source
    assert "Use kg_structured_prediction as the authoritative identification" in source
    assert "preserve null fields when the joint frame leaves an attribute unresolved" in source
