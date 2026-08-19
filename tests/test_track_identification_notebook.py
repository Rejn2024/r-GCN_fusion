import ast
import json
from pathlib import Path

NOTEBOOK = Path("notebooks/Track_identification.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_source():
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def test_track_identification_notebook_code_cells_parse():
    for index, cell in enumerate(_notebook()["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell {index}")


def test_track_notebook_has_one_joint_rao_head_and_observation_mode_head():
    source = _code_source()
    assert "class TrackAttentionPool" in source
    assert "class TrackRGCNHGTClassifier" in source
    assert "self.rao_head = head(hidden_dim, num_rao_classes)" in source
    assert "self.mode_head = head(2 * hidden_dim, num_mode_classes)" in source
    assert '"track_rao": rao_logits' in source
    assert '"radar_mode": mode_logits' in source
    assert "rao_mode_compatibility" in source


def test_track_notebook_balances_mode_loss_and_enforces_output_constraints():
    source = _code_source()
    assert "def mean_modes_per_track" in source
    assert "rao_target_by_series" in source
    assert "observation_track_index" in source
    assert "constrained_mode_logits" in source
    assert "RAO_VIOLATION_COUNT = 0" in source
    assert '"rao_invariant": True' in source


def test_dashboard_and_llm_keep_explanation_focused_on_track_rao():
    source = _code_source()
    assert '"track_rao_exact_accuracy"' in source
    assert '"mode_transition_f1"' in source
    assert (
        'final_outputs["evidential"]["radar_mode"]["uncertainty"]'
        "[test_observation_positions]" in source
    )
    assert "LLM_TRACK_INDEX" in source
    assert '"invariant_track_rao"' in source
    assert '"ordered_radar_mode_sequence"' not in source
    assert "Never imply that aircraft, radar, or operator changes" in source
    assert "radar mode is outside the scope of this explanation" in source
    assert "RAO identity, RAO evidence, RAO uncertainty" in source


def test_track_notebook_ingests_new_reports_by_proximity_and_links_kg_entities():
    source = _code_source()
    assert "demo_esm_observation_series_with_sightings_and_patterns.json" in source
    assert "report_observation_proximity" in source
    assert "proximity_by_observation" in source
    assert '"claim_asserts_kg_entity"' in source
    assert '"kg_entity_asserted_by_claim"' in source
    assert "kg_entity_node_indices" in source
    assert "report_kg_edges" in source


def test_candidate_recall_at_k_is_vectorised_and_visualised():
    source = _code_source()
    assert "RECALL_K_START = 5" in source
    assert "first_correct_rank" in source
    assert "np.bincount" in source
    assert "np.cumsum" in source
    assert 'candidate_recall_at_k.png' in source
    assert 'axis.plot(recall_k_values, recall_at_k' in source
