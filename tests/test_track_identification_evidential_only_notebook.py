import ast
import json
from pathlib import Path


NOTEBOOK = Path("notebooks/Track_identification_non_radar_rf_evidential_only.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_source():
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def test_evidential_only_notebook_code_cells_parse_and_outputs_are_clean():
    for index, cell in enumerate(_notebook()["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell {index}")
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_training_has_a_user_configurable_two_hour_hard_finish():
    source = _code_source()
    assert 'HARD_FINISH_HOURS = float(os.getenv("HARD_FINISH_HOURS", "2"))' in source
    assert "training_deadline = time.monotonic() + HARD_FINISH_HOURS * 60 * 60" in source
    assert "def hard_finish_reached():" in source
    assert "if hard_finish_reached():" in source
    assert '"hard_finish_hours": HARD_FINISH_HOURS' in source


def test_evidential_only_notebook_has_no_parallel_logit_heads():
    source = _code_source()
    assert "self.rao_evidential_head = head(hidden_dim, num_rao_classes)" in source
    assert "self.mode_evidential_head = head(2 * hidden_dim, num_mode_classes)" in source
    assert "self.rao_head" not in source
    assert "self.mode_head" not in source
    assert 'return {"evidential": {"track_rao": rao_evidential, "radar_mode": mode_evidential}' in source
    assert "RAO_CLASSIFICATION_WEIGHT" not in source
    assert "MODE_CLASSIFICATION_WEIGHT" not in source
    assert "F.cross_entropy" not in source


def test_evidential_probabilities_drive_conditioning_predictions_and_accuracies():
    source = _code_source()
    assert 'compatibility = rao_evidential["probabilities"] @ self.rao_mode_compatibility' in source
    assert 'final_outputs["evidential"]["track_rao"]["probabilities"].argmax' in source
    assert "constrained_mode_probabilities = mode_probabilities_tensor * observation_mode_mask" in source
    assert 'rao_probabilities = final_outputs["evidential"]["track_rao"]["probabilities"]' in source
    assert "def accuracy(probabilities, labels)" in source
    assert 'outputs["evidential"]["radar_mode"]["probabilities"][observations]' in source


def test_aircraft_radar_and_operator_marginals_remain_in_evidential_loss():
    source = _code_source()
    assert "def component_weighted_rao_edl(alpha, labels)" in source
    assert "RAO_AIRCRAFT_LOSS_WEIGHT = 1.0" in source
    assert "RAO_RADAR_LOSS_WEIGHT = 1.0" in source
    assert "RAO_OPERATOR_LOSS_WEIGHT = 1.0" in source
    assert "component_weighted_rao_edl(alpha, labels) * rao_component_loss_weights" in source
    assert 'for index, field in enumerate(("aircraft", "radar", "operator"))' in source
    assert '"rao_aircraft_acc": component_accuracy[0]' in source
    assert '"rao_radar_acc": component_accuracy[1]' in source
    assert '"rao_operator_acc": component_accuracy[2]' in source
