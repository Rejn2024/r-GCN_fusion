import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_advanced_network_diagram.py"
SPEC = importlib.util.spec_from_file_location("advanced_network_diagram", SCRIPT)
diagram = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diagram)


def test_extracts_current_notebook_architecture():
    architecture = diagram.extract_architecture(
        ROOT / "notebooks" / "observation_series_and_intel_rgcn_classification_advanced_network.ipynb"
    )

    assert architecture["graphsage_hidden_dims"] == [128, 119, 111, 102, 93, 84, 76, 67, 58, 49, 41, 32]
    assert architecture["settings"]["NUM_ATTENTION_HEADS"] == 4
    assert architecture["targets"] == ["aircraft_variant", "radar_mode", "radar_type", "operator_country"]


def test_svg_contains_accessible_architecture_labels():
    architecture = diagram.extract_architecture(
        ROOT / "notebooks" / "observation_series_and_intel_rgcn_classification_advanced_network.ipynb"
    )
    svg = diagram.render_svg(architecture)

    assert 'role="img"' in svg
    assert "12 mean-aggregation blocks" in svg
    assert "Relation-aware HGT" in svg
    assert "operator_country" in svg
