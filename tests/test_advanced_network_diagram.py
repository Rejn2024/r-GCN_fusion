import importlib.util
import ast
import json
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


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

    assert len(architecture["graphsage_hidden_dims"]) == 25
    assert architecture["graphsage_hidden_dims"][0] == 40
    assert architecture["graphsage_hidden_dims"][-1] == 32
    assert architecture["settings"]["NUM_ATTENTION_HEADS"] == 4
    assert architecture["settings"]["EVIDENTIAL_LOSS_WEIGHT"] == 0.1
    assert architecture["targets"] == ["aircraft_variant", "radar_mode", "radar_type", "operator_country"]


def test_svg_contains_accessible_architecture_labels():
    architecture = diagram.extract_architecture(
        ROOT / "notebooks" / "observation_series_and_intel_rgcn_classification_advanced_network.ipynb"
    )
    svg = diagram.render_svg(architecture)

    assert 'role="img"' in svg
    assert "25 mean-aggregation blocks" in svg
    assert "Relation-aware HGT" in svg
    assert "Dirichlet / DS evidence" in svg
    assert "operator_country" in svg


def test_advanced_classifier_emits_normalized_dirichlet_ds_outputs():
    notebook = json.loads(
        (ROOT / "notebooks" / "observation_series_and_intel_rgcn_classification_advanced_network.ipynb").read_text()
    )
    source = "\n\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    tree = ast.parse(source)
    class_nodes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in diagram.REQUIRED_CLASSES
    ]
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(compile(ast.Module(body=class_nodes, type_ignores=[]), "<advanced-notebook-classes>", "exec"), namespace)
    classifier = namespace["SeriesGraphSAGEHGTClassifier"](
        3, 4, 4, 1, {"task": 3}, num_message_passing_edges=1,
        num_hgt_layers=0, num_heads=1, dropout=0.0, task_head_hidden_dim=5,
    )

    outputs = classifier(torch.randn(2, 3), torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long))
    evidence = outputs["evidential"]["task"]

    assert outputs["task"].shape == (2, 3)
    assert torch.all(evidence["evidence"] >= 0)
    assert torch.all(evidence["alpha"] >= 1)
    assert torch.allclose(evidence["probabilities"].sum(-1), torch.ones(2))
    assert torch.allclose(evidence["belief"].sum(-1, keepdim=True) + evidence["uncertainty"], torch.ones(2, 1))
