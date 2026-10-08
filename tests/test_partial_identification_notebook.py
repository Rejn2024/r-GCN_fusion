import ast
import json
from pathlib import Path

import torch


NOTEBOOK = (
    Path(__file__).parents[1]
    / "notebooks"
    / "Track_identification_open_set_ufo_evidential_only_no_unknown_output.ipynb"
)


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _loss_functions():
    """Load the notebook's loss functions with a minimal synthetic joint frame."""
    tree = ast.parse(_notebook_source())
    wanted = {
        "expected_dirichlet_ce",
        "component_weighted_rao_edl",
        "combined_rao_edl",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    assert {node.name for node in functions} == wanted

    # World 1 differs from the target only at variant level. World 2 differs at
    # variant, type, radar, and operator levels.
    namespace = {
        "torch": torch,
        "RAO_SUPERVISION_FIELDS": (
            "aircraft_variant",
            "aircraft_type",
            "radar_type",
            "operator_country",
        ),
        "rao_component_ids": torch.tensor(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [2, 1, 1, 1],
            ],
            dtype=torch.long,
        ),
        "rao_component_loss_weights": torch.full((4,), 0.25),
        "rao_loss_weights": torch.full((2,), 0.5),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


def test_notebook_supervises_the_same_aircraft_taxonomy_used_at_inference():
    source = _notebook_source()

    assert 'RAO_SUPERVISION_FIELDS = ("aircraft_variant", "aircraft_type", "radar_type", "operator_country")' in source
    assert "RAO_AIRCRAFT_TYPE_LOSS_WEIGHT" in source
    assert "(variant, rao_family_by_variant[variant], radar, operator)" in source
    assert '"aircraft_type": rao_family_by_variant[variant]' in source


def test_taxonomy_loss_rewards_a_wrong_variant_in_the_correct_type_and_context():
    namespace = _loss_functions()
    labels = torch.tensor([0])
    sibling_variant_evidence = torch.tensor([[1.0, 8.0, 1.0]])
    unrelated_world_evidence = torch.tensor([[1.0, 1.0, 8.0]])

    sibling_components = namespace["component_weighted_rao_edl"](
        sibling_variant_evidence, labels
    )
    unrelated_components = namespace["component_weighted_rao_edl"](
        unrelated_world_evidence, labels
    )

    # Both predictions miss the exact variant, but only the sibling preserves
    # aircraft type, radar, and operator. Those three marginal losses, and hence
    # the complete blended objective, must reward the useful partial result.
    assert sibling_components[0, 1] < unrelated_components[0, 1]
    assert sibling_components[0, 2] < unrelated_components[0, 2]
    assert sibling_components[0, 3] < unrelated_components[0, 3]
    assert namespace["combined_rao_edl"](
        sibling_variant_evidence, labels
    ) < namespace["combined_rao_edl"](unrelated_world_evidence, labels)
