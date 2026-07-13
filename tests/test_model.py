import pytest
import torch

from rgcn_fusion.model import RGCNEvidenceModel


def test_rgcn_evidence_model_emits_ds_derived_classification_scores():
    model = RGCNEvidenceModel(
        in_features=3,
        hidden_features=5,
        num_relations=2,
        num_hypotheses=2,
        classification_tasks={
            "aircraft_variant": 2,
            "operator": 2,
        },
    )
    x = torch.randn(7, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_type = torch.tensor([0, 1, 0])

    outputs = model(x, edge_index, edge_type)
    expected_scores = model.interval_midpoints(outputs["masses"])

    assert outputs["masses"].shape == (7, 3)
    assert torch.allclose(outputs["masses"].sum(dim=-1), torch.ones(7), atol=1e-6)
    assert outputs["classification_logits"]["aircraft_variant"].shape == (7, 2)
    assert outputs["classification_logits"]["operator"].shape == (7, 2)
    assert torch.allclose(outputs["classification_logits"]["aircraft_variant"], expected_scores)
    assert torch.allclose(outputs["classification_logits"]["operator"], expected_scores)


def test_rgcn_evidence_model_adds_auxiliary_heads_for_non_hypothesis_class_counts():
    model = RGCNEvidenceModel(
        in_features=3,
        hidden_features=5,
        num_relations=2,
        num_hypotheses=2,
        classification_tasks={"radar_type": 4},
    )
    x = torch.randn(7, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_type = torch.tensor([0, 1, 0])

    outputs = model(x, edge_index, edge_type)

    assert outputs["classification_logits"]["radar_type"].shape == (7, 4)


def test_rgcn_evidence_model_supports_residual_basis_gated_dirichlet_stack():
    model = RGCNEvidenceModel(
        in_features=3,
        hidden_features=8,
        num_relations=3,
        num_hypotheses=2,
        num_layers=3,
        num_bases=2,
        relation_gates=True,
        task_head_hidden_features=6,
        mass_head_type="dirichlet",
        classification_tasks={"radar_type": 4},
    )
    x = torch.randn(7, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_type = torch.tensor([0, 1, 2, 0])

    outputs = model(x, edge_index, edge_type)

    assert len(model.layers) == 3
    assert model.layers[0].conv.num_bases == 2
    assert model.layers[0].conv.relation_gate_logits is not None
    assert outputs["masses"].shape == (7, 3)
    assert torch.allclose(outputs["masses"].sum(dim=-1), torch.ones(7), atol=1e-6)
    assert outputs["dirichlet_alpha"].shape == (7, 3)
    assert outputs["dirichlet_evidence"].shape == (7, 3)
    assert outputs["uncertainty"].shape == (7, 1)
    assert torch.all(outputs["dirichlet_alpha"] > 1.0)
    assert torch.all(outputs["uncertainty"] > 0.0)
    assert outputs["classification_logits"]["radar_type"].shape == (7, 4)


def test_rgcn_evidence_model_rejects_classification_tasks_with_too_few_classes():
    with pytest.raises(ValueError, match="at least two classes"):
        RGCNEvidenceModel(
            in_features=3,
            hidden_features=5,
            num_relations=2,
            num_hypotheses=2,
            classification_tasks={"radar_type": 1},
        )


def test_model_uses_singleton_uncertainty_masses_for_more_than_ten_hypotheses():
    model = RGCNEvidenceModel(
        in_features=3,
        hidden_features=4,
        num_relations=1,
        num_hypotheses=11,
    )
    assert model.num_masses == 12

    masses = torch.zeros(2, 12)
    masses[:, 3] = 0.8
    masses[:, -1] = 0.2
    midpoint_scores = model.interval_midpoints(masses)
    assert midpoint_scores.shape == (2, 11)
    assert torch.allclose(midpoint_scores[:, 3], torch.full((2,), 0.9))
    assert torch.allclose(midpoint_scores[:, 0], torch.full((2,), 0.1))
