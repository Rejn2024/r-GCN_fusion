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


def test_rgcn_evidence_model_rejects_classification_tasks_with_too_few_classes():
    with pytest.raises(ValueError, match="at least two classes"):
        RGCNEvidenceModel(
            in_features=3,
            hidden_features=5,
            num_relations=2,
            num_hypotheses=2,
            classification_tasks={"radar_type": 1},
        )
