import torch

from rgcn_fusion.model import RGCNEvidenceModel


def test_rgcn_evidence_model_emits_classification_heads():
    model = RGCNEvidenceModel(
        in_features=3,
        hidden_features=5,
        num_relations=2,
        num_hypotheses=2,
        classification_tasks={
            "radar_type": 4,
            "radar_mode": 3,
            "aircraft_variant": 6,
            "operator": 5,
        },
    )
    x = torch.randn(7, 3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_type = torch.tensor([0, 1, 0])

    outputs = model(x, edge_index, edge_type)

    assert outputs["masses"].shape == (7, 3)
    assert torch.allclose(outputs["masses"].sum(dim=-1), torch.ones(7), atol=1e-6)
    assert outputs["classification_logits"]["radar_type"].shape == (7, 4)
    assert outputs["classification_logits"]["radar_mode"].shape == (7, 3)
    assert outputs["classification_logits"]["aircraft_variant"].shape == (7, 6)
    assert outputs["classification_logits"]["operator"].shape == (7, 5)
