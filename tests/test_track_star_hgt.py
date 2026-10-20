"""Exercise virtual-track topology, HGT information flow, and notebook consumers."""

import copy
from dataclasses import replace
import unittest

import torch
import torch.nn.functional as F

from rgcn_fusion.graph_optimisation import TrackGraphBatch

try:
    from rgcn_fusion.track_star_hgt import (
        TrackStarHGTClassifier,
        prepare_track_star_graph,
    )
except ModuleNotFoundError as error:
    if not error.name.startswith("torch_geometric"):
        raise
    TrackStarHGTClassifier = None


BELONGS_TO = ("node", "belongs_to", "track")
CONTAINS = ("track", "contains", "node")


def make_batch(*, empty=False, dtype=torch.int32):
    """Global IDs deliberately differ from each local feature/track index."""
    edges = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 5, 2], [0, 1, 2, 3, 4, 5, 4, 1]], dtype=dtype
    )
    types = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)
    if empty:
        edges, types = edges[:, :0], types[:0]
    return TrackGraphBatch(
        track_indices=torch.tensor([17, 9]),
        node_indices=torch.tensor([10, 100, 5, 44, 33, 22], dtype=dtype),
        edge_index=edges,
        edge_types=types,
        observation_nodes=torch.tensor([4, 1, 3, 0]),
        observation_to_track=torch.tensor([1, 0, 1, 0]),
        observation_positions=torch.tensor([5, 4, 8, 0]),
    )


def make_model(*, checkpointing=False):
    return TrackStarHGTClassifier(
        4, 8, 3, 3, 2, torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
        num_hgt_layers=5, num_heads=2, dropout=0,
        task_head_hidden_dim=8, gradient_checkpointing=checkpointing,
    )


@unittest.skipUnless(TrackStarHGTClassifier is not None, "Optional torch-geometric unavailable")
class TrackStarHGTTest(unittest.TestCase):
    def test_incidence_uses_batch_local_membership_and_preserves_evidence(self):
        for dtype in (torch.int32, torch.int64):
            batch = make_batch(dtype=dtype)
            original_nodes = batch.node_indices.clone()
            graph = prepare_track_star_graph(batch, 3)
            self.assertEqual(graph.num_nodes, 6)
            self.assertEqual(graph.num_tracks, 2)
            expected = torch.stack((batch.observation_nodes, batch.observation_to_track))
            torch.testing.assert_close(graph.edge_index_dict[BELONGS_TO], expected)
            torch.testing.assert_close(graph.edge_index_dict[CONTAINS], expected.flip(0))
            for relation in range(3):
                actual = graph.edge_index_dict[("node", f"relation_{relation}", "node")]
                torch.testing.assert_close(
                    actual, batch.edge_index[:, batch.edge_types == relation].long()
                )
            self.assertEqual(graph.edge_index_dict[("node", "relation_2", "node")].shape, (2, 0))
            self.assertEqual(sum(edges.size(1) for edges in graph.edge_index_dict.values()), 16)
            torch.testing.assert_close(batch.node_indices, original_nodes)
            moved = graph.to("cpu", non_blocking=True)
            for key, edges in graph.edge_index_dict.items():
                torch.testing.assert_close(moved.edge_index_dict[key], edges)

    def test_invalid_incidence_is_rejected_before_message_passing(self):
        batch = make_batch()
        for invalid in (
            replace(batch, observation_to_track=torch.tensor([1, 0, 2, 0])),
            replace(batch, observation_to_track=torch.tensor([1, 0, -1, 0])),
            replace(batch, observation_to_track=torch.tensor([0, 0, 0, 0])),
            replace(batch, observation_nodes=torch.tensor([6, 1, 3, 0])),
            replace(batch, observation_to_track=torch.tensor([1, 0, 1])),
            replace(batch, edge_types=torch.tensor([0, 0, 0, 0, 0, 0, 1, 3])),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare_track_star_graph(invalid, 3)

    def test_five_hgt_layers_and_original_evidential_contract(self):
        torch.manual_seed(12)
        batch = make_batch()
        graph = prepare_track_star_graph(batch, 3)
        model = make_model().eval()
        self.assertEqual(len(model.hgt_layers), 5)
        self.assertFalse(hasattr(model, "rgcn_layers"))
        self.assertFalse(hasattr(model, "track_pool"))
        priors = torch.tensor([[.2, .3, .5], [.6, .1, .3]])
        output = model(torch.randn(6, 4), graph, batch.observation_nodes,
                       batch.observation_to_track, 2, priors)
        self.assertEqual(output["track_embeddings"].shape, (2, 8))
        self.assertEqual(output["observation_embeddings"].shape, (4, 8))
        self.assertEqual(set(output["evidential"]), {"track_rao", "radar_mode"})
        for name, rows, classes in (("track_rao", 2, 3), ("radar_mode", 4, 2)):
            opinion = output["evidential"][name]
            self.assertEqual(set(opinion), {"evidence", "alpha", "strength", "probabilities",
                                           "belief", "uncertainty", "base_rates", "prior_alpha"})
            self.assertEqual(opinion["probabilities"].shape, (rows, classes))
            torch.testing.assert_close(opinion["probabilities"].sum(-1), torch.ones(rows))
            torch.testing.assert_close(opinion["belief"].sum(-1, keepdim=True) +
                                       opinion["uncertainty"], torch.ones(rows, 1))
            self.assertTrue(torch.isfinite(opinion["alpha"]).all())
        torch.testing.assert_close(output["evidential"]["track_rao"]["base_rates"], priors)
        torch.testing.assert_close(output["evidential"]["track_rao"]["prior_alpha"], 3 * priors)

    def test_star_path_transmits_nonlocal_observations_without_mixing_tracks(self):
        torch.manual_seed(21)
        batch = make_batch(empty=True)
        graph = prepare_track_star_graph(batch, 3)
        model = make_model().double().eval()
        x = torch.randn(6, 4, dtype=torch.double, requires_grad=True)
        intermediate_nodes = []
        hooks = [layer.register_forward_hook(
            lambda _module, _inputs, result: intermediate_nodes.append(result[0])
        ) for layer in model.hgt_layers[:2]]
        encoded = model.encode(x, graph)
        for hook in hooks:
            hook.remove()
        # Nodes 0 and 1 share track 0; 3 and 4 belong to a disconnected track.
        probe = torch.arange(1, 9, dtype=torch.double)
        one_hop = torch.autograd.grad((intermediate_nodes[0][0] * probe).sum(), x,
                                      retain_graph=True)[0]
        two_hops = torch.autograd.grad((intermediate_nodes[1][0] * probe).sum(), x,
                                       retain_graph=True)[0]
        torch.testing.assert_close(one_hop[1], torch.zeros(4, dtype=torch.double), atol=0, rtol=0)
        self.assertGreater(float(two_hops[1].abs().max()), 1e-12)
        gradient = torch.autograd.grad((encoded["node"][0] * probe).sum(), x)[0]
        self.assertGreater(float(gradient[1].abs().max()), 1e-12)
        torch.testing.assert_close(gradient[[3, 4]], torch.zeros(2, 4, dtype=torch.double),
                                   atol=0, rtol=0)
        with torch.no_grad():
            changed = x.detach().clone()
            changed[1] += torch.tensor([2., -1., 3., -.5], dtype=torch.double)
            alternative = model.encode(changed, graph)
        self.assertGreater(float((alternative["node"][0] - encoded["node"][0].detach()).abs().max()), 1e-10)
        torch.testing.assert_close(alternative["node"][[3, 4]], encoded["node"][[3, 4]])
        torch.testing.assert_close(alternative["track"][1], encoded["track"][1])

    def test_checkpointed_backward_matches_plain_and_optimizer_updates(self):
        torch.manual_seed(32)
        batch = make_batch()
        graph = prepare_track_star_graph(batch, 3)
        model = make_model().train()
        checkpointed = copy.deepcopy(model)
        checkpointed.gradient_checkpointing = True
        x = torch.randn(6, 4)
        priors = torch.tensor([[.2, .3, .5], [.6, .1, .3]])
        outputs = []
        for active in (model, checkpointed):
            output = active(x, graph, batch.observation_nodes,
                            batch.observation_to_track, 2, priors)
            loss = F.nll_loss(output["evidential"]["track_rao"]["probabilities"].log(),
                              torch.tensor([1, 2]))
            loss = loss + F.nll_loss(output["evidential"]["radar_mode"]["probabilities"].log(),
                                     torch.tensor([0, 1, 1, 0]))
            loss.backward()
            outputs.append(output)
            self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                                for parameter in active.parameters()))
            before = next(active.parameters()).detach().clone()
            torch.optim.AdamW(active.parameters(), lr=.01).step()
            self.assertFalse(torch.equal(before, next(active.parameters()).detach()))
        for task in ("track_rao", "radar_mode"):
            torch.testing.assert_close(outputs[0]["evidential"][task]["alpha"],
                                       outputs[1]["evidential"][task]["alpha"])
        for (name, parameter), (_, other) in zip(model.named_parameters(), checkpointed.named_parameters()):
            self.assertEqual(parameter.grad is None, other.grad is None, name)
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, msg=name)


if __name__ == "__main__":
    unittest.main()
