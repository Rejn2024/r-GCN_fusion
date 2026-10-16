"""Compare cached/source-reusing HGT training against the original edge formula."""
import ast
import copy
import json
from pathlib import Path
from types import MethodType
import unittest

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
try:
    from torch_geometric.nn import HGTConv
except ModuleNotFoundError:
    HGTConv = None

from rgcn_fusion.graph_optimisation import TrackGraphBatch, segment_softmax
from rgcn_fusion.track_hgt_optimisation import (
    aggregate_relation_messages, prepare_relation_graph,
)


def make_batch(dtype=torch.int32):
    # Reused sources, duplicate edges, self loops, an empty relation, and node 7 isolated.
    edges = torch.tensor([[0, 1, 0, 2, 0, 1, 0, 2, 3, 4],
                          [1, 2, 2, 3, 1, 3, 0, 4, 5, 6]], dtype=dtype)
    types = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 2, 2])
    return TrackGraphBatch(torch.tensor([0, 1]), torch.arange(8, dtype=dtype),
                           edges, types, torch.tensor([0, 1, 2, 3]),
                           torch.tensor([0, 0, 1, 1]), torch.arange(4))


def legacy_aggregate(x, weights, graph, chunk_size=0):
    messages = x.new_zeros((x.size(0), weights.size(-1)))
    degree = x.new_zeros(x.size(0))
    for relation, edges in enumerate(graph.edges):
        for start in range(0, edges.size(1), chunk_size or max(1, edges.size(1))):
            src, dst = edges[:, start:start + (chunk_size or edges.size(1))]
            transformed = x[src] @ weights[relation]
            messages.index_add_(0, dst, transformed.to(messages.dtype))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    return messages / degree.clamp_min(1).unsqueeze(-1)


def notebook_classes():
    notebook = json.loads((Path(__file__).parents[1] / 'notebooks/track_hgt.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'class RelationalGraphConvolutionBlock' in ''.join(c['source']))
    classes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)]
    namespace = dict(torch=torch, nn=nn, F=F, HGTConv=HGTConv,
                     segment_softmax=segment_softmax,
                     aggregate_relation_messages=aggregate_relation_messages)
    exec(compile(ast.Module(body=classes, type_ignores=[]), 'track_hgt.ipynb', 'exec'), namespace)
    return namespace


class TrackHGTOptimisationTest(unittest.TestCase):
    def test_views_degree_cache_and_budget(self):
        for dtype in (torch.int32, torch.int64):
            original = make_batch(dtype)
            batch, graph = prepare_relation_graph(original, 4)
            torch.testing.assert_close(graph.degree, torch.tensor([1., 2., 2., 2., 1., 1., 1., 1.]))
            for edges in graph.edges:
                self.assertEqual(edges.untyped_storage().data_ptr(), batch.edge_index.untyped_storage().data_ptr())
            self.assertIsNotNone(graph.sources[0])
            self.assertIsNotNone(graph.sources[1])
            self.assertIsNone(graph.sources[2])  # No source reuse: preserve original path.
            self.assertEqual(graph.edges[3].numel(), 0)
            used = sum(t.numel() * t.element_size() for t in
                       (graph.degree, *graph.sources, *graph.source_inverse) if t is not None)
            budget = sum(t.numel() * t.element_size() for t in
                         (original.edge_index, original.edge_types, original.node_indices))
            self.assertLessEqual(used, budget)
            moved = graph.to('cpu')
            self.assertEqual(moved.edge_index.data_ptr(), graph.edge_index.data_ptr())
            torch.testing.assert_close(moved.degree, graph.degree)
            # Keep edge multiplicity and relation assignments, regardless of grouping.
            for r, edges in enumerate(graph.edges):
                torch.testing.assert_close(edges, original.edge_index[:, original.edge_types == r])

    def test_outputs_and_gradients_match_edge_formula(self):
        torch.manual_seed(4)
        for chunk_size in (0, 1, 2, 100):
            for dtype in (torch.int32, torch.int64):
                _, graph = prepare_relation_graph(make_batch(dtype), 4)
                x = torch.randn(8, 5, dtype=torch.double, requires_grad=True)
                w = torch.randn(4, 5, 3, dtype=torch.double, requires_grad=True)
                old = legacy_aggregate(x, w, graph, chunk_size)
                new = aggregate_relation_messages(x, w, graph, chunk_size)
                torch.testing.assert_close(new, old, atol=1e-12, rtol=1e-12)
                probe = torch.randn_like(new)
                old_grads = torch.autograd.grad((old * probe).sum(), (x, w))
                new_grads = torch.autograd.grad((new * probe).sum(), (x, w))
                for actual, expected in zip(new_grads, old_grads):
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_notebook_prepares_once_and_reuses_static_graph(self):
        from unittest.mock import Mock
        notebook = json.loads((Path(__file__).parents[1] / 'notebooks/track_hgt.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if 'def prepare_batches(' in ''.join(c['source']))
        functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                     and node.name in ('_pin_for_cuda', 'prepare_batches', 'model_forward_batch')]
        prepare = Mock(wraps=prepare_relation_graph)
        namespace = dict(torch=torch, prepare_relation_graph=prepare, RELATIONS=range(4),
                         PIN_BATCH_MEMORY=False, DEVICE=torch.device('cpu'), MODEL_DTYPE=torch.float32,
                         X=torch.randn(8, 5), rao_base_rates_cpu=torch.full((2, 3), 1/3),
                         model=lambda x, graph, *args: graph)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'track_hgt.ipynb', 'exec'), namespace)
        batches = [make_batch()]
        prepared = namespace['prepare_batches'](batches)
        self.assertIs(batches[0], prepared[0][0])
        cached = prepared[0][1]
        for _ in range(2):
            active = namespace['model_forward_batch'](prepared[0])
            self.assertEqual(active.edge_index.data_ptr(), cached.edge_index.data_ptr())
            self.assertEqual(active.degree.data_ptr(), cached.degree.data_ptr())
        self.assertEqual(prepare.call_count, 1)

    def test_empty_graph(self):
        batch = make_batch()
        from dataclasses import replace
        batch = replace(batch, edge_index=batch.edge_index[:, :0], edge_types=batch.edge_types[:0])
        _, graph = prepare_relation_graph(batch, 4)
        result = aggregate_relation_messages(torch.randn(8, 5), torch.randn(4, 5, 3), graph)
        torch.testing.assert_close(result, torch.zeros(8, 3))
        torch.testing.assert_close(graph.degree, torch.ones(8))

    def test_lookup_cache_respects_budget_on_many_relations(self):
        from dataclasses import replace
        batch = make_batch()
        batch = replace(batch, edge_index=torch.zeros((2, 200), dtype=torch.int32),
                        edge_types=torch.arange(100).repeat_interleave(2))
        _, graph = prepare_relation_graph(batch, 100)
        used = sum(t.numel() * t.element_size() for t in
                   (graph.degree, *graph.sources, *graph.source_inverse) if t is not None)
        budget = sum(t.numel() * t.element_size() for t in
                     (batch.edge_index, batch.edge_types, batch.node_indices))
        self.assertLessEqual(used, budget)

    def check_training(self, device='cpu', amp_dtype=None):
        torch.manual_seed(7)
        classes = notebook_classes()
        batch, graph = prepare_relation_graph(make_batch(), 4)
        graph = graph.to(device)
        for checkpointing in (False, True):
            model = classes['TrackRGCNHGTClassifier'](
                5, 8, 8, 4, 3, 2, torch.ones(3, 2), num_message_passing_edges=2,
                num_rgcn_bases=2, num_hgt_layers=1, num_heads=2, dropout=0,
                edge_chunk_size=4, gradient_checkpointing=checkpointing).to(device)
            reference = copy.deepcopy(model)
            def legacy_forward(layer, x, graph):
                weights = torch.einsum('rb,bio->rio', layer.coefficients, layer.bases)
                messages = legacy_aggregate(x, weights, graph, layer.edge_chunk_size)
                updated = layer.dropout(F.gelu(layer.root(x) + messages + layer.bias))
                return layer.norm(layer.residual(x) + updated)
            for layer in reference.rgcn_layers:
                layer.forward = MethodType(legacy_forward, layer)
            inputs = (torch.randn(8, 5, device=device), graph,
                      batch.observation_nodes.to(device), batch.observation_to_track.to(device),
                      2, torch.full((2, 3), 1/3, device=device))
            results = []
            for active in (reference, model):
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = active(*inputs)
                    loss = sum(head['alpha'].square().mean() for head in out['evidential'].values())
                    loss = loss + out['track_embeddings'].square().mean()
                loss.backward()
                results.append(out)
            tol = dict(atol=0.02, rtol=0.02) if amp_dtype else dict(atol=2e-5, rtol=2e-5)
            for key in ('track_rao', 'radar_mode'):
                torch.testing.assert_close(results[0]['evidential'][key]['alpha'],
                                           results[1]['evidential'][key]['alpha'], **tol)
            for (name, old), (_, new) in zip(reference.named_parameters(), model.named_parameters()):
                self.assertEqual(old.grad is None, new.grad is None, name)
                if old.grad is not None:
                    torch.testing.assert_close(new.grad, old.grad, msg=name, **tol)
            # Exercise the actual optimizer after the checkpointed backward pass.
            torch.optim.AdamW(model.parameters()).step()

    @unittest.skipUnless(HGTConv is not None, 'Optional torch-geometric unavailable')
    def test_notebook_training_and_checkpointed_backward(self):
        self.check_training()

    @unittest.skipUnless(HGTConv is not None, 'Optional torch-geometric unavailable')
    def test_cpu_mixed_precision(self):
        self.check_training(amp_dtype=torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available() and HGTConv is not None, 'CUDA/PyG unavailable')
    def test_cuda_mixed_precision(self):
        self.check_training('cuda', torch.float16)


if __name__ == '__main__':
    unittest.main()
