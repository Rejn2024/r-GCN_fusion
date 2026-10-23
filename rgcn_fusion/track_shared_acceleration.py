"""Exact feature compaction and packed projections for shared-candidate HGT.

Parameters retain the reference model's names/shapes. Only raw, immutable inputs
are compacted; learned projections are recomputed every layer and optimizer step.
"""
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from .track_shared_hgt import EdgeConditionedHGTLayer, SharedCandidateHGTClassifier


def compact_shared_batch_features(batch, feature_names):
    """Deduplicate one batch at a time, retaining only memmap row lookups.

    Every original edge survives. Its feature reference points to an exactly
    identical representative. Scratch space is bounded by one input batch.
    """
    graph = batch.graph
    if graph.feature_rows is None or graph.feature_groups is not None:
        raise ValueError("Compact a prepared CPU batch exactly once before staging")
    values = graph.edge_features.index_select(0, graph.feature_rows).numpy()
    _, representatives, inverse = np.unique(values, axis=0, return_index=True, return_inverse=True)
    representatives = torch.from_numpy(representatives.astype(np.int64, copy=False))
    inverse = torch.from_numpy(inverse.astype(np.int64, copy=False))
    refs = graph.feature_indices.clone()
    attributed = refs >= 0
    refs[attributed] = inverse[refs[attributed]]
    unique_values = values[representatives.numpy()]
    intelligence = {"coverage_fraction", "applicable_observation_count", "time_delta_hours", "distance_100km"}
    candidate_columns = [i for i, name in enumerate(feature_names) if name not in intelligence]
    has_candidate_values = (unique_values[:, candidate_columns] != 0).any(axis=1)
    groups = []
    for mask in (has_candidate_values, ~has_candidate_values):
        rows = np.flatnonzero(mask)
        if rows.size:
            # Even nonstandard mixed rows preserve every nonzero column.
            columns = np.flatnonzero((unique_values[rows] != 0).any(axis=0))
            groups.append((torch.from_numpy(rows), torch.from_numpy(columns)))
    compact = replace(graph, feature_indices=refs,
                      feature_rows=graph.feature_rows[representatives], feature_groups=tuple(groups))
    return replace(batch, graph=compact)


class PackedEdgeConditionedHGTLayer(EdgeConditionedHGTLayer):
    def _typed(self, modules, x, type_indices):
        result = torch.zeros_like(x)
        for module, indices in zip(modules, type_indices):
            # The new buffer is not consumed until all disjoint writes complete.
            result.index_copy_(0, indices, module(x[indices]).to(result.dtype))
        return result

    def _project_qkv(self, x, graph):
        packed = x.new_zeros((x.size(0), 3 * self.hidden_dim))
        for q, k, v, indices in zip(self.q, self.k, self.v, graph.type_indices):
            weight = torch.cat((q.weight, k.weight, v.weight))
            bias = torch.cat((q.bias, k.bias, v.bias))
            values = F.linear(x[indices], weight, bias).to(packed.dtype)
            packed.index_copy_(0, indices, values)
        return packed.split(self.hidden_dim, dim=-1)

    def _project_edges(self, graph):
        weight = torch.cat((self.edge_key.weight, self.edge_value.weight))
        if graph.feature_groups is None:
            return F.linear(graph.edge_features, weight).split(self.hidden_dim, dim=-1)
        packed = None
        for rows, columns in graph.feature_groups:
            values = F.linear(graph.edge_features[rows][:, columns], weight[:, columns])
            if packed is None:
                packed = values.new_zeros((graph.num_feature_rows, 2 * self.hidden_dim))
            packed.index_copy_(0, rows, values)
        if packed is None:
            packed = F.linear(graph.edge_features, weight)
        return packed.split(self.hidden_dim, dim=-1)


class AcceleratedSharedCandidateHGTClassifier(SharedCandidateHGTClassifier):
    attention_layer_type = PackedEdgeConditionedHGTLayer


def compile_packed_projections(model, **options):
    """Compile bounded projection regions without unrolling every relation/layer.

    Compiled callables wrap bound methods, not registered modules, so optimizer
    ownership and state-dictionary names remain unchanged.
    """
    for layer in model.hgt_layers:
        layer._project_qkv = torch.compile(layer._project_qkv, **options)
        layer._project_edges = torch.compile(layer._project_edges, **options)
    return model


def packed_l1(parameters):
    """One packed L1 reduction; autograd propagates to the original parameters."""
    return torch.cat([parameter.reshape(-1) for parameter in parameters]).abs().sum()
