"""Typed multihead graph attention with edge-conditioned keys and messages.

Unlike PyG HGTConv, observation/candidate match attributes participate in both
attention and message values. The evidential readout contract matches track_star.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .track_shared_graph import NODE_TYPES, NODE_TYPE_INDEX


class EdgeConditionedHGTLayer(nn.Module):
    """One synchronous typed attention update across all relations.

    Attention is normalized over all incoming relations for each destination and
    head. Missing edge attributes contribute zero. Unique incidence attributes
    are projected once per layer, reused by the two edge directions.
    """

    def __init__(self, hidden_dim, relation_metadata, edge_dim, num_heads=4, dropout=.2):
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.hidden_dim, self.num_heads = hidden_dim, num_heads
        self.head_dim = hidden_dim // num_heads
        self.relation_metadata = tuple(tuple(r) for r in relation_metadata)
        count = len(self.relation_metadata)
        self.q = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.k = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.v = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.out = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.norm = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in NODE_TYPES)
        self.relation_key = nn.Parameter(torch.empty(count, num_heads, self.head_dim, self.head_dim))
        self.relation_value = nn.Parameter(torch.empty(count, num_heads, self.head_dim, self.head_dim))
        self.relation_prior = nn.Parameter(torch.ones(count, num_heads))
        self.edge_key = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.edge_value = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_key)
        nn.init.xavier_uniform_(self.relation_value)

    def _typed(self, modules, x, type_indices):
        result = torch.zeros_like(x)
        for module, indices in zip(modules, type_indices):
            result = result.index_copy(0, indices, module(x[indices]).to(result.dtype))
        return result

    def forward(self, x, graph):
        if len(graph.relation_sizes) != len(self.relation_metadata):
            raise ValueError("Graph relation schema does not match the model")
        shape = (-1, self.num_heads, self.head_dim)
        q = self._typed(self.q, x, graph.type_indices).view(shape)
        k = self._typed(self.k, x, graph.type_indices).view(shape)
        v = self._typed(self.v, x, graph.type_indices).view(shape)
        edge_keys = self.edge_key(graph.edge_features).view(shape)
        edge_values = self.edge_value(graph.edge_features).view(shape)
        scores, messages = [], []
        offset = 0
        for relation, size in enumerate(graph.relation_sizes):
            stop = offset + size
            if size:
                src, dst = graph.edge_index[:, offset:stop]
                cache = graph.source_caches[relation]
                sources = src if cache is None else cache[0]
                key = torch.einsum("ehd,hdf->ehf", k[sources], self.relation_key[relation])
                value = torch.einsum("ehd,hdf->ehf", v[sources], self.relation_value[relation])
                if cache is not None:
                    key, value = key[cache[1]], value[cache[1]]
                refs = graph.feature_indices[offset:stop]
                # Select only attributed edges; -1 must never index the last feature.
                mode = graph.relation_feature_modes[relation]
                if mode == 1:
                    key = key + edge_keys[refs].to(key.dtype)
                    value = value + edge_values[refs].to(value.dtype)
                elif mode == 2:
                    positions = graph.attributed_positions[relation]
                    key = key.index_add(0, positions, edge_keys[refs[positions]].to(key.dtype))
                    value = value.index_add(0, positions, edge_values[refs[positions]].to(value.dtype))
                score = (q[dst] * key).sum(-1) * self.relation_prior[relation] / math.sqrt(self.head_dim)
                scores.append(score)
                messages.append(value)
            offset = stop
        if not scores:
            return x
        scores = torch.cat(scores).float()
        messages = torch.cat(messages)
        destinations = graph.edge_index[1]
        index = destinations[:, None].expand(-1, self.num_heads)
        maxima = scores.new_full((x.size(0), self.num_heads), -torch.inf)
        maxima.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
        weights = torch.exp(scores - maxima[destinations])
        denominators = weights.new_zeros(maxima.shape).scatter_add_(0, index, weights)
        weights = weights / denominators[destinations].clamp_min(torch.finfo(weights.dtype).tiny)
        weighted = messages * weights.to(messages.dtype).unsqueeze(-1)
        aggregate = messages.new_zeros((x.size(0), self.num_heads, self.head_dim))
        aggregate.index_add_(0, destinations, weighted)
        update = self._typed(self.out, aggregate.flatten(1).to(x.dtype), graph.type_indices)
        return self._typed(self.norm, x + self.dropout(F.gelu(update)), graph.type_indices)


class SharedCandidateHGTClassifier(nn.Module):
    """Five edge-conditioned HGT layers; one RAO opinion per complete track."""

    def __init__(self, in_dim, hidden_dim, num_relations, num_rao_classes, num_mode_classes,
                 rao_mode_compatibility, *, relation_metadata, edge_dim, num_kg_entities,
                 num_hgt_layers=5, num_heads=4, dropout=.2, task_head_hidden_dim=128,
                 gradient_checkpointing=False):
        super().__init__()
        if len(relation_metadata) != num_relations:
            raise ValueError("Relation metadata and count disagree")
        if tuple(rao_mode_compatibility.shape) != (num_rao_classes, num_mode_classes):
            raise ValueError("Invalid RAO/mode compatibility shape")
        if num_hgt_layers < 1:
            raise ValueError("At least one attention layer is required")
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden_dim)
        # Stable KG ids are domain vocabulary, not dataset track ids or labels.
        self.kg_embedding = nn.Embedding(max(1, num_kg_entities), hidden_dim)
        self.track_seed = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.track_seed, std=hidden_dim ** -.5)
        self.hgt_layers = nn.ModuleList(EdgeConditionedHGTLayer(
            hidden_dim, relation_metadata, edge_dim, num_heads, dropout) for _ in range(num_hgt_layers))
        self.gradient_checkpointing = gradient_checkpointing

        def head(dim, classes):
            return nn.Sequential(nn.Linear(dim, task_head_hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(task_head_hidden_dim, classes))

        self.rao_evidential_head = head(hidden_dim, num_rao_classes)
        self.mode_evidential_head = head(2 * hidden_dim, num_mode_classes)
        self.register_buffer("rao_mode_compatibility", rao_mode_compatibility.float())

    @staticmethod
    def dirichlet_ds_output(logits, base_rates=None):
        evidence = F.softplus(logits.float())
        classes = evidence.size(-1)
        rates = torch.full_like(evidence, 1 / classes) if base_rates is None else base_rates.to(evidence)
        prior = classes * rates
        alpha = evidence + prior
        strength = alpha.sum(-1, keepdim=True)
        return {"evidence": evidence, "alpha": alpha, "strength": strength,
                "probabilities": alpha / strength, "belief": evidence / strength,
                "uncertainty": prior.sum(-1, keepdim=True) / strength,
                "base_rates": rates, "prior_alpha": prior}

    def encode(self, x, graph):
        if x.size(0) != graph.num_nodes:
            raise ValueError("Feature rows do not match the prepared graph")
        if graph.feature_rows is not None:
            raise ValueError("Load the prepared graph with load_shared_batch before forwarding")
        hidden = self.input_projection(x) + self.type_embedding(graph.node_types)
        kg_nodes = graph.type_indices[NODE_TYPE_INDEX["kg_entity"]]
        hidden = hidden.index_add(0, kg_nodes, self.kg_embedding(graph.kg_indices[kg_nodes]))
        hidden = hidden.index_copy(0, graph.track_nodes, self.track_seed.expand(graph.num_tracks, -1))
        for layer in self.hgt_layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(lambda values, active=layer: active(values, graph), hidden, use_reentrant=False)
            else:
                hidden = layer(hidden, graph)
        return {"node": hidden, "track": hidden[graph.track_nodes]}

    def forward(self, x, graph, observation_nodes, observation_to_track, num_tracks, rao_base_rates):
        if num_tracks != graph.num_tracks:
            raise ValueError("Track count does not match the prepared graph")
        encoded = self.encode(x, graph)
        tracks, observations = encoded["track"], encoded["node"][observation_nodes]
        rao = self.dirichlet_ds_output(self.rao_evidential_head(tracks), rao_base_rates)
        features = torch.cat((observations, tracks[observation_to_track]), dim=-1)
        compatibility = rao["probabilities"] @ self.rao_mode_compatibility.float()
        bias = compatibility[observation_to_track].clamp_min(1e-8).log()
        mode = self.dirichlet_ds_output(self.mode_evidential_head(features) + bias)
        return {"evidential": {"track_rao": rao, "radar_mode": mode},
                "track_embeddings": tracks, "observation_embeddings": observations}
