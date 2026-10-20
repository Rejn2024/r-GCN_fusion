"""HGT evidence encoding with one virtual incidence node per complete track.

Track nodes live in hidden-feature space, so saved feature matrices and global
observation indices retain their original meaning.  Membership is represented
by two sparse relations rather than an observation clique.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch_geometric.nn import HGTConv

from .graph_optimisation import TrackGraphBatch


OBSERVATION_TO_TRACK = ("node", "belongs_to", "track")
TRACK_TO_OBSERVATION = ("track", "contains", "node")


def star_edge_types(num_relations: int) -> tuple[tuple[str, str, str], ...]:
    """Stable HGT metadata, including relations absent from an individual batch."""
    return (*(("node", f"relation_{i}", "node") for i in range(num_relations)),
            OBSERVATION_TO_TRACK, TRACK_TO_OBSERVATION)


@dataclass(frozen=True)
class PreparedTrackStarGraph:
    """Cached CPU topology with independent local indices for evidence and tracks."""

    num_nodes: int
    num_tracks: int
    evidence_edges: torch.Tensor
    relation_sizes: tuple[int, ...]
    incidence_edges: torch.Tensor
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = field(init=False)

    def __post_init__(self):
        original_types = star_edge_types(len(self.relation_sizes))[:-2]
        edges = dict(zip(original_types, self.evidence_edges.split(self.relation_sizes, dim=1)))
        edges[OBSERVATION_TO_TRACK] = self.incidence_edges
        edges[TRACK_TO_OBSERVATION] = self.incidence_edges.flip(0)
        object.__setattr__(self, "edge_index_dict", edges)

    @property
    def num_edges(self) -> int:
        return self.evidence_edges.size(1) + 2 * self.incidence_edges.size(1)

    @property
    def topology_bytes(self) -> int:
        # Evidence partitions are shared views, while reverse incidence is cached.
        return (self.evidence_edges.numel() * self.evidence_edges.element_size()
                + 2 * self.incidence_edges.numel() * self.incidence_edges.element_size())

    def to(self, device, *, non_blocking=False):
        # Move the evidence storage once, rather than transferring each view.
        return PreparedTrackStarGraph(
            self.num_nodes, self.num_tracks,
            self.evidence_edges.to(device, non_blocking=non_blocking),
            self.relation_sizes,
            self.incidence_edges.to(device, non_blocking=non_blocking),
        )


def prepare_track_star_graph(batch: TrackGraphBatch, num_relations: int) -> PreparedTrackStarGraph:
    """Cache all evidence relations plus exactly two incidence edges per observation.

    Virtual track ids index only ``batch.track_indices``; they are never used to
    index the persisted feature matrix.  Only observations receive membership
    edges. Candidate, claim, report and KG nodes retain their existing topology.
    """
    if num_relations < 1:
        raise ValueError("num_relations must be positive")
    num_nodes, num_tracks = batch.node_indices.numel(), batch.track_indices.numel()
    if num_tracks < 1:
        raise ValueError("A track batch must contain at least one track")
    edge_index, edge_types = batch.edge_index.cpu(), batch.edge_types.cpu()
    observations = batch.observation_nodes.cpu().long()
    membership = batch.observation_to_track.cpu().long()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if edge_types.ndim != 1 or edge_types.numel() != edge_index.size(1):
        raise ValueError("edge_types must have one relation id per edge")
    if observations.ndim != 1 or membership.ndim != 1 or observations.numel() != membership.numel():
        raise ValueError("observation_nodes and observation_to_track must be aligned vectors")
    if observations.numel() == 0:
        raise ValueError("Every mini-batch track must contain an observation")
    if int(observations.min()) < 0 or int(observations.max()) >= num_nodes:
        raise ValueError("observation_nodes contains an out-of-range local node id")
    if torch.unique(observations).numel() != observations.numel():
        raise ValueError("An observation must occur exactly once in the membership mapping")
    if int(membership.min()) < 0 or int(membership.max()) >= num_tracks:
        raise ValueError("observation_to_track contains an out-of-range local track id")
    if torch.any(torch.bincount(membership, minlength=num_tracks) == 0):
        raise ValueError("Every mini-batch track must contain an observation")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("edge_index contains an out-of-range local node id")
    if edge_types.numel() and (int(edge_types.min()) < 0 or int(edge_types.max()) >= num_relations):
        raise ValueError("edge_types contains an out-of-range relation id")
    order = torch.argsort(edge_types, stable=True)
    # HGT/PyG uses these tensors directly for gathers, so cache canonical int64.
    grouped_edges = edge_index[:, order].long().contiguous()
    sizes = tuple(torch.bincount(edge_types.long(), minlength=num_relations).tolist())
    incidence = torch.stack((observations, membership))
    return PreparedTrackStarGraph(num_nodes, num_tracks, grouped_edges, sizes, incidence)


class TrackStarHGTLayer(nn.Module):
    """One synchronous HGT update across evidence and virtual track node types."""

    def __init__(self, hidden_dim, num_relations, num_heads, dropout):
        super().__init__()
        self.conv = HGTConv(hidden_dim, hidden_dim,
                            metadata=(["node", "track"], list(star_edge_types(num_relations))),
                            heads=num_heads)
        self.norm = nn.ModuleDict({kind: nn.LayerNorm(hidden_dim) for kind in ("node", "track")})
        self.dropout = nn.Dropout(dropout)

    def forward(self, evidence, tracks, graph):
        inputs = {"node": evidence, "track": tracks}
        updated = self.conv(inputs, graph.edge_index_dict)
        outputs = {}
        for kind, previous in inputs.items():
            message = updated.get(kind)
            outputs[kind] = (previous if message is None else
                             self.norm[kind](previous + self.dropout(F.gelu(message))))
        return outputs["node"], outputs["track"]


class TrackStarHGTClassifier(nn.Module):
    """HGT-only encoding, direct track RAO readout and conditioned observation mode."""

    def __init__(self, in_dim, hidden_dim, num_relations, num_rao_classes, num_mode_classes,
                 rao_mode_compatibility, num_hgt_layers=5, num_heads=4, dropout=0.2,
                 task_head_hidden_dim=128, gradient_checkpointing=False):
        super().__init__()
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_relations < 1 or num_hgt_layers < 1:
            raise ValueError("num_relations and num_hgt_layers must be positive")
        if tuple(rao_mode_compatibility.shape) != (num_rao_classes, num_mode_classes):
            raise ValueError("rao_mode_compatibility must have shape [RAO classes, mode classes]")
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                              nn.GELU(), nn.Dropout(dropout))
        # Shared initialization supports unseen tracks; dataset ids are not features.
        self.track_seed = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.track_seed, std=hidden_dim ** -0.5)
        self.hgt_layers = nn.ModuleList([
            TrackStarHGTLayer(hidden_dim, num_relations, num_heads, dropout)
            for _ in range(num_hgt_layers)
        ])
        self.gradient_checkpointing = bool(gradient_checkpointing)

        def head(in_features, out_features):
            return nn.Sequential(nn.Linear(in_features, task_head_hidden_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(task_head_hidden_dim, out_features))

        self.rao_evidential_head = head(hidden_dim, num_rao_classes)
        self.mode_evidential_head = head(2 * hidden_dim, num_mode_classes)
        self.register_buffer("rao_mode_compatibility", rao_mode_compatibility.float())

    @staticmethod
    def dirichlet_ds_output(evidence_logits, base_rates=None):
        # Preserve the source notebook's float32 evidential output contract.
        evidence = F.softplus(evidence_logits.float())
        classes = evidence.size(-1)
        if base_rates is None:
            base_rates = torch.full_like(evidence, 1.0 / classes)
        else:
            base_rates = base_rates.to(device=evidence.device, dtype=evidence.dtype)
        prior_alpha = classes * base_rates
        alpha = evidence + prior_alpha
        strength = alpha.sum(-1, keepdim=True)
        return {"evidence": evidence, "alpha": alpha, "strength": strength,
                "probabilities": alpha / strength, "belief": evidence / strength,
                "uncertainty": prior_alpha.sum(-1, keepdim=True) / strength,
                "base_rates": base_rates, "prior_alpha": prior_alpha}

    def encode(self, x, graph: PreparedTrackStarGraph):
        if x.ndim != 2 or x.size(0) != graph.num_nodes:
            raise ValueError("x must contain exactly the original evidence nodes in this batch")
        evidence = self.input_projection(x)
        tracks = self.track_seed.expand(graph.num_tracks, -1)
        for layer in self.hgt_layers:
            if self.gradient_checkpointing and self.training:
                evidence, tracks = torch.utils.checkpoint.checkpoint(
                    lambda nodes, hubs, active=layer: active(nodes, hubs, graph),
                    evidence, tracks, use_reentrant=False)
            else:
                evidence, tracks = layer(evidence, tracks, graph)
        return {"node": evidence, "track": tracks}

    def forward(self, x, graph, observation_nodes, observation_to_track, num_tracks, rao_base_rates):
        if num_tracks != graph.num_tracks:
            raise ValueError("num_tracks does not match the prepared graph")
        encoded = self.encode(x, graph)
        observation_embeddings = encoded["node"][observation_nodes]
        track_embeddings = encoded["track"]
        rao_evidential = self.dirichlet_ds_output(self.rao_evidential_head(track_embeddings), rao_base_rates)
        mode_features = torch.cat((observation_embeddings, track_embeddings[observation_to_track]), dim=-1)
        compatibility = rao_evidential["probabilities"] @ self.rao_mode_compatibility.float()
        bias = compatibility[observation_to_track].clamp_min(1e-8).log()
        mode_evidential = self.dirichlet_ds_output(self.mode_evidential_head(mode_features) + bias)
        return {"evidential": {"track_rao": rao_evidential, "radar_mode": mode_evidential},
                "track_embeddings": track_embeddings, "observation_embeddings": observation_embeddings}
