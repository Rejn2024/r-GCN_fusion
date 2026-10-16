"""Static graph preparation and source-reusing aggregation for the HGT notebook."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .graph_optimisation import TrackGraphBatch


@dataclass(frozen=True)
class PreparedRelationGraph:
    """Relation views share one grouped edge tensor; lookup caches are bounded."""

    edge_index: torch.Tensor
    edges: tuple[torch.Tensor, ...]
    sources: tuple[torch.Tensor | None, ...]
    source_inverse: tuple[torch.Tensor | None, ...]
    degree: torch.Tensor

    def to(self, device, *, non_blocking=False):
        # Transfer the shared edge storage once rather than each relation separately.
        grouped = self.edge_index.to(device, non_blocking=non_blocking)
        sizes = [edges.size(1) for edges in self.edges]
        return PreparedRelationGraph(
            grouped, tuple(grouped.split(sizes, dim=1)),
            tuple(None if value is None else value.to(device, non_blocking=non_blocking)
                  for value in self.sources),
            tuple(None if value is None else value.to(device, non_blocking=non_blocking)
                  for value in self.source_inverse),
            self.degree.to(device, non_blocking=non_blocking),
        )


def prepare_relation_graph(batch: TrackGraphBatch, num_relations: int):
    """Replace batch edges with grouped storage and cache immutable topology.

    Extra persistent tensors (degree and source lookups) never exceed the bytes in
    the original edge, relation-id and node-index tensors. Only relations with at
    least two edges per distinct source receive a source lookup. Preparation is
    CPU-only and happens once, outside all training/checkpoint/evaluation passes.
    """
    if num_relations < 1:
        raise ValueError("num_relations must be positive")
    types = batch.edge_types.cpu()
    if types.numel() and (int(types.min()) < 0 or int(types.max()) >= num_relations):
        raise ValueError("edge_types contains an out-of-range relation id")
    order = torch.argsort(types, stable=True)
    grouped_edges = batch.edge_index.cpu()[:, order].contiguous()
    grouped_types = types[order]
    sizes = torch.bincount(grouped_types.long(), minlength=num_relations).tolist()
    relation_edges = tuple(grouped_edges.split(sizes, dim=1))
    # Count all incoming edges across relations, including duplicate/self edges.
    # Accumulate in int64 before casting; half-precision degree counting can overflow.
    degree = torch.bincount(grouped_edges[1].long(), minlength=batch.node_indices.numel())
    degree = degree.clamp_min_(1).float()
    budget = sum(t.numel() * t.element_size() for t in
                 (batch.edge_index, batch.edge_types, batch.node_indices))
    remaining = budget - degree.numel() * degree.element_size()
    sources, inverses = [], []
    for edges in relation_edges:
        nodes = inverse = None
        if edges.size(1):
            unique, lookup = torch.unique(edges[0], sorted=True, return_inverse=True)
            # Compact indices also bound transfer/storage costs on large graphs.
            index_dtype = torch.int32 if batch.node_indices.numel() < 2**31 else torch.int64
            cache_bytes = (unique.numel() + lookup.numel()) * torch.empty((), dtype=index_dtype).element_size()
            if unique.numel() * 2 <= edges.size(1) and cache_bytes <= remaining:
                nodes, inverse = unique.to(index_dtype), lookup.to(index_dtype)
                remaining -= cache_bytes
        sources.append(nodes)
        inverses.append(inverse)
    grouped_batch = replace(batch, edge_index=grouped_edges, edge_types=grouped_types)
    return grouped_batch, PreparedRelationGraph(
        grouped_edges, relation_edges, tuple(sources), tuple(inverses), degree)


def aggregate_relation_messages(x, relation_weights, graph: PreparedRelationGraph, edge_chunk_size=0):
    """Sum x[source] @ W[relation], reusing transforms for repeated sources.

    Gather/scatter remains edge-chunked. For cached relations the dense transform
    runs on distinct sources only; other relations use the original chunked path.
    Normalization remains by total incoming degree, never per-relation degree.
    """
    if edge_chunk_size < 0:
        raise ValueError("edge_chunk_size must be non-negative")
    messages = x.new_zeros((x.size(0), relation_weights.size(-1)))
    for relation_id, edges in enumerate(graph.edges):
        if not edges.numel():
            continue
        src, dst = edges
        chunk_size = edge_chunk_size or edges.size(1)
        nodes, inverse = graph.sources[relation_id], graph.source_inverse[relation_id]
        projected = None
        # Bound the added projection buffer by one original edge chunk.
        if nodes is not None and nodes.numel() <= chunk_size:
            projected = x[nodes] @ relation_weights[relation_id]
        for start in range(0, edges.size(1), chunk_size):
            stop = start + chunk_size
            transformed = (x[src[start:stop]] @ relation_weights[relation_id]
                           if projected is None else projected[inverse[start:stop]])
            messages.index_add_(0, dst[start:stop], transformed.to(messages.dtype))
    return (messages / graph.degree.unsqueeze(-1)).to(x.dtype)
