"""Bounded, synchronous staging of shared-candidate complete-track batches."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceSharedBatch:
    prepared: object
    x: torch.Tensor
    graph: object
    observation_nodes: torch.Tensor
    observation_to_track: torch.Tensor
    track_indices: torch.Tensor
    observation_positions: torch.Tensor
    rao_labels: torch.Tensor
    mode_labels: torch.Tensor
    rao_base_rates: torch.Tensor
    observation_counts: torch.Tensor

    @property
    def num_tracks(self):
        return self.prepared.num_tracks

    @property
    def num_observations(self):
        return self.prepared.num_observations


def prepare_shared_batches(batches, num_relations):
    """Topology and incidence feature tables are already prepared during batching."""
    batches = list(batches)
    for batch in batches:
        if len(batch.graph.relation_sizes) != num_relations:
            raise ValueError("Prepared batch has the wrong relation schema")
    return batches


def load_shared_batch(prepared, X, track_targets, mode_targets, rao_base_rates_cpu, *, device, dtype):
    device = torch.device(device)
    tracks = prepared.track_indices.to(device)
    positions = prepared.observation_positions.to(device)
    return DeviceSharedBatch(
        prepared, X.index_select(0, prepared.node_indices).to(device=device, dtype=dtype),
        prepared.graph.to(device), prepared.observation_nodes.to(device),
        prepared.observation_to_track.to(device), tracks, positions,
        track_targets.index_select(0, tracks.to(track_targets.device)).to(device),
        mode_targets.index_select(0, positions.to(mode_targets.device)).to(device),
        rao_base_rates_cpu.index_select(0, prepared.track_indices).to(device),
        prepared.observation_counts.to(device),
    )


@contextmanager
def iter_shared_batches(prepared_batches, X, track_targets, mode_targets, rao_base_rates_cpu, *,
                        device, dtype, prefetch=False):
    """Gather only active node features; explicit edge tables move with their graph."""
    if prefetch:
        raise ValueError("This experiment uses synchronous staging; set PREFETCH_BATCHES=False")
    iterator = (load_shared_batch(batch, X, track_targets, mode_targets, rao_base_rates_cpu,
                                  device=device, dtype=dtype) for batch in prepared_batches)
    try:
        yield iterator
    finally:
        iterator.close()
