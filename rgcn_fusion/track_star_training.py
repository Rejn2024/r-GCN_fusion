"""Bounded feature/topology staging for complete-track HGT batches.

Prepared batches retain sorted topology and compact lookup tensors, rather than
the original unsorted edge buffers. CUDA iteration keeps only two reusable
pinned host slots and one queued device batch. Model activations and learned
values are never cached here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch

from .graph_optimisation import TrackGraphBatch
from .track_star_hgt import (
    PreparedTrackStarGraph,
    TRACK_TO_OBSERVATION,
    prepare_track_star_graph,
)


@dataclass(frozen=True)
class PreparedStarBatch:
    """Persistent CPU topology and lookup tensors for one induced track group."""

    node_indices: torch.Tensor
    track_indices: torch.Tensor
    observation_nodes: torch.Tensor
    observation_to_track: torch.Tensor
    observation_positions: torch.Tensor
    observation_counts: torch.Tensor
    graph: PreparedTrackStarGraph

    @property
    def num_tracks(self) -> int:
        return self.track_indices.numel()

    @property
    def num_observations(self) -> int:
        return self.observation_positions.numel()


@dataclass(frozen=True)
class DeviceStarBatch:
    """One coherent active batch, with labels kept separate from model features."""

    prepared: PreparedStarBatch
    x: torch.Tensor
    graph: PreparedTrackStarGraph
    observation_nodes: torch.Tensor
    observation_to_track: torch.Tensor
    track_indices: torch.Tensor
    observation_positions: torch.Tensor
    rao_labels: torch.Tensor
    mode_labels: torch.Tensor
    rao_base_rates: torch.Tensor
    observation_counts: torch.Tensor

    @property
    def num_tracks(self) -> int:
        return self.prepared.num_tracks

    @property
    def num_observations(self) -> int:
        return self.prepared.num_observations

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Retain copy-stream allocations until their consumer work completes."""
        for name in (
            "x", "observation_nodes", "observation_to_track", "track_indices",
            "observation_positions", "rao_labels", "mode_labels", "rao_base_rates",
            "observation_counts",
        ):
            getattr(self, name).record_stream(stream)
        self.graph.evidence_edges.record_stream(stream)
        self.graph.incidence_edges.record_stream(stream)
        self.graph.edge_index_dict[TRACK_TO_OBSERVATION].record_stream(stream)


def prepare_star_batches(
    batches: Iterable[TrackGraphBatch], num_relations: int,
) -> list[PreparedStarBatch]:
    """Replace original edge/type storage with one canonical relation graph.

    The caller should drop its original batch collection after preparation; no
    ``TrackGraphBatch`` object or original edge/type tensor is retained here.
    All track and observation identifiers preserve their saved global ordering.
    """
    prepared = []
    for batch in batches:
        graph = prepare_track_star_graph(batch, num_relations)
        membership = batch.observation_to_track.detach().cpu().long().contiguous()
        prepared.append(PreparedStarBatch(
            node_indices=batch.node_indices.detach().cpu().contiguous(),
            track_indices=batch.track_indices.detach().cpu().long().contiguous(),
            observation_nodes=batch.observation_nodes.detach().cpu().long().contiguous(),
            observation_to_track=membership,
            observation_positions=batch.observation_positions.detach().cpu().long().contiguous(),
            observation_counts=torch.bincount(membership, minlength=graph.num_tracks),
            graph=graph,
        ))
    return prepared


class _PinnedSlot:
    """A reusable pinned buffer set, bounded by the largest staged batch."""

    def __init__(self):
        self.buffers: dict[str, torch.Tensor] = {}
        self.ready: torch.cuda.Event | None = None

    def wait_before_reuse(self) -> None:
        # A pinned host buffer cannot be overwritten while DMA still reads it.
        if self.ready is not None:
            self.ready.synchronize()

    def stage(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if value.device.type != "cpu":
            raise ValueError("Only CPU values may be staged in pinned buffers")
        buffer = self.buffers.get(name)
        if buffer is None or buffer.dtype != value.dtype or buffer.numel() < value.numel():
            buffer = torch.empty(value.numel(), dtype=value.dtype, pin_memory=True)
            self.buffers[name] = buffer
        view = buffer[:value.numel()].view(value.shape)
        view.copy_(value)
        return view


def _load_batch(
    prepared: PreparedStarBatch,
    X: torch.Tensor,
    track_targets: torch.Tensor,
    mode_targets: torch.Tensor,
    rao_base_rates_cpu: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    slot: _PinnedSlot | None = None,
) -> DeviceStarBatch:
    def move(name: str, value: torch.Tensor) -> torch.Tensor:
        if slot is not None:
            value = slot.stage(name, value)
        return value.to(device=device, non_blocking=slot is not None)

    # The feature source remains memmapped. Only an active batch is gathered.
    local_x = X.index_select(0, prepared.node_indices.long()).to(dtype=dtype).contiguous()
    x = move("x", local_x)
    indices = {
        name: move(name, getattr(prepared, name))
        for name in ("observation_nodes", "observation_to_track", "track_indices",
                     "observation_positions", "observation_counts")
    }
    graph = PreparedTrackStarGraph(
        prepared.graph.num_nodes, prepared.graph.num_tracks,
        move("evidence_edges", prepared.graph.evidence_edges),
        prepared.graph.relation_sizes,
        move("incidence_edges", prepared.graph.incidence_edges),
    )

    def labels(name, targets, index_name):
        # Targets are normally already on the accelerator. Reuse the indices
        # just copied rather than transferring them a second time for the loss.
        if targets.device == device:
            return targets.index_select(0, indices[index_name])
        host_indices = getattr(prepared, index_name)
        values = targets.index_select(0, host_indices.to(targets.device))
        return move(name, values)

    priors = rao_base_rates_cpu.index_select(0, prepared.track_indices)
    return DeviceStarBatch(
        prepared=prepared, x=x, graph=graph,
        **indices,
        rao_labels=labels("rao_labels", track_targets, "track_indices"),
        mode_labels=labels("mode_labels", mode_targets, "observation_positions"),
        rao_base_rates=move("rao_base_rates", priors),
    )


def load_star_batch(
    prepared: PreparedStarBatch,
    X: torch.Tensor,
    track_targets: torch.Tensor,
    mode_targets: torch.Tensor,
    rao_base_rates_cpu: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> DeviceStarBatch:
    """Load one batch synchronously, for direct forwards and final exports."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _load_batch(prepared, X, track_targets, mode_targets, rao_base_rates_cpu,
                       device=device, dtype=dtype)


@contextmanager
def iter_star_batches(
    prepared_batches: Iterable[PreparedStarBatch],
    X: torch.Tensor,
    track_targets: torch.Tensor,
    mode_targets: torch.Tensor,
    rao_base_rates_cpu: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    prefetch: bool = True,
) -> Iterator[Iterator[DeviceStarBatch]]:
    """Yield batches with at most current plus one prefetched CUDA batch.

    Use ``with iter_star_batches(...) as batches:`` so early termination and
    exceptions synchronize pending copies before pinned staging is discarded.
    CPU iteration and ``prefetch=False`` use the direct loader without pinning.
    """
    device = torch.device(device)
    if device.type != "cuda" or not prefetch:
        iterator = (
            load_star_batch(batch, X, track_targets, mode_targets, rao_base_rates_cpu,
                            device=device, dtype=dtype)
            for batch in prepared_batches
        )
        try:
            yield iterator
        finally:
            iterator.close()
        return

    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        # The global target tensors may have just been initialized on the
        # caller's stream. Establish that dependency once before indexing them.
        stream.wait_stream(torch.cuda.current_stream(device))
        slots = (_PinnedSlot(), _PinnedSlot())
        source = iter(prepared_batches)

        def enqueue(batch, slot):
            slot.wait_before_reuse()
            with torch.cuda.stream(stream):
                loaded = _load_batch(
                    batch, X, track_targets, mode_targets, rao_base_rates_cpu,
                    device=device, dtype=dtype, slot=slot,
                )
                ready = torch.cuda.Event()
                ready.record(stream)
                slot.ready = ready
            return loaded, ready

        def generate():
            first = next(source, None)
            if first is None:
                return
            pending = enqueue(first, slots[0])
            next_slot = 1
            while pending is not None:
                current, ready = pending
                consumer = torch.cuda.current_stream(device)
                consumer.wait_event(ready)
                current.record_stream(consumer)
                upcoming = next(source, None)
                pending = None if upcoming is None else enqueue(upcoming, slots[next_slot])
                next_slot = 1 - next_slot
                yield current
                # Do not retain the preceding device batch when loading the next.
                del current, ready

        iterator = generate()
        try:
            yield iterator
        finally:
            iterator.close()
            # Includes an in-flight prefetched batch after break or exceptions.
            stream.synchronize()
