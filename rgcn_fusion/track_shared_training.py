"""Bounded synchronous or double-buffered shared-candidate batch staging."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .track_shared_graph import SharedEvidenceGraph


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

    def record_stream(self, stream):
        for name in ("x", "observation_nodes", "observation_to_track", "track_indices",
                     "observation_positions", "rao_labels", "mode_labels", "rao_base_rates",
                     "observation_counts"):
            getattr(self, name).record_stream(stream)
        graph = self.graph
        for tensor in (*graph.tensors(), *graph.type_indices,
                       *(t for t in graph.attributed_positions if t is not None),
                       *(t for cache in graph.source_caches if cache is not None for t in cache),
                       *(t for group in (graph.feature_groups or ()) for t in group)):
            tensor.record_stream(stream)


def prepare_shared_batches(batches, num_relations):
    """Topology and incidence feature tables are already prepared during batching."""
    batches = list(batches)
    for batch in batches:
        if len(batch.graph.relation_sizes) != num_relations:
            raise ValueError("Prepared batch has the wrong relation schema")
    return batches


class _PinnedSlot:
    def __init__(self):
        self.buffers = {}
        self.ready = None

    def wait_before_reuse(self):
        if self.ready is not None:
            self.ready.synchronize()

    def stage(self, name, value):
        if value.device.type != "cpu":
            raise ValueError("Pinned staging expects CPU tensors")
        buffer = self.buffers.get(name)
        if buffer is None or buffer.dtype != value.dtype or buffer.numel() < value.numel():
            buffer = torch.empty(value.numel(), dtype=value.dtype, pin_memory=True)
            self.buffers[name] = buffer
        view = buffer[:value.numel()].view(value.shape)
        view.copy_(value)
        return view


def _load_batch(prepared, X, track_targets, mode_targets, rao_base_rates_cpu, *, device, dtype, slot=None):
    def move(name, value):
        if slot is not None:
            value = slot.stage(name, value)
        return value.to(device=device, non_blocking=slot is not None)

    indices = {name: move(name, getattr(prepared, name)) for name in (
        "track_indices", "observation_positions", "observation_nodes", "observation_to_track", "observation_counts")}
    original = prepared.graph
    attributes = (original.edge_features if original.feature_rows is None else
                  original.edge_features.index_select(0, original.feature_rows))
    graph = SharedEvidenceGraph(
        original.num_nodes, original.num_tracks,
        move("edge_index", original.edge_index), original.relation_sizes,
        move("feature_indices", original.feature_indices), move("edge_features", attributes),
        move("node_types", original.node_types), move("kg_indices", original.kg_indices),
        move("track_nodes", original.track_nodes), None,
        tuple(move(f"type_{i}", t) for i, t in enumerate(original.type_indices)),
        tuple(None if t is None else move(f"attributed_{i}", t)
              for i, t in enumerate(original.attributed_positions)), original.relation_feature_modes,
        tuple(None if cache is None else tuple(move(f"source_{i}_{j}", t) for j, t in enumerate(cache))
              for i, cache in enumerate(original.source_caches)),
        None if original.feature_groups is None else tuple(
            tuple(move(f"feature_group_{i}_{j}", t) for j, t in enumerate(group))
            for i, group in enumerate(original.feature_groups)),
    )

    def labels(name, targets, index_name):
        if targets.device == device:
            return targets.index_select(0, indices[index_name])
        selected = targets.index_select(0, getattr(prepared, index_name).to(targets.device))
        return selected.to(device) if selected.device.type != "cpu" else move(name, selected)

    return DeviceSharedBatch(
        prepared=prepared,
        x=move("x", X.index_select(0, prepared.node_indices).to(dtype=dtype)), graph=graph,
        **indices,
        rao_labels=labels("rao_labels", track_targets, "track_indices"),
        mode_labels=labels("mode_labels", mode_targets, "observation_positions"),
        rao_base_rates=move("rao_base_rates", rao_base_rates_cpu.index_select(0, prepared.track_indices)),
    )


def load_shared_batch(prepared, X, track_targets, mode_targets, rao_base_rates_cpu, *, device, dtype):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _load_batch(prepared, X, track_targets, mode_targets, rao_base_rates_cpu, device=device, dtype=dtype)


@contextmanager
def iter_shared_batches(prepared_batches, X, track_targets, mode_targets, rao_base_rates_cpu, *,
                        device, dtype, prefetch=False):
    """Keep at most the current device batch and one lookahead input batch.

    Two reusable pinned host slots wait for DMA completion before reuse. Tensor
    lifetimes are registered on the consumer stream, including every graph cache.
    Context exit drains pending copies after early stop or exceptions.
    """
    device = torch.device(device)
    if device.type != "cuda" or not prefetch:
        iterator = (load_shared_batch(batch, X, track_targets, mode_targets, rao_base_rates_cpu,
                                     device=device, dtype=dtype) for batch in prepared_batches)
        try:
            yield iterator
        finally:
            iterator.close()
        return
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        slots = (_PinnedSlot(), _PinnedSlot())
        source = iter(prepared_batches)

        def enqueue(batch, slot):
            slot.wait_before_reuse()
            with torch.cuda.stream(stream):
                loaded = _load_batch(batch, X, track_targets, mode_targets, rao_base_rates_cpu,
                                     device=device, dtype=dtype, slot=slot)
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
                del current, ready

        iterator = generate()
        try:
            yield iterator
        finally:
            iterator.close()
            stream.synchronize()
