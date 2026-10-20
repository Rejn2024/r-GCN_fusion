"""Verify compact batch ownership and bounded CPU/CUDA staging."""
import gc
import weakref
from collections import defaultdict
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from rgcn_fusion.graph_optimisation import TrackGraphBatch
from rgcn_fusion.track_star_hgt import TRACK_TO_OBSERVATION, prepare_track_star_graph
from rgcn_fusion.track_star_training import (
    _PinnedSlot,
    iter_star_batches,
    load_star_batch,
    prepare_star_batches,
)


def sample_batch(offset=0):
    # Distinguish node IDs, observation positions and global track IDs, and
    # deliberately interleave observations of tracks with unequal lengths.
    return TrackGraphBatch(
        track_indices=torch.tensor([2 + offset, 5 + offset]),
        node_indices=(torch.tensor([8, 1, 7, 4, 6], dtype=torch.int32) + offset) % 10,
        edge_index=torch.tensor([[3, 0, 1, 2, 0], [0, 1, 2, 3, 1]], dtype=torch.int32),
        edge_types=torch.tensor([1, 0, 1, 2, 0], dtype=torch.int32),
        observation_nodes=torch.tensor([2, 0, 3]),
        observation_to_track=torch.tensor([1, 0, 1]),
        observation_positions=torch.tensor([6, 3, 0]),
    )


def sources():
    return (
        torch.arange(40, dtype=torch.float16).reshape(10, 4),
        torch.arange(20) % 3,
        torch.arange(10) % 2,
        torch.tensor([[.2, .3, .5]] * 20),
    )


def assert_loaded(loaded, prepared, source, device):
    X, track_targets, mode_targets, priors = source
    assert loaded.prepared is prepared
    assert loaded.num_tracks == 2
    assert loaded.num_observations == 3
    assert loaded.x.dtype == torch.float32
    assert loaded.x.device == torch.device(device)
    torch.testing.assert_close(loaded.x.cpu(), X[prepared.node_indices].float())
    for name in ("track_indices", "observation_nodes", "observation_to_track",
                 "observation_positions", "observation_counts"):
        torch.testing.assert_close(getattr(loaded, name).cpu(), getattr(prepared, name))
    torch.testing.assert_close(loaded.rao_labels.cpu(), track_targets.cpu()[prepared.track_indices])
    torch.testing.assert_close(loaded.mode_labels.cpu(), mode_targets.cpu()[prepared.observation_positions])
    torch.testing.assert_close(loaded.rao_base_rates.cpu(), priors[prepared.track_indices])
    for kind, endpoints in prepared.graph.edge_index_dict.items():
        torch.testing.assert_close(loaded.graph.edge_index_dict[kind].cpu(), endpoints)


def test_preparation_drops_original_unsorted_topology():
    batch = sample_batch()
    expected = prepare_track_star_graph(batch, 4)
    original_edges = weakref.ref(batch.edge_index)
    original_types = weakref.ref(batch.edge_types)
    original_batch = weakref.ref(batch)
    prepared, = prepare_star_batches([batch], 4)
    torch.testing.assert_close(prepared.graph.evidence_edges, expected.evidence_edges)
    torch.testing.assert_close(prepared.graph.incidence_edges, expected.incidence_edges)
    torch.testing.assert_close(prepared.observation_counts, torch.tensor([1, 2]))
    assert not hasattr(prepared, "edge_index")
    assert not hasattr(prepared, "edge_types")
    del batch
    gc.collect()
    assert original_batch() is None
    assert original_edges() is None
    assert original_types() is None


def test_cpu_direct_and_iterator_keep_lookup_order_without_pinning(tmp_path):
    source = list(sources())
    path = tmp_path / "features.memmap"
    backing = np.memmap(path, mode="w+", dtype=np.float16, shape=(10, 4))
    backing[:] = source[0].numpy()
    backing.flush()
    source[0] = torch.from_numpy(np.memmap(path, mode="c", dtype=np.float16, shape=(10, 4)))
    prepared = prepare_star_batches([sample_batch(), sample_batch(1)], 4)
    before = source[0].clone()
    with patch.object(torch.Tensor, "pin_memory", side_effect=AssertionError("CPU pinning")):
        direct = load_star_batch(prepared[0], *source, device="cpu", dtype=torch.float32)
        assert_loaded(direct, prepared[0], source, "cpu")
        with iter_star_batches(prepared, *source, device="cpu", dtype=torch.float32) as batches:
            loaded = list(batches)
        assert len(loaded) == 2
        for active, expected in zip(loaded, prepared):
            assert_loaded(active, expected, source, "cpu")
    torch.testing.assert_close(source[0], before)
    np.testing.assert_array_equal(np.memmap(path, mode="r", dtype=np.float16,
                                         shape=(10, 4)), before.numpy())


def test_cpu_iterator_closes_on_break_and_empty_input():
    prepared = prepare_star_batches([sample_batch(), sample_batch(1)], 4)
    with iter_star_batches(prepared, *sources(), device="cpu", dtype=torch.float32) as batches:
        next(batches)
    with pytest.raises(StopIteration):
        next(batches)
    with iter_star_batches([], *sources(), device="cpu", dtype=torch.float32) as batches:
        assert list(batches) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("targets_on_device", [False, True])
def test_cuda_prefetch_matches_direct_and_reuses_two_pinned_slots(targets_on_device):
    source = list(sources())
    if targets_on_device:
        source[1:3] = [value.cuda() for value in source[1:3]]
    prepared = prepare_star_batches([sample_batch(index % 3) for index in range(7)], 4)
    slots = []
    original_init = _PinnedSlot.__init__

    def capture(slot):
        original_init(slot)
        slots.append(slot)

    with patch.object(_PinnedSlot, "__init__", capture):
        with iter_star_batches(prepared, *source, device="cuda", dtype=torch.float32) as batches:
            for index, active in enumerate(batches):
                assert_loaded(active, prepared[index], source, active.x.device)
                reference = load_star_batch(prepared[index], *source, device="cuda", dtype=torch.float32)
                # Include a consumer computation before dropping the yielded
                # input, exercising copy-stream allocation ownership.
                torch.testing.assert_close(active.x.square(), reference.x.square())
    assert len(slots) == 2
    assert all(slot.buffers for slot in slots)
    assert all(value.is_pinned() for slot in slots for value in slot.buffers.values())
    assert all(slot.ready.query() for slot in slots)
    # Both slots serve every same-sized batch without growing extra buffers.
    assert all(slot.buffers["x"].numel() == 20 for slot in slots)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("nondefault_consumer", [False, True])
def test_cuda_prefetch_async_consumers_keep_values_and_reuse_host_storage(nondefault_consumer):
    source = list(sources())
    originals = []
    for index in range(11):
        batch = sample_batch(index % 10)
        # Vary features, topology and incidence in successive same-sized batches
        # so premature pinned-slot reuse cannot pass with identical contents.
        originals.append(replace(
            batch, edge_index=(batch.edge_index + index) % 5,
            observation_nodes=(batch.observation_nodes + index) % 5,
        ))
    prepared = prepare_star_batches(originals, 4)

    def summarize(batch):
        edges = batch.graph.evidence_edges
        incidence = batch.graph.incidence_edges
        reverse = batch.graph.edge_index_dict[TRACK_TO_OBSERVATION]
        return torch.stack((
            batch.x.square().sum(),
            (batch.x[edges[0]] * (edges[1, :, None].float() + 1)).sum(),
            (incidence[0].float() * 7 + incidence[1]).sum(),
            (reverse[0].float() * 3 + reverse[1]).sum(),
            batch.rao_labels.float().sum(),
        ))

    expected = torch.stack([
        summarize(load_star_batch(batch, *source, device="cpu", dtype=torch.float32))
        for batch in prepared
    ])
    # Initialize global targets on the default stream. A non-default consumer
    # must establish their dependency before the prefetch stream indexes them.
    source[1:3] = [value.cuda() for value in source[1:3]]
    consumer = (torch.cuda.Stream() if nondefault_consumer
                else torch.cuda.current_stream())
    consumer.wait_stream(torch.cuda.current_stream())
    pointers = defaultdict(list)
    slots = []
    original_init = _PinnedSlot.__init__
    original_stage = _PinnedSlot.stage

    def capture_init(slot):
        original_init(slot)
        slots.append(slot)

    def capture_stage(slot, name, value):
        staged = original_stage(slot, name, value)
        pointers[id(slot), name].append(staged.data_ptr())
        return staged

    results = []
    with patch.object(_PinnedSlot, "__init__", capture_init), \
            patch.object(_PinnedSlot, "stage", capture_stage):
        with torch.cuda.stream(consumer):
            with iter_star_batches(prepared, *source, device="cuda", dtype=torch.float32) as batches:
                for active in batches:
                    # Queue consumer work without .cpu(), .item(), assertions or
                    # blocking reference loads between batches. Only derived
                    # small outputs survive; input allocations can be recycled.
                    if hasattr(torch.cuda, "_sleep"):
                        torch.cuda._sleep(100_000)
                    results.append(summarize(active))
                    del active
            actual = torch.stack(results).cpu()
    torch.testing.assert_close(actual, expected)
    assert len(slots) == 2
    assert all(len(values) >= 5 for values in pointers.values())
    assert all(len(set(values)) == 1 for values in pointers.values())
    assert all(slot.ready.query() for slot in slots)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_prefetch_early_break_and_exception_finish_pending_copy():
    prepared = prepare_star_batches([sample_batch(), sample_batch(1)], 4)
    slots = []
    original_init = _PinnedSlot.__init__

    def capture(slot):
        original_init(slot)
        slots.append(slot)

    with patch.object(_PinnedSlot, "__init__", capture):
        for fail in (False, True):
            try:
                with iter_star_batches(prepared, *sources(), device="cuda", dtype=torch.float32) as batches:
                    active = next(batches)
                    value = active.x.square().sum()
                    if fail:
                        raise RuntimeError("consumer failed")
            except RuntimeError as error:
                assert str(error) == "consumer failed"
            assert torch.isfinite(value).item()
    assert len(slots) == 4
    assert all(slot.ready.query() for slot in slots)
    assert active.graph.edge_index_dict[TRACK_TO_OBSERVATION].device.type == "cuda"
