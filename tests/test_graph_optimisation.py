import numpy as np
import pytest
import torch

from rgcn_fusion.graph_optimisation import (
    build_track_graph_batches,
    densify_feature_rows,
    partition_edges_by_relation,
    segment_softmax,
)


def test_densify_feature_rows_writes_only_known_sparse_values():
    result = densify_feature_rows(
        [{"b": 2.0}, {"a": 3.0, "ignored": 9.0}], ["a", "b"], dtype=np.float16
    )
    np.testing.assert_array_equal(result, np.array([[0, 2], [3, 0]], dtype=np.float16))


def test_partition_edges_by_relation_preserves_relation_order():
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]])
    partitions = partition_edges_by_relation(edges, torch.tensor([1, 0, 1]), 2)
    assert torch.equal(partitions[0], edges[:, 1:2])
    assert torch.equal(partitions[1], edges[:, [0, 2]])


def test_segment_softmax_normalizes_each_track():
    scores = torch.tensor([0.0, 1.0, 2.0, 2.0])
    segments = torch.tensor([0, 0, 1, 1])
    weights = segment_softmax(scores, segments, 2)
    totals = torch.zeros(2).index_add_(0, segments, weights)
    torch.testing.assert_close(totals, torch.ones(2))
    assert weights[1] > weights[0]


def test_segment_softmax_handles_autocast_promoting_exp(monkeypatch):
    original_exp = torch.exp
    monkeypatch.setattr(torch, "exp", lambda values: original_exp(values).float())
    scores = torch.tensor([0.0, 1.0, 2.0, 2.0], dtype=torch.float16)
    segments = torch.tensor([0, 0, 1, 1])

    weights = segment_softmax(scores, segments, 2)

    assert weights.dtype == scores.dtype
    totals = torch.zeros(2, dtype=weights.dtype).index_add_(0, segments, weights)
    torch.testing.assert_close(totals, torch.ones(2, dtype=weights.dtype))


def test_track_batches_keep_tracks_whole_and_reindex_shared_nodes():
    # nodes 0 and 3 are shared; tracks 0 and 1 own nodes 1 and 2 respectively
    edges = torch.tensor([[0, 1, 3, 2], [1, 0, 2, 3]])
    batches = build_track_graph_batches(
        edge_index=edges,
        edge_types=torch.tensor([0, 1, 0, 1]),
        node_track_index=torch.tensor([-1, 0, 1, -1]),
        observation_nodes=torch.tensor([1, 2]),
        observation_track_index=torch.tensor([0, 1]),
        selected_tracks=[0, 1],
        tracks_per_batch=1,
    )
    assert len(batches) == 2
    assert batches[0].track_indices.tolist() == [0]
    assert batches[0].observation_positions.tolist() == [0]
    assert batches[0].node_indices.tolist() == [0, 1]
    assert int(batches[0].edge_index.max()) < batches[0].node_indices.numel()


def test_track_batches_reject_invalid_batch_size():
    with pytest.raises(ValueError, match="tracks_per_batch"):
        build_track_graph_batches(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_types=torch.empty(0, dtype=torch.long),
            node_track_index=torch.tensor([-1]),
            observation_nodes=torch.empty(0, dtype=torch.long),
            observation_track_index=torch.empty(0, dtype=torch.long),
            selected_tracks=[],
            tracks_per_batch=0,
        )
