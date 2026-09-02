"""Memory-efficient graph construction and track-batching utilities."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import torch


def densify_feature_rows(
    rows: Sequence[Mapping[str, float]],
    feature_names: Sequence[str],
    *,
    dtype: np.dtype = np.dtype(np.float32),
    backing_file: str | PathLike[str] | None = None,
) -> np.ndarray:
    """Materialise sparse feature dictionaries without a list-of-lists.

    When ``backing_file`` is supplied, the result is a writable ``numpy.memmap``.
    This lets callers work with matrices larger than available RAM while retaining
    ordinary NumPy indexing and zero-copy ``torch.from_numpy`` compatibility.
    """
    columns = {name: index for index, name in enumerate(feature_names)}
    shape = (len(rows), len(feature_names))
    if backing_file is None:
        dense = np.zeros(shape, dtype=dtype)
    else:
        path = Path(backing_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        dense = np.memmap(path, mode="w+", shape=shape, dtype=dtype)
    for row_index, row in enumerate(rows):
        for name, value in row.items():
            column_index = columns.get(name)
            if column_index is not None:
                dense[row_index, column_index] = value
    return dense


def standardize_feature_matrix_in_place(
    dense: np.ndarray,
    *,
    chunk_rows: int,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Standardise a dense feature matrix using bounded-memory chunks.

    Population moments are merged with the parallel-variance formula, avoiding
    both a full float32 copy and catastrophic cancellation from ``E[x²]-E[x]²``.
    The normalized values are written back in the input dtype. This is especially
    useful for a disk-backed matrix returned by :func:`densify_feature_rows`.
    """
    if dense.ndim != 2:
        raise ValueError("dense must have shape [row_count, feature_count]")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    row_count, feature_count = dense.shape
    if row_count == 0:
        return dense

    work_device = torch.device(device)
    mean = torch.zeros(feature_count, dtype=torch.float32, device=work_device)
    m2 = torch.zeros_like(mean)
    count = 0
    with torch.inference_mode():
        for start in range(0, row_count, chunk_rows):
            values = torch.as_tensor(dense[start : start + chunk_rows]).to(
                device=work_device, dtype=torch.float32
            )
            batch_count = values.size(0)
            batch_mean = values.mean(dim=0)
            batch_m2 = ((values - batch_mean) ** 2).sum(dim=0)
            delta = batch_mean - mean
            combined_count = count + batch_count
            mean.add_(delta * (batch_count / combined_count))
            m2.add_(batch_m2 + delta.square() * (count * batch_count / combined_count))
            count = combined_count

        scale = torch.sqrt(m2 / count)
        scale.masked_fill_(scale == 0, 1.0)
        for start in range(0, row_count, chunk_rows):
            stop = min(start + chunk_rows, row_count)
            values = torch.as_tensor(dense[start:stop]).to(
                device=work_device, dtype=torch.float32
            )
            normalized = ((values - mean) / scale).to("cpu")
            dense[start:stop] = normalized.numpy()

    if isinstance(dense, np.memmap):
        dense.flush()
    return dense


def partition_edges_by_relation(
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    num_relations: int,
) -> tuple[torch.Tensor, ...]:
    """Group edges once so message-passing layers do not rebuild masks each pass."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if edge_types.ndim != 1 or edge_types.numel() != edge_index.size(1):
        raise ValueError("edge_types must contain one relation id per edge")
    if num_relations < 1:
        raise ValueError("num_relations must be positive")
    if edge_types.numel() and (
        int(edge_types.min()) < 0 or int(edge_types.max()) >= num_relations
    ):
        raise ValueError("edge_types contains an out-of-range relation id")
    return tuple(edge_index[:, edge_types == relation] for relation in range(num_relations))


def segment_softmax(
    scores: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Vectorised softmax over arbitrary segment ids using scatter reductions."""
    if scores.ndim != 1 or segment_ids.shape != scores.shape:
        raise ValueError("scores and segment_ids must be one-dimensional and aligned")
    maxima = scores.new_full((num_segments,), -torch.inf)
    maxima.scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
    # Autocast may promote ``exp`` to float32 while leaving half-precision
    # scores unchanged.  Build the reduction buffer from the operation result
    # so index_add_ always receives a source and destination of the same dtype.
    exponentials = torch.exp(scores - maxima[segment_ids])
    denominators = torch.zeros(
        num_segments, dtype=exponentials.dtype, device=exponentials.device
    )
    denominators.index_add_(0, segment_ids, exponentials)
    weights = exponentials / denominators[segment_ids].clamp_min(
        torch.finfo(exponentials.dtype).tiny
    )
    return weights.to(scores.dtype)


@dataclass(frozen=True)
class TrackGraphBatch:
    """An induced, locally indexed graph for a group of complete tracks."""

    track_indices: torch.Tensor
    node_indices: torch.Tensor
    edge_index: torch.Tensor
    edge_types: torch.Tensor
    observation_nodes: torch.Tensor
    observation_to_track: torch.Tensor
    observation_positions: torch.Tensor


def build_track_graph_batches(
    *,
    edge_index: torch.Tensor,
    edge_types: torch.Tensor,
    node_track_index: torch.Tensor,
    observation_nodes: torch.Tensor,
    observation_track_index: torch.Tensor,
    selected_tracks: Iterable[int],
    tracks_per_batch: int,
) -> list[TrackGraphBatch]:
    """Build deterministic induced mini-batches while keeping every track intact.

    Edges are owned by the track of either endpoint. Shared nodes (for example KG
    entities) are included only when connected to a selected track; unrelated
    shared-node self loops are intentionally omitted from mini-batches.
    """
    if tracks_per_batch < 1:
        raise ValueError("tracks_per_batch must be positive")
    edge_index = edge_index.cpu()
    edge_types = edge_types.cpu()
    node_track_index = node_track_index.cpu()
    observation_nodes = observation_nodes.cpu()
    observation_track_index = observation_track_index.cpu()
    tracks = sorted({int(track) for track in selected_tracks})
    batches: list[TrackGraphBatch] = []
    src_tracks = node_track_index[edge_index[0]]
    dst_tracks = node_track_index[edge_index[1]]
    edge_owners = torch.where(src_tracks >= 0, src_tracks, dst_tracks)

    for start in range(0, len(tracks), tracks_per_batch):
        batch_tracks = torch.tensor(tracks[start : start + tracks_per_batch], dtype=torch.long)
        owned_edges = torch.isin(edge_owners, batch_tracks)
        batch_edge_global = edge_index[:, owned_edges]
        batch_edge_types = edge_types[owned_edges]
        batch_observation_positions = torch.nonzero(
            torch.isin(observation_track_index, batch_tracks), as_tuple=False
        ).flatten()
        batch_observation_global = observation_nodes[batch_observation_positions]
        batch_nodes = torch.unique(
            torch.cat((batch_edge_global.flatten(), batch_observation_global)), sorted=True
        )
        global_to_local = torch.full((node_track_index.numel(),), -1, dtype=torch.long)
        global_to_local[batch_nodes] = torch.arange(batch_nodes.numel())
        local_edges = global_to_local[batch_edge_global]
        local_observations = global_to_local[batch_observation_global]
        local_track_lookup = torch.full(
            (int(batch_tracks.max()) + 1,), -1, dtype=torch.long
        )
        local_track_lookup[batch_tracks] = torch.arange(batch_tracks.numel())
        local_observation_tracks = local_track_lookup[
            observation_track_index[batch_observation_positions]
        ]
        batches.append(
            TrackGraphBatch(
                track_indices=batch_tracks,
                node_indices=batch_nodes,
                edge_index=local_edges,
                edge_types=batch_edge_types,
                observation_nodes=local_observations,
                observation_to_track=local_observation_tracks,
                observation_positions=batch_observation_positions,
            )
        )
    return batches
