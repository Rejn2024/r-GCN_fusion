"""Reload the existing shared-candidate graph bundle without feature copies."""
from pathlib import Path

import numpy as np
import torch


def load_shared_graph(directory, *, input_fingerprints, graph_settings):
    """Read trusted local artifacts saved by the source notebook's cache cell.

    Feature paths are relative to the bundle, so all four files can move together.
    Copy-on-write mappings protect the cached features and topology on disk.
    Metadata remains normal Python memory; no graph/scoring fallback is performed.
    """
    directory = Path(directory)
    metadata = torch.load(directory / "graph_construction_outputs.pt",
                          map_location="cpu", weights_only=False)
    if metadata.get("version") != 1:
        raise ValueError("Unsupported shared-candidate graph version")
    context = metadata.pop("cache_context", None)
    if context is None:
        raise ValueError("Cache provenance missing; run the source notebook's save cell once")
    if context.get("input_fingerprints") != input_fingerprints:
        raise ValueError("Cached graph belongs to different dataset/KG inputs")
    if context.get("graph_settings") != graph_settings:
        raise ValueError("Cached graph was built with different graph settings")
    for key, descriptor_key in (("X", "feature_matrix"), ("edge_features", "edge_feature_matrix")):
        descriptor = metadata.pop(descriptor_key)
        path = (directory / descriptor["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Feature path must stay inside the cache bundle")
        shape = tuple(descriptor["shape"])
        dtype = np.dtype(descriptor["dtype"])
        if len(shape) != 2 or any(not isinstance(n, int) or n < 1 for n in shape) or dtype != np.float32:
            raise ValueError(f"Invalid feature descriptor: {descriptor_key}")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Incorrect feature file size: {path}")
        metadata[key] = torch.from_numpy(np.memmap(path, mode="c", dtype=dtype, shape=shape))
    topology = torch.load(directory / "relational_graph_outputs.pt", map_location="cpu",
                          weights_only=True, mmap=True)
    edges = topology["edge_index"]
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype not in (torch.int32, torch.int64):
        raise ValueError("Invalid cached edge_index")
    for key in ("edge_types", "edge_feature_indices"):
        if topology[key].shape != (edges.shape[1],) or topology[key].dtype not in (torch.int32, torch.int64):
            raise ValueError(f"Invalid cached {key}")
    if tuple(metadata["X"].shape) != (len(metadata["node_meta"]), len(metadata["feature_names"])):
        raise ValueError("Cached node features and metadata disagree")
    if metadata["edge_features"].shape[1] != len(metadata["edge_feature_names"]):
        raise ValueError("Cached incidence features and schema disagree")
    # Validate indices in bounded chunks without a full-sized int64 temporary.
    nodes, features = metadata["X"].shape[0], metadata["edge_features"].shape[0]
    relations = len(metadata["relation_metadata"])
    for start in range(0, edges.shape[1], 1_000_000):
        stop = start + 1_000_000
        e = edges[:, start:stop]
        types = topology["edge_types"][start:stop]
        refs = topology["edge_feature_indices"][start:stop]
        if ((e < 0) | (e >= nodes)).any() or ((types < 0) | (types >= relations)).any():
            raise ValueError("Cached topology contains out-of-range nodes or relations")
        if ((refs < -1) | (refs >= features)).any():
            raise ValueError("Cached topology contains out-of-range incidence references")
    metadata.update({key: topology[key] for key in ("edge_index", "edge_types", "edge_feature_indices")})
    return metadata
