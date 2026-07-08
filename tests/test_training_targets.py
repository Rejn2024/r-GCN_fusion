import numpy as np

from rgcn_fusion.neo4j_loader import GraphData
from rgcn_fusion.train import DEFAULT_CLASSIFICATION_TARGETS, _classification_label_properties, _encode_classification_targets


def test_default_classification_targets_include_radar_type():
    assert _classification_label_properties({"classification": True}) == DEFAULT_CLASSIFICATION_TARGETS
    assert DEFAULT_CLASSIFICATION_TARGETS["radar_type"] == "radar_id"


def test_encode_classification_targets_preserves_missing_values():
    graph = GraphData(
        node_ids=["a", "b", "c"],
        node_features=np.zeros((3, 1), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_type=np.zeros((0,), dtype=np.int64),
        relation_names=[],
        classification_labels={"radar_type": np.asarray(["r1", None, "r2"], dtype=object)},
    )

    encoded, vocabularies = _encode_classification_targets(graph)

    assert vocabularies == {"radar_type": ["r1", "r2"]}
    assert encoded["radar_type"].tolist() == [0, -1, 1]
