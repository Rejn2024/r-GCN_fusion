import json
from pathlib import Path

import numpy as np

from rgcn_fusion.neo4j_loader import GraphData
from rgcn_fusion.train import (
    DEFAULT_CLASSIFICATION_TARGETS,
    DEFAULT_CLASSIFICATION_TASK_LOSS_WEIGHTS,
    RECOMMENDED_CANDIDATE_FEATURES,
    _classification_label_properties,
    _classification_task_loss_weights,
    _encode_classification_targets,
    _feature_properties,
)


def test_default_classification_targets_include_radar_type_and_operator_country():
    assert _classification_label_properties({"classification": True}) == DEFAULT_CLASSIFICATION_TARGETS
    assert DEFAULT_CLASSIFICATION_TARGETS["radar_type"] == "radar_id"
    assert DEFAULT_CLASSIFICATION_TARGETS["operator_country"] == "operator_country"


def test_default_classification_task_loss_weights_prioritize_requested_metrics():
    weights = _classification_task_loss_weights({})

    assert weights == DEFAULT_CLASSIFICATION_TASK_LOSS_WEIGHTS
    assert weights["aircraft_variant"] > 1.0
    assert weights["operator_country"] > 1.0


def test_classification_task_loss_weights_can_be_overridden():
    weights = _classification_task_loss_weights({
        "classification_task_loss_weights": {"aircraft_variant": 3, "radar_type": 0.5}
    })

    assert weights == {"aircraft_variant": 3.0, "radar_type": 0.5}


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


def test_feature_properties_can_append_recommended_candidate_features():
    features = _feature_properties({
        "recommended_candidate_features": True,
        "feature_properties": ["degree_score", "custom_score"],
    })

    assert features[:2] == ["degree_score", "custom_score"]
    assert "radar_interval_overlap_score" in features
    assert "candidate_ambiguity_count" in features
    assert len(features) == len(set(features))
    assert set(RECOMMENDED_CANDIDATE_FEATURES).issubset(features)


def test_observation_etl_notebook_node_query_keeps_node_variable_for_projection():
    notebook = json.loads(Path("notebooks/observation_etl_rgcn_end_to_end.ipynb").read_text())
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "WITH n, n {.*, aircraft_type: coalesce(family.id, n.aircraft_id), operator_country: n.operator} AS props" in source
    assert "WITH n {.*, aircraft_type: coalesce(family.id, n.aircraft_id)} AS props" not in source
