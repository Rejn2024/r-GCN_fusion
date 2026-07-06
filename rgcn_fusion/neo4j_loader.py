"""Neo4j knowledge graph extraction for r-GCN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from neo4j import GraphDatabase


@dataclass
class GraphData:
    """In-memory graph tensors before conversion to torch."""

    node_ids: list[str]
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_type: np.ndarray
    relation_names: list[str]
    labels: np.ndarray | None = None


class Neo4jGraphLoader:
    """Load a property graph from Neo4j into arrays suitable for an r-GCN."""

    def __init__(self, uri: str, user: str, password: str, database: str | None = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def load(
        self,
        *,
        feature_properties: list[str],
        label_property: str | None = None,
        node_query: str | None = None,
        edge_query: str | None = None,
    ) -> GraphData:
        """Fetch nodes, directed relationships, optional features, and optional labels."""
        node_query = node_query or "MATCH (n) RETURN elementId(n) AS id, properties(n) AS props ORDER BY id"
        edge_query = edge_query or (
            "MATCH (s)-[r]->(t) RETURN elementId(s) AS source, elementId(t) AS target, "
            "type(r) AS type ORDER BY source, target, type"
        )
        with self.driver.session(database=self.database) as session:
            nodes = list(session.run(node_query))
            edges = list(session.run(edge_query))

        node_ids = [record["id"] for record in nodes]
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        features = np.asarray(
            [[float(record["props"].get(prop, 0.0)) for prop in feature_properties] for record in nodes],
            dtype=np.float32,
        )

        labels = None
        if label_property:
            raw = [record["props"].get(label_property) for record in nodes]
            labels = np.asarray([
                value if isinstance(value, list) else [] for value in raw
            ], dtype=np.float32)

        relation_names = sorted({record["type"] for record in edges})
        rel_to_idx = {name: idx for idx, name in enumerate(relation_names)}
        edge_pairs: list[tuple[int, int]] = []
        edge_types: list[int] = []
        for record in edges:
            if record["source"] not in id_to_idx or record["target"] not in id_to_idx:
                continue
            edge_pairs.append((id_to_idx[record["source"]], id_to_idx[record["target"]]))
            edge_types.append(rel_to_idx[record["type"]])

        edge_index = np.asarray(edge_pairs, dtype=np.int64).T if edge_pairs else np.zeros((2, 0), dtype=np.int64)
        return GraphData(
            node_ids=node_ids,
            node_features=features,
            edge_index=edge_index,
            edge_type=np.asarray(edge_types, dtype=np.int64),
            relation_names=relation_names,
            labels=labels,
        )
