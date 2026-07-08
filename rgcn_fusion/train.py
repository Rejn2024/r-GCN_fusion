"""Train an r-GCN evidential mass and node classification model from Neo4j."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dempster_shafer import belief_plausibility, validate_masses
from .model import RGCNEvidenceModel
from .neo4j_loader import GraphData, Neo4jGraphLoader

IGNORE_CLASS_INDEX = -1
DEFAULT_CLASSIFICATION_TARGETS = {
    "radar_type": "radar_id",
    "radar_mode": "mode_id",
    "aircraft_variant": "aircraft_id",
    "operator": "operator",
}


def _tensor_graph(graph: GraphData, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(graph.node_features, dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    edge_type = torch.as_tensor(graph.edge_type, dtype=torch.long, device=device)
    return x, edge_index, edge_type


def _classification_label_properties(data_cfg: dict[str, Any]) -> dict[str, str]:
    """Return configured categorical prediction targets.

    ``classification_label_properties`` maps model task names to Neo4j node
    properties.  If the config only enables classification without supplying a
    mapping, the training pipeline uses the aircraft/radar ESM label fields,
    including radar type.
    """
    configured = data_cfg.get("classification_label_properties")
    if configured is None and data_cfg.get("classification", False):
        return dict(DEFAULT_CLASSIFICATION_TARGETS)
    if configured is None:
        return {}
    if not isinstance(configured, dict):
        raise TypeError("classification_label_properties must map task names to node property names")
    return {str(task_name): str(property_name) for task_name, property_name in configured.items()}


def _encode_classification_targets(
    graph: GraphData,
    class_values: dict[str, list[Any]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]]]:
    """Encode raw categorical labels as integer arrays, preserving missing labels as -1."""
    if not graph.classification_labels:
        return {}, {}

    encoded: dict[str, np.ndarray] = {}
    vocabularies: dict[str, list[Any]] = {}
    configured_values = class_values or {}
    for task_name, raw_values in graph.classification_labels.items():
        present_values = [value for value in raw_values.tolist() if value is not None]
        if task_name in configured_values:
            vocabulary = list(configured_values[task_name])
        else:
            vocabulary = sorted(set(present_values), key=str)
        if len(vocabulary) < 2:
            raise ValueError(f"classification task '{task_name}' requires at least two classes")
        value_to_idx = {value: idx for idx, value in enumerate(vocabulary)}
        labels = np.full(raw_values.shape[0], IGNORE_CLASS_INDEX, dtype=np.int64)
        for idx, value in enumerate(raw_values.tolist()):
            if value is None:
                continue
            if value not in value_to_idx:
                raise ValueError(f"label value {value!r} for task '{task_name}' is missing from class_values")
            labels[idx] = value_to_idx[value]
        encoded[task_name] = labels
        vocabularies[task_name] = vocabulary
    return encoded, vocabularies


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    """Load Neo4j data, train the r-GCN, and write masses/class predictions to disk."""
    neo4j_cfg = config["neo4j"]
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})
    classification_label_properties = _classification_label_properties(data_cfg)

    loader = Neo4jGraphLoader(
        neo4j_cfg["uri"], neo4j_cfg["user"], neo4j_cfg["password"], neo4j_cfg.get("database")
    )
    try:
        graph = loader.load(
            feature_properties=data_cfg["feature_properties"],
            label_property=data_cfg.get("label_property"),
            classification_label_properties=classification_label_properties,
            node_query=data_cfg.get("node_query"),
            edge_query=data_cfg.get("edge_query"),
        )
    finally:
        loader.close()

    if graph.labels is None or graph.labels.size == 0:
        raise ValueError("training requires a node label property containing target mass vectors")

    hypotheses = data_cfg["hypotheses"]
    targets_np = validate_masses(graph.labels)
    expected_masses = 2 ** len(hypotheses) - 1
    if targets_np.shape[1] != expected_masses:
        raise ValueError(f"labels must contain {expected_masses} masses per node")

    encoded_classes, class_vocabularies = _encode_classification_targets(
        graph, data_cfg.get("class_values")
    )

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    x, edge_index, edge_type = _tensor_graph(graph, device)
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    class_targets = {
        task_name: torch.as_tensor(labels, dtype=torch.long, device=device)
        for task_name, labels in encoded_classes.items()
    }

    model = RGCNEvidenceModel(
        in_features=x.shape[1],
        hidden_features=int(model_cfg.get("hidden_features", 64)),
        num_relations=max(len(graph.relation_names), 1),
        num_hypotheses=len(hypotheses),
        dropout=float(model_cfg.get("dropout", 0.1)),
        classification_tasks={task_name: len(values) for task_name, values in class_vocabularies.items()},
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    classification_loss_weight = float(train_cfg.get("classification_loss_weight", 1.0))
    epochs = int(train_cfg.get("epochs", 200))
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, edge_index, edge_type)
        masses = outputs["masses"]
        mass_loss = F.kl_div(torch.log(masses.clamp_min(1e-9)), targets, reduction="batchmean")
        class_loss = torch.zeros((), dtype=torch.float32, device=device)
        for task_name, labels in class_targets.items():
            if not torch.any(labels != IGNORE_CLASS_INDEX):
                continue
            logits = outputs["classification_logits"][task_name]
            class_loss = class_loss + F.cross_entropy(logits, labels, ignore_index=IGNORE_CLASS_INDEX)
        loss = mass_loss + classification_loss_weight * class_loss
        loss.backward()
        optimizer.step()
        history.append({
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "mass_loss": float(mass_loss.detach().cpu()),
            "classification_loss": float(class_loss.detach().cpu()),
        })

    model.eval()
    with torch.no_grad():
        outputs = model(x, edge_index, edge_type)
        predictions = outputs["masses"].cpu().numpy()
        classification_probabilities = {
            task_name: F.softmax(logits, dim=-1).cpu().numpy()
            for task_name, logits in outputs["classification_logits"].items()
        }

    output_dir = Path(output_cfg.get("directory", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "relations": graph.relation_names,
            "class_vocabularies": class_vocabularies,
        },
        output_dir / "rgcn_evidence_model.pt",
    )

    rows = []
    for node_index, (node_id, mass) in enumerate(zip(graph.node_ids, predictions)):
        intervals = [interval.__dict__ for interval in belief_plausibility(mass, hypotheses)]
        classifications = {}
        for task_name, probabilities in classification_probabilities.items():
            class_index = int(probabilities[node_index].argmax())
            classifications[task_name] = {
                "label": class_vocabularies[task_name][class_index],
                "probability": float(probabilities[node_index, class_index]),
                "probabilities": probabilities[node_index].tolist(),
            }
        rows.append({
            "node_id": node_id,
            "masses": mass.tolist(),
            "intervals": intervals,
            "classifications": classifications,
        })
    (output_dir / "node_evidence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "final_loss": history[-1]["loss"],
        "nodes": len(graph.node_ids),
        "classification_tasks": list(class_vocabularies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = train_model(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
