"""Train an r-GCN evidential mass model from a Neo4j knowledge graph."""

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


def _tensor_graph(graph: GraphData, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(graph.node_features, dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(graph.edge_index, dtype=torch.long, device=device)
    edge_type = torch.as_tensor(graph.edge_type, dtype=torch.long, device=device)
    return x, edge_index, edge_type


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    """Load Neo4j data, train the r-GCN, and write masses/intervals to disk."""
    neo4j_cfg = config["neo4j"]
    data_cfg = config["data"]
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    loader = Neo4jGraphLoader(
        neo4j_cfg["uri"], neo4j_cfg["user"], neo4j_cfg["password"], neo4j_cfg.get("database")
    )
    try:
        graph = loader.load(
            feature_properties=data_cfg["feature_properties"],
            label_property=data_cfg.get("label_property"),
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

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    x, edge_index, edge_type = _tensor_graph(graph, device)
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)

    model = RGCNEvidenceModel(
        in_features=x.shape[1],
        hidden_features=int(model_cfg.get("hidden_features", 64)),
        num_relations=max(len(graph.relation_names), 1),
        num_hypotheses=len(hypotheses),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 200))
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        masses = model(x, edge_index, edge_type)
        loss = F.kl_div(torch.log(masses.clamp_min(1e-9)), targets, reduction="batchmean")
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})

    model.eval()
    with torch.no_grad():
        predictions = model(x, edge_index, edge_type).cpu().numpy()

    output_dir = Path(output_cfg.get("directory", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": config, "relations": graph.relation_names},
        output_dir / "rgcn_evidence_model.pt",
    )

    rows = []
    for node_id, mass in zip(graph.node_ids, predictions):
        intervals = [interval.__dict__ for interval in belief_plausibility(mass, hypotheses)]
        rows.append({"node_id": node_id, "masses": mass.tolist(), "intervals": intervals})
    (output_dir / "node_evidence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"output_dir": str(output_dir), "final_loss": history[-1]["loss"], "nodes": len(graph.node_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = train_model(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
