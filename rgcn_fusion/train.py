"""Train an r-GCN evidential mass and node classification model from Neo4j."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional plotting dependency guard
    plt = None
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency guard
    SummaryWriter = None
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency guard
    def tqdm(iterable, desc=None):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        label = desc or "Progress"
        print(f"{label}: starting" + (f" ({total} steps)" if total else ""))
        for item in iterable:
            yield item
        print(f"{label}: done")

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
RECOMMENDED_CANDIDATE_FEATURES = [
    "degree_score",
    "text_score",
    "recency_score",
    "radar_interval_overlap_score",
    "waveform_match_score",
    "scan_type_match_score",
    "center_frequency_residual",
    "prf_residual",
    "bandwidth_residual",
    "pulse_width_residual",
    "speed_consistency_score",
    "altitude_consistency_score",
    "heading_consistency_score",
    "observation_uncertainty_width",
    "candidate_ambiguity_count",
    "missing_feature_count",
]


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


def _feature_properties(data_cfg: dict[str, Any]) -> list[str]:
    """Return configured numeric node features, optionally using the richer ESM candidate set."""
    configured = data_cfg.get("feature_properties")
    if configured is None:
        configured = RECOMMENDED_CANDIDATE_FEATURES if data_cfg.get("recommended_candidate_features", False) else []
    features = [str(feature) for feature in configured]
    if data_cfg.get("recommended_candidate_features", False):
        features = list(dict.fromkeys([*features, *RECOMMENDED_CANDIDATE_FEATURES]))
    if not features:
        raise ValueError("data.feature_properties must contain at least one numeric feature")
    return features


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
            feature_properties=_feature_properties(data_cfg),
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
        num_layers=int(model_cfg.get("num_layers", 2)),
        num_bases=model_cfg.get("num_bases"),
        residual=bool(model_cfg.get("residual", True)),
        normalization=model_cfg.get("normalization", "layernorm"),
        relation_gates=bool(model_cfg.get("relation_gates", False)),
        task_head_hidden_features=model_cfg.get("task_head_hidden_features"),
        mass_head_type=str(model_cfg.get("mass_head_type", "softmax")),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    classification_loss_weight = float(train_cfg.get("classification_loss_weight", 1.0))
    epochs = int(train_cfg.get("epochs", 200))
    seed = int(train_cfg.get("seed", 42))
    train_fraction = float(train_cfg.get("train_fraction", 0.5))
    test_fraction = float(train_cfg.get("test_fraction", 0.3))
    val_fraction = float(train_cfg.get("val_fraction", 0.2))
    if not math.isclose(train_fraction + test_fraction + val_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_fraction, test_fraction, and val_fraction must sum to 1.0")

    output_dir = Path(output_cfg.get("directory", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = output_dir / "tensorboard"
    ckpt_path = output_dir / "best_rgcn_evidence_model.pt"

    perm = np.random.default_rng(seed).permutation(len(graph.node_ids))
    n_train = int(round(train_fraction * len(perm)))
    n_test = int(round(test_fraction * len(perm)))
    split_indices = {
        "train": torch.as_tensor(perm[:n_train], dtype=torch.long, device=device),
        "test": torch.as_tensor(perm[n_train:n_train + n_test], dtype=torch.long, device=device),
        "val": torch.as_tensor(perm[n_train + n_test:], dtype=torch.long, device=device),
    }
    print({name: int(indices.numel()) for name, indices in split_indices.items()})

    def _classification_loss(outputs: dict[str, Any], indices: torch.Tensor) -> torch.Tensor:
        class_loss = torch.zeros((), dtype=torch.float32, device=device)
        for task_name, labels in class_targets.items():
            split_labels = labels[indices]
            if not torch.any(split_labels != IGNORE_CLASS_INDEX):
                continue
            logits = outputs["classification_logits"][task_name][indices]
            class_loss = class_loss + F.cross_entropy(logits, split_labels, ignore_index=IGNORE_CLASS_INDEX)
        return class_loss

    def _classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
        valid = labels != IGNORE_CLASS_INDEX
        if not torch.any(valid):
            return float("nan")
        predictions = torch.argmax(logits[valid], dim=1)
        return float((predictions == labels[valid]).float().mean().cpu())

    def split_losses_and_metrics(split: str) -> dict[str, float]:
        model.eval()
        indices = split_indices[split]
        with torch.no_grad():
            outputs = model(x, edge_index, edge_type)
            masses = outputs["masses"]
            mass_loss = F.kl_div(
                torch.log(masses[indices].clamp_min(1e-9)),
                targets[indices],
                reduction="batchmean",
            )
            class_loss = _classification_loss(outputs, indices)
            metrics = {
                "loss": float((mass_loss + classification_loss_weight * class_loss).cpu()),
                "mass_loss": float(mass_loss.cpu()),
                "classification_loss": float(class_loss.cpu()),
            }
            for task_name, labels in class_targets.items():
                metrics[f"{task_name}_acc"] = _classification_accuracy(
                    outputs["classification_logits"][task_name][indices],
                    labels[indices],
                )
            return metrics

    if SummaryWriter is None:
        raise ImportError("TensorBoard tracking requires the tensorboard package. Install dependencies with `pip install -r requirements.txt`.")
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    writer.add_text("run/config", json.dumps({
        "epochs": epochs,
        "learning_rate": float(train_cfg.get("learning_rate", 1e-3)),
        "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
        "hidden_features": int(model_cfg.get("hidden_features", 64)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
        "classification_loss_weight": classification_loss_weight,
        "seed": seed,
    }, indent=2))

    best_val = math.inf
    bad_epochs = 0
    patience = int(train_cfg.get("patience", epochs + 1))
    history: list[dict[str, float]] = []
    try:
        for epoch in tqdm(range(1, epochs + 1), desc="Training r-GCN"):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, edge_index, edge_type)
            train_indices = split_indices["train"]
            masses = outputs["masses"]
            mass_loss = F.kl_div(
                torch.log(masses[train_indices].clamp_min(1e-9)),
                targets[train_indices],
                reduction="batchmean",
            )
            class_loss = _classification_loss(outputs, train_indices)
            loss = mass_loss + classification_loss_weight * class_loss
            loss.backward()
            optimizer.step()

            train_metrics = split_losses_and_metrics("train")
            test_metrics = split_losses_and_metrics("test")
            val_metrics = split_losses_and_metrics("val")
            history_row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(history_row)
            writer.add_scalar("loss/train_total", train_metrics["loss"], epoch)
            writer.add_scalar("loss/train_mass", train_metrics["mass_loss"], epoch)
            writer.add_scalar("loss/train_classification", train_metrics["classification_loss"], epoch)
            writer.add_scalar("loss/test_total", test_metrics["loss"], epoch)
            writer.add_scalar("loss/test_mass", test_metrics["mass_loss"], epoch)
            writer.add_scalar("loss/test_classification", test_metrics["classification_loss"], epoch)
            writer.add_scalar("loss/validation_total", val_metrics["loss"], epoch)
            writer.add_scalar("loss/validation_mass", val_metrics["mass_loss"], epoch)
            writer.add_scalar("loss/validation_classification", val_metrics["classification_loss"], epoch)
            for task_name in class_targets:
                writer.add_scalar(f"accuracy_train/{task_name}", train_metrics[f"{task_name}_acc"], epoch)
                writer.add_scalar(f"accuracy_test/{task_name}", test_metrics[f"{task_name}_acc"], epoch)
                writer.add_scalar(f"accuracy_validation/{task_name}", val_metrics[f"{task_name}_acc"], epoch)
            writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch)

            diagnostic_parts = [
                f"epoch={epoch:04d}",
                f"train_loss={train_metrics['loss']:.6f}",
                f"test_loss={test_metrics['loss']:.6f}",
                f"val_loss={val_metrics['loss']:.6f}",
                f"train_mass_loss={train_metrics['mass_loss']:.6f}",
                f"test_mass_loss={test_metrics['mass_loss']:.6f}",
                f"train_classification_loss={train_metrics['classification_loss']:.6f}",
                f"test_classification_loss={test_metrics['classification_loss']:.6f}",
            ]
            for task_name in class_targets:
                diagnostic_parts.extend([
                    f"train_{task_name}_acc={train_metrics[f'{task_name}_acc']:.4f}",
                    f"test_{task_name}_acc={test_metrics[f'{task_name}_acc']:.4f}",
                ])
            print(" | ".join(diagnostic_parts))
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                bad_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "relations": graph.relation_names,
                        "class_vocabularies": class_vocabularies,
                        "split_indices": {name: indices.detach().cpu().tolist() for name, indices in split_indices.items()},
                        "best_val_loss": best_val,
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    finally:
        writer.flush()
        writer.close()

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    train_metrics = split_losses_and_metrics("train")
    test_metrics = split_losses_and_metrics("test")
    print("Train metrics:", train_metrics)
    print("Test metrics:", test_metrics)

    model.eval()
    with torch.no_grad():
        outputs = model(x, edge_index, edge_type)
        predictions = outputs["masses"].cpu().numpy()
        uncertainty = outputs["uncertainty"]
        uncertainty_values = uncertainty.cpu().numpy() if uncertainty is not None else None
        classification_probabilities = {
            task_name: F.softmax(logits, dim=-1).cpu().numpy()
            for task_name, logits in outputs["classification_logits"].items()
        }

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "relations": graph.relation_names,
            "class_vocabularies": class_vocabularies,
            "split_indices": {name: indices.detach().cpu().tolist() for name, indices in split_indices.items()},
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
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
            "uncertainty": float(uncertainty_values[node_index, 0]) if uncertainty_values is not None else None,
            "intervals": intervals,
            "classifications": classifications,
        })
    (output_dir / "node_evidence.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"train": train_metrics, "test": test_metrics}, indent=2), encoding="utf-8")

    metrics_plot_path = output_dir / "training_metrics.png"
    if plt is None:
        print("Training metrics plot skipped because matplotlib is not installed.")
    elif history:
        epochs_seen = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for split in ("train", "test", "val"):
            axes[0].plot(epochs_seen, [row[f"{split}_loss"] for row in history], label=f"{split} total")
            axes[0].plot(epochs_seen, [row[f"{split}_mass_loss"] for row in history], linestyle="--", label=f"{split} mass")
            axes[0].plot(epochs_seen, [row[f"{split}_classification_loss"] for row in history], linestyle=":", label=f"{split} class")
        axes[0].set_title("Loss by epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize="small")

        for task_name in class_targets:
            for split in ("train", "test"):
                axes[1].plot(
                    epochs_seen,
                    [row[f"{split}_{task_name}_acc"] for row in history],
                    label=f"{split} {task_name}",
                )
        axes[1].set_title("Train/test classification accuracy by epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(metrics_plot_path, dpi=150)
        print(f"Training metrics plot written to {metrics_plot_path}")
        plt.show()

    return {
        "output_dir": str(output_dir),
        "final_loss": history[-1]["val_loss"],
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
