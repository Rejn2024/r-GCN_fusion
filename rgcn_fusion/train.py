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
    def tqdm(iterable, desc=None, **_kwargs):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        label = desc or "Progress"
        print(f"{label}: starting" + (f" ({total} steps)" if total else ""))
        for item in iterable:
            yield item
        print(f"{label}: done")

from .dempster_shafer import belief_plausibility, subset_masks, validate_masses
from .model import RGCNEvidenceModel
from .neo4j_loader import GraphData, Neo4jGraphLoader

IGNORE_CLASS_INDEX = -1
DEFAULT_CLASSIFICATION_TARGETS = {
    "radar_type": "radar_id",
    "radar_mode": "mode_id",
    "aircraft_variant": "aircraft_id",
    "operator_country": "operator_country",
}
DEFAULT_CLASSIFICATION_TASK_LOSS_WEIGHTS = {
    "aircraft_variant": 10.0,
    "operator_country": 10.0,
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


def _classification_task_loss_weights(train_cfg: dict[str, Any]) -> dict[str, float]:
    """Return per-classification-task loss multipliers.

    The default emphasizes the aircraft-variant and operator-country targets so
    training optimizes more strongly for the accuracy metrics stakeholders care
    about most.  Users can override or extend these multipliers with
    ``training.classification_task_loss_weights``.
    """
    configured = train_cfg.get("classification_task_loss_weights")
    if configured is None:
        return dict(DEFAULT_CLASSIFICATION_TASK_LOSS_WEIGHTS)
    if not isinstance(configured, dict):
        raise TypeError("classification_task_loss_weights must map task names to numeric weights")
    weights = {str(task_name): float(weight) for task_name, weight in configured.items()}
    for task_name, weight in weights.items():
        if weight < 0.0:
            raise ValueError(f"classification_task_loss_weights[{task_name!r}] must be non-negative")
    return weights


def _bounded_fraction(train_cfg: dict[str, Any], key: str, default: float) -> float:
    """Return a configured regularization fraction in the inclusive [0, 1] range."""
    value = float(train_cfg.get(key, default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"training.{key} must be between 0.0 and 1.0")
    return value


def _smooth_mass_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Blend target masses with a uniform prior to discourage overconfident fits."""
    if smoothing == 0.0:
        return targets
    uniform = torch.full_like(targets, 1.0 / targets.shape[-1])
    return targets.mul(1.0 - smoothing).add(uniform, alpha=smoothing)


def _entropy_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Return mean categorical entropy for already-normalized probabilities."""
    return -(probabilities.clamp_min(1e-9).log() * probabilities).sum(dim=-1).mean()


def _nodes_with_labels(graph: GraphData, required_labels: list[str]) -> np.ndarray:
    """Return a boolean mask selecting nodes that have all required Neo4j labels."""
    if not required_labels:
        return np.ones(len(graph.node_ids), dtype=bool)
    if graph.node_labels is None:
        raise ValueError("node labels are required when data.supervised_node_labels is configured")
    required = set(required_labels)
    return np.asarray([required.issubset(set(labels)) for labels in graph.node_labels], dtype=bool)


def _supervised_node_mask(graph: GraphData, data_cfg: dict[str, Any]) -> np.ndarray:
    """Select nodes that are allowed to contribute supervised losses and metrics."""
    mask = _nodes_with_labels(graph, [str(label) for label in data_cfg.get("supervised_node_labels", [])])
    if graph.labels is not None:
        mask &= np.asarray([len(row) > 0 for row in graph.labels], dtype=bool)
    for values in (graph.classification_labels or {}).values():
        mask &= np.asarray([value is not None for value in values.tolist()], dtype=bool)
    if not np.any(mask):
        raise ValueError("no supervised nodes remain after applying label and node-label filters")
    return mask


def _grouped_split_indices(
    graph: GraphData,
    supervised_mask: np.ndarray,
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create train/test/validation splits, optionally grouping by a node property."""
    seed = int(train_cfg.get("seed", 42))
    train_fraction = float(train_cfg.get("train_fraction", 0.5))
    test_fraction = float(train_cfg.get("test_fraction", 0.3))
    val_fraction = float(train_cfg.get("val_fraction", 0.2))
    if not math.isclose(train_fraction + test_fraction + val_fraction, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train_fraction, test_fraction, and val_fraction must sum to 1.0")

    supervised_indices = np.flatnonzero(supervised_mask)
    split_group_property = data_cfg.get("split_group_property")
    rng = np.random.default_rng(seed)
    if split_group_property:
        if graph.node_properties is None:
            raise ValueError("node properties are required when data.split_group_property is configured")
        group_to_indices: dict[Any, list[int]] = {}
        for idx in supervised_indices.tolist():
            props = graph.node_properties[idx]
            group_value = props.get(split_group_property)
            if group_value is None:
                group_value = graph.node_ids[idx]
            group_to_indices.setdefault(group_value, []).append(idx)
        groups = np.asarray(list(group_to_indices), dtype=object)
        rng.shuffle(groups)
        n_train = int(round(train_fraction * len(groups)))
        n_test = int(round(test_fraction * len(groups)))
        split_groups = {
            "train": groups[:n_train],
            "test": groups[n_train:n_train + n_test],
            "val": groups[n_train + n_test:],
        }
        split_arrays = {
            name: np.asarray(
                [idx for group in groups_for_split.tolist() for idx in group_to_indices[group]],
                dtype=np.int64,
            )
            for name, groups_for_split in split_groups.items()
        }
    else:
        perm = rng.permutation(supervised_indices)
        n_train = int(round(train_fraction * len(perm)))
        n_test = int(round(test_fraction * len(perm)))
        split_arrays = {
            "train": perm[:n_train],
            "test": perm[n_train:n_train + n_test],
            "val": perm[n_train + n_test:],
        }
    return {
        name: torch.as_tensor(indices, dtype=torch.long, device=device)
        for name, indices in split_arrays.items()
    }


def _filter_graph_edges(graph: GraphData, data_cfg: dict[str, Any], split_indices: dict[str, torch.Tensor]) -> GraphData:
    """Drop configured leakage-prone relations and, optionally, cross-split edges."""
    excluded = {str(name) for name in data_cfg.get("exclude_relation_types", [])}
    keep_relation = np.ones(len(graph.edge_type), dtype=bool)
    if excluded:
        keep_relation &= np.asarray(
            [graph.relation_names[int(rel)] not in excluded for rel in graph.edge_type],
            dtype=bool,
        )

    if data_cfg.get("remove_cross_split_edges", False):
        split_by_node: dict[int, str] = {}
        for split, indices in split_indices.items():
            for idx in indices.detach().cpu().tolist():
                split_by_node[idx] = split
        keep_cross_split = []
        for source, target in graph.edge_index.T.tolist():
            source_split = split_by_node.get(source)
            target_split = split_by_node.get(target)
            keep_cross_split.append(
                source_split is None or target_split is None or source_split == target_split
            )
        keep_relation &= np.asarray(keep_cross_split, dtype=bool)

    if np.all(keep_relation):
        return graph

    kept_edge_type = graph.edge_type[keep_relation]
    kept_edge_index = graph.edge_index[:, keep_relation]
    used_relation_ids = sorted(set(kept_edge_type.tolist()))
    rel_remap = {old: new for new, old in enumerate(used_relation_ids)}
    remapped_edge_type = np.asarray([rel_remap[int(rel)] for rel in kept_edge_type], dtype=np.int64)
    relation_names = [graph.relation_names[old] for old in used_relation_ids]
    return GraphData(
        node_ids=graph.node_ids,
        node_features=graph.node_features,
        edge_index=kept_edge_index,
        edge_type=remapped_edge_type,
        relation_names=relation_names,
        labels=graph.labels,
        classification_labels=graph.classification_labels,
        node_properties=graph.node_properties,
        node_labels=graph.node_labels,
    )


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
    expected_masses = len(subset_masks(hypotheses))
    if targets_np.shape[1] != expected_masses:
        raise ValueError(f"labels must contain {expected_masses} masses per node")

    encoded_classes, class_vocabularies = _encode_classification_targets(
        graph, data_cfg.get("class_values")
    )

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    supervised_mask = _supervised_node_mask(graph, data_cfg)
    split_indices = _grouped_split_indices(graph, supervised_mask, data_cfg, train_cfg, device)
    graph = _filter_graph_edges(graph, data_cfg, split_indices)
    x, edge_index, edge_type = _tensor_graph(graph, device)
    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    mass_label_smoothing = _bounded_fraction(train_cfg, "mass_label_smoothing", 0.01)
    targets = _smooth_mass_targets(targets, mass_label_smoothing)
    class_targets = {
        task_name: torch.as_tensor(labels, dtype=torch.long, device=device)
        for task_name, labels in encoded_classes.items()
    }

    num_layers = max(5, int(model_cfg.get("num_layers", 5)))
    edge_chunk_size = int(model_cfg.get("edge_chunk_size", train_cfg.get("edge_chunk_size", 50000)))
    gradient_checkpointing = bool(model_cfg.get("gradient_checkpointing", train_cfg.get("gradient_checkpointing", True)))
    use_amp = bool(train_cfg.get("use_amp", device.type == "cuda"))
    amp_dtype_name = str(train_cfg.get("amp_dtype", "float16"))
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16

    model = RGCNEvidenceModel(
        in_features=x.shape[1],
        hidden_features=int(model_cfg.get("hidden_features", 64)),
        num_relations=max(len(graph.relation_names), 1),
        num_hypotheses=len(hypotheses),
        hypotheses=hypotheses,
        dropout=float(model_cfg.get("dropout", 0.1)),
        classification_tasks={task_name: len(values) for task_name, values in class_vocabularies.items()},
        num_layers=num_layers,
        num_bases=model_cfg.get("num_bases"),
        residual=bool(model_cfg.get("residual", True)),
        normalization=model_cfg.get("normalization", "layernorm"),
        relation_gates=bool(model_cfg.get("relation_gates", False)),
        task_head_hidden_features=model_cfg.get("task_head_hidden_features"),
        mass_head_type=str(model_cfg.get("mass_head_type", "softmax")),
        edge_chunk_size=edge_chunk_size,
        gradient_checkpointing=gradient_checkpointing,
    ).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda" and amp_dtype == torch.float16)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    classification_loss_weight = float(train_cfg.get("classification_loss_weight", 1.0))
    classification_task_loss_weights = _classification_task_loss_weights(train_cfg)
    classification_label_smoothing = _bounded_fraction(train_cfg, "classification_label_smoothing", 0.05)
    confidence_penalty_weight = float(train_cfg.get("confidence_penalty_weight", 0.01))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    l1_lambda = float(train_cfg.get("l1_lambda", 1e-5))
    epochs = int(train_cfg.get("epochs", 200))
    batch_size = int(train_cfg.get("batch_size", 0))
    seed = int(train_cfg.get("seed", 42))
    train_fraction = float(train_cfg.get("train_fraction", 0.5))
    test_fraction = float(train_cfg.get("test_fraction", 0.3))
    val_fraction = float(train_cfg.get("val_fraction", 0.2))

    output_dir = Path(output_cfg.get("directory", "artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = output_dir / "tensorboard"
    ckpt_path = output_dir / "best_rgcn_evidence_model.pt"

    print({name: int(indices.numel()) for name, indices in split_indices.items()})
    train_size = int(split_indices["train"].numel())
    if train_size < 1:
        raise ValueError("training split must contain at least one node")
    if batch_size == 0:
        batch_size = train_size
    elif batch_size < 0:
        raise ValueError("training.batch_size must be zero for full-batch training or greater than zero")
    batches_per_epoch = math.ceil(train_size / batch_size)
    print(f"training nodes={train_size:,}, batch_size={batch_size:,}, batches_per_epoch={batches_per_epoch:,}")

    def _epoch_train_batches(epoch: int) -> list[torch.Tensor]:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + epoch)
        shuffled = split_indices["train"][torch.randperm(train_size, generator=generator, device=device)]
        return [shuffled[start:start + batch_size] for start in range(0, train_size, batch_size)]

    if confidence_penalty_weight < 0.0:
        raise ValueError("training.confidence_penalty_weight must be non-negative")
    if max_grad_norm < 0.0:
        raise ValueError("training.max_grad_norm must be non-negative")

    def _classification_loss(outputs: dict[str, Any], indices: torch.Tensor) -> torch.Tensor:
        class_loss = torch.zeros((), dtype=torch.float32, device=device)
        for task_name, labels in class_targets.items():
            split_labels = labels[indices]
            if not torch.any(split_labels != IGNORE_CLASS_INDEX):
                continue
            logits = outputs["classification_logits"][task_name][indices]
            task_weight = classification_task_loss_weights.get(task_name, 1.0)
            class_loss = class_loss + task_weight * F.cross_entropy(
                logits,
                split_labels,
                ignore_index=IGNORE_CLASS_INDEX,
                label_smoothing=classification_label_smoothing,
            )
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
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
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
        "l1_lambda": l1_lambda,
        "mass_label_smoothing": mass_label_smoothing,
        "classification_label_smoothing": classification_label_smoothing,
        "confidence_penalty_weight": confidence_penalty_weight,
        "max_grad_norm": max_grad_norm,
        "hidden_features": int(model_cfg.get("hidden_features", 64)),
        "num_layers": num_layers,
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "train_fraction": train_fraction,
        "test_fraction": test_fraction,
        "val_fraction": val_fraction,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "classification_loss_weight": classification_loss_weight,
        "classification_task_loss_weights": classification_task_loss_weights,
        "edge_chunk_size": edge_chunk_size,
        "gradient_checkpointing": gradient_checkpointing,
        "use_amp": use_amp,
        "amp_dtype": amp_dtype_name,
        "seed": seed,
    }, indent=2))

    scheduler_enabled = bool(train_cfg.get("reduce_lr_on_plateau", True))
    lr_patience = int(train_cfg.get("lr_patience", max(1, int(train_cfg.get("patience", 10)) // 3)))
    lr_factor = float(train_cfg.get("lr_factor", 0.5))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    ) if scheduler_enabled else None

    best_val = math.inf
    bad_epochs = 0
    patience = int(train_cfg.get("patience", 10))
    min_delta = float(train_cfg.get("early_stopping_min_delta", 1e-4))
    history: list[dict[str, float]] = []
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_batches = _epoch_train_batches(epoch)
            batch_progress = tqdm(
                epoch_batches,
                desc=f"Epoch {epoch}/{epochs}",
                unit="batch",
                leave=True,
            )
            weighted_l1_penalty = 0.0
            for batch_indices in batch_progress:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
                    outputs = model(x, edge_index, edge_type)
                    masses = outputs["masses"]
                    mass_loss = F.kl_div(
                        torch.log(masses[batch_indices].clamp_min(1e-9)),
                        targets[batch_indices],
                        reduction="batchmean",
                    )
                    class_loss = _classification_loss(outputs, batch_indices)
                    l1_penalty = torch.zeros((), dtype=torch.float32, device=device)
                    if l1_lambda > 0.0:
                        l1_penalty = sum(parameter.abs().sum() for parameter in model.parameters())
                    batch_fraction = float(batch_indices.numel()) / train_size
                    confidence_penalty = torch.zeros((), dtype=torch.float32, device=device)
                    if confidence_penalty_weight > 0.0:
                        confidence_penalty = confidence_penalty - _entropy_from_probabilities(masses[batch_indices])
                        for logits in outputs["classification_logits"].values():
                            confidence_penalty = confidence_penalty - _entropy_from_probabilities(
                                F.softmax(logits[batch_indices], dim=-1)
                            )
                    loss = (
                        mass_loss
                        + classification_loss_weight * class_loss
                        + l1_lambda * l1_penalty * batch_fraction
                        + confidence_penalty_weight * confidence_penalty
                    )
                scaler.scale(loss).backward()
                if max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                weighted_l1_penalty += float(l1_penalty.detach().cpu()) * batch_fraction
                if hasattr(batch_progress, "set_postfix"):
                    batch_progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

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
            writer.add_scalar("regularization/l1_penalty", weighted_l1_penalty, epoch)

            print(f"Epoch no: {epoch}")
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
            if scheduler is not None:
                scheduler.step(val_metrics["loss"])
            if val_metrics["loss"] < best_val - min_delta:
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
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
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
