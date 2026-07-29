#!/usr/bin/env python3
"""Generate an SVG diagram from the advanced GraphSAGE/HGT notebook settings."""

from __future__ import annotations

import argparse
import ast
import html
import json
from pathlib import Path
from typing import Any


REQUIRED_CLASSES = {
    "NeighborSamplingGraphSAGEBlock",
    "RelationAwareHGTLayer",
    "SeriesGraphSAGEHGTClassifier",
}
SETTING_NAMES = {
    "MAXIMUM_HIDDEN_DIM",
    "MINIMUM_HIDDEN_DIM",
    "NUM_MESSAGE_PASSING_EDGES",
    "NUM_HGT_LAYERS",
    "NUM_ATTENTION_HEADS",
    "DROPOUT",
    "TASK_HEAD_HIDDEN_DIM",
    "GRAPHSAGE_FANOUT",
}


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def extract_architecture(path: Path) -> dict[str, Any]:
    """Extract literal configuration without executing the notebook."""
    tree = ast.parse(notebook_source(path), filename=str(path))
    settings: dict[str, Any] = {}
    targets: list[str] | None = None
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in SETTING_NAMES:
            settings[target.id] = ast.literal_eval(node.value)
        elif target.id == "TARGETS":
            targets = ast.literal_eval(node.value)

    missing_settings = sorted(SETTING_NAMES - settings.keys())
    missing_classes = sorted(REQUIRED_CLASSES - classes)
    if missing_settings or missing_classes or targets is None:
        problems = []
        if missing_settings:
            problems.append(f"settings: {', '.join(missing_settings)}")
        if missing_classes:
            problems.append(f"classes: {', '.join(missing_classes)}")
        if targets is None:
            problems.append("TARGETS")
        raise ValueError("Notebook architecture is incomplete; missing " + "; ".join(problems))

    layer_count = int(settings["NUM_MESSAGE_PASSING_EDGES"])
    maximum = int(settings["MAXIMUM_HIDDEN_DIM"])
    minimum = int(settings["MINIMUM_HIDDEN_DIM"])
    if layer_count == 1:
        hidden_dims = [maximum]
    else:
        hidden_dims = [
            round(maximum - (maximum - minimum) * index / (layer_count - 1))
            for index in range(layer_count)
        ]
    return {"settings": settings, "targets": targets, "graphsage_hidden_dims": hidden_dims}


def _text(x: int, y: int, value: str, *, size: int = 16, weight: int = 400, anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="#172033">{html.escape(value)}</text>'
    )


def _box(x: int, y: int, width: int, height: int, title: str, lines: list[str], color: str) -> str:
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" fill="{color}" stroke="#31415f" stroke-width="2"/>',
        _text(x + width // 2, y + 30, title, size=18, weight=700),
    ]
    for index, line in enumerate(lines):
        parts.append(_text(x + width // 2, y + 57 + index * 22, line, size=14))
    return "\n".join(parts)


def render_svg(architecture: dict[str, Any]) -> str:
    settings = architecture["settings"]
    dims = architecture["graphsage_hidden_dims"]
    targets = architecture["targets"]
    boxes = [
        (30, 145, 190, 150, "Node features", ["label-free ESM", "kinematics + time", "sensor + evidence"], "#e8f1ff"),
        (270, 145, 210, 150, "Input projection", [f"Linear → {settings['MAXIMUM_HIDDEN_DIM']}", "LayerNorm + GELU", f"Dropout {settings['DROPOUT']}"], "#e9f8f0"),
        (530, 120, 260, 200, "GraphSAGE encoder", [f"{len(dims)} mean-aggregation blocks", f"widths: {dims[0]} → {dims[-1]}", f"fanout: {settings['GRAPHSAGE_FANOUT']}", "residual + norm per block"], "#fff3d9"),
        (840, 120, 250, 200, "Relation-aware HGT", [f"{settings['NUM_HGT_LAYERS']} layer(s)", f"{settings['NUM_ATTENTION_HEADS']} attention heads", "typed key/value transforms", "full relational edge set"], "#f4e9ff"),
        (1140, 105, 260, 230, "Multitask heads", [f"MLP: {dims[-1]} → {settings['TASK_HEAD_HIDDEN_DIM']}", "GELU + dropout", *targets], "#ffe9ed"),
    ]
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1430" height="460" viewBox="0 0 1430 460" role="img" aria-labelledby="title description">',
        '<title id="title">Advanced GraphSAGE and HGT classifier architecture</title>',
        '<desc id="description">Node features flow through input projection, GraphSAGE blocks, relation-aware HGT attention, and four classification heads.</desc>',
        '<rect width="1430" height="460" fill="#ffffff"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#526887"/></marker></defs>',
        _text(715, 45, "Observation-series + intelligence GraphSAGE/HGT classifier", size=25, weight=700),
        _text(715, 76, "Configuration extracted from the notebook without executing training", size=15),
    ]
    for box in boxes:
        svg.append(_box(*box))
    for x1, x2 in ((220, 270), (480, 530), (790, 840), (1090, 1140)):
        svg.append(f'<line x1="{x1}" y1="220" x2="{x2 - 8}" y2="220" stroke="#526887" stroke-width="3" marker-end="url(#arrow)"/>')
    svg.extend([
        _text(660, 375, "Precomputed sampled inbound edges", size=14),
        _text(965, 375, "Self, temporal, emitter, candidate, report, claim, and contradiction relations", size=14),
        _text(715, 425, "All heads classify observation-node embeddings; ground-truth fields are not model inputs", size=15, weight=700),
        "</svg>",
    ])
    return "\n".join(svg) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=root / "notebooks" / "observation_series_and_intel_rgcn_classification_advanced_network.ipynb")
    parser.add_argument("--output", type=Path, default=root / "docs" / "advanced_network_architecture.svg")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(extract_architecture(args.notebook)), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
