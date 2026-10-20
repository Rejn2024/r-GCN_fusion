"""Run notebook consumers on complete synthetic tracks without training or services."""

import ast
import copy
from collections import Counter, defaultdict
from contextlib import ExitStack
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn
import torch.nn.functional as F

from rgcn_fusion.graph_optimisation import build_track_graph_batches_by_split


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "track_star_hgt.ipynb"


def notebook_cells():
    return ["".join(cell.get("source", []))
            for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"]


def execute(source, namespace):
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)


def test_all_notebook_cells_compile():
    for source in notebook_cells():
        compile(source, str(NOTEBOOK), "exec")


def synthetic_namespace(directory, plt):
    """Interleave global observation order across eight variable-length tracks."""
    series_ids = [f"track-{index}" for index in range(8)]
    membership = [track for step in range(4) for track in (5, 0, 7, 2, 4, 1, 6, 3)
                  if step < 2 + track % 3]
    sequences = Counter()
    metadata = []
    for position, track in enumerate(membership):
        sequence = sequences[track]
        sequences[track] += 1
        metadata.append({"node_kind": "observation", "series_id": series_ids[track],
                         "observation_id": f"observation-{position}",
                         "sequence_index": sequence})
    observation_count = len(metadata)
    metadata.append({"node_kind": "kg_entity"})
    node_count = len(metadata)
    # A shared KG node makes feature and observation counts deliberately differ.
    self_nodes = torch.arange(node_count)
    obs_nodes = torch.arange(observation_count)
    kg_nodes = torch.full((observation_count,), observation_count)
    edges = torch.cat((torch.stack((self_nodes, self_nodes)),
                       torch.stack((obs_nodes, kg_nodes)),
                       torch.stack((kg_nodes, obs_nodes))), dim=1)
    edge_types = torch.cat((torch.zeros(node_count, dtype=torch.long),
                            torch.ones(2 * observation_count, dtype=torch.long)))
    track_mapping = torch.tensor(membership)
    split_tracks = {"train": torch.tensor([0, 1]), "val": torch.tensor([2, 3]),
                    "test": torch.tensor([4, 5]), "open_set": torch.tensor([6, 7])}
    batches = build_track_graph_batches_by_split(
        edge_index=edges, edge_types=edge_types,
        node_track_index=torch.tensor([*membership, -1]),
        observation_nodes=obs_nodes, observation_track_index=track_mapping,
        selected_tracks_by_split=split_tracks, tracks_per_batch=2)
    vocab = [("Variant A", "Radar A", "Operator A"),
             ("Variant B", "Radar B", "Operator B")]
    worlds = [{"aircraft_variant": v, "aircraft_family": f"Family {i}",
               "radar_type": r, "operator_country": o}
              for i, (v, r, o) in enumerate(vocab)]
    targets = torch.tensor([0, 1, 0, 1, 0, 1, -1, -1])
    mode_targets = torch.tensor([
        -1 if track >= 6 else (metadata[index]["sequence_index"] + track) % 2
        for index, track in enumerate(membership)])
    kg_path = directory / "kg.json"
    kg_path.write_text(json.dumps({"worlds": worlds}), encoding="utf-8")
    generator = torch.Generator().manual_seed(43)
    return dict(
        torch=torch, nn=nn, F=F, np=np, os=os, math=math, gc=gc,
        json=json, hashlib=hashlib, Counter=Counter, defaultdict=defaultdict, time=time,
        plt=plt, DEVICE=torch.device("cpu"), SEED=42,
        ROOT=directory, ARTIFACT_DIR=directory, KG_JSON=kg_path,
        X=torch.randn(node_count, 5, generator=generator),
        node_meta=metadata, observation_node_indices=list(range(observation_count)),
        observation_nodes=obs_nodes, observation_nodes_cpu=obs_nodes,
        observation_track_index=track_mapping,
        observation_track_index_cpu=track_mapping,
        series_ids=series_ids, split_series={name: [series_ids[i] for i in ids]
                                           for name, ids in split_tracks.items()},
        track_splits=split_tracks,
        observation_splits={name: torch.nonzero(torch.isin(track_mapping, tracks)).flatten()
                            for name, tracks in split_tracks.items()},
        SUPERVISED_SPLITS=("train", "val", "test"), batches_by_split=batches,
        RELATIONS={"self": 0, "context": 1}, RELATION_NAMES=["self", "context"],
        relation_edge_counts=torch.bincount(edge_types, minlength=2).tolist(),
        edge_index=edges, edge_types=edge_types,
        rao_vocab=vocab, mode_vocab=["Mode A", "Mode B"],
        RAO_FIELDS=("aircraft_variant", "radar_type", "operator_country"),
        kg_valid_worlds=worlds,
        rao_family_by_variant={world["aircraft_variant"]: world["aircraft_family"]
                               for world in worlds},
        rao_mode_compatibility=torch.ones(2, 2),
        track_targets=targets, mode_targets=mode_targets,
        track_targets_by_series_id={sid: dict(zip(
            ("aircraft_variant", "radar_type", "operator_country"), vocab[i % 2]))
            for i, sid in enumerate(series_ids)},
        rao_base_rates_cpu=torch.tensor([[.7, .3], [.3, .7]] * 4),
        rao_prior_audit=[{"source_report_ids": [], "multiplier": 10.0}] * 8,
        RAO_OOB_BASE_RATE_MULTIPLIER=10.0,
        CLAIM_CANDIDATE_MIN_ABS_CONTRIBUTION=.15,
        COLLAPSE_RECIPROCAL_CLAIM_CANDIDATE_EDGES=True,
        TRACKS_PER_BATCH=2,
        mlflow=Mock(), log_numeric_results=Mock(),
        best_epoch=1, best_test_loss=1.0, best_ckpt_path=directory / "best.pt",
        # This sentinel must be released while downstream inference inputs survive.
        series_records=[{"construction_only": True}],
    )


def test_star_notebook_forward_cleanup_export_and_all_evaluation_cells(tmp_path):
    pytest.importorskip("torch_geometric")
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("sklearn")
    pytest.importorskip("plotly")
    pytest.importorskip("tensorboard")
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from plotly.basedatatypes import BaseFigure
    from rgcn_fusion.track_star_hgt import (
        TrackStarHGTClassifier, prepare_track_star_graph,
    )

    sources = notebook_cells()
    namespace = synthetic_namespace(tmp_path, plt)
    namespace.update(TrackStarHGTClassifier=TrackStarHGTClassifier,
                     prepare_track_star_graph=prepare_track_star_graph)
    thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ExitStack() as stack:
            # Exercise real numerical plotting code without rendering large exports
            # or opening a renderer. Network-backed logging is independently stubbed.
            stack.enter_context(patch.object(Figure, "savefig"))
            stack.enter_context(patch.object(BaseFigure, "write_html"))
            stack.enter_context(patch.object(BaseFigure, "write_image"))
            stack.enter_context(patch.object(BaseFigure, "show"))
            stack.enter_context(patch.object(plt, "show"))
            stack.enter_context(patch.dict(os.environ, {
                "DASHBOARD_TRACK_INDEX": "4", "LLM_TRACK_INDEX": "0",
                "PLOTLY_STATIC_EXPORT": "0", "PREFETCH_BATCHES": "0",
                "COMPILE_MODEL": "1",
            }))
            # Exercise Dynamo's dynamic capture while avoiding production GPU
            # code generation in this CPU correctness/integration test.
            compile_model = torch.compile
            compile_spy = stack.enter_context(patch.object(
                torch, "compile", side_effect=lambda module, **options:
                compile_model(module, backend="eager", **options)))

            model_source = next(source for source in sources
                                if "model = TrackStarHGTClassifier(" in source)
            execute(model_source, namespace)
            compile_spy.assert_called_once_with(namespace["model"], dynamic=True)
            model = namespace["model"]
            assert len(model.hgt_layers) == 5
            assert not hasattr(model, "rgcn_layers")
            model.eval()

            computation_source = next(source for source in sources
                                      if "def estimate_edge_computation_requirements(" in source)
            execute(computation_source, namespace)
            memory_source = next(source for source in sources
                                 if "def estimate_peak_training_gpu_memory(" in source)
            execute(memory_source, namespace)
            assert namespace["gpu_memory_preflight"]["training_can_start"]

            cleanup = next(source for source in sources
                           if "_CONSTRUCTION_ONLY_OBJECTS = (" in source)
            execute(cleanup, namespace)
            assert "series_records" not in namespace

            training = next(source for source in sources if "def prepare_batches(" in source)
            training_tree = ast.parse(training)
            # Definitions and preparation are real notebook code. Stop immediately
            # before training controls, MLflow run creation, and the epoch loop.
            prefix = []
            for node in training_tree.body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "EPOCHS"
                        for target in node.targets):
                    break
                prefix.append(node)
            execute(ast.Module(body=prefix, type_ignores=[]), namespace)

            with torch.inference_mode():
                gathered = namespace["model_forward"]()
                for batches in namespace["prepared_batches_by_split"].values():
                    for prepared in batches:
                        batch = namespace["load_batch"](prepared)
                        local = namespace["model_forward_batch"](batch)
                        torch.testing.assert_close(gathered["track_embeddings"][batch.track_indices],
                                                   local["track_embeddings"])
                        torch.testing.assert_close(
                            gathered["observation_embeddings"][batch.observation_positions],
                            local["observation_embeddings"])
            assert gathered["observation_embeddings"].size(0) == len(
                namespace["observation_node_indices"])
            assert gathered["track_embeddings"].size(0) == len(namespace["series_ids"])

            # Train one synthetic mini-batch through the notebook's actual loss;
            # every downstream consumer then receives outputs from the real model.
            model.train()
            prepared = namespace["prepared_batches_by_split"]["train"][0]
            batch = namespace["load_batch"](prepared)
            output = namespace["model_forward_batch"](batch)
            loss, _ = namespace["batch_hierarchical_loss"](output, batch, .005)
            loss.backward()
            assert torch.isfinite(loss)
            assert model.track_seed.grad is not None
            assert torch.isfinite(model.track_seed.grad).all()
            namespace["optimizer"].step()
            namespace["optimizer"].zero_grad(set_to_none=True)

            # Cover checkpoint reload with the changed state structure before export.
            torch.save({"model_state": model.state_dict(), "rao_vocab": namespace["rao_vocab"],
                        "mode_vocab": namespace["mode_vocab"], "epoch": 1, "test_loss": 1.0},
                       namespace["best_ckpt_path"])
            assert all(not key.startswith("_orig_mod.") for key in model.state_dict())
            reload_source = next(source for source in sources if source.startswith(
                "# Major step: Reload the best test-selected model for final evaluation"))
            execute(reload_source, namespace)

            policy = next(index for index, source in enumerate(sources) if source.startswith(
                "# Major step: Maximal safe partial-identification policy"))
            for source in sources[policy:]:
                if not source.strip():
                    continue
                if "def ollama_generate(" in source:
                    # Keep the complete explanation cell and its file-writing path;
                    # substitute only the external model call, leaving prompt logic.
                    tree = ast.parse(source)
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and node.name == "ollama_generate":
                            node.body = [ast.Return(value=ast.Constant(
                                "Synthetic evidence-grounded explanation."))]
                    execute(ast.fix_missing_locations(tree), namespace)
                elif "# Training curves for the two prediction levels." in source:
                    namespace["history"] = [{
                        "epoch": 1, "train_track_rao_acc": .5, "test_track_rao_acc": .5,
                        "train_radar_mode_acc": .5, "test_radar_mode_acc": .5,
                    }]
                    execute(source, namespace)
                else:
                    execute(source, namespace)

            records = json.loads((tmp_path / "track_predictions.json").read_text())
            assert [record["series_id"] for record in records] == namespace["series_ids"]
            for index, record in enumerate(records):
                expected_positions = [position for position, track in enumerate(
                    namespace["observation_track_index_cpu"].tolist()) if track == index]
                expected_ids = [namespace["node_meta"][position]["observation_id"]
                                for position in expected_positions]
                assert [item["observation_id"] for item in record["mode_sequence"]] == expected_ids
            summary = json.loads((tmp_path / "training_summary.json").read_text())
            assert set(summary["metrics"]) == {"train", "val", "test"}
            assert summary["open_set_evaluation"]["track_count"] == 2
            assert summary["rao_violation_count"] == 0
            assert namespace["diagnostic_tasks"]["Track RAO"]["arrays"]["test"]["targets"].size == 2
            assert namespace["track_explanation"]["series_id"] == namespace["series_ids"][0]
    finally:
        plt.close("all")
        torch.set_num_threads(thread_count)


def _replace_assignments(source, values):
    """Keep real notebook behavior while bounding synthetic training duration."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in values:
                node.value = ast.Constant(values[target.id])
    return ast.fix_missing_locations(tree)


def _training_namespace(directory):
    """Build uneven batches with a final one-track batch in train and test."""
    from rgcn_fusion.track_star_hgt import (
        TrackStarHGTClassifier, prepare_track_star_graph,
    )
    namespace = synthetic_namespace(directory, Mock())
    split_tracks = {
        "train": torch.tensor([0, 1, 2]), "test": torch.tensor([3, 4, 5]),
        "val": torch.tensor([6]), "open_set": torch.tensor([7]),
    }
    membership = namespace["observation_track_index_cpu"]
    namespace["track_targets"][6] = 0
    for position in torch.nonzero(membership == 6).flatten().tolist():
        namespace["mode_targets"][position] = position % 2
    namespace.update(
        TrackStarHGTClassifier=TrackStarHGTClassifier,
        prepare_track_star_graph=prepare_track_star_graph,
        track_splits=split_tracks,
        split_series={name: {namespace["series_ids"][track] for track in tracks.tolist()}
                      for name, tracks in split_tracks.items()},
        observation_splits={name: torch.nonzero(torch.isin(membership, tracks)).flatten()
                            for name, tracks in split_tracks.items()},
        batches_by_split=build_track_graph_batches_by_split(
            edge_index=namespace["edge_index"], edge_types=namespace["edge_types"],
            node_track_index=torch.cat((membership, torch.tensor([-1]))),
            observation_nodes=namespace["observation_nodes_cpu"],
            observation_track_index=membership,
            selected_tracks_by_split=split_tracks, tracks_per_batch=2),
        MLFLOW_RUN_NAME="synthetic-training", MLFLOW_TRACKING_URI="file://synthetic",
        DATA_PATH=directory / "dataset.json", GRAPH_INPUT_ARTIFACT=directory / "graph.pt",
        RELATIONAL_GRAPH_ARTIFACT=directory / "relations.pt", NOTEBOOK_LOGGER=Mock(),
    )
    namespace["mlflow"].start_run.return_value = SimpleNamespace(
        info=SimpleNamespace(run_id="synthetic-run"))
    source = next(source for source in notebook_cells()
                  if "model = TrackStarHGTClassifier(" in source)
    execute(_replace_assignments(source, {
        "HIDDEN_DIM": 8, "TASK_HEAD_HIDDEN_DIM": 8,
        "NUM_ATTENTION_HEADS": 2, "DROPOUT": 0.0,
    }), namespace)
    return namespace


def _definitions_before_training(namespace):
    source = next(source for source in notebook_cells() if "def prepare_batches(" in source)
    prefix = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "EPOCHS"
                for target in node.targets):
            break
        prefix.append(node)
    execute(ast.Module(body=prefix, type_ignores=[]), namespace)


def test_streaming_metrics_match_global_reference_on_uneven_batches(tmp_path):
    pytest.importorskip("torch_geometric")
    pytest.importorskip("tensorboard")
    original_compile = torch.compile
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch.dict(os.environ, {"COMPILE_MODEL": "1", "PREFETCH_BATCHES": "0"}), patch.object(
                torch, "compile", side_effect=lambda module, **options:
                original_compile(module, backend="eager", **options)):
            namespace = _training_namespace(tmp_path)
            _definitions_before_training(namespace)
            prepared = namespace["prepared_batches_by_split"]
            graph_ids = [id(batch.graph) for batches in prepared.values() for batch in batches]
            # Interactive reruns reuse the prepared topology after original
            # edge buffers and their container have already been released.
            assert "batches_by_split" not in namespace
            _definitions_before_training(namespace)
            assert namespace["prepared_batches_by_split"] is prepared
            assert [id(batch.graph) for batches in prepared.values() for batch in batches] == graph_ids
            memory_source = next(source for source in notebook_cells()
                                 if "def estimate_peak_training_gpu_memory(" in source)
            execute(memory_source, namespace)
            preflight = namespace["gpu_memory_preflight"]
            all_batches = [batch for batches in prepared.values() for batch in batches]
            assert preflight["maximum_batch_nodes"] == max(
                batch.node_indices.numel() + batch.track_indices.numel() for batch in all_batches)
            assert preflight["maximum_virtual_tracks"] == max(batch.num_tracks for batch in all_batches)
            assert preflight["maximum_batch_edges"] == max(batch.graph.num_edges for batch in all_batches)
            assert preflight["training_can_start"]
            namespace["model"].eval()
            with torch.inference_mode():
                gathered = namespace["model_forward"]()
                expected = namespace["metrics_from_outputs"](gathered, "test")
                actual = namespace["split_metrics"]("test")
            batches = namespace["prepared_batches_by_split"]["test"]
            assert [batch.track_indices.numel() for batch in batches] == [2, 1]
            assert set(actual) == set(expected)
            for key in expected:
                assert actual[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-6), key
            # No superseded unsorted edge buffers or TrackGraphBatch objects
            # survive through an alias in the notebook after preparation.
            assert "batches_by_split" not in namespace
            assert all(not hasattr(batch, "edge_index") and not hasattr(batch, "edge_types")
                       for batch in batches)
    finally:
        torch.set_num_threads(threads)


def test_compiled_initial_and_continuation_training_keep_raw_checkpoints(tmp_path):
    pytest.importorskip("torch_geometric")
    pytest.importorskip("tensorboard")
    original_compile = torch.compile
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    metric_calls = []

    def record_split_metric(function):
        def wrapped(name):
            metric_calls.append(name)
            return function(name)
        return wrapped

    try:
        with patch.dict(os.environ, {
            "COMPILE_MODEL": "1", "PREFETCH_BATCHES": "0", "METRICS_INTERVAL": "10",
            "CONTINUATION_EPOCHS": "1", "CONTINUATION_HARD_STOP_HOURS": "1",
            "HARD_FINISH_HOURS": "1",
        }), patch.object(torch, "compile", side_effect=lambda module, **options:
                         original_compile(module, backend="eager", **options)) as compiler, patch(
                             "torch.utils.tensorboard.SummaryWriter", return_value=Mock()):
            namespace = _training_namespace(tmp_path)
            compiler.assert_called_once_with(namespace["model"], dynamic=True)
            before = namespace["model"].track_seed.detach().clone()
            namespace["_record_split_metric"] = record_split_metric
            sources = notebook_cells()
            source = next(source for source in sources if "def prepare_batches(" in source)
            tree = _replace_assignments(source, {"EPOCHS": 2})
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "split_metrics":
                    node.decorator_list.insert(0, ast.Name(id="_record_split_metric", ctx=ast.Load()))
            execute(ast.fix_missing_locations(tree), namespace)
            continuation = next(source for source in sources if source.startswith(
                "# Major step: Reload the best model and continue training"))
            execute(continuation, namespace)
            assert [row["epoch"] for row in namespace["history"]] == [1, 2, 3]
            assert namespace["continuation_completed_epochs"] == 1
            assert metric_calls == ["train", "test", "test", "train", "test"]
            assert not torch.equal(before, namespace["model"].track_seed)
            assert all(math.isfinite(row["train_step_loss"]) for row in namespace["history"])
            state = torch.load(namespace["ckpt_path"], weights_only=False)
            best = torch.load(namespace["best_ckpt_path"], weights_only=False)
            assert state["epoch"] == 3
            assert best["selection_metric"] == "test_loss"
            assert best["epoch"] in {1, 2, 3}
            assert all(not key.startswith("_orig_mod.") for key in state["model_state"])
            restored = copy.deepcopy(namespace["model"])
            restored.load_state_dict(state["model_state"])
            restored.eval()
            namespace["model"].eval()
            prepared = namespace["prepared_batches_by_split"]["train"][0]
            batch = namespace["load_batch"](prepared)
            with torch.inference_mode():
                compiled_output = namespace["model_forward_batch"](batch)
                raw_output = restored(batch.x, batch.graph, batch.observation_nodes,
                                      batch.observation_to_track, batch.num_tracks,
                                      batch.rao_base_rates)
            for task in ("track_rao", "radar_mode"):
                torch.testing.assert_close(compiled_output["evidential"][task]["alpha"],
                                           raw_output["evidential"][task]["alpha"])
    finally:
        torch.set_num_threads(threads)


def test_epoch_loops_retain_allocator_cache_and_default_to_eight_tracks():
    sources = notebook_cells()
    training = next(source for source in sources if "def prepare_batches(" in source)
    continuation = next(source for source in sources if source.startswith(
        "# Major step: Reload the best model and continue training"))
    for source in (training, continuation):
        loops = [node for node in ast.parse(source).body if isinstance(node, ast.Try)]
        assert loops
        for loop in loops:
            for node in ast.walk(loop):
                if isinstance(node, ast.Call):
                    assert not (isinstance(node.func, ast.Name)
                                and node.func.id == "release_transient_memory")
                    assert not (isinstance(node.func, ast.Attribute)
                                and node.func.attr in {"collect", "empty_cache"})
    source = next(source for source in sources if "TRACKS_PER_BATCH =" in source)
    assignment = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.Assign) and any(
                          isinstance(target, ast.Name) and target.id == "TRACKS_PER_BATCH"
                          for target in node.targets))
    namespace = {"os": os}
    with patch.dict(os.environ):
        os.environ.pop("TRACKS_PER_BATCH", None)
        execute(ast.Module(body=[assignment], type_ignores=[]), namespace)
    assert namespace["TRACKS_PER_BATCH"] == 8
