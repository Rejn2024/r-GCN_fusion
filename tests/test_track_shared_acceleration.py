"""Graph-cache, numerical-gradient and bounded-staging regression checks."""
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from rgcn_fusion.track_shared_acceleration import (
    AcceleratedSharedCandidateHGTClassifier, compact_shared_batch_features,
    compile_packed_projections, packed_l1,
)
from rgcn_fusion.track_shared_cache import load_shared_graph
from rgcn_fusion.track_shared_graph import build_shared_evidence_graph
from rgcn_fusion.track_shared_hgt import SharedCandidateHGTClassifier
from rgcn_fusion.track_shared_training import iter_shared_batches, load_shared_batch
from test_track_shared_candidate import batches, fixture_inputs


FINGERPRINTS = {"data.json": "data-sha", "kg.json": "kg-sha"}
SETTINGS = {"MAX_KG_CANDIDATES": 18, "MAX_KG_RETRIEVAL_CANDIDATES": 320,
            "CLAIM_CANDIDATE_MIN_ABS_CONTRIBUTION": .15, "INTELLIGENCE_ROUTING": "selective"}


@pytest.fixture
def saved_graph(tmp_path):
    graph = build_shared_evidence_graph(*fixture_inputs(), tmp_path)
    path = tmp_path / "graph_construction_outputs.pt"
    metadata = torch.load(path, weights_only=False)
    metadata["cache_context"] = {"input_fingerprints": FINGERPRINTS, "graph_settings": SETTINGS}
    torch.save(metadata, path)
    return graph, tmp_path


def reload_graph(path, **kwargs):
    return load_shared_graph(path, input_fingerprints=kwargs.get("fingerprints", FINGERPRINTS),
                             graph_settings=kwargs.get("settings", SETTINGS))


def test_cache_roundtrip_and_copy_on_write(saved_graph):
    expected, path = saved_graph
    files = list(path.glob("*.pt")) + list(path.glob("*.memmap"))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    actual = reload_graph(path)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value
    actual["X"][0, 0] = 9
    actual["edge_features"][0, 0] = 9
    actual["edge_index"][0, 0] = 9
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files} == before


def test_cache_rejects_stale_or_incomplete_inputs(saved_graph):
    _, path = saved_graph
    with pytest.raises(ValueError, match="different dataset"):
        reload_graph(path, fingerprints={})
    with pytest.raises(ValueError, match="different graph settings"):
        reload_graph(path, settings={})
    with (path / "incidences.float32.memmap").open("r+b") as stream:
        stream.truncate(4)
    with pytest.raises(ValueError, match="file size"):
        reload_graph(path)


def test_cache_rejects_missing_provenance(saved_graph):
    _, path = saved_graph
    manifest_path = path / "graph_construction_outputs.pt"
    metadata = torch.load(manifest_path, weights_only=False)
    metadata.pop("cache_context")
    torch.save(metadata, manifest_path)
    with pytest.raises(ValueError, match="provenance"):
        reload_graph(path)


def load_fixture_batch(graph, prepared, device="cpu"):
    return load_shared_batch(prepared, graph["X"], torch.tensor([0, 1]),
                             torch.tensor([0, 1, 0, 1]), torch.tensor([[.4, .6], [.7, .3]]),
                             device=device, dtype=torch.float32)


def model_for(graph, cls):
    return cls(graph["X"].size(1), 8, len(graph["relation_metadata"]), 2, 2, torch.ones(2, 2),
               relation_metadata=graph["relation_metadata"], edge_dim=len(graph["edge_feature_names"]),
               num_kg_entities=graph["num_kg_entities"], num_hgt_layers=2,
               num_heads=2, dropout=0, task_head_hidden_dim=8)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))])
def test_packed_model_output_input_and_parameter_gradient_parity(saved_graph, compact, amp, device):
    graph, _ = saved_graph
    torch.manual_seed(29)
    original = batches(graph)["train"][0]
    prepared = compact_shared_batch_features(original, graph["edge_feature_names"]) if compact else original
    if compact:
        assert prepared.graph.num_feature_rows < original.graph.num_feature_rows
        assert prepared.graph.edge_features is original.graph.edge_features
        torch.testing.assert_close(prepared.graph.edge_index, original.graph.edge_index)
        assert prepared.graph.relation_sizes == original.graph.relation_sizes
        with pytest.raises(ValueError, match="exactly once"):
            compact_shared_batch_features(prepared, graph["edge_feature_names"])
    reference = model_for(graph, SharedCandidateHGTClassifier).to(device)
    packed = model_for(graph, AcceleratedSharedCandidateHGTClassifier).to(device)
    packed.load_state_dict(reference.state_dict(), strict=True)
    assert set(packed.state_dict()) == set(reference.state_dict())
    outputs, inputs = [], []
    for model, batch in ((reference, original), (packed, prepared)):
        loaded = load_fixture_batch(graph, batch, device)
        loaded.x.requires_grad_()
        with torch.autocast(device, dtype=torch.bfloat16 if device == "cpu" else torch.float16, enabled=amp):
            output = model(loaded.x, loaded.graph, loaded.observation_nodes,
                           loaded.observation_to_track, loaded.num_tracks, loaded.rao_base_rates)
            # Nontrivial probes expose all output dimensions and gradients.
            loss = sum((opinion["alpha"] * torch.tensor([.3, 1.7], device=device)).square().sum()
                       for opinion in output["evidential"].values())
        loss.backward()
        outputs.append(output)
        inputs.append(loaded.x.grad)
    tolerance = {"rtol": .04, "atol": .02} if amp else {"rtol": 3e-4, "atol": 3e-5}
    for task in outputs[0]["evidential"]:
        torch.testing.assert_close(outputs[0]["evidential"][task]["alpha"],
                                   outputs[1]["evidential"][task]["alpha"], **tolerance)
    torch.testing.assert_close(inputs[0], inputs[1], **tolerance)
    for (name, p), (other_name, q) in zip(reference.named_parameters(), packed.named_parameters()):
        assert name == other_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, **tolerance, msg=lambda msg: f"{name}: {msg}")


def test_packed_l1_gradient_and_adam_update():
    torch.manual_seed(4)
    reference = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    packed = copy.deepcopy(reference)
    for model, fast in ((reference, False), (packed, True)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.0001)
        penalty = packed_l1(tuple(model.parameters())) if fast else sum(p.abs().sum() for p in model.parameters())
        loss = model(torch.ones(2, 3)).square().mean() + .0005 * penalty / 7
        loss.backward()
        optimizer.step()
    for p, q in zip(reference.parameters(), packed.parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.parametrize("scope", ["projections", "model"])
def test_compiled_packed_model_dynamic_capture(saved_graph, scope):
    graph, _ = saved_graph
    prepared = compact_shared_batch_features(batches(graph)["train"][0], graph["edge_feature_names"])
    batch = load_fixture_batch(graph, prepared)
    model = model_for(graph, AcceleratedSharedCandidateHGTClassifier)
    compiled = (torch.compile(model, backend="eager", dynamic=True) if scope == "model"
                else compile_packed_projections(model, backend="eager", dynamic=True))
    output = compiled(batch.x, batch.graph, batch.observation_nodes,
                      batch.observation_to_track, batch.num_tracks, batch.rao_base_rates)
    output["evidential"]["track_rao"]["alpha"].sum().backward()
    assert model.track_seed.grad is not None
    assert all(not key.startswith("_orig_mod.") for key in model.state_dict())


@pytest.mark.parametrize("prefetch", [False, True])
def test_cpu_iterator_fallback(saved_graph, prefetch):
    graph, _ = saved_graph
    prepared = compact_shared_batch_features(batches(graph)["train"][0], graph["edge_feature_names"])
    with iter_shared_batches([prepared] * 3, graph["X"], torch.tensor([0, 1]), torch.tensor([0, 1, 0, 1]),
                             torch.full((2, 2), .5), device="cpu", dtype=torch.float32, prefetch=prefetch) as iterator:
        for loaded in iterator:
            torch.testing.assert_close(loaded.graph.edge_features,
                                       prepared.graph.edge_features[prepared.graph.feature_rows])
            assert loaded.graph.feature_rows is None
            assert loaded.graph.feature_groups is prepared.graph.feature_groups or len(loaded.graph.feature_groups) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_prefetch_parity_reuse_and_early_exit(saved_graph):
    graph, _ = saved_graph
    prepared = compact_shared_batch_features(batches(graph)["train"][0], graph["edge_feature_names"])
    smaller = [compact_shared_batch_features(batch, graph["edge_feature_names"])
               for batch in batches(graph, batch_size=1)["train"]]
    ordered = [prepared, *smaller] * 3
    for stop_early in (False, True):
        with iter_shared_batches(ordered, graph["X"], torch.tensor([0, 1]), torch.tensor([0, 1, 0, 1]),
                                 torch.tensor([[.4, .6], [.7, .3]]), device="cuda", dtype=torch.float32,
                                 prefetch=True) as iterator:
            for i, batch in enumerate(iterator):
                expected = load_fixture_batch(graph, ordered[i], "cuda")
                torch.testing.assert_close(batch.x, expected.x)
                torch.testing.assert_close(batch.graph.edge_features, expected.graph.edge_features)
                torch.testing.assert_close(batch.rao_labels, expected.rao_labels)
                for cache, wanted in zip(batch.graph.source_caches, expected.graph.source_caches):
                    if cache is not None:
                        for value, target in zip(cache, wanted):
                            torch.testing.assert_close(value, target)
                if stop_early and i == 2:
                    break
                del batch
        torch.cuda.synchronize()


def test_accelerated_notebook_cache_only_and_clean_cells():
    import nbformat
    path = Path(__file__).parents[1] / "notebooks/track_shared_candidate_hgt_accelerated.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert "build_shared_evidence_graph" not in code
    assert "load_shared_graph(" in code and '"cache_context"' not in code
    assert 'SOURCE_ARTIFACT_DIR' in code
    assert '"track_shared_candidate_hgt_accelerated"' in code
    assert 'os.getenv("METRICS_INTERVAL", "10")' in code
    assert code.count("L1_LAMBDA * l1_penalty()") == 2
    assert 'fused=DEVICE.type == "cuda"' in code
    assert "call" in notebook.cells[1].source and "vcvars" in notebook.cells[1].source
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"accelerated cell {i}", "exec")
            assert not cell.outputs and cell.execution_count is None


def test_cached_notebook_training_continuation_and_export(tmp_path, monkeypatch):
    """Run every code cell against a saved 12-track fixture, including both loops."""
    import mlflow
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from plotly.basedatatypes import BaseFigure
    from types import SimpleNamespace
    from unittest.mock import Mock
    from rgcn_fusion.data_privacy import strip_ground_truth

    templates, kg, candidate_rows = fixture_inputs()
    # Add a second valid world so the real evidential objective is nontrivial.
    second_kg = json.loads(json.dumps(kg).replace(":a", ":b").replace("Mode A", "Mode B")
                           .replace("Radar A", "Radar B").replace("Family A", "Family B")
                           .replace('"A"', '"B"').replace("Testland", "Otherland"))
    kg["nodes"].extend(second_kg["nodes"])
    kg["edges"].extend(second_kg["edges"])
    second_candidate = json.loads(json.dumps(candidate_rows[0]).replace(":a", ":b")
                                  .replace("Radar A", "Radar B").replace('"A"', '"B"')
                                  .replace("Testland", "Otherland"))
    candidate_rows.append(second_candidate)
    series = []
    for i in range(12):
        row = json.loads(json.dumps(templates[0]).replace("track-a", f"series-{i:02d}"))
        variant, radar, operator, mode = (("A", "Radar A", "Testland", "Mode A") if i % 2 == 0 else
                                          ("B", "Radar B", "Otherland", "Mode B"))
        if i >= 10:
            variant = "J-40"
        row["ground_truth_track_label"] = {"aircraft_variant": variant, "radar_type": radar,
                                            "operator_country": operator}
        for observation in row["observations"]:
            observation["ground_truth_label"] = {"aircraft_variant": variant, "radar": radar,
                                                   "operator_country": operator, "mode": mode}
        series.append(row)
    generated = tmp_path / "generated"
    generated.mkdir()
    data = generated / "demo_esm_observation_series_open_set_ufo.json"
    kg_file = generated / "aircraft_radar_kg.json"
    data.write_text(json.dumps({"observation_series": series}))
    kg_file.write_text(json.dumps(kg))
    cache = tmp_path / "artifacts/track_shared_candidate_hgt"
    build_shared_evidence_graph(strip_ground_truth(series), kg, candidate_rows, cache)
    metadata_path = cache / "graph_construction_outputs.pt"
    metadata = torch.load(metadata_path, weights_only=False)
    metadata["cache_context"] = {
        "input_fingerprints": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (data, kg_file)},
        "graph_settings": SETTINGS,
    }
    torch.save(metadata, metadata_path)
    cache_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.iterdir() if p.is_file()}
    monkeypatch.chdir(tmp_path)
    for name, value in {"EPOCHS": "2", "CONTINUATION_EPOCHS": "2", "METRICS_INTERVAL": "2",
                        "COMPILE_MODEL": "0", "PREFETCH_BATCHES": "0", "TRACKS_PER_BATCH": "2",
                        "RUN_LOCAL_LLM": "0", "PLOTLY_STATIC_EXPORT": "0"}.items():
        monkeypatch.setenv(name, value)
    for name in ("set_tracking_uri", "set_experiment", "set_tags", "set_tag", "log_params",
                 "log_metrics", "log_dict", "log_artifact", "end_run"):
        monkeypatch.setattr(mlflow, name, Mock())
    monkeypatch.setattr(mlflow, "start_run", Mock(return_value=SimpleNamespace(info=SimpleNamespace(run_id="fixture"))))
    for cls, methods in ((Figure, ("savefig",)), (BaseFigure, ("write_html", "write_image", "show"))):
        for name in methods:
            monkeypatch.setattr(cls, name, Mock())
    monkeypatch.setattr(plt, "show", Mock())
    # A cache-only consumer must never fall back to rebuilding.
    monkeypatch.setattr("rgcn_fusion.track_shared_graph.build_shared_evidence_graph",
                        Mock(side_effect=AssertionError("Notebook attempted to rebuild the graph")))
    path = Path(__file__).parents[1] / "notebooks/track_shared_candidate_hgt_accelerated.ipynb"
    namespace = {"__name__": "__main__"}
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for i, cell in enumerate(json.loads(path.read_text())["cells"]):
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), f"accelerated cell {i}", "exec"), namespace)
        assert len(namespace["history"]) == 4
        assert namespace["continuation_completed_epochs"] == 2
        output_dir = tmp_path / "artifacts/track_shared_candidate_hgt_accelerated"
        assert (output_dir / "best_track_model.pt").is_file()
        records = json.loads((output_dir / "track_predictions.json").read_text())
        assert len(records) == 12
        assert all(len(row["mode_sequence"]) == 2 for row in records)
        summary = json.loads((output_dir / "training_summary.json").read_text())
        assert set(summary["metrics"]) == {"train", "test", "val"}
        assert summary["open_set_evaluation"]["track_count"] == 2
        assert namespace["_feature_rows_after"] < namespace["_feature_rows_before"]
        assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cache_hashes} == cache_hashes
    finally:
        plt.close("all")
        for handler in namespace.get("NOTEBOOK_LOGGER", SimpleNamespace(handlers=[])).handlers:
            handler.close()
        torch.set_num_threads(threads)
