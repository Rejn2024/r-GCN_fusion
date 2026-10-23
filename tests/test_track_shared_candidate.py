"""Semantic checks for routing, incidence context, track isolation and edge learning."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rgcn_fusion.track_shared_graph import (
    NODE_TYPE_INDEX, SharedEvidenceGraph, build_shared_batches_by_split,
    build_shared_evidence_graph, intelligence_scope,
)
from rgcn_fusion.track_shared_hgt import EdgeConditionedHGTLayer, SharedCandidateHGTClassifier
from rgcn_fusion.track_shared_training import iter_shared_batches, load_shared_batch
from rgcn_fusion.track_star_loss import batch_hierarchical_evidential_loss


def fixture_inputs():
    kg = {"nodes": [
        {"id": "mode:a", "label": "RadarMode", "properties": {"name": "Mode A"}},
        {"id": "radar:a", "label": "Radar", "properties": {"name": "Radar A"}},
        {"id": "aircraft:a", "label": "AircraftVariant", "properties": {"variant": "A", "family": "Family A"}},
        {"id": "family:a", "label": "AircraftFamily", "properties": {"name": "Family A"}},
        {"id": "operator:a", "label": "Operator", "properties": {"name": "Testland"}},
    ], "edges": [
        {"source": "radar:a", "target": "mode:a", "relation": "HAS_MODE"},
        {"source": "aircraft:a", "target": "radar:a", "relation": "USES_RADAR"},
        {"source": "aircraft:a", "target": "family:a", "relation": "VARIANT_OF"},
        {"source": "operator:a", "target": "aircraft:a", "relation": "OPERATES"},
    ]}
    candidates = [{"mode_id": "mode:a", "mode_props": {
        "waveform": "pulse_doppler", "scan_type": "sector",
        "centre_frequency_min_ghz": 9.4, "centre_frequency_max_ghz": 9.8,
    }, "radar_id": "radar:a", "radar_props": {"name": "Radar A"},
        "aircraft_id": "aircraft:a", "aircraft_props": {"max_speed_mach": 2., "service_ceiling_m": 15000},
        "aircraft_uses_radar": True, "operator": "Testland"}]
    series = []
    for sid in ("track-a", "track-b"):
        observations = [{"series_id": sid, "observation_id": f"{sid}-{i}", "sequence_index": i,
            "timestamp_iso8601": f"2026-01-01T0{i}:00:00Z",
            "esm_radar_parameters": {"waveform": "pulse_doppler", "scan_type": "sector",
                                     "measured_centre_frequency_ghz": 9.6 + i * 2},
            "estimated_emitter_location": {"estimated_latitude_deg": 51.5,
                                           "estimated_longitude_deg": -.1, "area": "test"},
            "approximate_kinematics": {"speed_mach": 1.2, "altitude_m": 10000}}
            for i in range(2)]
        series.append({"series_id": sid, "observations": observations, "intelligence_reports": [
            {"report_id": sid + "-local", "report_type": "sighting_report", "credibility_score": .8,
             "sighting": {"last_observed_at": "2026-01-01T00:00:00Z",
                          "location": {"latitude_deg": 51.5, "longitude_deg": -.1}},
             "claims": [{"claim_id": sid + "-radar", "claim_type": "radar_type",
                         "object_id": "radar:a", "kg_entity_id": "radar:a", "stance": "supports",
                         "claim_confidence": 1.}]},
            {"report_id": sid + "-broad", "report_type": "theatre_aircraft_report", "credibility_score": .8,
             "valid_for_observation_ids": [o["observation_id"] for o in observations],
             "claims": [{"claim_id": sid + "-presence", "claim_type": "theatre_aircraft_presence",
                         "object_ids": ["aircraft:a"], "stance": "supports", "claim_confidence": .9}]},
        ]})
    return series, kg, candidates


@pytest.fixture
def graph_data(tmp_path):
    series, kg, candidates = fixture_inputs()
    graph = build_shared_evidence_graph(series, kg, candidates, tmp_path)
    return graph


def batches(graph, selected=None, batch_size=2):
    meta = graph["node_meta"]
    owner = {"track-a": 0, "track-b": 1}
    membership = torch.tensor([owner.get(m.get("series_id"), -1) for m in meta])
    observations = torch.tensor(graph["observation_node_indices"])
    return build_shared_batches_by_split(
        edge_index=graph["edge_index"], edge_types=graph["edge_types"],
        node_track_index=membership, observation_nodes=observations,
        observation_track_index=membership[observations],
        selected_tracks_by_split=selected or {"train": [0, 1]},
        tracks_per_batch=batch_size, graph_data=graph)


def relation_edges(graph, relation):
    matches = [i for i, r in enumerate(graph["relation_metadata"]) if r[1] == relation]
    mask = torch.isin(graph["edge_types"], torch.tensor(matches, dtype=graph["edge_types"].dtype))
    return graph["edge_index"][:, mask], graph["edge_feature_indices"][mask]


def test_sharing_reduces_nodes_but_keeps_observation_match_context(graph_data):
    graph = graph_data
    candidates = [(i, m) for i, m in enumerate(graph["node_meta"]) if m["node_kind"] == "candidate"]
    assert len(candidates) == 2  # 2 tracks x 1 hypothesis, not 4 observation candidates.
    assert candidates[0][1]["candidate_key"] == candidates[1][1]["candidate_key"]
    assert candidates[0][1]["series_id"] != candidates[1][1]["series_id"]
    edges, refs = relation_edges(graph, "has_candidate")
    assert edges.shape[1] == 4
    assert len(torch.unique(refs)) == 4
    scores = graph["edge_features"][refs.long()]
    assert not torch.equal(scores[0], scores[1])
    assert "candidate_intel_support_score" in graph["edge_feature_names"]
    support = graph["edge_feature_names"].index("retained_supporting_claim_count")
    assert scores[0, support] > scores[1, support]  # Sighting applies only at first observation.
    reverse, reverse_refs = relation_edges(graph, "candidate_for")
    torch.testing.assert_close(reverse, edges.flip(0))
    torch.testing.assert_close(reverse_refs, refs)
    node_kind = graph["feature_names"].index("node_kind_candidate")
    for i, _ in candidates:
        assert torch.count_nonzero(graph["X"][i]) == 1
        assert graph["X"][i, node_kind] > 0
    assert not any("claim_supports_candidate" == r[1] for r in graph["relation_metadata"])
    assert relation_edges(graph, "candidate_mode")[0].shape[1] == 2


def test_broad_to_tracks_local_to_observations_and_contains_once(graph_data):
    graph = graph_data
    meta = graph["node_meta"]
    for relation in ("report_for_track", "claim_for_track"):
        edges, refs = relation_edges(graph, relation)
        assert edges.shape[1] == 2
        assert all(meta[i]["node_kind"] == "track" for i in edges[1])
        assert all("broad" in meta[i].get("report_id", "") for i in edges[0])
        coverage = graph["edge_feature_names"].index("coverage_fraction")
        torch.testing.assert_close(graph["edge_features"][refs.long(), coverage], torch.full((2,), np.log(2), dtype=torch.float32))
    for relation in ("report_for_observation", "claim_for_observation"):
        edges, _ = relation_edges(graph, relation)
        assert edges.shape[1] == 2
        assert all(meta[i]["observation_id"].endswith("-0") for i in edges[1])
    contains, _ = relation_edges(graph, "report_contains_claim")
    assert contains.shape[1] == 4
    assert torch.unique(contains, dim=1).shape[1] == 4
    assert relation_edges(graph, "claim_asserts_kg_entity")[0].shape[1] == 2


def test_conservative_scope_and_partial_inventory_coverage(tmp_path):
    stable = {"claim_type": "aircraft_variant"}
    assert intelligence_scope({"report_type": "identity_report"}, stable, 2, 2) == "track"
    assert intelligence_scope({"report_type": "identity_report"}, stable, 1, 2) == "observation"
    assert intelligence_scope({"report_type": "sighting_report"}, stable, 2, 2) == "observation"
    assert intelligence_scope({"report_type": "pattern_of_life_report"}, stable, 2, 2) == "observation"
    assert intelligence_scope({"report_type": "identity_report"}, {"claim_type": "location"}, 2, 2) == "observation"
    assert intelligence_scope({"report_type": "theatre_aircraft_report"}, None, 0, 2) == "unlinked"
    series, kg, candidates = fixture_inputs()
    series[0]["intelligence_reports"][1]["valid_for_observation_ids"] = ["track-a-0"]
    graph = build_shared_evidence_graph(series, kg, candidates, tmp_path)
    audit = [json.loads(line) for line in (tmp_path / "intelligence_routing.jsonl").read_text().splitlines()]
    row = next(r for r in audit if r["kind"] == "intelligence_report" and r["series_id"] == "track-a" and r["scope"] == "track")
    assert row["coverage"] == .5
    assert row["observation_ids"] == ["track-a-0"]


def test_full_track_ablation_keeps_local_applicability_and_candidate_context(tmp_path):
    series, kg, candidates = fixture_inputs()
    graph = build_shared_evidence_graph(series, kg, candidates, tmp_path, intelligence_routing="track")
    assert relation_edges(graph, "report_for_observation")[0].shape[1] == 0
    assert relation_edges(graph, "claim_for_observation")[0].shape[1] == 0
    assert relation_edges(graph, "report_for_track")[0].shape[1] == 4
    audit = [json.loads(line) for line in (tmp_path / "intelligence_routing.jsonl").read_text().splitlines()]
    assert any(r["coverage"] == .5 and r["observation_ids"] == ["track-a-0"] for r in audit)
    _, refs = relation_edges(graph, "has_candidate")
    scores = graph["edge_features"][refs.long()]
    support = graph["edge_feature_names"].index("retained_supporting_claim_count")
    assert scores[0, support] > scores[1, support]


def test_batches_keep_candidates_owned_and_preserve_edge_attributes(graph_data):
    graph = graph_data
    split = batches(graph, {"train": [0], "val": [1]}, 1)
    for name, sid in (("train", "track-a"), ("val", "track-b")):
        batch = split[name][0]
        staged = batch.graph.to("cpu")
        assert batch.num_observations == 2 and batch.num_tracks == 1
        for i in batch.node_indices:
            assert graph["node_meta"][i].get("series_id", sid) == sid
        # Every reconstructed local (relation, edge, attributes) exists globally.
        offset = 0
        for rel, size in enumerate(batch.graph.relation_sizes):
            local_edges = batch.graph.edge_index[:, offset:offset + size]
            global_edges = batch.node_indices[local_edges]
            expected = graph["edge_index"][:, graph["edge_types"] == rel].long()
            for e, pair in enumerate(global_edges.t()):
                match = torch.nonzero((expected == pair[:, None]).all(0)).flatten()
                assert match.numel() == 1
                ref = batch.graph.feature_indices[offset + e]
                original_refs = graph["edge_feature_indices"][graph["edge_types"] == rel]
                original_ref = original_refs[match[0]]
                if ref >= 0:
                    torch.testing.assert_close(staged.edge_features[ref], graph["edge_features"][original_ref])
                else:
                    assert original_ref == -1
            offset += size


def test_edge_attributes_drive_attention_values_and_gradients():
    torch.manual_seed(1)
    layer = EdgeConditionedHGTLayer(8, [("observation", "has_candidate", "candidate")], 2, 2, 0)
    graph = SharedEvidenceGraph(3, 0, torch.tensor([[0, 1], [2, 2]]), (2,), torch.tensor([0, 1]),
                               torch.tensor([[.2, .8], [.9, .1]], requires_grad=True),
                               torch.tensor([0, 0, 1]), torch.full((3,), -1), torch.empty(0, dtype=torch.long))
    x = torch.randn(3, 8)
    output = layer(x, graph)
    probe = torch.arange(1., 9.)
    (output[2] * probe).sum().backward()
    assert graph.edge_features.grad.abs().max() > 0
    assert layer.edge_key.weight.grad.abs().max() > 0
    assert layer.edge_value.weight.grad.abs().max() > 0
    other = copy.copy(graph)
    object.__setattr__(other, "edge_features", graph.edge_features.detach().flip(0))
    assert not torch.allclose(output[2], layer(x, other)[2])


def test_full_training_loss_checkpointing_and_output_contract(graph_data):
    torch.manual_seed(23)
    torch.set_num_threads(1)
    graph = graph_data
    prepared = batches(graph)["train"][0]
    targets, modes, priors = torch.tensor([0, 1]), torch.tensor([0, 1, 0, 1]), torch.tensor([[.4, .6], [.7, .3]])
    loaded = load_shared_batch(prepared, graph["X"], targets, modes, priors, device="cpu", dtype=torch.float32)
    model = SharedCandidateHGTClassifier(graph["X"].size(1), 8, len(graph["relation_metadata"]), 2, 2,
        torch.ones(2, 2), relation_metadata=graph["relation_metadata"],
        edge_dim=graph["edge_features"].size(1), num_kg_entities=graph["num_kg_entities"],
        num_heads=2, dropout=0, task_head_hidden_dim=8)
    checkpointed = copy.deepcopy(model)
    checkpointed.gradient_checkpointing = True
    results = []
    for active in (model, checkpointed):
        active.train()
        output = active(loaded.x, loaded.graph, loaded.observation_nodes,
                        loaded.observation_to_track, loaded.num_tracks, loaded.rao_base_rates)
        loss, _ = batch_hierarchical_evidential_loss(output, rao_labels=loaded.rao_labels, mode_labels=loaded.mode_labels,
            observation_to_track=loaded.observation_to_track, observation_counts=loaded.observation_counts,
            component_ids=torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]), component_weights=torch.ones(4) / 4,
            rao_weights=torch.tensor([.9, .1]), kl_coefficient=.005)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in active.parameters())
        results.append(output)
    for name, p in model.named_parameters():
        q = dict(checkpointed.named_parameters())[name]
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
    for task, rows in (("track_rao", 2), ("radar_mode", 4)):
        opinion = results[0]["evidential"][task]
        assert opinion["probabilities"].shape == (rows, 2)
        torch.testing.assert_close(opinion["belief"].sum(-1, keepdim=True) + opinion["uncertainty"], torch.ones(rows, 1))
        torch.testing.assert_close(opinion["alpha"], results[1]["evidential"][task]["alpha"])
    before = model.hgt_layers[0].edge_value.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=.01).step()
    assert not torch.equal(before, model.hgt_layers[0].edge_value.weight)
    with iter_shared_batches([prepared], graph["X"], targets, modes, priors, device="cpu", dtype=torch.float32) as iterator:
        assert next(iterator).num_observations == 4


def test_ground_truth_rejected(tmp_path):
    series, kg, candidates = fixture_inputs()
    series[0]["ground_truth_track_label"] = {"aircraft_variant": "A"}
    with pytest.raises(ValueError, match="ground-truth-free"):
        build_shared_evidence_graph(series, kg, candidates, tmp_path)


@pytest.mark.parametrize("routing", ["selective", "track"])
def test_spawn_parallel_graph_is_identical_to_serial(tmp_path, routing):
    series, kg, candidates = fixture_inputs()
    # Make track sizes unequal and preserve arbitrary source ordering.
    series[0]["observations"] = series[0]["observations"][:1]
    series.reverse()
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    expected = build_shared_evidence_graph(series, kg, candidates, serial_dir,
        intelligence_routing=routing)
    actual = build_shared_evidence_graph(series, kg, candidates, parallel_dir,
        intelligence_routing=routing, num_workers=2)
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            assert torch.equal(actual[key], expected[key]), key
        else:
            assert actual[key] == expected[key], key
    for name in ("nodes.float32.memmap", "incidences.float32.memmap", "intelligence_routing.jsonl"):
        assert (serial_dir / name).read_bytes() == (parallel_dir / name).read_bytes()
    for name in ("graph_construction_outputs.pt", "relational_graph_outputs.pt"):
        expected_artifact = torch.load(serial_dir / name, weights_only=False)
        actual_artifact = torch.load(parallel_dir / name, weights_only=False)
        for key, value in expected_artifact.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(actual_artifact[key], value)
            else:
                assert actual_artifact[key] == value
    assert not list(parallel_dir.glob("shared-fragments-*"))


def test_parallel_worker_failure_cleans_spool_and_does_not_save_manifest(tmp_path):
    series, kg, candidates = fixture_inputs()
    del series[1]["observations"][0]["sequence_index"]
    with pytest.raises(KeyError, match="sequence_index"):
        build_shared_evidence_graph(series, kg, candidates, tmp_path, num_workers=2)
    assert not list(tmp_path.glob("shared-fragments-*"))
    assert not (tmp_path / "graph_construction_outputs.pt").exists()


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_invalid_graph_worker_count_rejected_before_writes(tmp_path, workers):
    with pytest.raises(ValueError, match="num_workers"):
        build_shared_evidence_graph(*fixture_inputs(), tmp_path, num_workers=workers)
    assert not list(tmp_path.iterdir())


def test_feature_spool_preserves_scalar_transform_across_chunks(tmp_path):
    from rgcn_fusion.track_shared_graph import _FeatureSpool
    spool = _FeatureSpool(tmp_path, "test")
    rows = [{"tiny": -1.e-35, "wide": 1.e50, "rounding": 1.0000000596}, {},
            {"negative": -123.456789, "zero": 0.}] * 5462
    for row in rows:
        spool.append(row)
    actual, names = spool.finish()
    expected = np.array([[np.sign(row.get(k, 0.)) * np.log1p(abs(row.get(k, 0.)))
                          for k in names] for row in rows], dtype=np.float32)
    assert np.array_equal(actual.numpy(), expected)


def test_fragment_submission_is_bounded(monkeypatch, tmp_path):
    import rgcn_fusion.track_shared_graph as module
    submitted, loaded = [], []

    class Future:
        def __init__(self, task):
            self.task = task

        def result(self):
            loaded.append(self.task[0])
            return self.task

        def cancel(self):
            pass

    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, task):
            submitted.append(task[0])
            assert len(submitted) - len(loaded) <= 4
            return Future(task)

    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(module, "load_spooled_series_fragment", lambda descriptor: descriptor)
    assert list(module._series_fragments(list(range(20)), {}, tmp_path, 2, "spawn")) == list(enumerate(range(20)))


def test_routing_reuses_proximity_for_report_and_claims(monkeypatch, tmp_path):
    import rgcn_fusion.track_shared_graph as module
    original = module.report_observation_proximity
    calls = []

    def measured(report, observation):
        calls.append((report["report_id"], observation["observation_id"]))
        return original(report, observation)

    monkeypatch.setattr(module, "report_observation_proximity", measured)
    build_shared_evidence_graph(*fixture_inputs(), tmp_path)
    assert len(calls) == 2  # Two local report/observation pairs, each with a claim.
    assert len(set(calls)) == len(calls)


def test_mixed_precision_forward_and_backward(graph_data):
    graph = graph_data
    prepared = batches(graph)["train"][0]
    loaded = load_shared_batch(prepared, graph["X"], torch.tensor([0, 1]), torch.tensor([0, 1, 0, 1]),
                               torch.full((2, 2), .5), device="cpu", dtype=torch.float32)
    model = SharedCandidateHGTClassifier(graph["X"].size(1), 8, len(graph["relation_metadata"]), 2, 2,
        torch.ones(2, 2), relation_metadata=graph["relation_metadata"], edge_dim=graph["edge_features"].size(1),
        num_kg_entities=graph["num_kg_entities"], num_heads=2, dropout=0)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(loaded.x, loaded.graph, loaded.observation_nodes, loaded.observation_to_track,
                       loaded.num_tracks, loaded.rao_base_rates)
    loss = result["evidential"]["track_rao"]["alpha"].sum() + result["evidential"]["radar_mode"]["alpha"].sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_new_notebook_same_inputs_and_independent_training():
    import nbformat
    path = Path(__file__).resolve().parents[1] / "notebooks/track_shared_candidate_hgt.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    assert '"demo_esm_observation_series_open_set_ufo.json"' in code
    assert '"aircraft_radar_kg.json"' in code
    assert 'ARTIFACT_DIR = ROOT / "artifacts" / "track_shared_candidate_hgt"' in code
    assert "build_shared_evidence_graph(\n    inference_records" in code
    assert "SOURCE_ARTIFACT_DIR" not in code
    assert "from rgcn_fusion.track_shared_training" in code
    assert "relation_metadata=relation_metadata" in code
    for i, c in enumerate(notebook.cells):
        if c.cell_type == "code":
            compile(c.source, f"notebook cell {i}", "exec")
            # The reference notebook includes the user's recorded experiment.
            # The accelerated derivative checks clean outputs independently.
