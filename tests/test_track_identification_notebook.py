import ast
import json
from pathlib import Path

NOTEBOOK = Path("notebooks/Track_identification.ipynb")


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_source():
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in _notebook()["cells"]
        if cell.get("cell_type") == "code"
    )


def test_track_identification_notebook_code_cells_parse():
    for index, cell in enumerate(_notebook()["cells"]):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell {index}")


def test_track_notebook_has_one_joint_rao_head_and_observation_mode_head():
    source = _code_source()
    assert "class TrackAttentionPool" in source
    assert "class TrackRGCNHGTClassifier" in source
    assert "self.rao_head, self.rao_evidential_head = head(hidden_dim, num_rao_classes), head(hidden_dim, num_rao_classes)" in source
    assert "self.mode_head = head(2 * hidden_dim, num_mode_classes)" in source
    assert '"track_rao": rao_logits' in source
    assert '"radar_mode": mode_logits' in source
    assert "rao_mode_compatibility" in source


def test_track_notebook_balances_mode_loss_and_enforces_output_constraints():
    source = _code_source()
    assert "def mean_modes_per_track" in source
    assert "rao_target_by_series" in source
    assert "observation_track_index" in source
    assert "constrained_mode_logits" in source
    assert "RAO_VIOLATION_COUNT = 0" in source
    assert '"rao_invariant": True' in source


def test_dashboard_and_llm_keep_explanation_focused_on_track_rao():
    source = _code_source()
    assert '"track_rao_exact_accuracy"' in source
    assert '"mode_transition_f1"' in source
    assert (
        'final_outputs["evidential"]["radar_mode"]["uncertainty"]'
        "[test_observation_positions]" in source
    )
    assert "LLM_TRACK_INDEX" in source
    assert '"invariant_track_rao"' in source
    assert '"ordered_radar_mode_sequence"' not in source
    assert "Never imply that aircraft, radar, or operator changes" in source
    assert "radar mode is outside the scope of this explanation" in source
    assert "RAO identity, RAO evidence, RAO uncertainty" in source


def test_track_notebook_ingests_new_reports_by_proximity_and_links_kg_entities():
    source = _code_source()
    assert "demo_esm_observation_series_with_sightings_and_patterns.json" in source
    assert "report_proximity_by_pair" in source
    assert "observation_nodes_by_series" in source
    assert "for node_idx, meta in enumerate(node_meta)" in source
    assert "for report_idx, meta in enumerate(node_meta)" not in source
    assert "for claim_idx, meta in enumerate(node_meta)" not in source
    assert "proximity_by_observation" in source
    assert '"claim_asserts_kg_entity"' in source
    assert '"kg_entity_asserted_by_claim"' in source
    assert "kg_entity_node_indices" in source
    assert "report_kg_edges" in source


def test_candidate_recall_at_k_is_vectorised_and_visualised():
    source = _code_source()
    assert "MAX_KG_RETRIEVAL_CANDIDATES = 320" in source
    assert "MAX_KG_CANDIDATES = 18" in source
    assert '"max_kg_retrieval_candidates": MAX_KG_RETRIEVAL_CANDIDATES' in source
    assert "RECALL_K_START = 1" in source
    assert "first_correct_rank" in source
    assert "np.bincount" in source
    assert "np.cumsum" in source
    assert 'candidate_recall_at_k.png' in source
    assert 'axis.plot(recall_k_values, recall_at_k' in source
    assert "radars_without_aircraft" in source
    assert 'aircraft_by_radar.get(radar_id) or [None]' in source
    assert "def kg_property(entity_id, property_name)" in source
    assert 'kg_property(score.aircraft_id, "variant")' in source
    assert 'kg_property(score.mode_id, "name")' in source
    assert 'kg_property(score.radar_id, "name")' in source
    assert "if entity_id is None" in source
    assert 'node = kg_nodes[entity_id]' in source


def test_graph_artifact_omits_redundant_feature_row_dictionaries():
    source = _code_source()
    assert '"version": 6' in source
    assert '"feature_rows": feature_rows' not in source
    assert '"observation_segment_indices": observation_segment_indices' in source
    assert 'graph_outputs["observation_segment_indices"]' in source
    assert 'dict(zip(observation_node_indices, observation_segment_indices.tolist()))' in source


def test_track_notebook_prunes_and_collapses_claim_candidate_edges():
    source = _code_source()
    assert "CLAIM_CANDIDATE_MIN_ABS_CONTRIBUTION = 0.15" in source
    assert '"claim_candidate_min_abs_contribution": CLAIM_CANDIDATE_MIN_ABS_CONTRIBUTION' in source
    assert "COLLAPSE_RECIPROCAL_CLAIM_CANDIDATE_EDGES = True" in source
    assert 'if not COLLAPSE_RECIPROCAL_CLAIM_CANDIDATE_EDGES:' in source
    assert "for signed_claim_index, candidate_idx in zip(" in source
    assert "for direct_edge in claim_candidate_edges:" not in source


def test_track_notebook_uses_partitioned_edges_vector_pooling_and_track_batches():
    source = _code_source()
    assert "partition_edges_by_relation" in source
    assert "for relation_id, edges in enumerate(relation_edges)" in source
    assert "mask = edge_types == relation_id" not in source
    assert "segment_softmax" in source
    assert 'scatter_reduce_(0, observation_to_track[:, None].expand_as(observations)' in source
    assert "build_track_graph_batches" in source
    assert "TRACKS_PER_BATCH" in source
    assert "model_forward_batch" in source
    assert "local_x = _pin_for_cuda(X[batch.node_indices].to(dtype=MODEL_DTYPE).contiguous())" in source
    assert "def split_metrics(name)" in source
    assert 'outputs = model_forward(prepared_batches_by_split[name])' in source
    assert 'METRICS_INTERVAL = max(1, int(os.getenv("METRICS_INTERVAL", "10")))' in source
    assert 'tensor.to(DEVICE, non_blocking=True) for tensor in device_indices' in source
    assert "epoch_started_at = time.perf_counter()" in source
    assert 'torch.cuda.synchronize(DEVICE)' in source
    assert 'row["epoch_time_seconds"] = time.perf_counter() - epoch_started_at' in source
    assert "print(f\"Epoch {epoch:03d}/{EPOCHS}: {row['epoch_time_seconds']:.2f} s\")" in source
    assert "attentive_contributions = (weights.unsqueeze(-1) * observations).to(attentive.dtype)" in source
    assert "observations.to(means.dtype)" in source
    assert "observations.to(maxima.dtype)" in source
    assert "values.to(totals.dtype)" in source
    assert 'enabled=USE_AMP and DEVICE.type == "cuda"' in source


def test_track_notebook_densifies_only_present_features():
    source = _code_source()
    assert "densify_feature_rows(feature_rows, feature_names, dtype=X_NP_DTYPE)" in source
    assert "row.get(name, 0.0) for name in feature_names" not in source


def test_track_notebook_ports_vacuity_dissonance_plots_by_rao_outcome():
    source = _code_source()
    assert "def _singleton_dissonance" in source
    assert 'final_outputs["evidential"]["track_rao"]["belief"]' in source
    assert 'final_outputs["evidential"]["track_rao"]["uncertainty"]' in source
    assert '(rao_correct, "Correct identifications"' in source
    assert '(~rao_correct, "Incorrect identifications"' in source
    assert "rao_vacuity[selected], rao_dissonance[selected]" in source
    assert "track_rao_vacuity_vs_dissonance_by_outcome" in source


def test_track_notebook_rao_loss_rewards_partially_correct_components():
    source = _code_source()
    assert "RAO_AIRCRAFT_LOSS_WEIGHT = 1.0" in source
    assert "RAO_RADAR_LOSS_WEIGHT = 1.0" in source
    assert "RAO_OPERATOR_LOSS_WEIGHT = 1.0" in source
    assert "RAO_COMPLETE_LOSS_WEIGHT = 1.0" in source
    assert "RAO_COMPONENT_LOSS_WEIGHT = 1.0" in source
    assert "def component_weighted_rao_nll" in source
    assert "def component_weighted_rao_edl" in source
    assert "matching_worlds" in source
    assert "torch.logsumexp" in source
    assert "def combined_rao_nll" in source
    assert "def combined_rao_edl" in source
    assert "F.cross_entropy(logits, labels, reduction=\"none\")" in source
    assert "expected_dirichlet_ce(alpha, labels)" in source
    assert 'rao_ce = combined_rao_nll(outputs["track_rao"], rao_labels).mean()' in source
    assert 'rao_edl = combined_rao_edl(rao_alpha, rao_labels).mean()' in source
