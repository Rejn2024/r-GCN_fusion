"""Track-local candidate sharing and selective intelligence routing.

Re-score the existing inference dataset: legacy artifacts do not identify their
candidate tuples. Scores belong to observation/candidate incidences, never to a
shared candidate. Disk-backed feature tables bound construction memory.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .data_privacy import ground_truth_paths
from .intelligence_reports import report_observation_proximity
from .parallel_graph_construction import build_series_fragment, _flatten_numeric

NODE_TYPES = ("observation", "candidate", "intelligence_report", "report_claim", "kg_entity", "track")
NODE_TYPE_INDEX = {name: i for i, name in enumerate(NODE_TYPES)}
BROAD_REPORT_TYPES = frozenset({"theatre_aircraft_report", "airborne_aircraft_report"})
STABLE_CLAIM_TYPES = frozenset({"aircraft_variant", "aircraft_family", "radar_type", "operator"})


def intelligence_scope(report, claim, applicable_count, observation_count):
    """Unknown or restricted evidence stays local, including stable sighting claims.

    Broad inventory reports describe an operating area, not an identification of
    the platform. They may feed the track with explicit coverage. Other stable
    identity claims move only if their parent applies to every observation and
    the parent has no sighting/pattern time/location semantics.
    """
    if not applicable_count:
        return "unlinked"
    if report.get("report_type") in BROAD_REPORT_TYPES:
        return "track" if claim is None or claim.get("claim_type") in (
            STABLE_CLAIM_TYPES | {"theatre_aircraft_presence", "airborne_aircraft_presence"}
        ) else "observation"
    local = report.get("report_type") in {"sighting_report", "pattern_of_life_report"}
    local = local or bool(report.get("sighting") or report.get("pattern_of_life"))
    if claim is not None and not local and applicable_count == observation_count:
        if claim.get("claim_type") in STABLE_CLAIM_TYPES:
            return "track"
    return "observation"


class _FeatureSpool:
    """Discover sparse columns while streaming rows; materialise without a full copy."""

    def __init__(self, directory, name):
        self.path = Path(directory) / (name + ".float32.memmap")
        self.spool = tempfile.TemporaryFile(mode="w+t", dir=directory)
        self.names = set()
        self.count = 0

    def append(self, row):
        values = {k: float(v) for k, v in row.items()}
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Non-finite graph feature")
        self.names.update(values)
        self.spool.write(json.dumps(values) + "\n")
        self.count += 1
        return self.count - 1

    def finish(self):
        names = sorted(self.names) or ["empty"]
        columns = {name: i for i, name in enumerate(names)}
        # At least one row makes an empty feature table a valid memmap.
        dense = np.memmap(self.path, mode="w+", dtype=np.float32,
                          shape=(max(1, self.count), len(names)))
        self.spool.seek(0)
        chunk_size = 16384
        for start in range(0, max(1, self.count), chunk_size):
            stop = min(start + chunk_size, max(1, self.count))
            chunk = np.zeros((stop - start, len(names)), dtype=np.float32)
            for i in range(stop - start):
                line = self.spool.readline()
                if line:
                    for key, value in json.loads(line).items():
                        # Fixed transform: no statistics fitted on held-out tracks.
                        chunk[i, columns[key]] = np.sign(value) * np.log1p(abs(value))
            dense[start:stop] = chunk
        dense.flush()
        self.spool.close()
        return torch.from_numpy(dense), names


def build_shared_evidence_graph(series_records, kg, candidate_rows, output_dir, *,
                                max_candidates=18, retrieval_candidates=320,
                                min_claim_contribution=.15, intelligence_routing="selective"):
    """Build new artifacts from the same label-free inputs as track_star_hgt.

    No candidate is shared between series. Direct claim/candidate links are
    replaced by the existing scorer's per-incidence evidence channels, augmented
    with retained positive/negative claim counts. Candidate identity connects to
    its KG components; no observation match score is stored on a candidate node.
    """
    if ground_truth_paths(series_records):
        raise ValueError("Graph construction requires a ground-truth-free inference view")
    if intelligence_routing not in {"selective", "track"}:
        raise ValueError("intelligence_routing must be selective or track")
    if max_candidates < 1 or retrieval_candidates < 1:
        raise ValueError("Candidate limits must be positive")
    if not np.isfinite(min_claim_contribution) or min_claim_contribution < 0:
        raise ValueError("Claim contribution threshold must be finite and nonnegative")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features, edge_features = _FeatureSpool(output_dir, "nodes"), _FeatureSpool(output_dir, "incidences")
    meta, observations = [], []
    src, dst, types, feature_indices = array("i"), array("i"), array("i"), array("i")
    relations, relation_ids = [], {}
    kg_index = {}
    operator_ids = {node.get("properties", {}).get("name"): node["id"]
                    for node in kg["nodes"] if node.get("label") == "Operator"}

    def node(metadata, values):
        i = len(meta)
        meta.append(metadata)
        features.append(values)
        return i

    def edge(a, b, relation, attributes=-1):
        signature = (meta[a]["node_kind"], relation, meta[b]["node_kind"])
        if signature not in relation_ids:
            relation_ids[signature] = len(relations)
            relations.append(signature)
        src.append(a); dst.append(b); types.append(relation_ids[signature]); feature_indices.append(attributes)

    def pair(a, b, forward, reverse, attributes=-1):
        edge(a, b, forward, attributes)
        edge(b, a, reverse, attributes)

    for entity in kg["nodes"]:
        if entity.get("label") not in {"AircraftVariant", "AircraftFamily", "Radar", "Operator", "RadarMode"}:
            continue
        values = {"node_kind_kg_entity": 1.}
        _flatten_numeric("kg", entity.get("properties", {}), values)
        kg_index[entity["id"]] = node({"node_kind": "kg_entity", "kg_id": entity["id"],
                                      "kg_label": entity["label"], "kg_index": len(kg_index)}, values)
    for e in kg["edges"]:
        if e["source"] in kg_index and e["target"] in kg_index:
            pair(kg_index[e["source"]], kg_index[e["target"]],
                 "kg_" + e["relation"], "kg_reverse_" + e["relation"])

    families = {e["source"]: e["target"] for e in kg["edges"] if e["relation"] == "VARIANT_OF"}
    variants = defaultdict(list)
    for row in candidate_rows:
        variants[(row["mode_id"], row["radar_id"], row["aircraft_id"])].append(row)
    context = {"candidate_templates": [{**rows[0], "operator": None} for rows in variants.values()],
               "candidate_variants": dict(variants), "aircraft_family_by_aircraft": families,
               "max_kg_retrieval_candidates": retrieval_candidates, "max_kg_candidates": max_candidates,
               "claim_candidate_min_abs_contribution": min_claim_contribution,
               "include_candidate_nodes": True, "include_intel_report_nodes": True,
               "segment_frequency_shift_ghz": .75}
    audit = Counter()
    route_audit = (output_dir / "intelligence_routing.jsonl").open("w", encoding="utf-8")
    seen_series = set()
    for ordinal, series in enumerate(series_records):
        sid = series["series_id"]
        if sid in seen_series:
            raise ValueError("Duplicate series_id")
        seen_series.add(sid)
        _, fragment = build_series_fragment((ordinal, series), context)
        track = node({"node_kind": "track", "series_id": sid}, {"node_kind_track": 1.})
        mapping = {}
        # The fragment contains one copy of each report and claim, plus local candidates.
        for offset, metadata in enumerate(fragment["node_meta"]):
            if metadata["node_kind"] == "candidate":
                continue
            mapping[offset] = node(dict(metadata), fragment["feature_rows"][offset])
        obs = [mapping[i] for i in fragment["observation_offsets"]]
        observations.extend(obs)
        for i in obs:
            pair(i, track, "belongs_to", "contains")
        for a, b in zip(obs, obs[1:]):
            pair(a, b, "next_observation", "prev_observation")
        for a, b in zip(obs, obs[2:]):
            pair(a, b, "same_emitter", "same_emitter")
        shared = {}
        for observation_offset, candidate_offset, signed_claims in fragment["candidate_links"]:
            m = fragment["node_meta"][candidate_offset]
            identity = tuple(m[key] for key in ("mode_id", "radar_id", "aircraft_id", "operator"))
            if identity not in shared:
                c = node({"node_kind": "candidate", "series_id": sid, "candidate_key": identity},
                         {"node_kind_candidate": 1.})
                shared[identity] = c
                for role, entity in zip(("mode", "radar", "aircraft", "operator"),
                                        (*identity[:3], operator_ids.get(identity[3]))):
                    if entity in kg_index:
                        pair(c, kg_index[entity], "candidate_" + role, role + "_of_candidate")
            c, o = shared[identity], mapping[observation_offset]
            match = {k: v for k, v in fragment["feature_rows"][candidate_offset].items()
                     if k.startswith("candidate_")}
            rf_observed = match.get("candidate_non_radar_rf_observed_count", 0) > 0
            rf_compatible = match.get("candidate_non_radar_rf_score", 0) >= .5
            match["candidate_has_compatible_rf_emission"] = float(rf_observed and rf_compatible)
            match["candidate_contradicts_rf_emission"] = float(rf_observed and not rf_compatible)
            match["retained_supporting_claim_count"] = sum(i > 0 for i in signed_claims)
            match["retained_refuting_claim_count"] = sum(i < 0 for i in signed_claims)
            attributes = edge_features.append(match)
            pair(o, c, "has_candidate", "candidate_for", attributes)
            audit["candidate_incidences"] += 1
        audit["shared_candidates"] += len(shared)
        audit["original_candidate_nodes"] += len(fragment["candidate_links"])
        raw_observations = {o["observation_id"]: o for o in series["observations"]}
        report_by_id = {r["report_id"]: r for r in series.get("intelligence_reports") or []}
        applicable = defaultdict(list)
        for r, o in fragment["report_links"]:
            applicable[r].append(o)
        claim_types = defaultdict(list)
        for offset, m in enumerate(fragment["node_meta"]):
            if m["node_kind"] not in {"intelligence_report", "report_claim"}:
                continue
            is_claim = m["node_kind"] == "report_claim"
            report_offset = m["report_node_offset"] if is_claim else offset
            report = report_by_id[m["report_id"]]
            claim = next((c for c in report.get("claims") or [] if c["claim_id"] == m.get("claim_id")), None) if is_claim else None
            targets = applicable[report_offset]
            scope = intelligence_scope(report, claim, len(targets), len(obs))
            if intelligence_routing == "track" and targets:
                scope = "track"
            i = mapping[offset]
            if is_claim:
                pair(mapping[report_offset], i, "report_contains_claim", "claim_from_report")
                meta[i]["report_node_idx"] = mapping[report_offset]
                meta[i].pop("report_node_offset", None)
                for o in targets:
                    claim_types[(o, m.get("claim_type"))].append(offset)
            coverage = len(targets) / max(1, len(obs))
            # Persist exact applicability, not just its compressed coverage statistic.
            route_audit.write(json.dumps({"series_id": sid, "node_index": i, "kind": m["node_kind"],
                                "scope": scope, "coverage": coverage,
                                "observation_ids": [fragment["node_meta"][o]["observation_id"] for o in targets]}) + "\n")
            prefix = "claim" if is_claim else "report"
            if scope == "track":
                attr = edge_features.append({"coverage_fraction": coverage,
                                             "applicable_observation_count": len(targets)})
                pair(i, track, prefix + "_for_track", "track_has_" + prefix, attr)
                audit[prefix + "_track_pairs"] += 1
            elif scope == "observation":
                for o in targets:
                    raw = raw_observations[fragment["node_meta"][o]["observation_id"]]
                    proximity = report_observation_proximity(report, raw)
                    attr = edge_features.append({"coverage_fraction": coverage,
                        "time_delta_hours": float(proximity["time_delta_s"]) / 3600,
                        "distance_100km": float(proximity["distance_km"]) / 100})
                    pair(i, mapping[o], prefix + "_for_observation", "observation_has_" + prefix, attr)
                    audit[prefix + "_observation_pairs"] += 1
            audit["original_" + prefix + "_observation_pairs"] += len(targets)
        # Preserve local contradictory claims, but deduplicate repeated pair edges.
        conflicts = set()
        for claims in claim_types.values():
            for pos, a in enumerate(claims):
                for b in claims[pos + 1:]:
                    if fragment["node_meta"][a].get("object_id") != fragment["node_meta"][b].get("object_id"):
                        conflicts.add((a, b))
        for a, b in sorted(conflicts):
            pair(mapping[a], mapping[b], "contradicts_claim", "contradicts_claim")
        for c, entity_id in fragment["report_kg_edges"]:
            if entity_id in kg_index:
                pair(mapping[c], kg_index[entity_id], "claim_asserts_kg_entity", "kg_entity_asserted_by_claim")
        if (ordinal + 1) % 25 == 0:
            print(f"Prepared {ordinal + 1} tracks; {audit['original_candidate_nodes']:,} candidate incidences -> {audit['shared_candidates']:,} candidates")

    if not observations:
        raise ValueError("At least one observation is required")
    # Self edges ensure isolated KG/report/claim nodes remain represented in batches.
    for i in range(len(meta)):
        edge(i, i, "self")
    X, feature_names = features.finish()
    edge_X, edge_feature_names = edge_features.finish()
    edges = torch.stack((torch.frombuffer(src, dtype=torch.int32), torch.frombuffer(dst, dtype=torch.int32)))
    result = {"version": 1, "X": X, "feature_names": feature_names, "node_meta": meta,
              "observation_node_indices": observations, "edge_index": edges,
              "edge_types": torch.frombuffer(types, dtype=torch.int32),
              "edge_feature_indices": torch.frombuffer(feature_indices, dtype=torch.int32),
              "edge_features": edge_X, "edge_feature_names": edge_feature_names,
              "relation_metadata": relations, "audit": {**audit, "intelligence_routing": intelligence_routing},
              "num_kg_entities": len(kg_index)}
    route_audit.close()
    manifest = {k: v for k, v in result.items() if k not in {"X", "edge_features", "edge_index", "edge_types", "edge_feature_indices"}}
    manifest["feature_matrix"] = {"path": features.path.name, "shape": list(X.shape), "dtype": "float32"}
    manifest["edge_feature_matrix"] = {"path": edge_features.path.name, "shape": list(edge_X.shape), "dtype": "float32"}
    torch.save(manifest, output_dir / "graph_construction_outputs.pt")
    torch.save({k: result[k] for k in ("edge_index", "edge_types", "edge_feature_indices")}, output_dir / "relational_graph_outputs.pt")
    return result


@dataclass(frozen=True)
class SharedEvidenceGraph:
    num_nodes: int
    num_tracks: int
    edge_index: torch.Tensor
    relation_sizes: tuple[int, ...]
    feature_indices: torch.Tensor
    edge_features: torch.Tensor
    node_types: torch.Tensor
    kg_indices: torch.Tensor
    track_nodes: torch.Tensor
    # On CPU, retain a shared memmap and compact row selection. Gather raw
    # attributes only for the active batch, rather than copying the full dataset
    # into persistent in-memory batch caches.
    feature_rows: torch.Tensor | None = None
    type_indices: tuple[torch.Tensor, ...] | None = None
    attributed_positions: tuple[torch.Tensor | None, ...] | None = None
    relation_feature_modes: tuple[int, ...] | None = None
    source_caches: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | None = None

    def __post_init__(self):
        # Build lookup caches on CPU once, not through CUDA nonzero/unique in
        # each forward or activation-checkpoint recomputation.
        if self.type_indices is None:
            object.__setattr__(self, "type_indices", tuple(
                torch.nonzero(self.node_types == i).flatten() for i in range(len(NODE_TYPES))))
        if self.relation_feature_modes is None:
            modes, positions, caches = [], [], []
            for edges, refs in zip(self.edge_index.split(self.relation_sizes, dim=1),
                                   self.feature_indices.split(self.relation_sizes)):
                attributed = torch.nonzero(refs >= 0).flatten()
                mode = 0 if not attributed.numel() else (1 if attributed.numel() == refs.numel() else 2)
                modes.append(mode)
                positions.append(attributed if mode == 2 else None)
                cache = None
                if edges.size(1):
                    unique, inverse = torch.unique(edges[0], sorted=True, return_inverse=True)
                    if unique.numel() * 2 <= edges.size(1):
                        cache = (unique, inverse)
                caches.append(cache)
            object.__setattr__(self, "relation_feature_modes", tuple(modes))
            object.__setattr__(self, "attributed_positions", tuple(positions))
            object.__setattr__(self, "source_caches", tuple(caches))

    @property
    def num_edges(self):
        return self.edge_index.size(1)

    @property
    def topology_bytes(self):
        attributes = self.num_feature_rows * self.edge_features.size(1) * self.edge_features.element_size()
        caches = [*self.type_indices, *(t for t in self.attributed_positions if t is not None),
                  *(t for cache in self.source_caches if cache is not None for t in cache)]
        return attributes + sum(t.numel() * t.element_size() for t in (
            self.edge_index, self.feature_indices, self.node_types, self.kg_indices, self.track_nodes, *caches))

    @property
    def num_feature_rows(self):
        return self.edge_features.size(0) if self.feature_rows is None else self.feature_rows.numel()

    def tensors(self):
        return (self.edge_index, self.feature_indices, self.edge_features,
                self.node_types, self.kg_indices, self.track_nodes)

    def to(self, device, *, non_blocking=False):
        attributes = (self.edge_features if self.feature_rows is None else
                      self.edge_features.index_select(0, self.feature_rows))
        return SharedEvidenceGraph(self.num_nodes, self.num_tracks,
            self.edge_index.to(device, non_blocking=non_blocking), self.relation_sizes,
            self.feature_indices.to(device, non_blocking=non_blocking),
            attributes.to(device, non_blocking=non_blocking),
            self.node_types.to(device, non_blocking=non_blocking),
            self.kg_indices.to(device, non_blocking=non_blocking),
            self.track_nodes.to(device, non_blocking=non_blocking), None,
            tuple(t.to(device, non_blocking=non_blocking) for t in self.type_indices),
            tuple(None if t is None else t.to(device, non_blocking=non_blocking) for t in self.attributed_positions),
            self.relation_feature_modes,
            tuple(None if cache is None else tuple(t.to(device, non_blocking=non_blocking) for t in cache)
                  for cache in self.source_caches))


@dataclass(frozen=True)
class SharedTrackBatch:
    node_indices: torch.Tensor
    track_indices: torch.Tensor
    observation_nodes: torch.Tensor
    observation_to_track: torch.Tensor
    observation_positions: torch.Tensor
    observation_counts: torch.Tensor
    graph: SharedEvidenceGraph

    @property
    def num_tracks(self):
        return self.track_indices.numel()

    @property
    def num_observations(self):
        return self.observation_positions.numel()


def build_shared_batches_by_split(*, edge_index, edge_types, node_track_index,
                                  observation_nodes, observation_track_index,
                                  selected_tracks_by_split, tracks_per_batch,
                                  graph_data):
    """Partition complete tracks once; preserve edge attributes through reindexing.

    KG entities and KG/KG edges are shared domain context. Candidate ownership is
    always a single track, even when other tracks contain an identical tuple.
    """
    if tracks_per_batch < 1:
        raise ValueError("tracks_per_batch must be positive")
    edges = edge_index
    ownership = node_track_index
    parts = defaultdict(list)
    shared_parts = []
    for start in range(0, edges.size(1), 1_000_000):
        chunk = edges[:, start:start + 1_000_000]
        a, b = ownership[chunk[0]], ownership[chunk[1]]
        if ((a >= 0) & (b >= 0) & (a != b)).any():
            raise ValueError("Cross-track evidence edge")
        owners = torch.where(a >= 0, a, b)
        for owner in torch.unique(owners).tolist():
            positions = (torch.nonzero(owners == owner).flatten() + start).to(torch.int32)
            (shared_parts if owner < 0 else parts[owner]).append(positions)
    shared = torch.cat(shared_parts).long() if shared_parts else torch.empty(0, dtype=torch.long)
    node_types = torch.tensor([NODE_TYPE_INDEX[m["node_kind"]] for m in graph_data["node_meta"]])
    kg_indices = torch.tensor([m.get("kg_index", -1) for m in graph_data["node_meta"]])
    track_lookup = {int(ownership[i]): i for i in torch.nonzero(node_types == NODE_TYPE_INDEX["track"]).flatten().tolist()}
    observation_order = torch.argsort(observation_track_index, stable=True)
    counts = torch.bincount(observation_track_index, minlength=len(track_lookup))
    observation_offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    results = {}
    for split, selected in selected_tracks_by_split.items():
        track_ids = sorted(set(map(int, selected)))
        batches = []
        for start in range(0, len(track_ids), tracks_per_batch):
            tracks = torch.tensor(track_ids[start:start + tracks_per_batch])
            positions = torch.cat([shared, *(part.long() for t in tracks.tolist() for part in parts[t])])
            obs_positions = torch.sort(torch.cat([
                observation_order[observation_offsets[t]:observation_offsets[t + 1]]
                for t in tracks.tolist()])).values
            obs_global = observation_nodes[obs_positions]
            global_edges = edges[:, positions]
            nodes = torch.unique(torch.cat((global_edges.flatten(), obs_global.to(global_edges.dtype))), sorted=True).long()
            # Sorted search avoids an O(global nodes) scratch tensor per batch.
            local_edges = torch.searchsorted(nodes, global_edges)
            local_types = edge_types[positions].long()
            order = torch.argsort(local_types, stable=True)
            sizes = tuple(torch.bincount(local_types, minlength=len(graph_data["relation_metadata"])).tolist())
            refs = graph_data["edge_feature_indices"][positions].long()[order]
            used_features = torch.unique(refs[refs >= 0], sorted=True)
            local_refs = refs.clone()
            local_refs[refs >= 0] = torch.searchsorted(used_features, refs[refs >= 0])
            local_obs = torch.searchsorted(nodes, obs_global)
            membership = torch.searchsorted(tracks, observation_track_index[obs_positions])
            track_nodes = torch.searchsorted(nodes, torch.tensor([track_lookup[t] for t in tracks.tolist()]))
            graph = SharedEvidenceGraph(nodes.numel(), tracks.numel(), local_edges[:, order].contiguous(), sizes,
                local_refs, graph_data["edge_features"], node_types[nodes],
                kg_indices[nodes], track_nodes, used_features)
            batches.append(SharedTrackBatch(nodes, tracks, local_obs, membership, obs_positions,
                                           torch.bincount(membership, minlength=tracks.numel()), graph))
        results[split] = batches
    return results
