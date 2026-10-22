"""Benchmark graph construction without changing notebook inputs or artifacts.

Run from the repository root. Use --series-json for the real dataset, or explicitly
request a separate synthetic workload with --synthetic-tracks. Graphs go into a
temporary directory; optional profiles use the requested path. Timings include
artifact writes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgcn_fusion.data_privacy import strip_ground_truth
from rgcn_fusion.track_shared_graph import _FeatureSpool, build_shared_evidence_graph


def candidate_rows(kg):
    """Use the notebook's same KG hypotheses, including RF equipment and exclusions."""
    nodes = {n["id"]: n for n in kg["nodes"]}
    aircraft, operators, equipment = defaultdict(list), defaultdict(list), defaultdict(list)
    rf = {"USES_DATA_LINK": "data_link", "USES_RADIO": "radio", "USES_RADAR_ALTIMETER": "radar_altimeter"}
    for e in kg["edges"]:
        if e["relation"] == "USES_RADAR":
            aircraft[e["target"]].append(e["source"])
        elif e["relation"] == "OPERATES":
            operators[e["target"]].append(nodes[e["source"]]["properties"].get("name"))
        elif e["relation"] in rf:
            equipment[e["source"]].append({"emission_type": rf[e["relation"]],
                "equipment_id": e["target"], **nodes[e["target"]].get("properties", {})})
    return [{"mode_id": e["target"], "mode_props": nodes[e["target"]]["properties"],
             "radar_id": e["source"], "radar_props": nodes[e["source"]]["properties"],
             "aircraft_id": a, "aircraft_props": nodes[a]["properties"] if a else None,
             "aircraft_uses_radar": a is not None, "operator": o,
             "rf_equipment": equipment.get(a, [])}
            for e in kg["edges"] if e["relation"] == "HAS_MODE"
            for a in (aircraft.get(e["source"]) or [None])
            for o in (operators.get(a, [None]) if a else [None])
            if a not in {"aircraft:j_40", "aircraft:j_35", "aircraft:su_75"}]


def graph_digest(graph, directory):
    digest = hashlib.sha256()
    for key in ("node_meta", "observation_node_indices", "feature_names", "edge_feature_names",
                "relation_metadata", "audit", "num_kg_entities"):
        for part in json.JSONEncoder(sort_keys=True).iterencode(graph[key]):
            digest.update(part.encode())
    for key in ("X", "edge_features", "edge_index", "edge_types", "edge_feature_indices"):
        values = memoryview(graph[key].numpy()).cast("B")
        for start in range(0, len(values), 1024 * 1024):
            digest.update(values[start:start + 1024 * 1024])
    with (directory / "intelligence_routing.jsonl").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_spool(rows, repeats, baseline):
    row = {f"candidate_feature_{i}": (-1 if i % 2 else 1) * (i + .123456789)
           for i in range(48)}
    implementations = [("binary_vectorized", _FeatureSpool)]
    if baseline:
        implementations.insert(0, ("baseline", baseline._FeatureSpool))
    result, reference = {}, None
    for label, cls in implementations:
        times = []
        for _ in range(repeats):
            with tempfile.TemporaryDirectory() as directory:
                started = time.perf_counter()
                spool = cls(directory, "features")
                for _ in range(rows):
                    spool.append(row)
                data, _ = spool.finish()
                times.append(time.perf_counter() - started)
                if reference is None:
                    reference = data.clone()
                assert torch.equal(data, reference)
                del data
        result[label] = {"seconds": times, "median_seconds": statistics.median(times)}
    print(json.dumps({"spool_rows": rows, "features_per_row": 48, "runs": result}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--series-json", type=Path)
    inputs.add_argument("--synthetic-tracks", type=int)
    inputs.add_argument("--spool-rows", type=int, help="Measure only feature append/materialisation")
    parser.add_argument("--duration", type=float, default=2.)
    parser.add_argument("--kg", type=Path, default=ROOT / "generated/aircraft_radar_kg.json")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--baseline-source", type=Path,
                        help="Unmodified track_shared_graph.py for before/after comparison")
    parser.add_argument("--profile", type=Path, help="Save a cProfile of the baseline or serial run")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(1)
    baseline = None
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location("rgcn_fusion._baseline_graph", args.baseline_source)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
    if args.spool_rows is not None:
        if args.spool_rows < 1:
            parser.error("--spool-rows must be positive")
        benchmark_spool(args.spool_rows, args.repeats, baseline)
        return
    if args.series_json:
        records = json.loads(args.series_json.read_text())["observation_series"]
    else:
        from esm_observation_series_generator import generate_observation_series_with_intelligence_reports
        records = generate_observation_series_with_intelligence_reports(
            count=args.synthetic_tracks, seed=7, min_duration_s=args.duration,
            max_duration_s=args.duration, workers=1)["observation_series"]
    records = strip_ground_truth(records)
    kg = json.loads(args.kg.read_text())
    candidates = candidate_rows(kg)
    builds = []
    if baseline:
        builds.append(("baseline", baseline.build_shared_evidence_graph, {}))
    builds += [(f"workers={w}", build_shared_evidence_graph,
                {"num_workers": w, "start_method": args.start_method}) for w in args.workers]
    result = {"tracks": len(records), "observations": sum(len(s["observations"]) for s in records),
              "candidate_rows": len(candidates), "start_method": args.start_method, "runs": {}}
    reference = None
    with tempfile.TemporaryDirectory(prefix="benchmark-shared-") as temporary:
        for label, build, options in builds:
            times = []
            for repeat in range(args.repeats):
                directory = Path(temporary) / f"{label}-{repeat}"
                started = time.perf_counter()
                graph = build(records, kg, candidates, directory, **options)
                times.append(time.perf_counter() - started)
                print(f"{label}, repeat {repeat + 1}: {times[-1]:.3f}s", flush=True)
                digest = graph_digest(graph, directory)
                if reference is None:
                    reference = digest
                assert digest == reference, f"Graph changed for {label}"
                del graph
            result["runs"][label] = {"seconds": times, "median_seconds": statistics.median(times)}
        if args.profile:
            # Profile an additional run so profiler overhead never skews timings.
            label, build, options = builds[0]
            with cProfile.Profile() as profile:
                build(records, kg, candidates, Path(temporary) / "profile", **options)
            profile.dump_stats(args.profile)
    baseline_time = next(iter(result["runs"].values()))["median_seconds"]
    for run in result["runs"].values():
        run["speedup"] = baseline_time / run["median_seconds"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
