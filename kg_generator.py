#!/usr/bin/env python3
"""Procedurally generate an aircraft/radar knowledge graph for r-GCN experiments.

The graph is intentionally lightweight and dependency-free.  It combines curated,
open-source-friendly seed data for well-known combat aircraft families with
procedural expansion into typed nodes and relations that can be consumed by graph
ML pipelines.  Numeric properties are representative and should be treated as
experiment inputs rather than authoritative performance claims.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RadarMode:
    name: str
    prf: str
    centre_frequency_ghz: float
    bandwidth_mhz: float
    waveform: str
    scan_type: str
    detection_range_km: int
    track_capacity: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class Radar:
    name: str
    band: str
    antenna: str
    modes: tuple[RadarMode, ...]


@dataclass(frozen=True)
class AircraftVariant:
    family: str
    variant: str
    role: str
    generation: str
    radar: str
    max_speed_mach: float
    service_ceiling_m: int
    combat_radius_km: int
    ferry_range_km: int
    hardpoints: int
    operators: tuple[str, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def radar_modes(base_freq: float, long_range: int, track_capacity: int) -> tuple[RadarMode, ...]:
    """Generate a reusable mode suite around a radar's nominal X-band frequency."""
    return (
        RadarMode("range_while_search", "medium", base_freq, 80.0, "pulse_doppler", "mechanical_or_electronic", long_range, track_capacity),
        RadarMode("track_while_scan", "medium/high", base_freq + 0.08, 120.0, "pulse_doppler", "sector", int(long_range * 0.85), track_capacity),
        RadarMode("single_target_track", "high", base_freq + 0.12, 60.0, "pulse_doppler", "focused", int(long_range * 0.95), 1),
        RadarMode("look_down_shoot_down", "high", base_freq - 0.05, 100.0, "doppler_filtering", "sector", int(long_range * 0.65), max(1, track_capacity // 2)),
        RadarMode("air_to_ground_mapping", "low/medium", base_freq - 0.1, 160.0, "synthetic_aperture_or_real_beam", "ground", int(long_range * 0.45), None),
    )


RADARS: dict[str, Radar] = {
    "N019 Rubin": Radar("N019 Rubin", "I/J (X)", "slotted planar array", radar_modes(9.3, 80, 10)),
    "N010 Zhuk": Radar("N010 Zhuk", "X", "slotted planar array", radar_modes(9.5, 100, 10)),
    "Zhuk-ME": Radar("Zhuk-ME", "X", "slotted planar array", radar_modes(9.6, 120, 10)),
    "Zhuk-AE": Radar("Zhuk-AE", "X", "AESA", radar_modes(9.7, 160, 30)),
    "N001 Myech": Radar("N001 Myech", "X", "cassegrain", radar_modes(9.4, 100, 10)),
    "N011M Bars": Radar("N011M Bars", "X", "PESA", radar_modes(9.6, 180, 15)),
    "Irbis-E": Radar("Irbis-E", "X", "PESA", radar_modes(9.8, 300, 30)),
    "Zaslon": Radar("Zaslon", "X", "PESA", radar_modes(9.2, 200, 10)),
    "Zaslon-M": Radar("Zaslon-M", "X", "PESA", radar_modes(9.25, 320, 24)),
    "CAPTOR-M": Radar("CAPTOR-M", "X", "mechanically scanned array", radar_modes(9.7, 185, 20)),
    "CAPTOR-E": Radar("CAPTOR-E", "X", "AESA", radar_modes(9.9, 220, 40)),
    "AN/APG-66": Radar("AN/APG-66", "X", "planar array", radar_modes(9.5, 75, 10)),
    "AN/APG-68": Radar("AN/APG-68", "X", "planar array", radar_modes(9.6, 160, 10)),
    "AN/APG-80": Radar("AN/APG-80", "X", "AESA", radar_modes(9.8, 180, 20)),
    "AN/APG-83 SABR": Radar("AN/APG-83 SABR", "X", "AESA", radar_modes(10.0, 200, 20)),
    "AN/APG-63": Radar("AN/APG-63", "X", "mechanically scanned array", radar_modes(9.5, 160, 14)),
    "AN/APG-63(V)3": Radar("AN/APG-63(V)3", "X", "AESA", radar_modes(9.9, 220, 20)),
    "AN/APG-70": Radar("AN/APG-70", "X", "mechanically scanned array", radar_modes(9.6, 180, 14)),
    "AN/APG-82(V)1": Radar("AN/APG-82(V)1", "X", "AESA", radar_modes(10.0, 240, 20)),
}


AIRCRAFT: tuple[AircraftVariant, ...] = (
    AircraftVariant("MiG-29", "MiG-29A", "multirole fighter", "4", "N019 Rubin", 2.25, 18000, 700, 2100, 6, ("Russia", "Ukraine", "India", "Poland", "Germany")),
    AircraftVariant("MiG-29", "MiG-29S", "multirole fighter", "4", "N019 Rubin", 2.25, 18000, 700, 2100, 6, ("Russia", "Ukraine", "Belarus")),
    AircraftVariant("MiG-29", "MiG-29SMT", "multirole fighter", "4+", "Zhuk-ME", 2.25, 18000, 1000, 2400, 6, ("Russia", "Algeria")),
    AircraftVariant("MiG-29", "MiG-29K", "carrier multirole fighter", "4+", "Zhuk-ME", 2.0, 17500, 850, 2000, 8, ("Russia", "India")),
    AircraftVariant("MiG-29", "MiG-35", "multirole fighter", "4++", "Zhuk-AE", 2.25, 17500, 1000, 3100, 9, ("Russia",)),
    AircraftVariant("Su-27", "Su-27S", "air superiority fighter", "4", "N001 Myech", 2.35, 19000, 1340, 3530, 10, ("Russia", "Ukraine", "China", "Kazakhstan")),
    AircraftVariant("Su-27", "Su-30MKI", "multirole fighter", "4+", "N011M Bars", 2.0, 17300, 1500, 3000, 12, ("India",)),
    AircraftVariant("Su-27", "Su-30SM", "multirole fighter", "4+", "N011M Bars", 2.0, 17300, 1500, 3000, 12, ("Russia", "Kazakhstan", "Belarus")),
    AircraftVariant("Su-27", "Su-35S", "air superiority fighter", "4++", "Irbis-E", 2.25, 18000, 1600, 3600, 12, ("Russia", "China", "Egypt")),
    AircraftVariant("MiG-31", "MiG-31B", "interceptor", "4", "Zaslon", 2.83, 20600, 720, 3000, 6, ("Russia", "Kazakhstan")),
    AircraftVariant("MiG-31", "MiG-31BM", "interceptor", "4+", "Zaslon-M", 2.83, 20600, 720, 3000, 8, ("Russia", "Kazakhstan")),
    AircraftVariant("Typhoon", "Typhoon Tranche 1", "multirole fighter", "4+", "CAPTOR-M", 2.0, 19800, 1389, 2900, 13, ("United Kingdom", "Germany", "Italy", "Spain", "Austria")),
    AircraftVariant("Typhoon", "Typhoon Tranche 3", "multirole fighter", "4+", "CAPTOR-E", 2.0, 19800, 1389, 2900, 13, ("United Kingdom", "Germany", "Italy", "Spain", "Kuwait", "Qatar", "Saudi Arabia")),
    AircraftVariant("F-16", "F-16A/B", "multirole fighter", "4", "AN/APG-66", 2.05, 15240, 550, 4220, 9, ("United States", "Belgium", "Netherlands", "Norway", "Denmark", "Israel")),
    AircraftVariant("F-16", "F-16C/D Block 50", "multirole fighter", "4", "AN/APG-68", 2.05, 15240, 550, 4220, 9, ("United States", "Turkey", "Greece", "Poland", "South Korea")),
    AircraftVariant("F-16", "F-16E/F Block 60", "multirole fighter", "4+", "AN/APG-80", 2.0, 15240, 550, 4220, 11, ("United Arab Emirates",)),
    AircraftVariant("F-16", "F-16V Block 70/72", "multirole fighter", "4+", "AN/APG-83 SABR", 2.0, 15240, 550, 4220, 9, ("Bahrain", "Slovakia", "Bulgaria", "Taiwan", "Greece")),
    AircraftVariant("F-15", "F-15C", "air superiority fighter", "4", "AN/APG-63", 2.5, 20000, 1061, 5550, 9, ("United States", "Japan", "Saudi Arabia", "Israel")),
    AircraftVariant("F-15", "F-15SA", "strike fighter", "4+", "AN/APG-63(V)3", 2.5, 20000, 1270, 3900, 11, ("Saudi Arabia",)),
    AircraftVariant("F-15", "F-15E Strike Eagle", "strike fighter", "4", "AN/APG-70", 2.5, 18200, 1270, 3900, 11, ("United States", "South Korea", "Singapore", "Qatar")),
    AircraftVariant("F-15", "F-15EX Eagle II", "strike fighter", "4++", "AN/APG-82(V)1", 2.5, 18200, 1270, 3900, 12, ("United States",)),
)


def add_node(nodes: dict[str, dict[str, Any]], node_id: str, label: str, **properties: Any) -> None:
    nodes.setdefault(node_id, {"id": node_id, "label": label, "properties": properties})


def add_edge(edges: list[dict[str, str]], source: str, relation: str, target: str) -> None:
    edge = {"source": source, "relation": relation, "target": target}
    if edge not in edges:
        edges.append(edge)


def generate_graph() -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    for radar in RADARS.values():
        radar_id = f"radar:{slug(radar.name)}"
        add_node(nodes, radar_id, "Radar", name=radar.name, band=radar.band, antenna=radar.antenna)
        for mode in radar.modes:
            mode_id = f"radar_mode:{slug(radar.name)}:{slug(mode.name)}"
            add_node(
                nodes,
                mode_id,
                "RadarMode",
                name=mode.name,
                pulse_repetition_frequency=mode.prf,
                centre_frequency_ghz=mode.centre_frequency_ghz,
                bandwidth_mhz=mode.bandwidth_mhz,
                waveform=mode.waveform,
                scan_type=mode.scan_type,
                detection_range_km=mode.detection_range_km,
                track_capacity=mode.track_capacity,
                notes=mode.notes,
            )
            add_edge(edges, radar_id, "HAS_MODE", mode_id)

    for aircraft in AIRCRAFT:
        family_id = f"aircraft_family:{slug(aircraft.family)}"
        variant_id = f"aircraft:{slug(aircraft.variant)}"
        radar_id = f"radar:{slug(aircraft.radar)}"
        add_node(nodes, family_id, "AircraftFamily", name=aircraft.family)
        add_node(
            nodes,
            variant_id,
            "AircraftVariant",
            family=aircraft.family,
            variant=aircraft.variant,
            role=aircraft.role,
            generation=aircraft.generation,
            max_speed_mach=aircraft.max_speed_mach,
            service_ceiling_m=aircraft.service_ceiling_m,
            combat_radius_km=aircraft.combat_radius_km,
            ferry_range_km=aircraft.ferry_range_km,
            hardpoints=aircraft.hardpoints,
            tags=list(aircraft.tags),
        )
        add_edge(edges, variant_id, "VARIANT_OF", family_id)
        add_edge(edges, variant_id, "USES_RADAR", radar_id)
        for operator in aircraft.operators:
            operator_id = f"operator:{slug(operator)}"
            add_node(nodes, operator_id, "Operator", name=operator)
            add_edge(edges, operator_id, "OPERATES", variant_id)

    return {"metadata": {"schema_version": "1.0", "node_count": len(nodes), "edge_count": len(edges)}, "nodes": list(nodes.values()), "edges": edges}


def write_json(graph: dict[str, Any], output: Path) -> None:
    output.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_triples(graph: dict[str, Any], output: Path) -> None:
    lines = ["source,relation,target"]
    lines.extend(f"{edge['source']},{edge['relation']},{edge['target']}" for edge in graph["edges"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a combat-aircraft/radar knowledge graph.")
    parser.add_argument("--json", type=Path, default=Path("generated/aircraft_radar_kg.json"), help="JSON graph output path")
    parser.add_argument("--triples", type=Path, default=Path("generated/aircraft_radar_triples.csv"), help="CSV triples output path")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    graph = generate_graph()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.triples.parent.mkdir(parents=True, exist_ok=True)
    write_json(graph, args.json)
    write_triples(graph, args.triples)
    print(f"Wrote {graph['metadata']['node_count']} nodes and {graph['metadata']['edge_count']} edges")
    print(f"JSON: {args.json}")
    print(f"Triples: {args.triples}")


if __name__ == "__main__":
    main()
