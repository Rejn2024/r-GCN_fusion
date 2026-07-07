#!/usr/bin/env python3
"""Procedurally generate synthetic ESM observations of aircraft radar emissions.

The observations are designed to line up with the aircraft/radar knowledge graph
produced by :mod:`kg_generator`.  Values are representative simulation inputs,
not authoritative sensor or platform-performance data.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from kg_generator import AIRCRAFT, RADARS, AircraftVariant, RadarMode, slug


@dataclass(frozen=True)
class ObservationLabel:
    aircraft_family: str
    aircraft_variant: str
    aircraft_id: str
    operator: str
    radar: str
    radar_id: str
    mode: str
    mode_id: str


@dataclass(frozen=True)
class ErrorValue:
    value: float
    error: float
    min: float
    max: float
    units: str


TRAINING_AREAS = (
    ("North Sea", 56.25, 2.25),
    ("Eastern Mediterranean", 34.50, 31.00),
    ("Baltic Sea", 55.25, 18.50),
    ("Arabian Gulf", 25.75, 52.75),
    ("South China Sea", 16.50, 114.00),
    ("Bay of Bengal", 15.00, 88.00),
    ("Sea of Japan", 39.50, 135.00),
    ("Western Pacific", 24.00, 145.00),
)

DIRECTLY_MEASURABLE_MODE_FIELDS = (
    ("measured_centre_frequency_ghz", "centre_frequency", "ghz", 0.006),
    ("measured_bandwidth_mhz", "bandwidth", "mhz", 0.08),
    ("measured_prf_hz", "prf", "hz", 0.025),
    ("measured_pulse_width_us", "pulse_width", "us", 0.08),
    ("measured_duty_cycle", "duty_cycle", "", 0.08),
    ("measured_coherent_processing_interval_ms", "coherent_processing_interval", "ms", 0.08),
    ("measured_dwell_time_ms", "dwell_time", "ms", 0.08),
)


def _uniform_error(value: float, relative: float, absolute_floor: float = 0.0) -> ErrorValue:
    error = max(abs(value) * relative, absolute_floor)
    return ErrorValue(round(value, 6), round(error, 6), round(value - error, 6), round(value + error, 6), "")


def _mode_bounds(mode: RadarMode) -> dict[str, Any]:
    # Re-use the generated KG, rather than duplicating its range-spread logic.
    from kg_generator import generate_graph

    mode_id = f"radar_mode:{slug(next(r.name for r in RADARS.values() if mode in r.modes))}:{slug(mode.name)}"
    nodes = {node["id"]: node for node in generate_graph()["nodes"]}
    return nodes[mode_id]["properties"]


def _sample_between(rng: random.Random, low: float, high: float) -> float:
    return rng.uniform(float(low), float(high))


def _measured_from_mode_range(rng: random.Random, props: dict[str, Any], prefix: str, units: str, rel_error: float) -> dict[str, Any]:
    low = float(props[f"{prefix}_min_{units}"] if units else props[f"{prefix}_min"])
    high = float(props[f"{prefix}_max_{units}"] if units else props[f"{prefix}_max"])
    value = _sample_between(rng, low, high)
    ev = _uniform_error(value, rel_error, 1e-9)
    return {"value": ev.value, "error": ev.error, "min": ev.min, "max": ev.max}


def _location(rng: random.Random) -> dict[str, Any]:
    area, lat, lon = rng.choice(TRAINING_AREAS)
    lat += rng.uniform(-1.5, 1.5)
    lon += rng.uniform(-1.5, 1.5)
    lat_err = rng.uniform(0.03, 0.35)
    lon_err = rng.uniform(0.03, 0.35)
    return {
        "area": area,
        "estimated_latitude_deg": round(lat, 6),
        "estimated_longitude_deg": round(lon, 6),
        "error_box": {
            "min_latitude_deg": round(lat - lat_err, 6),
            "max_latitude_deg": round(lat + lat_err, 6),
            "min_longitude_deg": round(lon - lon_err, 6),
            "max_longitude_deg": round(lon + lon_err, 6),
        },
    }


def _kinematics(rng: random.Random, aircraft: AircraftVariant) -> dict[str, Any]:
    # Rough Mach-to-kph conversion at altitude; sufficient for synthetic labels.
    max_kph = aircraft.max_speed_mach * 1060.0
    if aircraft.generation == "rotary":
        speed = rng.uniform(120, min(310, max_kph))
        altitude = rng.uniform(100, min(4500, aircraft.service_ceiling_m * 0.75))
    elif "bomber" in aircraft.role:
        speed = rng.uniform(650, min(1050, max_kph * 0.9))
        altitude = rng.uniform(7000, aircraft.service_ceiling_m * 0.9)
    else:
        speed = rng.uniform(450, min(max_kph * 0.85, 1450))
        altitude = rng.uniform(1500, aircraft.service_ceiling_m * 0.9)
    speed_error = rng.uniform(25, 140)
    altitude_error = rng.uniform(100, 900)
    return {
        "ground_speed_kph": round(speed, 1),
        "ground_speed_error_kph": round(speed_error, 1),
        "ground_speed_min_kph": round(max(0, speed - speed_error), 1),
        "ground_speed_max_kph": round(speed + speed_error, 1),
        "altitude_m": round(altitude, 1),
        "altitude_error_m": round(altitude_error, 1),
        "altitude_min_m": round(max(0, altitude - altitude_error), 1),
        "altitude_max_m": round(min(aircraft.service_ceiling_m, altitude + altitude_error), 1),
        "heading_deg": round(rng.uniform(0, 360), 1),
        "heading_error_deg": round(rng.uniform(2, 20), 1),
    }


def _ambiguous_candidates(aircraft: AircraftVariant, operator: str, mode_name: str) -> list[dict[str, str]]:
    candidates = []
    for other in AIRCRAFT:
        if other.radar == aircraft.radar or (other.family == aircraft.family and mode_name != "single_target_track"):
            candidates.append({
                "aircraft_family": other.family,
                "aircraft_variant": other.variant,
                "aircraft_id": f"aircraft:{slug(other.variant)}",
                "operator": operator if operator in other.operators else other.operators[0],
                "radar": other.radar,
            })
    return candidates[:10]


def generate_observation(index: int, rng: random.Random, start: datetime, end: datetime) -> dict[str, Any]:
    aircraft = rng.choice(AIRCRAFT)
    operator = rng.choice(aircraft.operators)
    radar = RADARS[aircraft.radar]
    mode = rng.choice(radar.modes)
    props = _mode_bounds(mode)
    span = int((end - start).total_seconds())
    ts = start + timedelta(seconds=rng.randrange(max(1, span)))
    label = ObservationLabel(aircraft.family, aircraft.variant, f"aircraft:{slug(aircraft.variant)}", operator, radar.name, f"radar:{slug(radar.name)}", mode.name, f"radar_mode:{slug(radar.name)}:{slug(mode.name)}")

    esm = {
        "observed_waveform": props["waveform"],
        "observed_scan_type": props["scan_type"],
    }
    for field_name, prefix, units, rel_error in DIRECTLY_MEASURABLE_MODE_FIELDS:
        esm[field_name] = _measured_from_mode_range(rng, props, prefix, units, rel_error)
    prf_hz = esm["measured_prf_hz"]["value"]
    pri = _uniform_error(1_000_000.0 / prf_hz, 0.025)
    esm["measured_pulse_repetition_interval_us"] = {"value": pri.value, "error": pri.error, "min": pri.min, "max": pri.max}

    return {
        "observation_id": f"esm_obs_{index:05d}",
        "timestamp_unix": int(ts.timestamp()),
        "timestamp_iso8601": ts.isoformat().replace("+00:00", "Z"),
        "sensor": {"type": "passive_esm", "platform": "synthetic_ground_or_airborne_collector"},
        "estimated_emitter_location": _location(rng),
        "approximate_kinematics": _kinematics(rng, aircraft),
        "esm_radar_parameters": esm,
        "candidate_labels_from_shared_kg_features": _ambiguous_candidates(aircraft, operator, mode.name),
        "ground_truth_label": asdict(label),
    }


def generate_observations(count: int, seed: int = 7, start: datetime | None = None, end: datetime | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    end = end or datetime(2026, 1, 1, tzinfo=UTC)
    observations = [generate_observation(i + 1, rng, start, end) for i in range(count)]
    return {"metadata": {"schema_version": "1.0", "count": count, "seed": seed, "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z")}, "observations": observations}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic ESM observations consistent with the aircraft/radar KG.")
    parser.add_argument("--count", type=int, default=100, help="Number of observations to emit")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible output")
    parser.add_argument("--output", type=Path, default=Path("generated/esm_observations.json"), help="JSON output path")
    parser.add_argument("--start", default="2024-01-01T00:00:00Z", help="Inclusive UTC start timestamp")
    parser.add_argument("--end", default="2026-01-01T00:00:00Z", help="Exclusive UTC end timestamp")
    return parser.parse_args(argv)


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be positive")
    data = generate_observations(args.count, args.seed, _parse_utc(args.start), _parse_utc(args.end))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.count} ESM observations to {args.output}")


if __name__ == "__main__":
    main()
