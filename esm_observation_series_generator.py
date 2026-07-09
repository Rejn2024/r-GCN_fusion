#!/usr/bin/env python3
"""Generate synthetic single-emitter ESM observation series.

Each generated entry represents one aircraft/emitter track containing repeated
observations spaced at roughly 0.5 seconds.  Per-observation records preserve the
same top-level fields emitted by :mod:`esm_observation_generator` so they can be
reused by existing ETL/scoring experiments, while the series wrapper adds track
metadata and temporal ordering for single-emitter sequence experiments.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from esm_observation_generator import (
    DIRECTLY_MEASURABLE_MODE_FIELDS,
    ObservationLabel,
    _ambiguous_candidates,
    _kinematics,
    _location,
    _measured_from_mode_range,
    _mode_bounds,
    _uniform_error,
)
from kg_generator import AIRCRAFT, RADARS, slug

DEFAULT_SERIES_COUNT = 2500
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.5
DEFAULT_MIN_DURATION_SECONDS = 1.0
DEFAULT_MAX_DURATION_SECONDS = 60.0


def _sample_esm_parameters(rng: random.Random, props: dict[str, Any]) -> dict[str, Any]:
    esm: dict[str, Any] = {
        "observed_waveform": props["waveform"],
        "observed_scan_type": props["scan_type"],
    }
    for field_name, prefix, units, rel_error in DIRECTLY_MEASURABLE_MODE_FIELDS:
        esm[field_name] = _measured_from_mode_range(rng, props, prefix, units, rel_error)
    prf_hz = esm["measured_prf_hz"]["value"]
    pri = _uniform_error(1_000_000.0 / prf_hz, 0.025)
    esm["measured_pulse_repetition_interval_us"] = {
        "value": pri.value,
        "error": pri.error,
        "min": pri.min,
        "max": pri.max,
    }
    return esm


def _move_location(location: dict[str, Any], heading_deg: float, speed_kph: float, elapsed_s: float) -> dict[str, Any]:
    # Equirectangular approximation is sufficient for sub-minute synthetic tracks.
    import math

    lat = float(location["estimated_latitude_deg"])
    lon = float(location["estimated_longitude_deg"])
    distance_km = speed_kph * elapsed_s / 3600.0
    heading_rad = math.radians(heading_deg)
    delta_north_km = math.cos(heading_rad) * distance_km
    delta_east_km = math.sin(heading_rad) * distance_km
    moved_lat = lat + delta_north_km / 111.0
    moved_lon = lon + delta_east_km / max(1e-6, 111.0 * math.cos(math.radians(lat)))
    lat_err = (location["error_box"]["max_latitude_deg"] - location["error_box"]["min_latitude_deg"]) / 2.0
    lon_err = (location["error_box"]["max_longitude_deg"] - location["error_box"]["min_longitude_deg"]) / 2.0
    return {
        "area": location["area"],
        "estimated_latitude_deg": round(moved_lat, 6),
        "estimated_longitude_deg": round(moved_lon, 6),
        "error_box": {
            "min_latitude_deg": round(moved_lat - lat_err, 6),
            "max_latitude_deg": round(moved_lat + lat_err, 6),
            "min_longitude_deg": round(moved_lon - lon_err, 6),
            "max_longitude_deg": round(moved_lon + lon_err, 6),
        },
    }


def _evolve_kinematics(rng: random.Random, base: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    kin = dict(base)
    speed = max(0.0, float(base["ground_speed_kph"]) + rng.gauss(0.0, 0.6) * elapsed_s)
    altitude = max(0.0, float(base["altitude_m"]) + rng.gauss(0.0, 1.8) * elapsed_s)
    heading = (float(base["heading_deg"]) + rng.gauss(0.0, 0.04) * elapsed_s) % 360.0
    speed_error = float(base["ground_speed_error_kph"])
    altitude_error = float(base["altitude_error_m"])
    kin.update({
        "ground_speed_kph": round(speed, 1),
        "ground_speed_min_kph": round(max(0, speed - speed_error), 1),
        "ground_speed_max_kph": round(speed + speed_error, 1),
        "altitude_m": round(altitude, 1),
        "altitude_min_m": round(max(0, altitude - altitude_error), 1),
        "altitude_max_m": round(altitude + altitude_error, 1),
        "heading_deg": round(heading, 1),
    })
    return kin


def _observation_count_for_duration(duration_s: float, sample_interval_s: float) -> int:
    return max(2, int(round(duration_s / sample_interval_s)) + 1)


def generate_observation_series(
    count: int = DEFAULT_SERIES_COUNT,
    seed: int = 7,
    min_duration_s: float = DEFAULT_MIN_DURATION_SECONDS,
    max_duration_s: float = DEFAULT_MAX_DURATION_SECONDS,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    start: datetime | None = None,
    end: datetime | None = None,
) -> dict[str, Any]:
    """Generate single-emitter ESM observation series."""
    if count < 1:
        raise ValueError("count must be positive")
    if min_duration_s < sample_interval_s:
        raise ValueError("min_duration_s must be at least one sample interval")
    if max_duration_s < min_duration_s:
        raise ValueError("max_duration_s must be greater than or equal to min_duration_s")
    rng = random.Random(seed)
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    end = end or datetime(2026, 1, 1, tzinfo=UTC)
    span = int((end - start).total_seconds())
    if span <= max_duration_s:
        raise ValueError("time range must be longer than max_duration_s")

    series_entries = []
    for series_index in range(1, count + 1):
        aircraft = rng.choice(AIRCRAFT)
        operator = rng.choice(aircraft.operators)
        radar = RADARS[aircraft.radar]
        modes = list(radar.modes)
        initial_mode = rng.choice(modes)
        duration_s = rng.uniform(min_duration_s, max_duration_s)
        obs_count = _observation_count_for_duration(duration_s, sample_interval_s)
        track_start = start + timedelta(seconds=rng.randrange(max(1, int(span - duration_s))))
        base_location = _location(rng)
        base_kinematics = _kinematics(rng, aircraft)
        shift_index = rng.randrange(1, obs_count) if len(modes) > 1 and rng.random() < 0.35 else None
        shifted_mode = (
            rng.choice([mode for mode in modes if mode.name != initial_mode.name])
            if shift_index
            else initial_mode
        )

        observations = []
        for obs_index in range(obs_count):
            elapsed_s = obs_index * sample_interval_s + rng.uniform(-0.04, 0.04)
            elapsed_s = max(0.0, elapsed_s)
            timestamp = track_start + timedelta(seconds=elapsed_s)
            mode = shifted_mode if shift_index is not None and obs_index >= shift_index else initial_mode
            props = _mode_bounds(mode)
            label = ObservationLabel(
                aircraft.family,
                aircraft.variant,
                f"aircraft:{slug(aircraft.variant)}",
                operator,
                radar.name,
                f"radar:{slug(radar.name)}",
                mode.name,
                f"radar_mode:{slug(radar.name)}:{slug(mode.name)}",
            )
            kin = _evolve_kinematics(rng, base_kinematics, elapsed_s)
            observations.append({
                "observation_id": f"esm_series_{series_index:05d}_obs_{obs_index + 1:03d}",
                "series_id": f"esm_series_{series_index:05d}",
                "sequence_index": obs_index,
                "elapsed_time_s": round(elapsed_s, 3),
                "timestamp_unix": timestamp.timestamp(),
                "timestamp_iso8601": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "sensor": {"type": "passive_esm", "platform": "synthetic_ground_or_airborne_collector"},
                "estimated_emitter_location": _move_location(
                    base_location,
                    kin["heading_deg"],
                    kin["ground_speed_kph"],
                    elapsed_s,
                ),
                "approximate_kinematics": kin,
                "esm_radar_parameters": _sample_esm_parameters(rng, props),
                "candidate_labels_from_shared_kg_features": _ambiguous_candidates(aircraft, operator, mode.name),
                "ground_truth_label": asdict(label),
            })
        series_entries.append({
            "series_id": f"esm_series_{series_index:05d}",
            "emitter_type": "aircraft",
            "sample_interval_s": sample_interval_s,
            "duration_s": round(observations[-1]["elapsed_time_s"] - observations[0]["elapsed_time_s"], 3),
            "observation_count": len(observations),
            "mode_shift_sequence_index": shift_index,
            "ground_truth_track_label": asdict(
                ObservationLabel(
                    aircraft.family,
                    aircraft.variant,
                    f"aircraft:{slug(aircraft.variant)}",
                    operator,
                    radar.name,
                    f"radar:{slug(radar.name)}",
                    "multiple" if shift_index else initial_mode.name,
                    "multiple" if shift_index else f"radar_mode:{slug(radar.name)}:{slug(initial_mode.name)}",
                )
            ),
            "observations": observations,
        })
    return {
        "metadata": {
            "schema_version": "1.0",
            "series_count": count,
            "default_count": DEFAULT_SERIES_COUNT,
            "seed": seed,
            "min_duration_s": min_duration_s,
            "max_duration_s": max_duration_s,
            "sample_interval_s": sample_interval_s,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
        "observation_series": series_entries,
    }


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic single-emitter ESM observation series.")
    parser.add_argument("--count", type=int, default=DEFAULT_SERIES_COUNT, help="Number of emitter series entries to emit")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible output")
    parser.add_argument("--min-duration-s", type=float, default=DEFAULT_MIN_DURATION_SECONDS, help="Minimum series duration in seconds")
    parser.add_argument("--max-duration-s", type=float, default=DEFAULT_MAX_DURATION_SECONDS, help="Maximum series duration in seconds")
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS, help="Nominal interval between observations in seconds")
    parser.add_argument("--output", type=Path, default=Path("generated/esm_observation_series.json"), help="JSON output path")
    parser.add_argument("--start", default="2024-01-01T00:00:00Z", help="Inclusive UTC start timestamp")
    parser.add_argument("--end", default="2026-01-01T00:00:00Z", help="Exclusive UTC end timestamp")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    data = generate_observation_series(
        args.count,
        args.seed,
        args.min_duration_s,
        args.max_duration_s,
        args.sample_interval_s,
        _parse_utc(args.start),
        _parse_utc(args.end),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.count} ESM observation series to {args.output}")


if __name__ == "__main__":
    main()
