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
from json import JSONDecodeError
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor
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


def _json_error_context(
    document: str, error: JSONDecodeError, *, context_lines: int = 2
) -> str:
    """Return a compact source excerpt around a JSON decoding error."""
    lines = document.splitlines() or [""]
    line_index = max(0, min(error.lineno - 1, len(lines) - 1))
    start = max(0, line_index - context_lines)
    end = min(len(lines), line_index + context_lines + 1)
    width = len(str(end))
    excerpt: list[str] = []
    for idx in range(start, end):
        marker = ">" if idx == line_index else " "
        excerpt.append(f"{marker} {idx + 1:{width}d} | {lines[idx]}")
        if idx == line_index:
            caret_column = max(error.colno, 1)
            excerpt.append(f"  {' ' * width} | {' ' * (caret_column - 1)}^")
    return "\n".join(excerpt)


def _json_error_hint(document: str, error: JSONDecodeError) -> str:
    """Return a targeted remediation hint for common JSON mistakes."""
    lines = document.splitlines() or [""]
    line_index = max(0, min(error.lineno - 1, len(lines) - 1))
    candidate_lines = [lines[line_index]]
    if line_index > 0:
        candidate_lines.append(lines[line_index - 1])
    if (
        error.msg == "Expecting ':' delimiter"
        and any(
            (stripped := line.strip()).startswith('"') and stripped.endswith('"')
            for line in candidate_lines
        )
    ):
        return (
            "Hint: this line looks like a JSON object key without a trailing "
            "colon. Add ':' after the closing quote, then re-run the loader."
        )
    return "Hint: regenerate this file or fix the JSON syntax near the highlighted line."


def load_observation_series_json(path: str | Path) -> dict[str, object]:
    """Load an observation-series JSON file with line-level error context.

    This wraps :func:`json.loads` so notebooks fail with an actionable message
    when a generated or hand-edited dataset is malformed.
    """
    source_path = Path(path)
    document = source_path.read_text(encoding="utf-8")
    try:
        data = json.loads(document)
    except JSONDecodeError as exc:
        context = _json_error_context(document, exc)
        hint = _json_error_hint(document, exc)
        raise ValueError(
            f"Could not parse JSON in {source_path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}.\n{context}\n{hint}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(
        data.get("observation_series"), list
    ):
        raise ValueError(
            f"{source_path} must contain an object with an 'observation_series' list"
        )
    return data


def _sample_esm_parameters(rng: random.Random, props: dict[str, Any]) -> dict[str, Any]:
    esm: dict[str, Any] = {
        "observed_waveform": props["waveform"],
        "observed_scan_type": props["scan_type"],
    }
    for field_name, prefix, units, rel_error in DIRECTLY_MEASURABLE_MODE_FIELDS:
        esm[field_name] = _measured_from_mode_range(
            rng, props, prefix, units, rel_error
        )
    prf_hz = esm["measured_prf_hz"]["value"]
    pri = _uniform_error(1_000_000.0 / prf_hz, 0.025)
    esm["measured_pulse_repetition_interval_us"] = {
        "value": pri.value,
        "error": pri.error,
        "min": pri.min,
        "max": pri.max,
    }
    return esm


def _move_location(
    location: dict[str, Any], heading_deg: float, speed_kph: float, elapsed_s: float
) -> dict[str, Any]:
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
    lat_err = (
        location["error_box"]["max_latitude_deg"]
        - location["error_box"]["min_latitude_deg"]
    ) / 2.0
    lon_err = (
        location["error_box"]["max_longitude_deg"]
        - location["error_box"]["min_longitude_deg"]
    ) / 2.0
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


def _evolve_kinematics(
    rng: random.Random, base: dict[str, Any], elapsed_s: float
) -> dict[str, Any]:
    kin = dict(base)
    speed = max(0.0, float(base["ground_speed_kph"]) + rng.gauss(0.0, 0.6) * elapsed_s)
    altitude = max(0.0, float(base["altitude_m"]) + rng.gauss(0.0, 1.8) * elapsed_s)
    heading = (float(base["heading_deg"]) + rng.gauss(0.0, 0.04) * elapsed_s) % 360.0
    speed_error = float(base["ground_speed_error_kph"])
    altitude_error = float(base["altitude_error_m"])
    kin.update(
        {
            "ground_speed_kph": round(speed, 1),
            "ground_speed_min_kph": round(max(0, speed - speed_error), 1),
            "ground_speed_max_kph": round(speed + speed_error, 1),
            "altitude_m": round(altitude, 1),
            "altitude_min_m": round(max(0, altitude - altitude_error), 1),
            "altitude_max_m": round(altitude + altitude_error, 1),
            "heading_deg": round(heading, 1),
        }
    )
    return kin


def _observation_count_for_duration(duration_s: float, sample_interval_s: float) -> int:
    return max(2, int(round(duration_s / sample_interval_s)) + 1)


def _mode_label(
    aircraft: Any, operator: str, radar: Any, mode: Any
) -> ObservationLabel:
    return ObservationLabel(
        aircraft.family,
        aircraft.variant,
        f"aircraft:{slug(aircraft.variant)}",
        operator,
        radar.name,
        f"radar:{slug(radar.name)}",
        mode.name,
        f"radar_mode:{slug(radar.name)}:{slug(mode.name)}",
    )


def _sample_mode_schedule(
    rng: random.Random,
    modes: list[Any],
    observation_count: int,
    switch_probability: float,
    max_switches: int | None,
) -> tuple[list[Any], list[int]]:
    """Sample a per-observation radar-mode schedule and transition indices.

    Transition indices are sequence indices where the mode first differs from the
    previous observation. Index 0 is never a transition.
    """
    if not 0.0 <= switch_probability <= 1.0:
        raise ValueError("mode_switch_probability must be between 0.0 and 1.0")
    if max_switches is not None and max_switches < 0:
        raise ValueError("max_mode_switches must be non-negative when provided")

    initial_mode = rng.choice(modes)
    if len(modes) <= 1 or observation_count <= 1 or switch_probability == 0.0:
        return [initial_mode] * observation_count, []

    switch_indices = [
        idx for idx in range(1, observation_count) if rng.random() < switch_probability
    ]
    if max_switches is not None and len(switch_indices) > max_switches:
        switch_indices = sorted(rng.sample(switch_indices, max_switches))

    schedule = []
    current_mode = initial_mode
    switch_set = set(switch_indices)
    for obs_index in range(observation_count):
        if obs_index in switch_set:
            current_mode = rng.choice(
                [mode for mode in modes if mode.name != current_mode.name]
            )
        schedule.append(current_mode)
    return schedule, switch_indices


def observations_without_ground_truth(
    series_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return observations suitable for inference by removing truth-only fields.

    The generator records per-observation ground truth for evaluation, but callers
    should pass this stripped view to candidate scoring or classification code to
    avoid label leakage.
    """
    inference_rows = []
    for observation in series_entry["observations"]:
        row = dict(observation)
        row.pop("ground_truth_label", None)
        inference_rows.append(row)
    return inference_rows


def _generate_single_observation_series(
    args: tuple[
        int,
        int,
        float,
        float,
        float,
        datetime,
        int,
        float,
        int | None,
    ],
) -> dict[str, Any]:
    (
        series_index,
        series_seed,
        min_duration_s,
        max_duration_s,
        sample_interval_s,
        start,
        span,
        mode_switch_probability,
        max_mode_switches,
    ) = args
    rng = random.Random(series_seed)
    aircraft = rng.choice(AIRCRAFT)
    operator = rng.choice(aircraft.operators)
    radar = RADARS[aircraft.radar]
    modes = list(radar.modes)
    duration_s = rng.uniform(min_duration_s, max_duration_s)
    obs_count = _observation_count_for_duration(duration_s, sample_interval_s)
    mode_schedule, shift_indices = _sample_mode_schedule(
        rng,
        modes,
        obs_count,
        mode_switch_probability,
        max_mode_switches,
    )
    track_start = start + timedelta(
        seconds=rng.randrange(max(1, int(span - duration_s)))
    )
    base_location = _location(rng)
    base_kinematics = _kinematics(rng, aircraft)

    observations = []
    for obs_index in range(obs_count):
        elapsed_s = obs_index * sample_interval_s + rng.uniform(-0.04, 0.04)
        elapsed_s = max(0.0, elapsed_s)
        timestamp = track_start + timedelta(seconds=elapsed_s)
        mode = mode_schedule[obs_index]
        props = _mode_bounds(mode)
        label = _mode_label(aircraft, operator, radar, mode)
        kin = _evolve_kinematics(rng, base_kinematics, elapsed_s)
        observations.append(
            {
                "observation_id": f"esm_series_{series_index:05d}_obs_{obs_index + 1:03d}",
                "series_id": f"esm_series_{series_index:05d}",
                "sequence_index": obs_index,
                "elapsed_time_s": round(elapsed_s, 3),
                "timestamp_unix": timestamp.timestamp(),
                "timestamp_iso8601": timestamp.isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z"),
                "sensor": {
                    "type": "passive_esm",
                    "platform": "synthetic_ground_or_airborne_collector",
                },
                "estimated_emitter_location": _move_location(
                    base_location,
                    kin["heading_deg"],
                    kin["ground_speed_kph"],
                    elapsed_s,
                ),
                "approximate_kinematics": kin,
                "esm_radar_parameters": _sample_esm_parameters(rng, props),
                "candidate_labels_from_shared_kg_features": _ambiguous_candidates(
                    aircraft, operator, mode.name
                ),
                "ground_truth_label": asdict(label),
            }
        )
    return {
        "series_id": f"esm_series_{series_index:05d}",
        "emitter_type": "aircraft",
        "sample_interval_s": sample_interval_s,
        "duration_s": round(
            observations[-1]["elapsed_time_s"] - observations[0]["elapsed_time_s"], 3
        ),
        "observation_count": len(observations),
        "mode_shift_sequence_indices": shift_indices,
        "mode_shift_sequence_index": shift_indices[0] if shift_indices else None,
        "ground_truth_mode_sequence": [
            {
                "sequence_index": index,
                "mode": mode.name,
                "mode_id": f"radar_mode:{slug(radar.name)}:{slug(mode.name)}",
            }
            for index, mode in enumerate(mode_schedule)
        ],
        "ground_truth_track_label": asdict(
            ObservationLabel(
                aircraft.family,
                aircraft.variant,
                f"aircraft:{slug(aircraft.variant)}",
                operator,
                radar.name,
                f"radar:{slug(radar.name)}",
                "multiple" if shift_indices else mode_schedule[0].name,
                (
                    "multiple"
                    if shift_indices
                    else f"radar_mode:{slug(radar.name)}:{slug(mode_schedule[0].name)}"
                ),
            )
        ),
        "observations": observations,
    }


def generate_observation_series(
    count: int = DEFAULT_SERIES_COUNT,
    seed: int = 7,
    min_duration_s: float = DEFAULT_MIN_DURATION_SECONDS,
    max_duration_s: float = DEFAULT_MAX_DURATION_SECONDS,
    sample_interval_s: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    start: datetime | None = None,
    end: datetime | None = None,
    mode_switch_probability: float = 0.03,
    max_mode_switches: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Generate single-emitter ESM observation series.

    ``mode_switch_probability`` is applied independently between adjacent
    observations, so one entry may contain zero, one, or several radar-mode
    transitions at random sequence indices. Ground-truth mode labels remain in
    ``ground_truth_label`` for evaluation only; use
    :func:`observations_without_ground_truth` before passing rows to inference or
    classification code.

    ``workers`` controls process-level parallelism across independent series.
    Per-series seeds are derived deterministically from ``seed`` before work is
    dispatched, so results are stable regardless of worker count.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if min_duration_s < sample_interval_s:
        raise ValueError("min_duration_s must be at least one sample interval")
    if max_duration_s < min_duration_s:
        raise ValueError(
            "max_duration_s must be greater than or equal to min_duration_s"
        )
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    end = end or datetime(2026, 1, 1, tzinfo=UTC)
    span = int((end - start).total_seconds())
    if span <= max_duration_s:
        raise ValueError("time range must be longer than max_duration_s")
    if workers is not None and workers < 1:
        raise ValueError("workers must be positive when provided")

    seed_rng = random.Random(seed)
    tasks = [
        (
            series_index,
            seed_rng.getrandbits(64),
            min_duration_s,
            max_duration_s,
            sample_interval_s,
            start,
            span,
            mode_switch_probability,
            max_mode_switches,
        )
        for series_index in range(1, count + 1)
    ]
    worker_count = min(count, workers or (os.cpu_count() or 1))
    if worker_count == 1:
        series_entries = [_generate_single_observation_series(task) for task in tasks]
    else:
        chunksize = max(1, count // (worker_count * 4))
        mp_context = (
            multiprocessing.get_context("fork") if hasattr(os, "fork") else None
        )
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=mp_context
        ) as executor:
            series_entries = list(
                executor.map(
                    _generate_single_observation_series, tasks, chunksize=chunksize
                )
            )
    return {
        "metadata": {
            "schema_version": "1.0",
            "series_count": count,
            "default_count": DEFAULT_SERIES_COUNT,
            "seed": seed,
            "min_duration_s": min_duration_s,
            "max_duration_s": max_duration_s,
            "sample_interval_s": sample_interval_s,
            "mode_switch_probability": mode_switch_probability,
            "max_mode_switches": max_mode_switches,
            "workers": worker_count,
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
        "observation_series": series_entries,
    }


def generate_observation_series_with_intelligence_reports(
    *args: Any,
    intelligence_seed: int | None = None,
    min_reports_per_observation: int = 10,
    max_reports_per_observation: int = 12,
    **kwargs: Any,
) -> dict[str, Any]:
    """Generate ESM observation series enriched with synthetic intelligence reports."""
    from rgcn_fusion.intelligence_reports import add_intelligence_reports_to_series

    data = generate_observation_series(*args, **kwargs)
    seed = int(data["metadata"].get("seed", 7) if intelligence_seed is None else intelligence_seed)
    return add_intelligence_reports_to_series(
        data,
        seed=seed,
        min_reports=min_reports_per_observation,
        max_reports=max_reports_per_observation,
    )


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic single-emitter ESM observation series."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_SERIES_COUNT,
        help="Number of emitter series entries to emit",
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for reproducible output"
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=DEFAULT_MIN_DURATION_SECONDS,
        help="Minimum series duration in seconds",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=DEFAULT_MAX_DURATION_SECONDS,
        help="Maximum series duration in seconds",
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
        help="Nominal interval between observations in seconds",
    )
    parser.add_argument(
        "--mode-switch-probability",
        type=float,
        default=0.03,
        help="Probability of a radar-mode transition between adjacent observations",
    )
    parser.add_argument(
        "--max-mode-switches",
        type=int,
        default=None,
        help="Optional cap on radar-mode transitions per series",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("generated/esm_observation_series.json"),
        help="JSON output path",
    )
    parser.add_argument(
        "--start", default="2024-01-01T00:00:00Z", help="Inclusive UTC start timestamp"
    )
    parser.add_argument(
        "--end", default="2026-01-01T00:00:00Z", help="Exclusive UTC end timestamp"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes to use; defaults to available CPU cores",
    )
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
        args.mode_switch_probability,
        args.max_mode_switches,
        args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.count} ESM observation series to {args.output}")


if __name__ == "__main__":
    main()
