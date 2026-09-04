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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RadarMode:
    name: str
    prf_min_hz: int
    prf_max_hz: int
    centre_frequency_ghz: float
    bandwidth_mhz: float
    waveform: str
    scan_type: str
    detection_range_km: int
    pulse_width_us: float
    duty_cycle: float
    coherent_processing_interval_ms: float
    dwell_time_ms: float
    azimuth_coverage_deg: int
    elevation_coverage_deg: int
    range_resolution_m: float
    velocity_resolution_mps: float
    instrumented_range_km: int
    peak_power_kw: float
    average_power_kw: float
    noise_figure_db: float
    probability_of_detection: float
    false_alarm_rate: float
    track_capacity: int | None = None
    notes: str = ""
    pri_modulation: str = "stable"
    intrapulse_modulation: str = "unmodulated"
    frequency_pattern: str = "fixed"
    frequency_agility_mhz: float = 0.0
    scan_period_s: float = 1.0
    polarization: str = "linear"


@dataclass(frozen=True)
class Radar:
    name: str
    band: str
    antenna: str
    modes: tuple[RadarMode, ...]
    polarization: str = "linear"


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


def numeric_range(value: int | float, spread: float = 0.1, *, floor: float = 0.0, ceiling: float | None = None) -> tuple[int | float, int | float]:
    """Return representative minimum/maximum bounds around a nominal value."""
    lower = max(floor, value * (1.0 - spread))
    upper = value * (1.0 + spread)
    if ceiling is not None:
        upper = min(ceiling, upper)
    if isinstance(value, int):
        return int(round(lower)), int(round(upper))
    return round(lower, 6), round(upper, 6)


def radar_modes(base_freq: float, long_range: int, track_capacity: int) -> tuple[RadarMode, ...]:
    """Generate a reusable mode suite around a radar's nominal X-band frequency.

    Values are representative experiment features. Numeric mode parameters are
    emitted as lower/upper bounds so downstream models can reason over numeric
    intervals rather than single-point estimates or string-only categories.
    """
    instrumented_range = int(long_range * 1.2)
    return (
        RadarMode("range_while_search", 3000, 8000, base_freq, 80.0, "pulse_doppler", "mechanical_or_electronic", long_range, 2.5, 0.020, 16.0, 80.0, 120, 60, 45.0, 18.0, instrumented_range, 6.0, 1.2, 3.5, 0.85, 1e-6, track_capacity),
        RadarMode("track_while_scan", 6000, 18000, base_freq + 0.08, 120.0, "pulse_doppler", "sector", int(long_range * 0.85), 1.8, 0.032, 24.0, 120.0, 70, 45, 30.0, 12.0, instrumented_range, 6.5, 1.6, 3.3, 0.88, 5e-7, track_capacity),
        RadarMode("single_target_track", 12000, 30000, base_freq + 0.12, 60.0, "pulse_doppler", "focused", int(long_range * 0.95), 1.2, 0.036, 32.0, 160.0, 20, 20, 20.0, 8.0, instrumented_range, 7.0, 1.8, 3.0, 0.92, 1e-7, 1),
        RadarMode("look_down_shoot_down", 15000, 45000, base_freq - 0.05, 100.0, "doppler_filtering", "sector", int(long_range * 0.65), 1.0, 0.045, 28.0, 140.0, 80, 50, 35.0, 6.0, int(instrumented_range * 0.8), 6.8, 1.9, 3.8, 0.80, 1e-6, max(1, track_capacity // 2)),
        RadarMode("air_to_ground_mapping", 1000, 5000, base_freq - 0.1, 160.0, "synthetic_aperture_or_real_beam", "ground", int(long_range * 0.45), 10.0, 0.050, 64.0, 250.0, 60, 30, 5.0, 3.0, int(instrumented_range * 0.6), 5.5, 1.5, 4.0, 0.78, 1e-5, None),
    )


def _radar_mode_signature(radar_index: int, mode_index: int) -> dict[str, float | int]:
    """Return deterministic offsets that make every radar/mode signature distinct.

    The seed table in :func:`radar_modes` captures mode-relative behaviour, while
    these small per-radar offsets ensure each concrete ``Radar``/``RadarMode``
    pair has unique observable interval bounds in the KG and synthetic ESM data.
    """
    return {
        "prf_min_hz": radar_index * 137 + mode_index * 17,
        "prf_max_hz": radar_index * 311 + mode_index * 37,
        "centre_frequency_ghz": radar_index * 0.0031 + mode_index * 0.0007,
        "bandwidth_mhz": radar_index * 0.91 + mode_index * 0.13,
        "pulse_width_us": radar_index * 0.041 + mode_index * 0.009,
        "duty_cycle": radar_index * 0.00017 + mode_index * 0.000031,
        "coherent_processing_interval_ms": radar_index * 0.83 + mode_index * 0.19,
        "dwell_time_ms": radar_index * 1.73 + mode_index * 0.31,
    }


def _with_distinct_mode_signatures(radars: dict[str, Radar]) -> dict[str, Radar]:
    """Return radars whose modes have unique observable parameter values."""
    distinct: dict[str, Radar] = {}
    mode_characteristics = {
        "range_while_search": ("staggered", "linear_fm", "stepped", 42.0, 3.8),
        "track_while_scan": ("jittered", "linear_fm", "hopping", 68.0, 2.4),
        "single_target_track": ("stable", "phase_coded", "fixed", 8.0, 0.8),
        "look_down_shoot_down": ("staggered", "phase_coded", "hopping", 54.0, 2.0),
        "air_to_ground_mapping": ("sliding", "linear_fm", "chirped", 96.0, 4.6),
    }
    for radar_index, (radar_name, radar) in enumerate(radars.items(), start=1):
        antenna = radar.antenna.lower()
        polarization = "dual_linear" if "aesa" in antenna or "pesa" in antenna else "linear"
        radar = replace(radar, polarization=polarization)
        modes: list[RadarMode] = []
        for mode_index, mode in enumerate(radar.modes, start=1):
            offsets = _radar_mode_signature(radar_index, mode_index)
            pri_modulation, intrapulse_modulation, frequency_pattern, agility, scan_period = mode_characteristics[mode.name]
            modes.append(
                replace(
                    mode,
                    prf_min_hz=mode.prf_min_hz + int(offsets["prf_min_hz"]),
                    prf_max_hz=mode.prf_max_hz + int(offsets["prf_max_hz"]),
                    centre_frequency_ghz=round(mode.centre_frequency_ghz + float(offsets["centre_frequency_ghz"]), 6),
                    bandwidth_mhz=round(mode.bandwidth_mhz + float(offsets["bandwidth_mhz"]), 6),
                    pulse_width_us=round(mode.pulse_width_us + float(offsets["pulse_width_us"]), 6),
                    duty_cycle=round(mode.duty_cycle + float(offsets["duty_cycle"]), 6),
                    coherent_processing_interval_ms=round(
                        mode.coherent_processing_interval_ms
                        + float(offsets["coherent_processing_interval_ms"]),
                        6,
                    ),
                    dwell_time_ms=round(mode.dwell_time_ms + float(offsets["dwell_time_ms"]), 6),
                    pri_modulation=pri_modulation,
                    intrapulse_modulation=intrapulse_modulation,
                    frequency_pattern=frequency_pattern,
                    frequency_agility_mhz=round(agility + radar_index * 0.37 + mode_index * 0.11, 6),
                    scan_period_s=round(scan_period + radar_index * 0.013 + mode_index * 0.007, 6),
                    polarization=polarization,
                )
            )
        distinct[radar_name] = replace(radar, modes=tuple(modes))
    return distinct


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
    "KLJ-7A": Radar("KLJ-7A", "X", "AESA", radar_modes(9.8, 170, 20)),
    "KLJ-10": Radar("KLJ-10", "X", "mechanically scanned array", radar_modes(9.5, 120, 10)),
    "KLJ-7": Radar("KLJ-7", "X", "slotted planar array", radar_modes(9.4, 105, 10)),
    "Type 1473": Radar("Type 1473", "X", "pulse-doppler array", radar_modes(9.5, 120, 10)),
    "Type 1493 AESA": Radar("Type 1493 AESA", "X", "AESA", radar_modes(9.9, 200, 24)),
    "Type 1475 AESA": Radar("Type 1475 AESA", "X", "AESA", radar_modes(10.0, 240, 30)),
    "NRIET AESA": Radar("NRIET AESA", "X", "AESA", radar_modes(9.9, 220, 24)),
    "EL/M-2032": Radar("EL/M-2032", "X", "multimode planar array", radar_modes(9.5, 150, 10)),
    "EL/M-2052": Radar("EL/M-2052", "X", "AESA", radar_modes(9.8, 200, 20)),
    "Uttam AESA": Radar("Uttam AESA", "X", "AESA", radar_modes(9.9, 180, 20)),
    "Kopyo-M": Radar("Kopyo-M", "X", "pulse-doppler array", radar_modes(9.3, 75, 8)),
    "RDI": Radar("RDI", "X", "pulse-doppler array", radar_modes(9.4, 120, 8)),
    "RDY-2": Radar("RDY-2", "X", "pulse-doppler array", radar_modes(9.6, 150, 10)),
    "AN/APG-65": Radar("AN/APG-65", "X", "slotted planar array", radar_modes(9.5, 120, 10)),
    "AN/APG-73": Radar("AN/APG-73", "X", "slotted planar array", radar_modes(9.6, 150, 10)),
    "AN/APG-79": Radar("AN/APG-79", "X", "AESA", radar_modes(10.0, 220, 20)),
    "AN/APG-77": Radar("AN/APG-77", "X", "AESA", radar_modes(10.0, 240, 30)),
    "AN/APG-81": Radar("AN/APG-81", "X", "AESA", radar_modes(10.0, 240, 20)),
    "AN/APG-78": Radar("AN/APG-78", "Ka", "millimetre-wave fire-control radar", radar_modes(34.0, 16, 16)),
    "AN/APQ-164": Radar("AN/APQ-164", "Ku", "PESA", radar_modes(15.0, 240, 20)),
    "AN/APQ-181": Radar("AN/APQ-181", "Ku", "AESA", radar_modes(15.5, 240, 20)),
    "RBE2": Radar("RBE2", "X", "PESA", radar_modes(9.7, 160, 20)),
    "RBE2-AA": Radar("RBE2-AA", "X", "AESA", radar_modes(9.9, 220, 40)),
    "PS-05/A": Radar("PS-05/A", "X", "mechanically scanned array", radar_modes(9.6, 120, 10)),
    "Raven ES-05": Radar("Raven ES-05", "X", "AESA", radar_modes(9.9, 200, 20)),
    "Blue Vixen": Radar("Blue Vixen", "X", "pulse-doppler array", radar_modes(9.5, 150, 10)),
    "Blue Fox": Radar("Blue Fox", "I", "monopulse radar", radar_modes(9.2, 60, 2)),
    "Foxhunter": Radar("Foxhunter", "I/J", "pulse-doppler array", radar_modes(9.4, 185, 12)),
    "ECR-90": Radar("ECR-90", "X", "mechanically scanned array", radar_modes(9.6, 150, 10)),
    "Cyrano IV": Radar("Cyrano IV", "I/J", "monopulse radar", radar_modes(9.2, 70, 2)),
    "Anemone": Radar("Anemone", "I/J", "pulse-doppler array", radar_modes(9.4, 100, 8)),
}


RADARS = _with_distinct_mode_signatures(RADARS)


AIRCRAFT: tuple[AircraftVariant, ...] = (
    AircraftVariant("MiG-29", "MiG-29A", "multirole fighter", "4", "N019 Rubin", 2.25, 18000, 700, 2100, 6, ("Ukraine", "Poland")),
    AircraftVariant("MiG-29", "MiG-29S", "multirole fighter", "4", "N019 Rubin", 2.25, 18000, 700, 2100, 6, ("Russia", "Belarus")),
    AircraftVariant("MiG-29", "MiG-29SMT", "multirole fighter", "4+", "Zhuk-ME", 2.25, 18000, 1000, 2400, 6, ("Russia", "Algeria")),
    AircraftVariant("MiG-29", "MiG-29K", "carrier multirole fighter", "4+", "Zhuk-ME", 2.0, 17500, 850, 2000, 8, ("Russia", "India")),
    AircraftVariant("MiG-29", "MiG-35", "multirole fighter", "4++", "Zhuk-AE", 2.25, 17500, 1000, 3100, 9, ("Russia",)),
    AircraftVariant("Su-27", "Su-27S", "air superiority fighter", "4", "N001 Myech", 2.35, 19000, 1340, 3530, 10, ("Russia", "Ukraine")),
    AircraftVariant("Su-27", "Su-30MKI", "multirole fighter", "4+", "N011M Bars", 2.0, 17300, 1500, 3000, 12, ("India",)),
    AircraftVariant("Su-27", "Su-30SM", "multirole fighter", "4+", "N011M Bars", 2.0, 17300, 1500, 3000, 12, ("Russia", "Kazakhstan", "Belarus")),
    AircraftVariant("Su-27", "Su-35S", "air superiority fighter", "4++", "Irbis-E", 2.25, 18000, 1600, 3600, 12, ("Russia", "China")),
    AircraftVariant("MiG-31", "MiG-31BM", "interceptor", "4+", "Zaslon-M", 2.83, 20600, 720, 3000, 8, ("Russia",)),
    AircraftVariant("Typhoon", "Typhoon Tranche 1", "multirole fighter", "4+", "CAPTOR-M", 2.0, 19800, 1389, 2900, 13, ("United Kingdom", "Germany", "Italy", "Spain", "Austria")),
    AircraftVariant("Typhoon", "Typhoon Tranche 3", "multirole fighter", "4+", "CAPTOR-M", 2.0, 19800, 1389, 2900, 13, ("United Kingdom", "Germany", "Italy", "Spain")),
    AircraftVariant("Typhoon", "Typhoon Tranche 3A (export)", "multirole fighter", "4+", "CAPTOR-E", 2.0, 19800, 1389, 2900, 13, ("Kuwait", "Qatar")),
    AircraftVariant("F-16", "F-16C/D Block 50", "multirole fighter", "4", "AN/APG-68", 2.05, 15240, 550, 4220, 9, ("United States", "Turkey", "Greece", "Poland", "South Korea")),
    AircraftVariant("F-16", "F-16E/F Block 60", "multirole fighter", "4+", "AN/APG-80", 2.0, 15240, 550, 4220, 11, ("United Arab Emirates",)),
    AircraftVariant("F-16", "F-16V Block 70/72", "multirole fighter", "4+", "AN/APG-83 SABR", 2.0, 15240, 550, 4220, 9, ("Bahrain", "Slovakia", "Bulgaria")),
    AircraftVariant("F-15", "F-15C", "air superiority fighter", "4", "AN/APG-63", 2.5, 20000, 1061, 5550, 9, ("United States", "Saudi Arabia", "Israel")),
    AircraftVariant("F-15", "F-15SA", "strike fighter", "4+", "AN/APG-63(V)3", 2.5, 20000, 1270, 3900, 11, ("Saudi Arabia",)),
    AircraftVariant("F-15", "F-15E Strike Eagle", "strike fighter", "4", "AN/APG-70", 2.5, 18200, 1270, 3900, 11, ("United States",)),
    AircraftVariant("F-15", "F-15EX Eagle II", "strike fighter", "4++", "AN/APG-82(V)1", 2.5, 18200, 1270, 3900, 12, ("United States",)),
    # China and Chinese-origin combat aircraft
    AircraftVariant("J-10", "J-10A", "multirole fighter", "4", "KLJ-10", 2.2, 18000, 550, 1850, 11, ("China",)),
    AircraftVariant("J-10", "J-10B", "multirole fighter", "4+", "Type 1473", 2.2, 18000, 550, 1850, 11, ("China",)),
    AircraftVariant("J-10", "J-10C", "multirole fighter", "4+", "Type 1493 AESA", 2.2, 18000, 550, 1850, 11, ("China", "Pakistan")),
    AircraftVariant("JF-17", "JF-17 Block I", "light multirole fighter", "4", "KLJ-7", 1.6, 16900, 1350, 2037, 7, ("Pakistan", "Myanmar", "Nigeria")),
    AircraftVariant("JF-17", "JF-17 Block III", "light multirole fighter", "4+", "KLJ-7A", 1.6, 16900, 1350, 2037, 7, ("Pakistan")),
    AircraftVariant("J-11", "J-11B", "air superiority fighter", "4", "Type 1473", 2.35, 19000, 1500, 3530, 10, ("China",)),
    AircraftVariant("J-15", "J-15", "carrier multirole fighter", "4+", "Type 1473", 2.4, 20000, 1200, 3500, 12, ("China",)),
    AircraftVariant("J-16", "J-16", "multirole strike fighter", "4+", "NRIET AESA", 2.0, 17300, 1500, 3000, 12, ("China",)),
    AircraftVariant("J-20", "J-20A", "stealth air superiority fighter", "5", "Type 1475 AESA", 2.0, 20000, 1200, 5500, 6, ("China",)),

    # India-operated and Indian-origin combat aircraft
    AircraftVariant("Tejas", "Tejas Mk1", "light multirole fighter", "4", "EL/M-2032", 1.6, 15200, 500, 1850, 8, ("India",)),
    AircraftVariant("MiG-29", "MiG-29UPG", "multirole fighter", "4+", "Zhuk-ME", 2.25, 18000, 1000, 2400, 6, ("India",)),
    AircraftVariant("Mirage 2000", "Mirage 2000I", "multirole fighter", "4", "RDY-2", 2.2, 17060, 740, 3335, 9, ("India",)),
    AircraftVariant("Rafale", "Rafale EH/DH", "multirole fighter", "4+", "RBE2-AA", 1.8, 15240, 1850, 3700, 14, ("India",)),

    # United States combat aircraft
    AircraftVariant("F/A-18", "F/A-18C Hornet", "carrier multirole fighter", "4", "AN/APG-73", 1.8, 15240, 740, 3300, 9, ("United States", "Finland", "Switzerland", "Spain", "Kuwait")),
    AircraftVariant("F/A-18", "F/A-18E Super Hornet", "carrier multirole fighter", "4+", "AN/APG-79", 1.8, 15240, 722, 3330, 11, ("United States", "Australia", "Kuwait")),
    AircraftVariant("F/A-18", "F/A-18F Super Hornet", "carrier multirole fighter", "4+", "AN/APG-79", 1.8, 15240, 722, 3330, 11, ("United States", "Australia", "Kuwait")),
    AircraftVariant("EA-18G", "EA-18G Growler", "electronic attack aircraft", "4+", "AN/APG-79", 1.8, 15240, 722, 3330, 9, ("United States", "Australia")),
    AircraftVariant("F-22", "F-22A Raptor", "stealth air superiority fighter", "5", "AN/APG-77", 2.25, 19800, 850, 2960, 4, ("United States",)),
    AircraftVariant("F-35", "F-35A Lightning II", "stealth multirole fighter", "5", "AN/APG-81", 1.6, 15240, 1239, 2200, 10, ("United States", "United Kingdom", "Italy", "Netherlands", "Norway", "Denmark", "Belgium", "Poland", "Germany", "Finland", "Switzerland")),
    AircraftVariant("F-35", "F-35B Lightning II", "STOVL stealth multirole fighter", "5", "AN/APG-81", 1.6, 15240, 935, 1670, 10, ("United States", "United Kingdom", "Italy")),
    AircraftVariant("F-35", "F-35C Lightning II", "carrier stealth multirole fighter", "5", "AN/APG-81", 1.6, 15240, 1240, 2520, 10, ("United States",)),
    AircraftVariant("AV-8B", "AV-8B Harrier II Plus", "V/STOL attack aircraft", "4", "AN/APG-65", 0.9, 15240, 556, 3300, 7, ("United States", "Italy", "Spain")),
    AircraftVariant("AH-64", "AH-64E Apache Guardian", "attack helicopter", "rotary", "AN/APG-78", 0.29, 6100, 480, 1900, 4, ("United States", "United Kingdom", "Netherlands", "Greece", "India")),
    AircraftVariant("B-1", "B-1B Lancer", "strategic bomber", "bomber", "AN/APQ-164", 1.25, 18000, 5543, 12000, 8, ("United States",)),
    AircraftVariant("B-2", "B-2A Spirit", "stealth strategic bomber", "bomber", "AN/APQ-181", 0.95, 15240, 6000, 11100, 2, ("United States",)),

    # Western European combat aircraft and variants
    AircraftVariant("Rafale", "Rafale C", "multirole fighter", "4+", "RBE2-AA", 1.8, 15240, 1850, 3700, 14, ("France",)),
    AircraftVariant("Rafale", "Rafale B", "multirole fighter", "4+", "RBE2-AA", 1.8, 15240, 1850, 3700, 14, ("France",)),
    AircraftVariant("Rafale", "Rafale M", "carrier multirole fighter", "4+", "RBE2-AA", 1.8, 15240, 1850, 3700, 13, ("France",)),
    AircraftVariant("Mirage 2000", "Mirage 2000-5", "multirole fighter", "4", "RDY-2", 2.2, 17060, 740, 3335, 9, ("France", "Greece", "Qatar", "Taiwan")),
    AircraftVariant("Gripen", "JAS 39C Gripen", "multirole fighter", "4+", "PS-05/A", 2.0, 15240, 800, 3200, 8, ("Sweden", "Czech Republic", "Hungary", "South Africa", "Thailand")),
    AircraftVariant("Gripen", "JAS 39E Gripen", "multirole fighter", "4+", "Raven ES-05", 2.0, 16000, 1300, 4000, 10, ("Sweden", "Brazil")),
    AircraftVariant("Tornado", "Tornado IDS", "interdictor/strike aircraft", "4", "ECR-90", 2.2, 15240, 1390, 3890, 9, ("Germany", "Italy", "Saudi Arabia")),

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

    # Only emit radars that participate in a current aircraft/radar/operator
    # combination.  This keeps retired and developmental systems from remaining
    # selectable as orphan classes after the aircraft inventory is pruned.
    active_radar_names = {aircraft.radar for aircraft in AIRCRAFT}
    for radar in (radar for name, radar in RADARS.items() if name in active_radar_names):
        radar_id = f"radar:{slug(radar.name)}"
        agility_bounds = [
            bound
            for mode in radar.modes
            for bound in numeric_range(mode.frequency_agility_mhz)
        ]
        scan_period_bounds = [
            bound for mode in radar.modes for bound in numeric_range(mode.scan_period_s)
        ]
        add_node(
            nodes,
            radar_id,
            "Radar",
            name=radar.name,
            band=radar.band,
            antenna=radar.antenna,
            polarization=radar.polarization,
            supported_pri_modulations=sorted(
                {mode.pri_modulation for mode in radar.modes}
            ),
            supported_intrapulse_modulations=sorted(
                {mode.intrapulse_modulation for mode in radar.modes}
            ),
            supported_frequency_patterns=sorted(
                {mode.frequency_pattern for mode in radar.modes}
            ),
            frequency_agility_min_mhz=min(agility_bounds),
            frequency_agility_max_mhz=max(agility_bounds),
            scan_period_min_s=min(scan_period_bounds),
            scan_period_max_s=max(scan_period_bounds),
        )
        for mode in radar.modes:
            mode_id = f"radar_mode:{slug(radar.name)}:{slug(mode.name)}"
            add_node(
                nodes,
                mode_id,
                "RadarMode",
                name=mode.name,
                prf_min_hz=mode.prf_min_hz,
                prf_max_hz=mode.prf_max_hz,
                centre_frequency_min_ghz=numeric_range(mode.centre_frequency_ghz, 0.02)[0],
                centre_frequency_max_ghz=numeric_range(mode.centre_frequency_ghz, 0.02)[1],
                bandwidth_min_mhz=numeric_range(mode.bandwidth_mhz)[0],
                bandwidth_max_mhz=numeric_range(mode.bandwidth_mhz)[1],
                waveform=mode.waveform,
                scan_type=mode.scan_type,
                pri_modulation=mode.pri_modulation,
                intrapulse_modulation=mode.intrapulse_modulation,
                frequency_pattern=mode.frequency_pattern,
                frequency_agility_min_mhz=numeric_range(mode.frequency_agility_mhz)[0],
                frequency_agility_max_mhz=numeric_range(mode.frequency_agility_mhz)[1],
                scan_period_min_s=numeric_range(mode.scan_period_s)[0],
                scan_period_max_s=numeric_range(mode.scan_period_s)[1],
                polarization=mode.polarization,
                detection_range_min_km=numeric_range(mode.detection_range_km, 0.25)[0],
                detection_range_max_km=numeric_range(mode.detection_range_km, 0.25)[1],
                pulse_width_min_us=numeric_range(mode.pulse_width_us)[0],
                pulse_width_max_us=numeric_range(mode.pulse_width_us)[1],
                duty_cycle_min=numeric_range(mode.duty_cycle)[0],
                duty_cycle_max=numeric_range(mode.duty_cycle)[1],
                coherent_processing_interval_min_ms=numeric_range(mode.coherent_processing_interval_ms)[0],
                coherent_processing_interval_max_ms=numeric_range(mode.coherent_processing_interval_ms)[1],
                dwell_time_min_ms=numeric_range(mode.dwell_time_ms)[0],
                dwell_time_max_ms=numeric_range(mode.dwell_time_ms)[1],
                azimuth_coverage_min_deg=numeric_range(mode.azimuth_coverage_deg)[0],
                azimuth_coverage_max_deg=numeric_range(mode.azimuth_coverage_deg)[1],
                elevation_coverage_min_deg=numeric_range(mode.elevation_coverage_deg)[0],
                elevation_coverage_max_deg=numeric_range(mode.elevation_coverage_deg)[1],
                range_resolution_min_m=numeric_range(mode.range_resolution_m)[0],
                range_resolution_max_m=numeric_range(mode.range_resolution_m)[1],
                velocity_resolution_min_mps=numeric_range(mode.velocity_resolution_mps)[0],
                velocity_resolution_max_mps=numeric_range(mode.velocity_resolution_mps)[1],
                instrumented_range_min_km=numeric_range(mode.instrumented_range_km, 0.25)[0],
                instrumented_range_max_km=numeric_range(mode.instrumented_range_km, 0.25)[1],
                peak_power_min_kw=numeric_range(mode.peak_power_kw)[0],
                peak_power_max_kw=numeric_range(mode.peak_power_kw)[1],
                average_power_min_kw=numeric_range(mode.average_power_kw)[0],
                average_power_max_kw=numeric_range(mode.average_power_kw)[1],
                noise_figure_min_db=numeric_range(mode.noise_figure_db)[0],
                noise_figure_max_db=numeric_range(mode.noise_figure_db)[1],
                probability_of_detection_min=numeric_range(mode.probability_of_detection, ceiling=1.0)[0],
                probability_of_detection_max=numeric_range(mode.probability_of_detection, ceiling=1.0)[1],
                false_alarm_rate_min=numeric_range(mode.false_alarm_rate, 0.5)[0],
                false_alarm_rate_max=numeric_range(mode.false_alarm_rate, 0.5)[1],
                track_capacity_min=mode.track_capacity,
                track_capacity_max=mode.track_capacity,
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

    return {"metadata": {"schema_version": "1.0", "service_snapshot_year": 2026, "node_count": len(nodes), "edge_count": len(edges)}, "nodes": list(nodes.values()), "edges": edges}


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
