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
    # China and Chinese-origin combat aircraft
    AircraftVariant("J-7", "J-7G", "light fighter", "3", "KLJ-7", 2.0, 17500, 850, 2200, 5, ("China", "Bangladesh", "Myanmar", "Nigeria")),
    AircraftVariant("J-8", "J-8F", "interceptor", "3+", "Type 1493 AESA", 2.2, 20000, 800, 2200, 7, ("China",)),
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
    AircraftVariant("Tejas", "Tejas Mk1A", "light multirole fighter", "4+", "EL/M-2052", 1.6, 15200, 500, 1850, 8, ("India",)),
    AircraftVariant("Tejas", "Tejas Mk2", "multirole fighter", "4+", "Uttam AESA", 1.8, 17000, 1500, 2500, 11, ("India",)),
    AircraftVariant("MiG-21", "MiG-21 Bison", "interceptor", "3+", "Kopyo-M", 2.05, 17500, 370, 1210, 5, ("India",)),
    AircraftVariant("MiG-29", "MiG-29UPG", "multirole fighter", "4+", "Zhuk-ME", 2.25, 18000, 1000, 2400, 6, ("India",)),
    AircraftVariant("Mirage 2000", "Mirage 2000I", "multirole fighter", "4", "RDY-2", 2.2, 17060, 740, 3335, 9, ("India",)),
    AircraftVariant("Jaguar", "Jaguar IS DARIN III", "strike aircraft", "3+", "EL/M-2052", 1.6, 14000, 850, 3524, 5, ("India",)),
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
    AircraftVariant("Rafale", "Rafale C", "multirole fighter", "4+", "RBE2-AA", 1.8, 15240, 1850, 3700, 14, ("France", "Greece", "Croatia", "Egypt", "Qatar", "India", "United Arab Emirates", "Indonesia")),
    AircraftVariant("Rafale", "Rafale B", "multirole fighter", "4+", "RBE2-AA", 1.8, 15240, 1850, 3700, 14, ("France", "Greece", "Egypt", "Qatar", "India")),
    AircraftVariant("Rafale", "Rafale M", "carrier multirole fighter", "4+", "RBE2-AA", 1.8, 15240, 1850, 3700, 13, ("France",)),
    AircraftVariant("Mirage 2000", "Mirage 2000C", "interceptor", "4", "RDI", 2.2, 17060, 740, 3335, 9, ("France", "Greece", "Taiwan")),
    AircraftVariant("Mirage 2000", "Mirage 2000-5", "multirole fighter", "4", "RDY-2", 2.2, 17060, 740, 3335, 9, ("France", "Greece", "Qatar", "Taiwan", "United Arab Emirates")),
    AircraftVariant("Gripen", "JAS 39C Gripen", "multirole fighter", "4+", "PS-05/A", 2.0, 15240, 800, 3200, 8, ("Sweden", "Czech Republic", "Hungary", "South Africa", "Thailand", "Brazil")),
    AircraftVariant("Gripen", "JAS 39E Gripen", "multirole fighter", "4+", "Raven ES-05", 2.0, 16000, 1300, 4000, 10, ("Sweden", "Brazil")),
    AircraftVariant("Sea Harrier", "Sea Harrier FA2", "carrier fighter", "4", "Blue Vixen", 1.2, 16000, 750, 3600, 5, ("United Kingdom", "India")),
    AircraftVariant("Harrier", "Harrier GR.9", "V/STOL attack aircraft", "4", "Blue Fox", 0.9, 15240, 556, 3300, 7, ("United Kingdom",)),
    AircraftVariant("Tornado", "Tornado F3", "interceptor", "4", "Foxhunter", 2.2, 15240, 1390, 3890, 8, ("United Kingdom", "Italy", "Saudi Arabia")),
    AircraftVariant("Tornado", "Tornado IDS", "interdictor/strike aircraft", "4", "ECR-90", 2.2, 15240, 1390, 3890, 9, ("Germany", "Italy", "United Kingdom", "Saudi Arabia")),
    AircraftVariant("Mirage F1", "Mirage F1CR", "reconnaissance fighter", "3+", "Cyrano IV", 2.2, 20000, 425, 3300, 5, ("France", "Spain", "Morocco")),
    AircraftVariant("Super Etendard", "Super Etendard Modernise", "carrier strike aircraft", "3+", "Anemone", 1.3, 13700, 850, 1820, 5, ("France", "Argentina")),
    AircraftVariant("AMX", "AMX A-11B", "light attack aircraft", "3+", "EL/M-2032", 0.86, 13000, 889, 3330, 5, ("Italy", "Brazil")),

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
                prf_min_hz=mode.prf_min_hz,
                prf_max_hz=mode.prf_max_hz,
                centre_frequency_min_ghz=numeric_range(mode.centre_frequency_ghz, 0.02)[0],
                centre_frequency_max_ghz=numeric_range(mode.centre_frequency_ghz, 0.02)[1],
                bandwidth_min_mhz=numeric_range(mode.bandwidth_mhz)[0],
                bandwidth_max_mhz=numeric_range(mode.bandwidth_mhz)[1],
                waveform=mode.waveform,
                scan_type=mode.scan_type,
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
