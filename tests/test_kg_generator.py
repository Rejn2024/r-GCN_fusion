from kg_generator import AIRCRAFT, generate_graph


RETIRED_VARIANTS = {
    "Sea Harrier FA2",
    "Harrier GR.9",
    "Tornado F3",
    "Mirage F1CR",
    "Super Etendard Modernise",
    "MiG-21 Bison",
    "J-7G",
    "J-8F",
}


def test_inventory_excludes_retired_and_not_yet_in_service_variants():
    variants = {aircraft.variant for aircraft in AIRCRAFT}

    assert variants.isdisjoint(RETIRED_VARIANTS)
    assert "Tejas Mk2" not in variants


def test_graph_only_emits_radars_used_by_current_aircraft():
    graph = generate_graph()
    radar_names = {
        node["properties"]["name"]
        for node in graph["nodes"]
        if node["label"] == "Radar"
    }

    assert radar_names == {aircraft.radar for aircraft in AIRCRAFT}
    assert graph["metadata"]["service_snapshot_year"] == 2026


def test_corrected_variant_operator_combinations():
    operators_by_variant = {
        aircraft.variant: set(aircraft.operators) for aircraft in AIRCRAFT
    }

    assert operators_by_variant["F-15E Strike Eagle"] == {"United States"}
    assert operators_by_variant["F-16V Block 70/72"] == {
        "Bahrain",
        "Bulgaria",
        "Slovakia",
    }
    assert "Egypt" not in operators_by_variant["Su-35S"]


def test_every_variant_has_parameterized_non_radar_rf_equipment():
    graph = generate_graph()
    nodes = {node["id"]: node for node in graph["nodes"]}
    edges = graph["edges"]
    for aircraft in AIRCRAFT:
        variant_id = "aircraft:" + "".join(
            char.lower() if char.isalnum() else "_" for char in aircraft.variant
        ).strip("_")
        related = {edge["relation"]: nodes[edge["target"]] for edge in edges if edge["source"] == variant_id}
        assert {"USES_DATA_LINK", "USES_RADIO", "USES_RADAR_ALTIMETER"} <= related.keys()
        assert related["USES_DATA_LINK"]["properties"]["data_rate_kbps"] > 0
        assert related["USES_RADIO"]["properties"]["output_power_w"] > 0
        assert related["USES_RADAR_ALTIMETER"]["properties"]["measurement_max_m"] > 0
