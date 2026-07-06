# r-GCN Fusion Knowledge Graph Generator

This repository contains a dependency-free Python generator for a combat-aircraft
knowledge graph intended for r-GCN and Dempster-Shafer fusion experiments.

The generated graph includes:

- aircraft family and variant nodes for Russian, Chinese, Indian, US, and
  Western European combat aircraft, including MiG-29, Su-27/Su-30/Su-35,
  MiG-31, J-7/J-8/J-10/J-11/J-15/J-16/J-20, JF-17, Tejas, Rafale,
  Mirage, Gripen, Tornado, Harrier, Typhoon, F-16, F-15, F/A-18, F-22,
  F-35, and bomber/attack-helicopter examples;
- radar nodes known to be associated with those aircraft variants;
- radar-mode nodes with representative numeric pulse repetition frequency lower/upper bounds, centre frequencies,
  bandwidths, pulse widths, duty cycles, dwell times, coverage angles, resolution
  estimates, power/noise figures, detection probabilities, false-alarm rates, and
  track capacities;
- kinematic aircraft properties including maximum Mach number, service ceiling,
  combat radius, ferry range, and hardpoints;
- operator nation/organisation nodes and `OPERATES` relationships.

> Note: numeric radar and kinematic values are representative experiment inputs
> assembled from commonly published open-source descriptions. They are not a
> substitute for authoritative technical data.

## Usage

```bash
python kg_generator.py
```

By default this writes:

- `generated/aircraft_radar_kg.json` with typed nodes, properties, and edges;
- `generated/aircraft_radar_triples.csv` with `source,relation,target` triples.

Custom output paths can be supplied:

```bash
python kg_generator.py --json /tmp/kg.json --triples /tmp/triples.csv
```

## Neo4j Notebook

A Jupyter notebook at `notebooks/neo4j_kg_creation.ipynb` demonstrates how to
generate the graph, connect to a Neo4j 5 database, create uniqueness constraints,
load typed nodes and relationships, and run example Cypher inspections.

## Schema

Nodes have the following shape:

```json
{
  "id": "aircraft:f_16v_block_70_72",
  "label": "AircraftVariant",
  "properties": {
    "variant": "F-16V Block 70/72",
    "max_speed_mach": 2.0
  }
}
```

Edges have the following shape:

```json
{
  "source": "aircraft:f_16v_block_70_72",
  "relation": "USES_RADAR",
  "target": "radar:an_apg_83_sabr"
}
```

The relation vocabulary is deliberately small for r-GCN experimentation:

- `VARIANT_OF`
- `USES_RADAR`
- `HAS_MODE`
- `OPERATES`
