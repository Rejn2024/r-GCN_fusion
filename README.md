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
- radar-mode nodes with representative numeric lower/upper bounds for pulse repetition frequency,
  centre frequency, bandwidth, detection/instrumented range, pulse width, duty cycle,
  dwell time, coverage angles, resolution estimates, power/noise figures, detection
  probabilities, false-alarm rates, and track capacities;
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


## Synthetic ESM Observation Generator

The repository also includes `esm_observation_generator.py`, which creates
synthetic passive-ESM observations whose radar parameters are sampled from the
radar-mode bounds already present in the KG. Each observation includes:

- ESM-deducible radar features such as PRF ranges, centre frequency, bandwidth,
  waveform, scan type, pulse width, duty cycle, CPI, dwell time, coverage,
  resolution, power/noise estimates, and track-capacity hints where present;
- an estimated emitter latitude/longitude with an error box;
- approximate kinematic estimates for speed, altitude, and heading with errors;
- Unix and ISO-8601 UTC timestamps;
- a ground-truth aircraft variant/operator/radar/mode label;
- additional candidate labels sharing KG features, so some observations are
  intentionally compatible with multiple aircraft or variants.

Generate a reproducible sample with:

```bash
python esm_observation_generator.py --count 100 --seed 7 --output generated/esm_observations.json
```

Date bounds can be customized with UTC ISO-8601 timestamps:

```bash
python esm_observation_generator.py --start 2025-01-01T00:00:00Z --end 2025-02-01T00:00:00Z
```

A Jupyter walkthrough at `notebooks/esm_observation_generation_demo.ipynb`
demonstrates in-memory generation, inspection of uncertainty fields, KG label
validation, ambiguity candidates, and JSON export for downstream experiments.


## r-GCN Training Targets

The training pipeline now supports multi-task node classification in addition to
Dempster-Shafer mass prediction. Enable `data.classification` or provide
`data.classification_label_properties` to train shared r-GCN embeddings with
categorical heads for:

- `radar_type` (for example, a `radar_id` target);
- `radar_mode` (for example, a `mode_id` target);
- `aircraft_variant` (for example, an `aircraft_id` target);
- `operator` (for example, an operator name or id target).

Classification loss is added to the evidential KL-divergence objective and can
be scaled with `training.classification_loss_weight`. The emitted
`node_evidence.json` includes Dempster-Shafer masses, belief/plausibility
intervals, and per-task class predictions.

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
