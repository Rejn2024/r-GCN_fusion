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


## Observation-to-Neo4j ETL

`rgcn_fusion.observation_etl` fills the bridge between synthetic ESM
observations and r-GCN training. It compares each observation against the
`RadarMode` nodes already loaded in Neo4j, scores interval overlap for measured
radar parameters plus waveform/scan-type and kinematic consistency, and writes
an `EvidenceEntity` subgraph for training.

Run the ETL after loading the base KG into Neo4j:

```bash
rgcn-fusion-load-observations --observations generated/esm_observations.json \
  --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password password
```

The ETL creates `Observation` and `CandidateEvidence` nodes, both labelled
`EvidenceEntity`, with `degree_score`, `text_score`, `recency_score`,
`ds_masses`, and optional class-label properties (`radar_id`, `mode_id`,
`aircraft_id`, and `operator`). It also creates `HAS_CANDIDATE`,
`CONTRADICTS_CANDIDATE`, `GROUND_TRUTH_CANDIDATE`, and `SHARES_BEST_MODE`
relationships so `rgcn_fusion.train` can load the evidence subgraph with the
example `EvidenceEntity` Cypher queries. `CONTRADICTS_CANDIDATE` is directed
from a stronger-scored candidate to a weaker incompatible candidate for the same
observation, with `score_delta` and `reason` properties explaining which
hypothesis fields differ.

A Jupyter walkthrough at
`notebooks/observation_etl_rgcn_end_to_end.ipynb` shows the full process:
base KG loading, observation generation, ETL, evidence-graph inspection, and
r-GCN training.


## r-GCN Training Targets

The training pipeline supports node classification in addition to
Dempster-Shafer mass prediction. Enable `data.classification` or provide
`data.classification_label_properties` to train the shared r-GCN for classification targets such as:

- `radar_type` (for example, a `radar_id` target);
- `radar_mode` (for example, a `mode_id` target);
- `aircraft_variant` (for example, an `aircraft_id` target);
- `operator_country` (for example, an operator country/name target).

Classification loss is added to the evidential KL-divergence objective and can
be scaled globally with `training.classification_loss_weight`. Individual tasks
can be emphasized with `training.classification_task_loss_weights`; by default,
`aircraft_variant` and `operator_country` receive 2.0x multipliers so their
accuracy metrics are prioritized during training. Tasks whose class count
matches the configured hypotheses are scored from the midpoint of each singleton
hypothesis' Dempster-Shafer belief/plausibility interval. Targets with a
different vocabulary size use lightweight per-task classifier heads on the
shared r-GCN embedding, so metadata labels such as radar type or aircraft
variant do not need to match the hypothesis set. The emitted `node_evidence.json`
includes Dempster-Shafer masses, belief/plausibility intervals, and per-task
class predictions.

For leakage-safe observation-level radar-mode experiments, the example config
filters supervised loss/metrics to `Observation` nodes, groups splits by
`series_id` when that property is present, removes cross-split edges, and drops
truth-only or candidate-derived shortcut relations such as
`GROUND_TRUTH_CANDIDATE` and `SHARES_BEST_MODE`. This prevents candidate-node
`mode_id` properties from being treated as observation truth labels and avoids
message passing across train/test/validation series.

The example training config enables the enhanced architecture for noisy
candidate-evidence graphs:

- `data.recommended_candidate_features: true` appends a richer feature set for
  ESM/candidate matching signals, including radar interval overlap,
  waveform/scan-type matches, numeric residuals, kinematic consistency,
  uncertainty width, ambiguity count, and missing-feature count. Missing Neo4j
  properties are projected as `0.0`, so the same config can run before every
  feature has been materialised.
- `model.num_layers`, `model.residual`, and `model.normalization` build a
  configurable residual r-GCN stack instead of a fixed two-layer encoder;
  training enforces a minimum of five r-GCN layers.
- `model.num_bases` enables r-GCN basis decomposition to share parameters
  across relations.
- `model.relation_gates` learns an importance gate for each relation type.
- `model.task_head_hidden_features` replaces single-linear auxiliary heads with
  small MLP task heads.
- `model.mass_head_type: dirichlet` predicts non-negative evidence, Dirichlet
  concentration parameters, normalized masses, and an uncertainty scalar for
  each node.
- `training.l1_lambda`, `training.mass_label_smoothing`,
  `training.classification_label_smoothing`, `training.confidence_penalty_weight`,
  `training.max_grad_norm`, `training.reduce_lr_on_plateau`, `training.patience`,
  and `training.early_stopping_min_delta` combine L1 regularization, softened
  targets, entropy-based confidence penalties, gradient clipping, plateau-based
  learning-rate reduction, and aggressive validation-loss early stopping to slow
  rapid overfitting.

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
