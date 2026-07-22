# Presentation Brief: r-GCN Fusion Knowledge Graph

## Purpose and scope

This document is source material for an AI-assisted PowerPoint presentation about
this repository.  It describes the implemented experimental workflow, separates
implemented capabilities from proposed extensions, and supplies a suggested slide
story.  The project is a research/prototyping pipeline for **aircraft/radar
identification under uncertainty**: a knowledge graph (KG) provides structured
context, synthetic passive ESM observations provide controlled data, an r-GCN
learns relational evidence patterns, and Dempster-Shafer (DS) theory expresses
both support and residual uncertainty.

**Important framing:** the aircraft and radar numbers are representative,
open-source-inspired simulation inputs.  They are not authoritative technical
performance data and the repository is not an operational identification system.

## Core concepts

- **Knowledge graph:** a typed graph of entities and relationships, rather than
  an unstructured table.  It makes the chain from aircraft variant to radar to
  radar mode to operator explicit.
- **Passive ESM observation:** a simulated measurement of radar-emission
  characteristics.  It can constrain candidate emitters but is intentionally
  uncertain and can be compatible with several candidates.
- **Evidence fusion:** candidate-match evidence is represented as DS masses, so
  the workflow can distinguish evidence *for*, evidence *against*, and
  uncommitted/ambiguous evidence.
- **Relational learning:** a relational graph convolutional network (r-GCN)
  exchanges messages over typed edges.  This lets a node's prediction use its
  own matching features and the structure of related evidence nodes.

## End-to-end methodology

Use this as the central workflow diagram in a presentation:

```text
Curated representative seed tables
        -> procedural KG generator -> JSON/triples -> Neo4j
                                                    |
Synthetic ESM generator -> uncertain observations -+-> candidate-scoring ETL
                                                        -> evidence subgraph
                                                        -> leakage-safe splits
                                                        -> r-GCN
                                                        -> DS masses, belief/
                                                           plausibility, and
                                                           optional class labels
```

1. Encode radar, radar-mode, aircraft-variant, family, and operator facts in a
   procedural Python seed model.
2. Generate a KG with typed nodes, stable IDs, properties, and a small,
   interpretable relation vocabulary.
3. Generate reproducible synthetic ESM records by sampling values from the KG
   mode intervals and adding measurement uncertainty.
4. Score each observation against KG radar-mode candidates using interval,
   categorical, kinematic, and optional external-context evidence.
5. Load observations and candidates into Neo4j as an evidence graph.
6. Train a shared r-GCN to predict DS masses and optional categorical labels.
7. Report normalized masses, uncertainty, belief/plausibility intervals, class
   probabilities, train/test metrics, and training history.

## Knowledge-graph ontology

### Entity types

| Node label | Meaning | Examples of important properties |
|---|---|---|
| `AircraftFamily` | Coarse platform family | `name` |
| `AircraftVariant` | Specific platform variant | family, variant, role, generation, maximum Mach, service ceiling, combat/ferry range, hardpoints, tags |
| `Radar` | Radar associated with one or more variants | name, band, antenna type |
| `RadarMode` | Operating mode of a radar | waveform, scan type, interval-valued radar parameters, track capacity |
| `Operator` | Nation/organisation operating a variant | name |
| `Observation` + `EvidenceEntity` | Synthetic ESM observation after ETL | scores, DS masses, best candidate fields; optional offline labels |
| `CandidateEvidence` + `EvidenceEntity` | A ranked KG interpretation of one observation | candidate identity, scores, matching/residual/uncertainty features, DS masses |

### Base KG relation vocabulary

```text
(:AircraftVariant)-[:VARIANT_OF]->(:AircraftFamily)
(:AircraftVariant)-[:USES_RADAR]->(:Radar)
(:Radar)-[:HAS_MODE]->(:RadarMode)
(:Operator)-[:OPERATES]->(:AircraftVariant)
```

The deliberately small vocabulary is suitable for r-GCN experimentation.  A
fresh generated graph currently contains 458 nodes and 566 edges: 34 aircraft
families, 66 variants, 50 radars, 250 radar modes, and 58 operators.  This is
not a claim about real-world force structure; it is the current generated
experiment graph.

### Radar-mode representation

Modes include range-while-search, track-while-scan, single-target-track,
look-down/shoot-down, and air-to-ground mapping.  Numeric mode attributes are
stored as lower/upper intervals rather than point values.  They include PRF,
centre frequency, bandwidth, detection/instrumented range, pulse width, duty
cycle, coherent processing interval, dwell time, coverage angles, range and
velocity resolution, power, noise figure, detection probability, false-alarm
rate, and track capacity.  This makes interval-based matching possible.

The generator starts from reusable mode templates, then applies deterministic
per-radar/per-mode offsets to observable fields.  Therefore, every concrete
radar/mode pair has a distinct signature while retaining mode-relative
behaviour.  Nominal numeric attributes are expanded into representative bounds
(usually +/-10%; selected values use different spreads and physical ceilings).

## Synthetic data preparation

### Single observations

The synthetic generator selects an aircraft variant, one of its operators, its
associated radar, and a radar mode.  It samples directly measurable parameters
from that mode's KG intervals, then attaches an uncertainty interval to each
measurement.  Directly measured fields are centre frequency, bandwidth, PRF,
pulse width, duty cycle, coherent processing interval, and dwell time; PRI is
derived from sampled PRF.

Each record also contains:

- a timestamp sampled within a configurable UTC range (default 2024-01-01 to
  2026-01-01);
- a synthetic passive-ESM sensor descriptor;
- an estimated location in one of several training areas plus latitude/longitude
  error box;
- approximate speed, altitude, and heading with error bounds;
- a ground-truth label for offline evaluation only; and
- alternative candidates sharing a radar or, in certain modes, aircraft-family
  characteristics, deliberately creating ambiguity.

The generator accepts a random seed (default 7), so datasets can be reproduced.
Ground truth must not be used as an inference feature.

### Single-emitter time series

The repository also implements a series generator.  A series is one emitter
track with repeated observations at a default 0.5-second interval and a default
duration range of 1--60 seconds.  Location is moved using an equirectangular
approximation; kinematics evolve with small Gaussian perturbations.  A sampled
mode schedule can include radar-mode switches.  The generator provides a helper
that strips `ground_truth_label` before inference.

This supports a useful presentation point: a mode change should not be mistaken
for a platform change.  Mode evidence may need segmentation, whereas parent
radar, aircraft, and operator evidence can remain continuous.

## Observation ETL and evidence-graph preparation

The ETL queries Neo4j for each `RadarMode` and its associated radar, aircraft,
and operator context.  It ranks up to five candidates per observation by a
weighted score:

```text
total = 0.75 * radar-mode match
      + 0.15 * aircraft compatibility
      + 0.10 * optional external operator/context prior
```

- **Radar-mode match:** interval-overlap scoring for measured values, plus exact
  waveform and scan-type agreement.  When intervals do not overlap, the score
  degrades according to normalized centre distance rather than becoming a hard
  rejection.
- **Aircraft compatibility:** 80% speed/altitude kinematic consistency and 20%
  KG aircraft-to-radar compatibility.
- **Operator prior:** optional external deployment/context input only; it does
  not read `ground_truth_label`.

For each observation, the ETL writes one `Observation` node and ranked
`CandidateEvidence` nodes.  It adds `HAS_CANDIDATE` edges, directed
`CONTRADICTS_CANDIDATE` edges from a materially stronger incompatible candidate
to a weaker one, and `SHARES_BEST_MODE` links between observations with the same
best mode.  Offline-only truth edges can be enabled explicitly.  Candidate nodes
include baseline scores plus interval-overlap, waveform/scan match, normalized
residual, kinematic-consistency, uncertainty-width, ambiguity-count, and
missing-feature signals.

## Dempster-Shafer theory in this repository

### Why DS is used

Ordinary probabilities force all support onto singleton classes.  DS theory
allows mass on a set of hypotheses, representing evidence that supports a group
without resolving which member is correct.  For a two-hypothesis candidate
match frame, the mass-vector order is:

```text
[{non-match}, {match}, {non-match, match}]
```

The ETL derives these masses from candidate score and ambiguity.  The first two
masses are committed support against/for a match; the last is uncertainty.  The
mass vector is normalized to sum to 1.

For a singleton hypothesis H:

- **Belief(H)** is the mass assigned exactly to H: conservative committed
  support.
- **Plausibility(H)** is the sum of all masses whose focal set intersects H:
  the maximum support still compatible with H.
- The interval `[belief, plausibility]` makes ambiguity visible.

### Combination rule

The DS utility combines two normalized mass vectors using Dempster's normalized
rule.  It multiplies focal-element masses, assigns products with empty set
intersection to conflict, accumulates non-empty intersections, and divides by
`1 - conflict`.  Total conflict is rejected as undefined.  This is suitable for
combining independent evidence sources, but the presentation should state that
independence and calibration need careful validation in any real deployment.

### Constrained frame of discernment

The full frame of discernment is the configured list of mutually exclusive
singleton hypotheses.  Each focal element is represented by a bit mask.  The
implementation constrains the representation to avoid exponential growth:

- For **10 or fewer** hypotheses, it uses every non-empty subset: `2^n - 1`
  focal elements.
- For **more than 10**, it uses singleton masks, inferred multi-variant aircraft
  type/family group masks where labels make those groups available (for example,
  MiG-29 variants), and one full-frame uncertainty mask.

This is an engineering approximation: it preserves direct identity support,
coarse type-level ambiguity, and "I do not know" uncertainty without attempting
to learn all possible subsets.  It is not the unrestricted power set for large
frames.  A two-hypothesis example in `configs/example.yaml` (`benign`,
`suspicious`) therefore has exactly three masses in the order shown above.

## Neural-network architecture

### Encoder

The model starts with a linear input projection, optional LayerNorm, GELU, and
dropout.  It then applies a configurable stack of residual r-GCN blocks (the
training pipeline enforces a minimum of five blocks).  Each block performs:

```text
self-loop linear transform + degree-normalized sum of relation-specific messages
-> GELU -> optional LayerNorm -> dropout -> optional residual addition
```

Relation weights can use basis decomposition: a small set of basis matrices is
mixed into one transform per relation, reducing parameters and sharing
statistical strength across relation types.  Optional sigmoid relation gates
learn a relation-specific message importance.  Edge chunking limits temporary
message memory, and optional gradient checkpointing trades extra computation for
lower activation memory.

### Output heads and objectives

The shared embedding feeds an evidential mass head and optional classification
heads.  The mass head can use softmax or the recommended Dirichlet approach:
softplus logits become non-negative evidence, `alpha = evidence + 1`, normalized
alpha values become masses, and uncertainty is `number_of_masses / sum(alpha)`.

The primary objective is KL divergence between predicted and target DS masses.
Optional tasks predict radar type, radar mode, aircraft variant, and operator
country.  If a task has the same cardinality as the DS hypothesis frame, its
scores are belief/plausibility midpoints; otherwise it uses a small MLP (or
linear) task head.  The total training loss adds weighted classification loss,
optional L1 regularization, and an entropy-based penalty that discourages
overconfident outputs.  Label smoothing, gradient clipping, AdamW, learning-rate
reduction on validation plateaus, and validation-loss early stopping are used to
reduce rapid overfitting.

## Data splits, evaluation, and leakage controls

The configured example uses a deterministic seed (default 42) and a 50% / 30%
/ 20% **train / test / validation** split.  Fractions are configurable and must
sum to one.  The split is created only from supervised nodes; the example
restricts supervised loss and metrics to nodes carrying the `Observation` label.

When `series_id` is present, all supervised observations in a series are placed
in the same split.  This prevents nearly adjacent records from the same emitter
track leaking across train and test.  With `remove_cross_split_edges: true`,
message-passing edges connecting supervised nodes in different splits are
removed.  The example also excludes `GROUND_TRUTH_CANDIDATE` and
`SHARES_BEST_MODE` relations from message passing, because they can create
truth-derived or candidate-derived shortcuts.  Candidate evidence may remain in
the graph for structural context but is not treated as an observation truth
label.

Training reports train, test, and validation losses each epoch; the best
checkpoint is selected by validation loss.  Final artifacts include model
checkpoints, `node_evidence.json` (masses, intervals, uncertainty, and classes),
history/metrics JSON, TensorBoard logs, and a metrics plot when matplotlib is
available.  The project reports test metrics, but a future presentation should
not imply a held-out operational performance claim without a documented dataset
and experimental results.

## Features of note and caveats

- The base KG generator has no runtime dependencies and emits both JSON and CSV
  triples; Neo4j is used for the observation/evidence graph and training loader.
- Stable, typed IDs make the generated KG, synthetic labels, and Neo4j graph
  joinable.
- The candidate scoring code explicitly avoids truth labels unless the
  offline-only option is enabled.
- The repository contains notebooks for KG creation, ESM generation, ETL,
  classification, advanced-network experiments, and DS identification demos.
- A documented **proposed extension** recommends `EmitterTrack` and
  `ModeSegment` nodes, temporal links, and conflict-aware segmentation.  Present
  it as a design proposal unless the corresponding graph/ETL implementation is
  added; it is not the same as the existing series generator.
- Synthetic observations are valuable for reproducible pipeline tests and
  ablations, but they do not establish real-world sensor performance,
  distributional validity, robustness to adversarial emissions, or calibrated
  DS independence assumptions.

## Suggested 12-slide deck

1. **Title and objective** — identification under uncertainty with KG + r-GCN + DS.
2. **Problem** — ambiguous passive ESM observations; explain why a radar mode is
   not a complete platform identity.
3. **End-to-end workflow** — use the pipeline diagram above.
4. **KG ontology** — entity/relation diagram and current generated graph counts.
5. **KG preparation methodology** — representative seed data, interval-valued
   parameters, deterministic unique mode signatures.
6. **Synthetic ESM data** — sampled KG-consistent measurements, error intervals,
   kinematics, locations, timestamps, and ambiguity.
7. **Candidate scoring and ETL** — formula, evidence nodes, support and
   contradiction edges.
8. **Dempster-Shafer** — focal elements, masses, belief/plausibility, conflict.
9. **Constrained frame** — bit masks; full subsets for <=10 vs compact
   singleton/group/uncertainty frame for larger identity sets.
10. **r-GCN architecture** — relation-aware messages, residual stack, basis
    decomposition/gates, Dirichlet mass head, multitask heads.
11. **Leakage-safe training/evaluation** — observation-only supervision,
    grouped series split, removed shortcut/cross-split edges, 50/30/20 default.
12. **Results/artifacts, limitations, and roadmap** — artifact outputs; clarify
    synthetic/representative status; track/segment extension and real-data
    validation next steps.

## Visual guidance for the presentation generator

- Use diagrams and equations rather than treating the KG as a long list of
  platforms.
- Mark synthetic or representative data clearly with a visible disclaimer.
- Use three colors consistently: KG context, observed evidence, and model/DS
  outputs.
- For the DS slide, show the two-hypothesis mass vector and a belief-plausibility
  interval, not only a probability bar chart.
- For the split slide, draw series as grouped blocks that never cross train/test/
  validation boundaries and visually cross out leakage-prone edges.
