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

The current repository also generates **synthetic intelligence reports** for
observation series. Report and claim provenance, credibility, recency, and
contradictions are represented as evidence nodes instead of overwriting the
canonical KG. In addition to the packaged r-GCN training path, an advanced
notebook experiments with a deeper GraphSAGE-plus-HGT classifier for the
combined series, candidate, and intelligence-report graph.

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
Synthetic intelligence reports -> claims/provenance -+-> evidence subgraph
                                                        -> leakage-safe splits
                                                        -> packaged r-GCN or
                                                           notebook GraphSAGE/HGT
                                                        -> DS evidence outputs
                                                           (packaged r-GCN) and/or
                                                           class predictions
```

1. Encode radar, radar-mode, aircraft-variant, family, and operator facts in a
   procedural Python seed model.
2. Generate a KG with typed nodes, stable IDs, properties, and a small,
   interpretable relation vocabulary.
3. Generate reproducible synthetic ESM records by sampling values from the KG
   mode intervals and adding measurement uncertainty.
4. Score each observation against KG radar-mode candidates using interval,
   categorical, kinematic, and optional external-context evidence.
5. Optionally generate one shared set of 10--12 synthetic intelligence reports
   per emitter series, including at least two order-of-battle assessments and
   supportive, contradictory, or refuting claims.
6. Load observations, candidates, reports, and claims into Neo4j as an evidence
   graph without changing canonical KG facts.
7. Train the packaged shared r-GCN to predict DS masses and optional categorical
   labels, or run the advanced notebook's classification-only GraphSAGE/HGT
   experiment.
8. Report normalized masses, uncertainty, belief/plausibility intervals, class
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
| `IntelligenceReport` + `EvidenceEntity` | A provenance-bearing report shared by a series | source/type, collection/publication times, credibility, recency, DS masses |
| `ReportClaim` + `EvidenceEntity` | A structured claim extracted from a report | claim type, stance, object, confidence, extraction confidence, support score, DS masses |

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

### Intelligence reports for a series

The series API can enrich each emitter series with one shared set of 10--12
reports; reports are deliberately stored once on the series wrapper rather than
copied into every observation. The report mix covers operator,
aircraft-variant, aircraft-family, radar-type, radar-mode, location, and
relationship claims. At least two reports are order-of-battle assessments. The
synthetic mix includes correct, incorrect, supporting, and refuting claims so
that corroboration and contradiction can be tested. Synthetic truth fields
exist only to evaluate the generator and must not become inference features.

Claim scoring blends claim confidence (25%), source credibility (20%), recency
(15%), extraction confidence (15%), optional external prior (10%), KG
consistency (10%), and specificity (5%). Recency uses exponential decay with a
14-day default half-life. As with ESM generation, fixed seeds make the dataset
reproducible.

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

The separate report ETL writes `IntelligenceReport` and `ReportClaim` evidence
nodes. `REPORT_CONTAINS_CLAIM` preserves provenance,
`CLAIM_SUPPORTS_OBSERVATION` connects every shared series claim to every
observation for which it is valid (and retains the claim stance), and directed
`CONTRADICTS_CLAIM` edges connect incompatible same-type claims from the
stronger-scored claim to the weaker. Reports and claims receive baseline
features and two-hypothesis DS masses, allowing them to participate in the same
Neo4j evidence graph as observations and candidates.

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

The presentation should lead with the architecture in
`observation_series_and_intel_rgcn_classification_advanced_network.ipynb`: a
deep GraphSAGE encoder followed by relation-aware HGT attention.  It is designed
for a heterogeneous series graph containing observations, scored candidates,
intelligence reports, and report claims.  The installable CLI's residual r-GCN
is a separate, production-facing path and should be mentioned only as the
current implementation of the DS/Dirichlet output described below; do not imply
that the notebook already has an evidential head.

### Node properties and linear input projection

Before message passing, each node is a sparse row in one shared feature matrix.
An observation row contains measured ESM values and uncertainty bounds,
kinematics, location/error-box values, elapsed time, relative sequence position,
series duration/count, and an optional measurement-derived segment index.  A
candidate row represents match, residual, ambiguity, kinematic, rank, and DS
scores.  Report rows encode credibility, recency, rank, and claim count; claim
rows encode confidence, extraction confidence, specificity, KG consistency,
text-derived support, and supporting/refuting stance.  One-hot node-kind flags
tell the network which of these property sets is meaningful.  Missing fields are
zero-filled, and ground-truth fields are excluded from the input.

A **linear input projection** is a learned affine map, not a manual feature
selection or a graph operation.  If node properties are
$\mathbf{x}_v\in\mathbb{R}^{F}$, the notebook computes

$$
\mathbf{h}^{(0)}_v =
\mathrm{Dropout}\!\left(\mathrm{GELU}\!\left(
\mathrm{LayerNorm}(\mathbf{W}_{in}\mathbf{x}_v+\mathbf{b}_{in})
\right)\right).
$$

**Presentation wording:** project the raw property vector with a learned weight
matrix and bias, then apply LayerNorm, GELU, and dropout.

Thus $\mathbf{W}_{in}\in\mathbb{R}^{128\times F}$ learns weighted mixtures
of differently named properties and places every node type in the same
128-dimensional latent coordinate system.  Layer normalization controls scale,
GELU introduces non-linearity, and dropout regularizes the representation.  A
single projected coordinate need not correspond to one human-readable field;
it can represent a learned combination such as frequency residual, measurement
uncertainty, report credibility, and node kind.

### Deep GraphSAGE representation

The notebook deterministically precomputes at most 12 inbound neighbours per
node and reuses those edges for 12 GraphSAGE layers.  At layer $\ell$, it forms
the mean inbound-neighbour representation and combines a transform of that mean
with a separate transform of the node itself:

$$
\bar{\mathbf{h}}^{(\ell)}_{N(v)}=
\frac{1}{|N(v)|}\sum_{u\in N(v)}\mathbf{h}^{(\ell)}_u,
\qquad
\tilde{\mathbf{h}}^{(\ell+1)}_v =
\mathbf{W}^{(\ell)}_{self}\mathbf{h}^{(\ell)}_v+
\mathbf{W}^{(\ell)}_{nbr}\bar{\mathbf{h}}^{(\ell)}_{N(v)}.
$$

**Presentation wording:** average the inbound neighbour encodings, transform the
node and neighbour average separately, and add the two results.

After GELU and dropout, a residual projection carries the previous state around
the update and LayerNorm produces the next state.  Layer widths taper from 128
to 32 dimensions.  Consequently, an observation's final encoding represents
both its own measurements and progressively more distant context: adjacent
observations and same-emitter continuity, candidate matches and contradictions,
and report/claim provenance, support, and contradiction.  The residual path
preserves local node properties while the narrowing stack compresses this
multi-hop evidence into a compact representation.

### Relation-aware HGT refinement and task heads

One four-head HGT-style layer then revisits the full typed edge set.  For edge
$u\xrightarrow{r}v$, each head compares the destination query with a
relation-transformed source key and transforms the source value with a second
relation matrix:

$$
s^{r,h}_{uv}=\frac{\left(\mathbf{q}^{h}_v\right)^\top
\mathbf{R}^{r,h}_{K}\mathbf{k}^{h}_u}{\sqrt{d_h}}\,\mu_{r,h},
\qquad
\mathbf{m}^{r,h}_{uv}=\sigma(s^{r,h}_{uv})
\mathbf{R}^{r,h}_{V}\mathbf{v}^{h}_u.
$$

**Presentation wording:** score each source-to-destination message using the
edge type and attention head, use that score as a gate, and pass a
relation-transformed source value to the destination.

Messages are mean-aggregated at the destination, projected, passed through
GELU/dropout, and added residually before LayerNorm.  Relation-specific key and
value transforms plus the learned priority $\mu_{r,h}$ let, for example, a
`CONTRADICTS_CLAIM` edge affect a node differently from a temporal or self-loop
edge.  Separate two-layer MLP heads map the shared 32-dimensional encoding to
aircraft-variant, radar-mode, radar-type, and operator-country logits.  The
notebook trains these heads with summed cross-entropy and optional L1
regularization, using one full-graph step per epoch.  Edge chunking, CUDA mixed
precision, gradient checkpointing, and early stopping control memory and
overfitting.

### Dirichlet evidential output: formulae and advantages

The advanced notebook currently ends in ordinary classification logits.  A
natural evidential extension is to replace or augment a $K$-class task head
with the Dirichlet construction already implemented by the packaged r-GCN.  For
head logits $\mathbf{z}$, define non-negative evidence and concentration as

$$
e_k=\mathrm{softplus}(z_k),\qquad
\alpha_k=e_k+1,\qquad S=\sum_{j=1}^{K}\alpha_j.
$$

**Presentation wording:** turn each logit into non-negative evidence, add one to
obtain its Dirichlet concentration, and sum all concentrations to obtain the
total evidence strength.

The expected class probability is $\mathbb{E}[p_k]=\alpha_k/S$.  In the
subjective-logic/DS view, committed singleton belief and uncommitted uncertainty
are

$$
b_k=\frac{e_k}{S},\qquad u=\frac{K}{S},\qquad
\sum_{k=1}^{K}b_k+u=1.
$$

**Presentation wording:** divide each class's evidence by the total strength to
obtain committed belief; the unevidenced share is the number of classes divided
by that same strength.

For the repository's DS mass head, whose $K$ outputs are focal-element masses,
the implemented prediction is $m_k=\alpha_k/S$, with the same concentration
diagnostic $u=K/S$.  This distinction should be stated on a slide: normalized
Dirichlet means over focal elements are the model's masses, whereas
$b_k=e_k/S$ and $u=K/S$ give the conventional evidential decomposition.

The approach has four presentation-worthy advantages over a bare softmax:

- it cannot create negative evidence, and $\alpha_k\geq1$ provides a clear
  zero-evidence prior;
- total concentration $S$ records how much evidence the network has gathered,
  so identical class rankings can carry different uncertainty;
- uncertainty rises toward one when all evidence is weak and falls only when
  accumulated evidence is strong, making ignorance explicit rather than forcing
  all output mass among classes; and
- the output can be connected to DS focal-element masses and therefore to
  belief/plausibility reporting and evidence-fusion workflows.

These quantities are useful uncertainty indicators, not automatic guarantees of
calibration.  Evidence from correlated graph neighbours or reports must not be
treated as independent without validation, and an evidential version of the
advanced notebook would require calibration and out-of-distribution evaluation.

### Equation encoding and presentation reuse

The equations above use GitHub-supported Markdown math delimiters: single dollar
signs for inline expressions (`$...$`) and double dollar signs on their own lines
for display equations (`$$ ... $$`).  Their contents are LaTeX math commands;
GitHub renders that combination in Chrome, whereas the generic LaTeX delimiters
`\\(...\\)` and `\\[...\\]` are not reliably recognized by GitHub Markdown.
Function names use the supported `\\mathrm{...}` styling command rather than
`\\operatorname{...}`, which GitHub's math renderer rejects in this context.

For presentation generation, use the bold **Presentation wording** below each
formula as speaker notes or a plain-text fallback.  For a visual equation on a
slide, paste the contents between the dollar-sign delimiters into PowerPoint's
equation editor in LaTeX input mode; do not paste the dollar signs themselves.
This keeps the source readable in raw Markdown, rendered GitHub notes, and slide
software without relying on a screenshot of the equation.

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
- Synthetic report generation and a `rgcn-fusion-load-reports` CLI are
  implemented. Reports are shared at series level, claims are linked to every
  observation in that series, and canonical KG facts remain unchanged.
- The repository contains notebooks for KG creation, ESM generation, ETL,
  observation-series-plus-intelligence classification, advanced GraphSAGE/HGT
  experiments, and DS identification demos.
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
7. **Evidence graph ETL** — candidate formula plus report/claim provenance,
   support, stance, and contradiction edges.
8. **Dempster-Shafer** — focal elements, masses, belief/plausibility, conflict.
9. **Constrained frame** — bit masks; full subsets for <=10 vs compact
   singleton/group/uncertainty frame for larger identity sets.
10. **Advanced network** — linear property projection, 12-hop tapered
    GraphSAGE, relation-aware four-head HGT, multitask heads, and the Dirichlet
    evidential-head extension; identify clearly what is notebook-local versus
    already implemented in the packaged r-GCN.
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
