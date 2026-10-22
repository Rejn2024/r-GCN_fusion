# Selective intelligence and track-local shared candidates

Run `notebooks/track_shared_candidate_hgt.ipynb` from the repository root or its
`notebooks/` directory. The notebook requires the **existing** files used by
`track_star_hgt.ipynb`:

- `generated/demo_esm_observation_series_open_set_ufo.json`
- `generated/aircraft_radar_kg.json`

It does not regenerate those inputs or load the original feature/graph artifacts.
Install the repository dependencies, `scikit-learn` and `tensorboard`. The new
attention implementation uses native PyTorch; it does not require PyG. All
experiment outputs go to `artifacts/track_shared_candidate_hgt/`.

## Candidate identity and incidence evidence

Candidate identity is the full `(mode_id, radar_id, aircraft_id, operator)` tuple,
scoped by `series_id`. A tuple appearing in two tracks creates two candidate nodes.
Candidates are the union of the existing scorer's per-observation shortlists:
sharing does not introduce a hypothesis link at an observation where it was not
retrieved. Scoring/retrieval limits remain 320 retrieved and 18 retained by default.

The original artifact metadata omits candidate identity. The small additive
change in `parallel_graph_construction.py` now exports those fields without
changing scores or legacy node/edge construction. The new notebook re-scores the
ground-truth-free inference records so it cannot accidentally merge candidates
by rank or infer identity from a match-score vector.

Shared candidate node inputs contain only a kind indicator. Their mode, radar,
aircraft and operator connect to KG entity nodes through distinct component
relations. KG entities use numeric properties and learned embeddings indexed by
the fixed KG vocabulary, never by track IDs or evaluation labels. Relevant KG
domain relations and their reverse directions are included.

All original candidate match columns, including sensor/RF compatibility, rank,
field-level scores, intelligence support/refutation/conflict and fused masses,
move onto the observation–candidate incidence. Forward and reverse edges reference
the same attribute row. Fixed signed-log transforms handle scale without fitting
normalisation statistics on validation/test observations.

The existing scorer still determines claim applicability and recency per
observation/candidate pair. Its aggregated intelligence features and retained
positive/negative claim counts form incidence attributes. There are **no direct
claim–shared-candidate edges**, which would erase the observation context.
RF compatibility relation semantics are retained as incidence indicators.
Rank-based candidate–candidate contradiction edges are omitted because their
meaning was observation-specific; incoming candidate messages compete through
attention at their observation, and rank remains an incidence attribute.

Sharing introduces an additional route for observations within a track to
exchange information. Shared KG nodes can also relay evidence between tracks in
a batch. Candidate nodes themselves always belong to exactly one track. The
original experiment's shared KG context means batch grouping can affect results;
keep `TRACKS_PER_BATCH` fixed when comparing runs.

## Intelligence routing

Default `INTELLIGENCE_ROUTING=selective` uses this conservative policy:

| Evidence | Destination |
|---|---|
| Applicable theatre/airborne inventory report | Track, with coverage |
| Inventory or stable identity claim from a broad inventory report | Track, with coverage |
| Other stable identity claim, applicable to every observation, whose report has no sighting/pattern restriction | Track |
| Sighting or pattern-of-life report and its claims | Applicable observations |
| Other restricted, behavioural, location/time or unknown evidence | Applicable observations |
| Evidence with no applicable observation | No observation/track link |

Stable claim types are `aircraft_variant`, `aircraft_family`, `radar_type` and
`operator`. The parent report's temporal/spatial restriction takes precedence:
a stable aircraft attribute mentioned in a brief sighting does not by itself
establish that the sighting identifies the whole track. Existing proximity logic
determines applicability; membership in a track's report container is insufficient.

For `INTELLIGENCE_ROUTING=track`, all applicable report/claim links move to tracks.
This is the optional full-replacement experiment. Coverage remains on edges and
exact applicability remains in the audit; the model no longer has direct local
report/claim edges. Candidate incidence scores still preserve local applicability.

Every routed relation is bidirectional. Track relations are `report_for_track` /
`track_has_report` and `claim_for_track` / `track_has_claim`; local equivalents are
`report_for_observation` / `observation_has_report` and `claim_for_observation` /
`observation_has_claim`. Report–claim pairs are emitted once. Claim–KG relations
and contradictions between co-applicable claims are retained. Duplicate copies
of a contradictory claim pair are removed; contradictory pairs are bidirectional.

Coverage is not treated as a proof of platform identity. Broad inventories also
retain the original order-of-battle prior, so compare uniform-prior control runs
when investigating evidence/prior interactions.

## Attention, batching and evaluation

Each of five layers uses type-specific Q/K/V projections, relation-specific key
and value transforms, and multihead attention normalised over all incoming edges
at a destination. Incidence features affect both the key and message value.
Repeated source transforms and topology lookups are cached once per batch.
Forward/reverse incidences reuse the same projected feature row within a layer.
This replaces the original `HGTConv`, which did not consume arbitrary edge features.

Raw node and incidence features are disk-backed. Batch preparation retains compact
row selections into the shared incidence memmap. Only the active batch gathers
and transfers its feature rows; staging is synchronous and bounded to one batch.
Graph sizes, attributes and cached lookup tensors contribute to the conservative
memory estimate. Fewer candidate nodes do not guarantee faster training: explicit
edge attributes and attention still cost memory and compute.

The original split procedure, open-set exclusions, RAO/mode vocabularies, priors,
losses, partial identification, dashboards and prediction formats are retained.
The latest source defaults include hidden width 120 and five attention layers.
Checkpoints are selected using **test loss**, matching the original experiment;
use the untouched validation split for independent evaluation. Checkpoint reload
checks the full model/feature configuration and input fingerprints. Original
checkpoints are incompatible, so retrain this architecture.

Controls include `TRACKS_PER_BATCH`, `EPOCHS` (default 150), `METRICS_INTERVAL`,
`RAO_OOB_BASE_RATE_MULTIPLIER`, and `INTELLIGENCE_ROUTING`. Optional continuation
defaults to zero epochs. Local Ollama explanations are opt-in using
`RUN_LOCAL_LLM=1`; training/evaluation does not require that service.

## Audits and verification

The notebook writes input SHA-256 fingerprints, node/incidence memmaps, graph
metadata/topology, `graph_reduction_audit.json`, `edge_compute_requirements.json`
and `intelligence_routing.jsonl`. Routing records include the original applicable
observation IDs, selected scope and raw coverage, even when edges are compressed
to a track link. Graph metadata includes feature column order and relation endpoint
types. Model checkpoints also retain those schemas and input fingerprints.

Run focused checks with:

```bash
python -m pytest -q tests/test_track_shared_candidate.py tests/test_parallel_graph_construction.py tests/test_track_star_loss.py
```

Checks cover candidate ownership/deduplication, distinct incidence scores, claim
context, selective and full routing, coverage, reindexing, train/validation batch
separation, edge-feature gradients, mixed precision, activation checkpointing,
evidential losses and notebook syntax/schema. Every notebook cell was also run
on an isolated 12-track fixture for one training epoch, through checkpoint reload,
prediction export and dashboards. That verifies execution, not scientific results.
The original generated dataset is not present in the development environment;
full-dataset training, GPU memory measurements and accuracy comparisons remain
to be performed with the user's existing inputs.
