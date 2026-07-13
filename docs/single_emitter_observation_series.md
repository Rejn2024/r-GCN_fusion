# Single-Emitter Observation Series Ingestion

This note summarizes a proposed extension to the existing ESM observation,
Neo4j ETL, r-GCN, and Dempster-Shafer workflow for handling repeated
observations of a single emitter. The goal is to let repeated observations
reinforce or counter identification of radar types, aircraft variants, and
operators while preserving the possibility that the emitter changes radar mode
during the observation series.

## Current workflow context

The current workflow ingests individual synthetic ESM observations containing
measured radar parameters, approximate kinematics, timestamps, labels, and
ambiguous candidate labels. The ETL scores each observation against knowledge
graph `RadarMode` nodes using interval overlap for measured radar features,
waveform and scan-type agreement, kinematic consistency, and operator context.
It then writes `Observation` and `CandidateEvidence` nodes labelled as
`EvidenceEntity`, along with Dempster-Shafer mass vectors and optional labels
for radar, mode, aircraft, and operator classification.

This per-observation structure should remain the foundation for series
handling. A series should not be collapsed into one averaged observation,
because averaging across a radar-mode transition could weaken or distort the
underlying evidence.

## Recommended graph representation

Represent a repeated set of observations from one emitter as a track-level
entity containing time-ordered observations:

```text
(:EmitterTrack)-[:HAS_OBSERVATION {sequence_index, delta_t_s}]->(:Observation)
```

Each `Observation` should continue to receive its own ranked
`CandidateEvidence` nodes through the existing candidate-scoring process. A
higher-level grouping can then be added for mode-consistent intervals:

```text
(:EmitterTrack)-[:HAS_SEGMENT]->(:ModeSegment)
(:ModeSegment)-[:CONTAINS_OBSERVATION]->(:Observation)
(:ModeSegment)-[:BEST_MODE]->(:RadarMode)
(:ModeSegment)-[:BEST_RADAR]->(:Radar)
```

Useful temporal and evidential relationship types include:

- `NEXT_OBSERVATION`
- `SAME_EMITTER`
- `SAME_MODE_SEGMENT`
- `POSSIBLE_MODE_SHIFT`
- `SUPPORTS_RADAR`
- `SUPPORTS_AIRCRAFT`
- `SUPPORTS_OPERATOR`
- `CONTRADICTS_CANDIDATE`

These relationships would expose both temporal continuity and candidate support
to the r-GCN. `CONTRADICTS_CANDIDATE` is emitted by the observation ETL for
same-observation candidates when a stronger candidate supports incompatible
mode, radar, aircraft, or operator hypotheses.

## Candidate scoring and fusion levels

Each observation in the series should first be scored independently against the
knowledge graph. The resulting evidence can then be fused at different
hypothesis levels:

### Radar mode

Radar-mode evidence should be fused only within mode-consistent segments. If
adjacent observations strongly support different modes, the series should be
segmented rather than smoothed into a single mode estimate.

### Radar type

Radar-type evidence can accumulate across mode shifts when the supported modes
belong to the same radar. For example, a sequence that transitions from search
to track-while-scan to single-target track may counter a single-mode hypothesis
but reinforce the parent radar hypothesis.

### Aircraft variant

Aircraft-variant evidence should accumulate from repeated support for radars
used by the same variant, while remaining constrained by kinematic feasibility
and any other contextual evidence.

### Operator

Operator evidence should be treated as slower-moving and more conservative. It
should be reinforced only when radar, aircraft, geographic, temporal, and other
contextual priors remain consistent.

## Mode-shift handling

The workflow should explicitly detect possible mode shifts instead of hiding
them through averaging. A simple segmentation approach is:

1. Sort observations by timestamp.
2. Score top candidate modes, radars, aircraft, and operators for each
   observation.
3. Compare adjacent observations' top-N candidates or Dempster-Shafer masses.
4. Start a new mode segment when the best mode changes sharply, the previous
   mode confidence drops below a threshold, or adjacent evidence has high
   conflict.
5. Preserve radar-level continuity if the new and old modes belong to the same
   radar.

Dempster-Shafer conflict should be retained as a diagnostic. High conflict may
mean either a true contradiction or an expected radar-mode transition. The graph
should distinguish these cases by linking observations through
`POSSIBLE_MODE_SHIFT` when the conflict is explainable by a mode change within a
plausible radar.

## Reinforcement and counter-evidence

Repeated observations should update cumulative belief as follows:

- Reinforce a hypothesis when consecutive or repeated observations support the
  same radar mode, radar type, aircraft variant, or operator.
- Counter a hypothesis when observations support incompatible alternatives.
  Within one observation, the ETL makes this explicit with directed
  `CONTRADICTS_CANDIDATE` edges from higher-scored candidates to lower-scored
  incompatible candidates.
- Preserve uncertainty when candidate scores are close or ambiguous.
- Split mode segments when the mode changes but retain radar, aircraft, and
  operator continuity when appropriate.

Conceptually:

```text
For each observation:
  score top-N candidates

For each hypothesis level:
  radar_mode: combine only inside mode-consistent segments
  radar_type: combine across segments when modes belong to the same radar
  aircraft_variant: combine across segments using radar and kinematic support
  operator: combine cautiously using external priors and deployment context

If conflict rises:
  if conflict is explained by a plausible mode transition:
    split or mark a mode segment boundary
  else:
    treat it as counter-evidence against the current hypothesis
```

## Label leakage caution

For synthetic experiments, existing observation labels can be used to validate
candidate ranking and model training. In an operational or realistic inference
setting, however, `ground_truth_label` fields must not be used as scoring
inputs. Operator priors should instead come from external context such as region,
order of battle, deployment windows, mission area, or other non-label sources.

## Example interpretation

A single-emitter series might produce this pattern:

```text
Observation t1 -> radar A / search / aircraft X / operator O1
Observation t2 -> radar A / track-while-scan / aircraft X / operator O1
Observation t3 -> radar A / single-target-track / aircraft X / operator O1
```

The appropriate interpretation is not that the mode estimate is unstable.
Instead, the series suggests a mode transition while reinforcing the parent
radar, compatible aircraft variant, and plausible operator.
