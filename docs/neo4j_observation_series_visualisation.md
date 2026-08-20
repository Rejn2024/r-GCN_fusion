# Visualise an Observation Series in Neo4j

This guide renders the evidence graph for one synthetic ESM observation series:
each `Observation` node, its ranked `CandidateEvidence` nodes, associated
`IntelligenceReport` and `ReportClaim` nodes, and any candidate-to-candidate
contradiction edges.  It is intended for Neo4j Browser, Neo4j Workspace, or
Bloom's Cypher perspective.

## Load series data with its track fields

The observation ETL accepts both the original flat `observations` JSON and the
wrapper produced by `esm_observation_series_generator.py`.  For a series file,
it flattens `observation_series[*].observations` while retaining `series_id`,
`sequence_index`, and `elapsed_time_s` on the evidence nodes:

```bash
rgcn-fusion-load-observations \
  --observations generated/esm_observation_series.json \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password
```

List the available series and choose one value for `$seriesId`:

```cypher
MATCH (observation:Observation)
WHERE observation.series_id IS NOT NULL
RETURN observation.series_id AS series_id,
       count(*) AS observation_count,
       min(observation.timestamp_iso8601) AS first_seen,
       max(observation.timestamp_iso8601) AS last_seen
ORDER BY first_seen, series_id;
```

In Browser/Workspace, set the parameter before running any query below:

```cypher
:param seriesId => 'esm_series_00001';
```

## Visualise observations and candidates

Run this query in graph view. It returns a `path`, rather than scalar values or
maps, so Neo4j Browser renders the observation/candidate subgraph directly. It
limits each observation to its three highest ranked candidates, which keeps long
series readable. Increase or reduce `$candidateLimit` to change that limit.

```cypher
:param candidateLimit => 3;

MATCH path = (observation:Observation {series_id: $seriesId})
             -[has_candidate:HAS_CANDIDATE]->(candidate:CandidateEvidence)
WHERE has_candidate.rank <= $candidateLimit
RETURN path
ORDER BY observation.sequence_index, has_candidate.rank;
```

## Visualise the full sequence, candidates, and intelligence reports

After loading both the observation ETL and report ETL outputs, run the following
query to display the complete series evidence graph. The `OPTIONAL MATCH`
clauses are intentional: every observation remains in the result even when it
has no candidate, nearby report, or applicable claim. No candidate limit is
applied, so this returns the full candidate set for every observation.

```cypher
MATCH (observation:Observation {series_id: $seriesId})
OPTIONAL MATCH (observation)-[has_candidate:HAS_CANDIDATE]
               ->(candidate:CandidateEvidence)
OPTIONAL MATCH (report:IntelligenceReport)
               -[near:REPORT_NEAR_OBSERVATION]->(observation)
OPTIONAL MATCH (report)-[contains:REPORT_CONTAINS_CLAIM]
               ->(claim:ReportClaim)
OPTIONAL MATCH (claim)-[applies:CLAIM_SUPPORTS_OBSERVATION]
               ->(observation)
OPTIONAL MATCH (claim)-[candidate_evidence:CLAIM_SUPPORTS_CANDIDATE|CLAIM_REFUTES_CANDIDATE]
               ->(candidate)
RETURN observation,
       has_candidate,
       candidate,
       report,
       near,
       contains,
       claim,
       applies,
       candidate_evidence
ORDER BY observation.sequence_index, has_candidate.rank,
         report.published_at, claim.claim_id;
```

`REPORT_NEAR_OBSERVATION` identifies reports associated with each observation.
The two claim matches expose both observation-level applicability and the
positive or negative intelligence evidence attached to a particular candidate.
Neo4j's graph view deduplicates nodes and relationships that recur in rows, so
the repeated tabular rows caused by multiple candidates and claims do not create
duplicate graph elements. For a very large series, use the candidate-limited
query above first or add `WHERE has_candidate.rank <= $candidateLimit` after the
first `OPTIONAL MATCH`.

To inspect only the contradiction links as a graph, run this path query
separately (the source candidate is limited to the same top ranks):

```cypher
MATCH path = (observation:Observation {series_id: $seriesId})
             -[:HAS_CANDIDATE]->(candidate:CandidateEvidence)
             -[:CONTRADICTS_CANDIDATE]->(other:CandidateEvidence)
WHERE candidate.rank <= $candidateLimit
  AND other.series_id = $seriesId
RETURN path;
```

`HAS_CANDIDATE.rank` is the score rank (one is best) and its `score` property
is the total candidate score.  `CONTRADICTS_CANDIDATE` is directed from the
stronger incompatible candidate to the weaker one; its `score_delta` and
`reason` properties explain the relationship.

### Suggested graph styling

Configure these captions and colours in the graph style panel:

| Element | Caption | Suggested colour |
| --- | --- | --- |
| `Observation` | `sequence_index` | blue |
| `CandidateEvidence` | `rank` | orange |
| `HAS_CANDIDATE` | `rank` | grey |
| `CONTRADICTS_CANDIDATE` | `reason` | red |

Open an observation node to inspect `timestamp_iso8601`, `best_candidate_mode_id`,
and `best_candidate_score`.  Open a candidate node to compare `mode_id`,
`radar_id`, `aircraft_id`, `operator`, and its individual consistency scores.

## Tabular timeline (recommended for long series)

The following result is easier to scan than a dense graph and shows the top
candidate at every observation in temporal order:

```cypher
MATCH (observation:Observation {series_id: $seriesId})
MATCH (observation)-[has_candidate:HAS_CANDIDATE {rank: 1}]->(candidate:CandidateEvidence)
RETURN observation.sequence_index AS sequence,
       observation.elapsed_time_s AS elapsed_seconds,
       observation.timestamp_iso8601 AS timestamp,
       candidate.mode_id AS mode,
       candidate.radar_id AS radar,
       candidate.aircraft_id AS aircraft,
       candidate.operator AS operator,
       has_candidate.score AS score,
       observation.best_candidate_score AS observation_score
ORDER BY sequence;
```

For databases loaded before the track fields were added, reload the source
series through the ETL above.  Existing nodes are matched by stable evidence
node ID and receive the new properties on reload.
