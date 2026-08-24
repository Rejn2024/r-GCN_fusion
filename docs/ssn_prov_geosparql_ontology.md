# Alternative SSN, PROV-O, and GeoSPARQL ontology

This design is an RDF/OWL alternative to the property-graph schema documented
in [Evidence Graph Ontology](evidence_graph_ontology.md). Its normative local
terms are provided in [`ontology/esm-evidence.ttl`](../ontology/esm-evidence.ttl).
It reuses rather than duplicates three established vocabularies:

* **SOSA/SSN** describes ESM sensing: observations, receivers, observed signal
  properties, procedures, features of interest, and results.
* **PROV-O** describes reports, sources, extraction activities, claims, and
  derivation. It makes the distinction between a document and a proposition
  explicit.
* **GeoSPARQL 1.1** describes emitters, tracks, trajectories, reported places,
  estimated locations, and uncertainty regions in a queryable CRS.

The design is intentionally an alternative interchange and reasoning model. It
does not require replacing the current Neo4j labels or r-GCN relation names.

The graph-population notebooks call
`rgcn_fusion.ontology_projection.materialize_ontology_view` after loading their
canonical or evidential data. The projection retains legacy labels and edges
for existing queries while adding ontology class IRIs, stable resource IRIs,
standards-aligned relationship aliases, named evidence masses, and GeoSPARQL
geometry nodes. This makes the ontology part of population rather than a
documentation-only mapping.

## Design rules

1. An ESM sample is an `esm:ESMObservation` and therefore a
   `sosa:Observation`. Link it to the receiver with `sosa:madeBySensor`, the
   emitter with `sosa:hasFeatureOfInterest`, the measured property with
   `sosa:observedProperty`, the result with `sosa:hasResult`, and the event time
   with `sosa:resultTime`.
2. Measurement values belong to `sosa:Result` resources. This permits pulse
   repetition interval, radio frequency, pulse width, scan period, and other
   results to carry their own value, unit, and uncertainty.
3. An intelligence report and each extracted claim are separate `prov:Entity`
   resources. A claim uses `prov:wasDerivedFrom` (through
   `esm:claimInReport`) and may have a qualified `prov:Generation` identifying
   the extraction activity and time. The report uses `prov:wasAttributedTo`
   for its source.
4. A claim's quoted proposition is represented by `esm:claimSubject`,
   `esm:claimPredicate`, and `esm:claimObject`. The proposition is **not** added
   directly as an RDF triple. Thus a false or refuting claim never pollutes the
   canonical domain graph.
5. Every geometry carries an explicit CRS in `geo:asWKT`. Use a point only when
   its precision is defensible; represent an ESM error box or reported area as
   a polygon. A trajectory is a `geo:Feature` whose default geometry can be a
   `LINESTRING`, while observation points retain event times and ordering.
6. Dempster-Shafer masses remain three separate decimal properties so their
   meanings survive RDF export. Application validation must enforce values in
   `[0,1]` and a sum of one; OWL alone cannot express that arithmetic rule.

## Core model

```mermaid
classDiagram
    sosa_Observation <|-- ESMObservation
    sosa_Sensor <|-- ESMReceiver
    sosa_FeatureOfInterest <|-- Emitter
    geo_Feature <|-- Emitter
    geo_Feature <|-- Track
    geo_Feature <|-- Trajectory
    prov_Entity <|-- IntelligenceReport
    prov_Entity <|-- Claim
    prov_Activity <|-- ClaimExtraction
    prov_Agent <|-- Source
    ESMReceiver --> ESMObservation : sosa:madeObservation
    ESMObservation --> Emitter : sosa:hasFeatureOfInterest
    ESMObservation --> SignalMeasurement : sosa:hasResult
    Track --> ESMObservation : esm:hasObservation
    Track --> Trajectory : esm:hasTrajectory
    IntelligenceReport --> Claim : esm:containsClaim
    Claim --> IntelligenceReport : prov:wasDerivedFrom
    ClaimExtraction --> IntelligenceReport : prov:used
    Claim --> ClaimExtraction : prov:wasGeneratedBy
```

## Example instance

The example keeps the report's assertion quoted. Notice that it does **not**
assert `:emitter-7 :hasOperator :operator-blue` as a canonical fact.

```turtle
@prefix : <https://example.test/data/> .
@prefix esm: <https://w3id.org/rgcn-fusion/esm#> .
@prefix geo: <http://www.opengis.net/ont/geosparql#> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

:obs-42 a esm:ESMObservation ;
    sosa:madeBySensor :receiver-a ;
    sosa:hasFeatureOfInterest :emitter-7 ;
    sosa:observedProperty :radioFrequency ;
    sosa:resultTime "2026-08-24T12:00:00Z"^^xsd:dateTime ;
    sosa:hasResult :rf-result-42 ;
    esm:estimatedLocation :error-box-42 ;
    esm:inTrack :track-7 ;
    esm:sequenceIndex 42 .

:rf-result-42 a esm:SignalMeasurement ;
    esm:numericValue 9350.0 ;
    esm:unit <http://qudt.org/vocab/unit/MegaHZ> .

:error-box-42 a geo:Geometry ;
    geo:asWKT "<http://www.opengis.net/def/crs/OGC/1.3/CRS84> POLYGON((-1.1 51.4,-0.9 51.4,-0.9 51.6,-1.1 51.6,-1.1 51.4))"^^geo:wktLiteral .

:report-9 a esm:IntelligenceReport ;
    prov:wasAttributedTo :source-3 ;
    prov:generatedAtTime "2026-08-24T12:05:00Z"^^xsd:dateTime ;
    esm:containsClaim :claim-9-1 .

:claim-9-1 a esm:Claim ;
    esm:claimInReport :report-9 ;
    esm:claimSubject :emitter-7 ;
    esm:claimPredicate :hasOperator ;
    esm:claimObject :operator-blue ;
    esm:stance "supports" ;
    esm:claimConfidence 0.82 ;
    prov:wasGeneratedBy :extraction-9 .

:extraction-9 a esm:ClaimExtraction ;
    prov:used :report-9 ;
    prov:endedAtTime "2026-08-24T12:06:00Z"^^xsd:dateTime .
```

GeoSPARQL longitude precedes latitude for `CRS84`; exporters must not silently
reverse the coordinates. Observation time is kept on the observation rather
than embedded in the WKT trajectory, because standard GeoSPARQL WKT geometries
do not encode per-vertex time.

## Mapping from the current property graph

| Current graph element | RDF representation |
| --- | --- |
| `Observation` | `esm:ESMObservation` |
| `CandidateEvidence` | `esm:CandidateHypothesis`, with `prov:wasDerivedFrom` the observation |
| `IntelligenceReport` | `esm:IntelligenceReport` |
| `ReportClaim` | `esm:Claim` |
| `Track` | `esm:Track` and optionally `esm:hasTrajectory` |
| `REPORT_CONTAINS_CLAIM` | `esm:containsClaim` / `esm:claimInReport` |
| `TRACK_HAS_OBSERVATION` | `esm:hasObservation` / `esm:inTrack` |
| `next_observation` | `esm:nextObservation` |
| `CLAIM_SUPPORTS_CANDIDATE` | `esm:supports` |
| `CLAIM_REFUTES_CANDIDATE` | `esm:refutes` |
| `CONTRADICTS_CLAIM` | symmetric semantic relation `esm:contradicts`; retain score-based direction separately if required by the r-GCN |
| coordinate/error-box properties | `esm:estimatedLocation` to a `geo:Geometry` |
| `ds_masses` array | `esm:massNonMatch`, `esm:massMatch`, `esm:massUncertain` |

`REPORT_NEAR_OBSERVATION` is best treated as a materialized analytical result,
not primitive geography. It can remain an application edge with distance and
time-delta annotations; spatial eligibility should be reproducible with
GeoSPARQL functions such as `geof:distance` or `geof:sfWithin`, followed by the
application's temporal-window rule.

## Competency questions

The ontology should support the following implementation tests:

* Which receiver made each observation, using which procedure, and what signal
  property and result were recorded?
* Which observations fall within a reported polygon and within a report's time
  window?
* Which ordered observations form a track, and what line geometry summarizes
  its trajectory?
* Which source report and extraction activity produced a claim?
* Which claims from independent sources support or refute the same candidate,
  and which claims contradict each other?
* What evidence masses belong to an observation, claim, or candidate without
  treating a prediction as a canonical aircraft/radar fact?

## Validation and implementation notes

Before production interchange, add SHACL shapes for required cardinalities,
confidence and mass ranges, the mass-sum constraint, permitted stance values,
geometry datatypes, and monotonically increasing track sequence indexes. Keep
units as URIs from one selected unit vocabulary. Treat `owl:imports` as logical
dependencies; deployments that cannot dereference the web should load pinned
local copies of SSN/SOSA, PROV-O, and GeoSPARQL for deterministic validation.
