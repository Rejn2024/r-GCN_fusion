# SKOS alignment assessment

## Scope and conclusion

This assessment compares both graphs in this repository—the canonical
aircraft/radar graph and the evidence graph—with the W3C **SKOS Simple
Knowledge Organization System Reference**. SKOS is an RDF vocabulary for
representing thesauri, taxonomies, classification schemes, and other knowledge
organization systems. It is not a general-purpose replacement for a domain
ontology or an event/provenance model.

The present model **does not embody SKOS as an implementation**. It has some
SKOS-like qualities (stable identifiers, named entities, a family/variant
hierarchy, and typed associative links), but it neither uses the SKOS
vocabulary nor satisfies SKOS's core concept-scheme, labelling, semantic
relation, and mapping conventions. This is not inherently a defect: aircraft,
radars, observations, reports, and claims are domain individuals or events,
whereas `skos:Concept` is intended for entries in a knowledge organization
system. The best alignment is therefore a **SKOS terminology layer alongside
the operational and evidence graphs**, not a wholesale conversion of every
node and edge to SKOS.

Normative comparison points in this note come from the
[W3C SKOS Reference](https://www.w3.org/TR/skos-reference/), with the
[SKOS Primer](https://www.w3.org/TR/skos-primer/) providing non-normative
implementation guidance.

## Current node and edge inventory

### Canonical domain graph

| Current node type | Current role | Closest SKOS interpretation | Assessment |
| --- | --- | --- | --- |
| `AircraftFamily` | Aircraft family entity | A concept in an aircraft taxonomy | Good candidate for a parallel `skos:Concept`, provided it denotes the category rather than a physical aircraft. |
| `AircraftVariant` | Specific platform variant | A narrower concept, or a domain class/entity linked to one | Potential concept, but its technical and operational properties remain in the domain model. |
| `Radar` | A radar design associated with variants | A concept in a radar-design scheme, or a domain entity | Do not infer a SKOS hierarchy merely from equipment use. |
| `RadarMode` | A radar's concrete operating mode with numeric bounds | A radar-mode concept plus a domain specification | The repeated mode names could be controlled concepts, while each radar-specific parameter set remains a domain resource. |
| `Operator` | Nation or organization operating a variant | A domain organization/place entity | Usually not a SKOS concept; link it to an external authority rather than force it into a taxonomy. |

The canonical relations are:

- `VARIANT_OF`: the only relation with a plausible hierarchical reading;
- `USES_RADAR`, `HAS_MODE`, and `OPERATES`: domain predicates, not SKOS
  semantic relations.

### Evidence graph

The concrete evidence types are `Observation`, `CandidateEvidence`,
`IntelligenceReport`, `ReportClaim`, and `Track`, with `EvidenceEntity` used as
an umbrella Neo4j label. Their edges express containment, temporal adjacency,
candidate ranking, support, refutation, contradiction, proximity, grounding,
and track membership.

These resources are observations, hypotheses, information artifacts, and
assertions—not terms in a controlled vocabulary. Relations such as
`HAS_CANDIDATE`, `REPORT_CONTAINS_CLAIM`, `CLAIM_SUPPORTS_CANDIDATE`, and
`CONTRADICTS_CLAIM` must therefore remain domain/evidence predicates. In
particular, `CONTRADICTS_*` must not be replaced by `skos:related`: SKOS's
associative relation does not encode evidential opposition, direction,
strength, or provenance.

## Comparison with SKOS principles

| SKOS area | SKOS expectation | Present ontology | Result |
| --- | --- | --- | --- |
| RDF resources | Concepts and schemes are RDF resources identified by IRIs. | The generated graph is JSON/CSV and Neo4j property-graph data. IDs such as `aircraft_family:f_16` are application identifiers, not declared RDF IRIs. | Not implemented. |
| Concept schemes | Concepts are organized with `skos:ConceptScheme`, `skos:inScheme`, `skos:hasTopConcept`, and `skos:topConceptOf`. | No explicit scheme, scheme membership, version, or top concepts exist. | Not implemented. |
| Concepts | Knowledge-organization entries are instances of `skos:Concept`. | Neo4j labels describe domain/evidence types; no node is typed `skos:Concept`. | Not implemented. |
| Lexical labels | Human-readable terms use `skos:prefLabel`, `skos:altLabel`, and `skos:hiddenLabel`, normally language-tagged; a concept has no more than one preferred label per language. | Most canonical nodes have a single untagged `name`; aliases, spelling variants, acronyms, and language are not modeled. Evidence identifiers and values are ordinary properties. | Not implemented. |
| Notations | Codes are represented with `skos:notation`, preferably as typed literals. | Machine IDs, display names, and designations are not explicitly distinguished. | Not implemented. |
| Documentation | Definitions, scope notes, examples, change notes, and provenance can use the SKOS documentation properties. | Numeric `notes` exist on radar modes, but there are no controlled definitions or SKOS notes on concepts. | Mostly absent. |
| Hierarchy | `skos:broader`/`skos:narrower` express direct conceptual hierarchy; transitive closure is exposed separately. | `VARIANT_OF` resembles a direct hierarchy but has domain-specific entity semantics and no inverse or transitive SKOS view. | Partially analogous, not SKOS-conformant. |
| Associative links | `skos:related` expresses a symmetric, non-hierarchical conceptual association. | Domain links and directed r-GCN evidence links have more precise semantics and are not declared symmetric. | Correctly kept domain-specific, but no SKOS association layer exists. |
| Integrity | SKOS declares, among other conditions, disjoint label properties and disjoint hierarchical/associative relations. | No SKOS constraints or validation profile is present. | Not implemented. |
| Mapping schemes | Cross-scheme alignment uses `skos:exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch`, and `relatedMatch`. | There is no explicit external-vocabulary mapping or confidence/provenance model for mappings. | Not implemented. |
| Collections | `skos:Collection` and `skos:OrderedCollection` group concepts without asserting a hierarchy. | Tags, candidate lists, tracks, and observation series are domain groupings, not SKOS collections. | Not implemented—and should not be mechanically converted. |

Accordingly, the graph follows useful **knowledge-graph design principles** but
cannot currently be advertised as a SKOS vocabulary. Its typed nodes and edges
alone do not establish SKOS conformance.

## Recommended target design

### 1. Add a separate terminology layer

Create explicit schemes such as:

- `AircraftTypeScheme` for family and variant concepts;
- `RadarTypeScheme` for radar-design concepts;
- `RadarModeScheme` for generic operating-mode concepts;
- optionally, `ClaimTypeScheme`, `StanceScheme`, `WaveformScheme`, and
  `ScanTypeScheme` for values that are currently strings.

Give every scheme and concept a dereferenceable, persistent HTTP(S) IRI in an
owned namespace. Retain current IDs as `skos:notation` values or internal keys,
rather than treating prefixed strings as globally unique web identifiers.

### 2. Separate concepts from operational resources

Keep the existing domain/evidence types and predicates. Link them to terminology
concepts with an application predicate such as `ex:classifiedAs` (or an
appropriate standard vocabulary predicate). For example, a radar-specific mode
resource containing PRF bounds can be classified as the generic “track while
scan” concept. Likewise, an `AircraftVariant` domain resource may be linked to
an aircraft-variant concept without asserting that an observed aircraft or an
evidence candidate is itself a `skos:Concept`.

Do **not** make `Observation`, `CandidateEvidence`, `IntelligenceReport`,
`ReportClaim`, or `Track` into SKOS concepts. SKOS also does not replace the
Dempster-Shafer masses, temporal links, provenance, claim stance, contradiction,
or scoring properties.

### 3. Publish labels, definitions, and codes

For each concept:

- provide exactly one `skos:prefLabel` per supported language;
- use `skos:altLabel` for acronyms and genuine synonyms, and
  `skos:hiddenLabel` for search-only misspellings or legacy forms;
- use language tags, for example `"track while scan"@en`;
- keep designations/codes in `skos:notation` with a datatype that identifies
  the code system where practical;
- add `skos:definition`, `skos:scopeNote`, and `skos:example` where a term's
  intended use or boundary is not self-evident.

This also resolves the current overloading of `name`: a preferred display term,
an alternate designation, and a stable identifier become different fields.

### 4. Expose only genuine conceptual relations as SKOS

If domain review confirms that aircraft variants are narrower categories of
aircraft families, publish the terminology relation as:

```turtle
ex:aircraft-variant/f-16c-d-block-50
    a skos:Concept ;
    skos:inScheme ex:scheme/aircraft-types ;
    skos:prefLabel "F-16C/D Block 50"@en ;
    skos:broader ex:aircraft-family/f-16 .
```

The existing `VARIANT_OF` edge can remain the authoritative domain predicate
and be mapped to this SKOS view through an explicit export rule. Do not map
`USES_RADAR`, `HAS_MODE`, or `OPERATES` to `skos:broader`, and do not assume
that graph connectivity is a taxonomy. Add explicit top concepts and direct
hierarchical links; compute `skos:broaderTransitive` for query convenience
rather than using it to encode direct parentage.

### 5. Treat external alignment as mapping, not identity by default

Add mappings to maintained external schemes only after review. Prefer
`skos:closeMatch` when concepts are merely similar; reserve `skos:exactMatch`
for concepts that can safely be used interchangeably. Record mapping source,
reviewer, date, and confidence in a provenance-capable statement model (for
example RDF-star or a mapping resource), because a bare SKOS mapping triple
cannot carry the evidence fields used elsewhere in this project.

### 6. Define and validate an application profile

Document which SKOS constructs the project requires, then validate exports with
SHACL. A minimum profile should check:

1. every concept has an IRI, one scheme membership, and one English preferred
   label;
2. no concept has more than one preferred label for the same language;
3. preferred, alternate, and hidden labels do not reuse the same literal on one
   concept;
4. every scheme identifies at least one top concept;
5. broader/narrower links remain within the expected scheme unless explicitly
   permitted;
6. no pair is both hierarchically related and `skos:related`;
7. mapping predicates connect concepts in different schemes; and
8. domain/evidence resources are not accidentally typed as SKOS concepts.

Run an RDF validator/reasoner as well as SHACL: SHACL enforces the project's
closed-world quality rules, while SKOS's RDF/OWL axioms provide the standard
inferences (such as inverse and transitive semantic-relation views).

## Migration sequence

1. **Inventory and govern terms.** Decide which current values denote reusable
   concepts, assign scheme owners, and distinguish canonical labels from codes.
2. **Create a small pilot.** Model aircraft families/variants and the five
   generic radar modes first; preserve all current graph data unchanged.
3. **Add export mappings.** Generate RDF/Turtle (or JSON-LD) SKOS resources from
   the governed terminology while keeping Neo4j domain edges available to the
   r-GCN.
4. **Link, do not conflate.** Add classification links from domain resources and
   claim objects to concept IRIs. Keep evidence provenance and uncertainty on
   the evidence model.
5. **Validate in CI.** Check the application profile and a set of expected SKOS
   entailments on every vocabulary release.
6. **Version and publish.** Give schemes release metadata, a change policy, and
   stable IRIs; deprecate concepts without reusing their identifiers.

This approach brings the controlled vocabulary into line with SKOS while
preserving the precise node and relation semantics on which candidate scoring,
evidence fusion, and r-GCN message passing depend.
