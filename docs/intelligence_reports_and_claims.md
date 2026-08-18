# Intelligence Reports and Claims

## Report versus claim

An **intelligence report** is the provenance-bearing container for intelligence.
It records where and when information came from and how credible the source is.
A **claim** is one structured assertion extracted from that report. In short:

> **Report = source and context. Claim = assertion being evaluated.**

For example, a sighting report can record that a particular aircraft, operator,
and radar were last observed near a track location at a given time. The report
holds the observer source, timestamps, report type, and overall credibility.
Separate claims represent the asserted operator, aircraft variant, radar,
location, and last-observed time, each with its own stance and confidence.

| Aspect | Intelligence report | Report claim |
|---|---|---|
| Represents | An intelligence product or source record | One proposition extracted from a report |
| Answers | Who reported it, when, and with what credibility? | What exactly is being asserted? |
| Key fields | Source/type, collection and publication times, credibility | Type, subject, predicate, object, stance, and confidence |
| Graph role | Contains claims and preserves provenance | Supports/refutes observations and may contradict other claims |
| Neo4j label | `IntelligenceReport` + `EvidenceEntity` | `ReportClaim` + `EvidenceEntity` |

Reports are stored once for an observation series rather than copied into each
observation. Current generation creates two complementary products:

- **Sighting reports** describe aircraft near the generated track, their
  operators and radars, the reported location, and when they were last seen.
  Most are correct, while controlled identity, location, and time errors create
  contradictory evidence for evaluation.
- **Pattern-of-life reports** describe expected behavior for an aircraft family
  and variant, including its operator, radar, role, typical radar modes,
  operating area, and performance ceilings.

Each report contains multiple structured claims. During ETL,
`REPORT_CONTAINS_CLAIM` links each report to its claims, while
`REPORT_NEAR_OBSERVATION` and `CLAIM_SUPPORTS_OBSERVATION` link reports and
claims only to observations that satisfy their temporal/geographical
applicability test. Sighting reports require nearby coordinates and time;
pattern-of-life reports require the expected operating area and a wider time
window. `REPORT_APPLIES_TO_TRACK` then connects an applicable report to the
track containing those observations. `CLAIM_ASSERTS_KG_ENTITY` also links
structured claims directly to existing `AircraftVariant`, `AircraftFamily`,
`Radar`, `RadarMode`, or `Operator` nodes. Contradictions are modeled
between incompatible claims, not between their containing reports.

Keeping these entities separate preserves provenance, permits multiple sources
to corroborate the same proposition, and lets assertions from one report carry
different confidence or extraction quality.

## Claim scoring

Each claim receives a bounded support/quality score from 0 to 1:

```text
score = 0.25 * claim confidence
      + 0.20 * source credibility
      + 0.15 * recency
      + 0.15 * extraction confidence
      + 0.10 * optional external prior
      + 0.10 * KG consistency
      + 0.05 * specificity
```

Recency follows exponential decay with a 14-day half-life. `collected_at` is
preferred over `published_at`, and a missing timestamp receives a recency score
of 0.25. An absent external prior is neutral at 0.5. Inputs and the final score
are bounded to `[0, 1]`.

This score measures the quality or strength of an extracted assertion; it is
not a learned probability and does not by itself establish that the assertion
is true. Synthetic truth metadata is for generator evaluation and is excluded
from inference features.

## Baseline features

Reports and claims receive a small common numeric representation so they can
participate as `EvidenceEntity` nodes alongside observations and candidate
evidence:

- A report has credibility, recency, a degree score (number of claims), a text
  score (currently credibility), and DS masses.
- A claim has inherited credibility and recency, claim and extraction
  confidence, a degree score (currently 1), a text score (the composite claim
  score), and DS masses.

Claims also retain their type, stance, asserted object, report ID, and series
ID. These are called *baseline* features because they are simpler than the
domain-specific interval, waveform, scan, residual, and kinematic features used
for ESM candidate matching.

## Two-hypothesis Dempster-Shafer masses

A score `s` and ambiguity `u` are converted to the normalized mass vector:

```text
[m(non-match), m(match), m({non-match, match})]
```

using:

```text
m(uncertain) = u
m(match)     = (1 - u) * s
m(non-match) = (1 - u) * (1 - s)
```

The third entry assigns mass to the complete two-hypothesis frame. It represents
uncommitted uncertainty, not a third outcome. Ambiguity is bounded to
`[0.05, 0.60]`.

Report masses use `credibility * recency` as the score and ambiguity 0.25.
Claim masses use the composite claim score and ambiguity 0.20 for supporting
claims or 0.35 for other stances. For example, a supporting claim scored 0.80
produces:

```text
[0.16, 0.64, 0.20]
```

This commits 0.16 against a match, 0.64 for a match, and leaves 0.20 uncertain.
Its match belief is 0.64 and its match plausibility is 0.84.

The claim's substantive direction is not encoded solely by this mass vector.
The graph also retains its stance on the claim-to-observation edge and represents
incompatible same-type assertions with directed `CONTRADICTS_CLAIM` edges.

## Candidate compatibility and fusion

During observation ingestion, report claims are quality-scored in memory and
compared with every KG candidate before the shortlist is limited and before any
candidate nodes are produced. Report ingestion subsequently materialises the
report/claim provenance graph and idempotently refreshes the same candidate
fusion fields. Compatibility follows the **final
signed** convention in `[-1, 1]`: its sign already includes both object
alignment and claim stance. The quality-weighted contribution is therefore:

```text
contribution(q, c) = claim_quality(q) * compatibility(q, c)
```

There is intentionally no additional `stance_sign`; adding one would count
stance twice. Direct matches and refutations receive full magnitude. Refuting a
different alternative receives only weak positive weight because eliminating
one alternative does not establish the candidate under consideration.

The aggregation retains all non-neutral claim-to-candidate edges for provenance
and graph learning, while using only the strongest contribution per source for
numeric fusion. It exposes separate sensor, support, refutation, net,
uncertainty, conflict, source-count, and final-score features. Claim evidence is
also converted into candidate-specific two-hypothesis masses and combined with
the sensor mass. The default scalar blend caps intelligence influence at 15%
and reduces it further when fewer than three independent sources apply.
