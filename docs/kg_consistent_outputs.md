# KG-consistent classification outputs

The current multi-task heads predict radar, aircraft variant, and operator
independently. Taking `argmax` for each head can therefore compose a tuple that
never occurs in the knowledge graph. Accuracy improvements alone cannot enforce
this cross-task invariant.

## Recommended design

1. **Make the KG tuple the output space.** At model-load time, query complete
   paths such as `(operator)-[:OPERATES]->(aircraft)-[:USES_RADAR]->(radar)`
   (using the repository's actual relationship directions and names). Deduplicate
   their `(radar_type, aircraft_variant, operator_country)` values and version the
   resulting allow-list with the model artifact. Do not build this allow-list
   from test labels.
2. **Use constrained joint decoding.** Pass the task probabilities, checkpoint
   vocabularies, and allow-list to `decode_kg_constrained`. It scores only full
   tuples present in the KG, so an impossible Cartesian-product combination can
   never be emitted. Return the selected tuple as one atomic result rather than
   returning three independently selected labels.
3. **Reject an unknown identity atomically.** Tune `unknown_threshold` and
   `min_task_probability` on a validation set containing held-out classes and
   out-of-distribution observations. If rejection fires, return `labels: null`
   and `status: "unknown"`; do not retain any of the independently predicted
   fields, which could still imply an impossible combination.
4. **Preserve uncertainty.** Include the constrained confidence, the best few
   valid alternatives, and the Dempster-Shafer uncertainty in downstream output.
   A KG match means “represented in the KG”, not necessarily “true”.

```python
from rgcn_fusion import decode_kg_constrained

result = decode_kg_constrained(
    probabilities=classification_probabilities_for_one_node,
    vocabularies=checkpoint["class_vocabularies"],
    valid_combinations=kg_allow_list,
    unknown_threshold=0.65,
    min_task_probability=0.20,
)
payload = result.to_dict()
```

## Novel radar modes for a known identity

Radar/aircraft/operator compatibility is a different question from whether a
mode has appeared in the KG. Treating every field as one closed-world tuple
would discard useful identity evidence whenever a radar uses a previously unseen
mode. Instead, use hierarchical decoding:

1. Jointly constrain the stable identity fields (`radar_type`,
   `aircraft_variant`, and `operator_country`) to a KG path.
2. Decode `radar_mode` only among modes connected to the selected radar identity.
3. Give the mode its own open-set rejection. If it is novel, emit the known
   identity plus `radar_mode: null`, `status: "partially_known"`, and
   `unknown_tasks: ["radar_mode"]`. Never add the proposed mode to the KG solely
   because the model predicted it; retain the observation for analyst review and
   controlled KG curation.

```python
from rgcn_fusion import decode_kg_hierarchical

result = decode_kg_hierarchical(
    probabilities=classification_probabilities_for_one_node,
    vocabularies=checkpoint["class_vocabularies"],
    valid_combinations=kg_rows_with_modes,
    identity_tasks=("radar_type", "aircraft_variant", "operator_country"),
    open_set_tasks=("radar_mode",),
    attribute_thresholds={"radar_mode": 0.55},
    novelty_scores={"radar_mode": calibrated_mode_ood_score},
    novelty_thresholds={"radar_mode": 0.70},
)
```

A maximum softmax probability is not a reliable novelty detector: neural
classifiers can be confident on out-of-distribution inputs. The
`novelty_scores` value should come from a detector calibrated on held-out modes,
for example an energy score, embedding-distance/density model, conformal
prediction, or a separately trained unknown/background class. Validate the mode
threshold independently from the identity threshold and report known-mode
accuracy, novel-mode recall, false-known rate, and coverage.

### What “energy score” means here

An energy score is a scalar OOD signal calculated from the radar-mode head's
**pre-softmax logits**, not from its winning probability. If the mode logits for
an observation are \(z_1, \ldots, z_K\), a common definition is:

\[
E(x) = -T \log \sum_{k=1}^{K} \exp(z_k / T),
\]

where \(T > 0\) is a temperature chosen during calibration. With this sign
convention, known/in-distribution examples generally have lower (often more
negative) energy because at least one known-mode logit is strongly activated;
unfamiliar examples tend to have higher energy because none of the known modes
is well supported. Thus, a calibrated rule can reject the mode as novel when
`energy > threshold`. Some libraries report the negative of this value, so the
score direction must be verified rather than assumed.

Energy uses the absolute scale of all logits, information that softmax removes.
For example, logits `[20, 19]` and `[2, 1]` produce the same softmax
probabilities, even though their energy values differ substantially. This can
make energy more useful than maximum softmax probability for detecting an input
that does not resemble any training mode. It is still only a detection signal:
it cannot name or characterize the novel mode, and an uncalibrated neural model
can remain overconfident on OOD data.

Fit \(T\) and the rejection threshold without using the test set, preferably
with validation examples that hold out entire radar modes and cover expected
sensor noise and operating conditions. The decoder's `novelty_scores` contract
assumes **larger means more novel**. Either pass raw energy with a threshold on
the same raw scale, or transform energy into a calibrated novelty probability
(for example, `0.92`) and use the corresponding probability threshold. Do not
compare a raw energy value with the illustrative `0.70` probability threshold
shown above.

## A structured frame of discernment

Yes—a more nuanced Dempster-Shafer frame can express high confidence in the
operator/radar/mode while leaving the aircraft variant unresolved. The key is
not to make the independently predicted labels the elementary hypotheses.
Instead, define the elementary worlds in \(\Theta\) as the **KG-valid joint
configurations**, for example:

```text
w1 = (India, MiG-29, MiG-29UPG, Zhuk-ME, track-while-scan)
w2 = (India, MiG-29, MiG-29K,   Zhuk-ME, track-while-scan)
w3 = (...another KG-valid configuration...)
```

The worlds are mutually exclusive, while a focal element may contain several
worlds. Evidence that establishes India, the MiG-29 family, Zhuk-ME, and the
mode—but cannot distinguish the variant—can place mass on `{w1, w2}` rather
than splitting or forcing that mass onto either singleton. This makes the
variant ambiguity explicit without weakening the shared claims.

Project (marginalize) the same joint mass function onto each attribute to report
a separate assessment for `radar_mode`, `radar`, `aircraft_variant`,
`aircraft_family`, and `operator`. For every attribute value, report:

- **belief**: mass whose focal worlds all have that value;
- **plausibility**: mass whose focal worlds include at least one world with it;
- **uncertainty/imprecision**: `plausibility - belief`;
- **pignistic probability**: a decision-oriented probability obtained by
  distributing each focal mass equally over its member worlds.

`attribute_assessments` performs this projection. In the example above, the
operator, radar, family, and mode can have high belief, while each variant has
low or zero belief and high plausibility. The pignistic probabilities across
variants still sum to one:

```python
from rgcn_fusion import attribute_assessments

variant_evidence = attribute_assessments(
    masses=joint_masses,
    worlds=kg_valid_worlds,
    attribute="aircraft_variant",
    focal_masks=sparse_focal_masks,
)
operator_evidence = attribute_assessments(
    masses=joint_masses,
    worlds=kg_valid_worlds,
    attribute="operator",
    focal_masks=sparse_focal_masks,
)
```

These are separate **marginal assessments**, not statistically independent
predictions: they retain the KG correlations because they come from one joint
frame. Training five unrelated frames would provide separate confidence values
but would reintroduce impossible combinations unless a consistency constraint
were applied afterwards.

A complete powerset has \(2^{|\Theta|}-1\) focal elements and quickly becomes
intractable. In production, use a sparse focal family containing only useful
sets: singleton worlds, groups sharing operator/radar/family/mode, groups sharing
operator/radar/family, and the full frame for total ignorance. Construct those
groups from explicit KG identifiers and taxonomy edges rather than inferring
families from names. If a genuinely novel mode is possible, add an explicit
`UNKNOWN_MODE`/open-world state or retain the hierarchical OOD rejection above;
a frame containing only known modes cannot assign belief to an unseen one.

## Further improvements

- Train a single classifier over KG tuple IDs (plus an `unknown` class), or add a
  structured loss that sums probability assigned to invalid tuples. Retain the
  inference constraint even after doing this: a learned penalty is not a hard
  guarantee.
- Distinguish “unknown because confidence is low” from “KG incomplete” in
  telemetry, while exposing both externally as unknown. Maintain temporal KG
  validity on edges when equipment/operator relationships change over time.
- Evaluate **joint tuple accuracy**, invalid-tuple rate (which must be zero after
  decoding), unknown precision/recall, coverage, and calibration. Split by
  observation series and consider holding out entire valid tuples to measure
  open-set behaviour.
- Fail closed when the allow-list is missing, stale, empty, or incompatible with
  checkpoint vocabularies. Never silently fall back to independent `argmax`.

## Maximal safe partial identification

Atomic rejection is the safest fallback for a decoder that can emit only a
complete tuple, but it leaves useful evidence unused. A production system
should additionally support a **set-valued, taxonomy-aware result**: retain the
KG-valid worlds that remain credible, then publish every attribute that is
shared by enough of those worlds. This permits, for example, `MiG-29` to be
reported when `MiG-29K` versus `MiG-29UPG` is unresolved, or `India` to be
reported when the aircraft type is unresolved. It must not be implemented by
independently thresholding task-head maxima, because that can recreate an
impossible combination.

### Recommended inference flow

1. **Represent useful levels explicitly.** Add stable KG identifiers and
   taxonomy edges for variant -> type/family -> role, and preserve explicit
   operator organisation -> nation relationships. Do not derive a family by
   splitting a variant's display name. Keep radar family, radar model, mode,
   platform type, platform variant, operator organisation, and operator nation
   as separate attributes.
2. **Score joint worlds, not isolated labels.** Construct the versioned set of
   valid worlds for the observation time and score each world from the relevant
   sensor, kinematic, contextual, intelligence, and negative evidence. Calibrate
   these scores into a posterior, Dempster-Shafer mass, or conformal prediction
   set. Evidence missing for an attribute should widen the retained world set;
   it should not count as evidence against a value.
3. **Retain a calibrated credible set.** Keep worlds using a conformal coverage
   rule, cumulative posterior-mass target, or calibrated plausibility threshold,
   with a maximum-size safety policy. Preserve the residual probability/mass as
   `unknown` rather than renormalising it away. Hard removal should be limited
   to reliable physical or KG-temporal impossibilities.
4. **Project upward and across attributes.** Marginalise the retained joint
   worlds onto each attribute and its ancestors. Publish a value only when its
   calibrated belief/probability meets that attribute's threshold and all
   material surviving worlds agree with it. Thus two credible variants of the
   same type yield the type, while aircraft ambiguity can still yield an
   operator nation if the credible aircraft worlds share that nation.
5. **Return the most specific supported value per branch.** Walk each taxonomy
   from specific to general and stop at the deepest accepted node. A rejected
   variant can fall back to aircraft type, then family or role. Apply this
   independently to operator, platform, radar, and mode branches, but always
   compute the answers by projection from the same joint world set.
6. **Run a consistency closure before publishing.** Intersect the KG worlds
   compatible with all proposed assertions. If the intersection is empty,
   remove the least-supported assertion until it is non-empty. Return the
   remaining world identifiers or a digest of the allow-list version so an
   analyst can reproduce the result.

The agreement rule should normally operate on a calibrated credible set rather
than literally every non-zero softmax entry. Neural softmax assigns a non-zero
score to almost everything, while prematurely discarding low-scoring worlds can
make a broad assertion look falsely certain. Thresholds therefore need separate
calibration for each level: the cost of wrongly naming an operator nation may
not equal the cost of wrongly naming an aircraft variant.

### Suggested output contract

Do not overload `null` to mean missing input, unresolved ambiguity, novelty,
and contradiction. Emit explicit assertions and unresolved alternatives:

```json
{
  "status": "partially_known",
  "assertions": {
    "aircraft_type": {
      "value": "MiG-29",
      "belief": 0.91,
      "plausibility": 0.98,
      "resolution": "type"
    },
    "aircraft_variant": {
      "value": null,
      "reason": "ambiguous",
      "alternatives": ["MiG-29K", "MiG-29UPG"]
    },
    "operator_nation": {
      "value": "India",
      "belief": 0.96,
      "plausibility": 0.99,
      "resolution": "nation"
    }
  },
  "credible_world_coverage": 0.95,
  "unknown_mass": 0.04,
  "kg_snapshot": "<version>"
}
```

Include provenance and source-specific contributions behind each assertion,
plus explicit reasons such as `insufficient_evidence`, `ambiguous`,
`out_of_distribution`, `conflicting_evidence`, or `kg_incomplete`. Consumers
can then distinguish a useful coarse classification from an unsupported guess.

### Network and training changes that support this output

- Add supervised heads at every useful hierarchy level (variant, type/family,
  operator organisation/nation, radar/mode), while retaining a joint-world head
  or structured decoder. Auxiliary coarse-level losses provide a learning signal
  even when fine labels are absent.
- Make fine heads conditional: predict variant given type, mode given radar,
  and operator organisation given nation. Mask impossible children using the KG
  and train a dedicated stop/abstain decision at every branch.
- Train with partially labelled examples by marginalising over every valid world
  consistent with the known label. An example labelled only `MiG-29` should
  reward the sum of its variant probabilities, not be discarded or assigned a
  fabricated variant.
- Fuse a track over time using reliability-aware attention or Bayesian/DS
  accumulation. Stable attributes such as operator and aircraft type can pool
  over the track, while mode remains time varying. Discount repeated correlated
  observations and reports so volume is not mistaken for independent support.
- Supply missingness masks, measurement uncertainty, sensor identity/quality,
  report provenance, and observation age as model inputs. Learn or calibrate
  source reliability, while retaining explicit conflict and ignorance rather
  than collapsing both into low confidence.
- Add open-set detectors at several levels. A novel variant may still belong to
  a known type, while a novel platform may still have a supported operator
  nation. Include an open-world bucket at each relevant branch instead of one
  global `unknown` class.

### Evaluation and rollout

Evaluate more than exact full-tuple accuracy. Report accuracy and calibration
at every hierarchy depth, semantic distance from truth, credible-set coverage
and size, correct-partial rate, over-specific error rate, abstention/coverage
curves, and the rate of internally inconsistent assertions (which must remain
zero). Stress-test held-out variants within known types, held-out types, shared
radars across platforms, multi-nationally operated aircraft, missing modalities,
contradictory reports, sensor degradation, and KG snapshot drift.

A practical delivery sequence is: first add taxonomy and nation projections to
the existing joint-world assessment; next introduce calibrated credible sets
and the structured result contract; then train hierarchy-level auxiliary heads
and abstention decisions; finally compare track-level fusion and learned source
gating against the transparent projection baseline. This preserves a usable,
auditable baseline while allowing the network to exploit progressively more of
the available information.
