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
