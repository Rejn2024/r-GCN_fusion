# Track-level RAO classification design

## Objective and current mismatch

The LLM-enabled advanced notebook currently produces four outputs for every
observation node: aircraft variant, radar type, operator country, and radar
mode. Although observations in a series exchange messages over temporal and
`same_emitter` edges, each observation embedding is passed independently to all
four task heads and every observation label contributes separately to the loss.
KG-constrained decoding makes each *individual* output tuple valid, but it does
not force two observations in the same track to select the same
aircraft/radar/operator (RAO) combination.

The modelling unit should instead match the domain assumptions:

* one latent, time-invariant **RAO identity per track**; and
* one **radar mode per observation**, allowed to vary through the track.

The recommended change is a hierarchical track/observation model. This makes
invariance structural rather than relying on a soft consistency penalty or a
post-processing vote.

## Recommended architecture

### 1. Retain the relational observation encoder

Keep the existing feature projection, r-GCN stack, and relation-aware HGT
layers. They remain useful for combining measurements, candidate hypotheses,
reports, claims, contradictions, and temporal context. Their output is an
embedding `h_i` for each observation `i`.

As an efficiency follow-up, consider reducing the 15-layer receptive field once
explicit track pooling is present. Pooling gives every identity decision direct
access to the entire track, so very deep temporal message passing is no longer
the only way for distant observations to interact. Select depth by validation,
not by assumption.

### 2. Pool observation embeddings into one track embedding

For track `s`, combine its observation embeddings into `z_s` using a
mask-aware pooling module. A gated attention pool is a good default:

```text
a_i = softmax_i(w^T tanh(W_h h_i + W_q q_s))
z_s = LayerNorm(sum_{i in s} a_i h_i + MLP([mean(h_s), max(h_s)]))
```

Here `q_s` can be a learned track token or a summary of track-level report
evidence. Attention should be normalized **within each series**, never across a
batch. Mean/max pooling is a simpler baseline and should be retained as an
ablation. Include observation time deltas or positional encodings if order is
not already sufficiently represented by typed temporal edges.

The pooling path must only consume observations available at inference time.
For online operation, replace bidirectional whole-track attention with a causal
GRU/Transformer or a running gated accumulator and report identity after each
prefix. Train on prefixes if prefix-time predictions are required.

### 3. Predict one joint RAO class from the track embedding

Replace the three observation-level heads for `aircraft_variant`, `radar_type`,
and `operator_country` with a single track-level head:

```text
rao_logits_s = RAOHead(z_s)              # [number of KG-valid RAO worlds]
rao_evidence_s = softplus(RAOEvidence(z_s))
```

The class vocabulary should be the allow-list of KG-valid
`(aircraft_variant, radar_type, operator_country)` combinations, not the
Cartesian product. A single joint head has two important properties: it cannot
emit an impossible combination, and there is only one RAO result to attach to
all observations in the track. Retain a full-frame ignorance mass (or an
explicit unknown/open-set policy) so an unsupported combination is not forced
into a known class.

Project the joint distribution or mass back to aircraft, radar, and operator
marginals for dashboards and explanations. Those marginals are diagnostic
views of one joint prediction; they are not independently decoded identities.

If the RAO vocabulary becomes too large or sparsely observed, use a
KG-factorized energy model as a second choice:

```text
score_s(r) = score_aircraft(a | z_s)
           + score_radar(r | a, z_s)
           + score_operator(o | a, r, z_s),  r=(a,r,o) in KG
```

Normalize scores only over KG-valid RAO worlds. This shares parameters across
worlds while retaining hard validity and one track-level choice. Three
unconstrained marginal heads plus an argmax or a consistency penalty are not a
recommended primary design because they can still disagree and handle unseen
combinations poorly.

### 4. Predict radar mode per observation, conditioned on the track

Keep radar mode at observation resolution. Its head should see both local
evidence and the pooled identity context:

```text
u_i = MLP([h_i, z_series(i), h_i * P(z_series(i))])
mode_logits_i[m] = ModeHead(u_i)[m] + compatibility(rao_s, m)
```

Apply a hard mask (or a very negative logit) to modes that are not attached in
the KG to the candidate radar in each RAO world. During training, a differentiable
version should marginalize over RAO uncertainty rather than condition on the
argmax:

```text
p(m_i | track) = sum_r p(r | track) p(m_i | h_i, r)
```

This lets uncertain identity hypotheses contribute gradients and prevents
early RAO errors from irreversibly masking the true mode. At inference, return
the shared track RAO plus each observation's mode distribution.

A linear-chain CRF or small temporal Transformer can optionally model mode
transitions. It must allow self-transitions and genuine changes; it should learn
or encode only physically meaningful transition preferences, not impose
constant mode. Start with the conditionally independent mode head above and add
transition modelling only if sequence metrics improve.

### 5. Keep evidential outputs at their natural level

Move the aircraft/radar/operator Dirichlet/DS output to the joint track RAO
frame. Keep a separate per-observation mode evidential head. This yields:

* `RAO evidence, belief, plausibility, uncertainty`: once per track; and
* `mode evidence, belief, plausibility, uncertainty`: once per observation.

Do not repeatedly combine observation-level RAO Dirichlet opinions as though
they were independent sensors: adjacent embeddings share the same graph and
reports, so that would count correlated evidence multiple times. Pool learned
embeddings/evidence first and generate one track opinion instead.

## Targets, splits, and loss

### Construct targets at the correct granularity

Build one `rao_target[series_id]` from `ground_truth_track_label`. Validate that
all observation-level aircraft/radar/operator labels agree with it and fail data
preparation if they do not. Keep `mode_target[observation_id]` from each
observation label.

Some data may contain a radar replacement or a mistaken track association. If
that is a real operational case, it is a track-boundary problem: split the
series at the identity change or introduce an explicit association/change-point
model. Do not silently teach the invariant RAO head contradictory targets.

Continue splitting by `series_id`; stratify on the **joint RAO class**, with
group-aware allocation for rare classes. Fit label vocabularies, calibration,
class weights, and any preprocessing statistics on training tracks only. No
track, report-derived node, or edge path should connect training and evaluation
graphs. In particular, globally linked claim nodes should be duplicated or
partitioned by split if they could relay information between series.

### Use a track-balanced multi-task objective

One suitable objective is:

```text
L = lambda_rao * mean_s CE(rao_logits_s, r_s)
  + lambda_mode * mean_s mean_{i in s} CE(mode_logits_i, m_i)
  + lambda_rao_ev * mean_s EDL(rao_alpha_s, r_s)
  + lambda_mode_ev * mean_s mean_{i in s} EDL(mode_alpha_i, m_i)
  + regularization
```

The inner per-track mean is important: without it, long tracks dominate the
mode objective. Sample tracks, not observations, and pad/mask variable-length
series or use a packed `observation_to_series` index with scatter operations.
Use class-balanced sampling or loss weights for rare RAO worlds and radar
modes. Tune the four task weights on validation data or use a learned
uncertainty/gradient-balancing method; raw sums can otherwise make the much more
numerous mode labels overwhelm the single RAO target.

For evidential learning, add a scheduled incorrect-evidence regularizer (for
example, an annealed KL term toward the non-informative Dirichlet) and validate
calibration. Expected cross-entropy alone encourages evidence for the correct
class but does not sufficiently penalize unsupported evidence on wrong classes.

Do not add a same-RAO consistency loss in the recommended model: the shared
track head guarantees equality exactly. Such a penalty is useful only as a
short-lived baseline when retaining observation heads:

```text
L_consistency = mean_s mean_{i in s} JS(p_rao_i, mean_j p_rao_j)
```

It improves smoothness but does not guarantee identical inference outputs, so
the deployed result would still need a track-level constrained decoder.

## Training and inference procedure

1. Build disconnected train, validation, and test graphs from whole tracks.
2. Encode nodes with the r-GCN/HGT encoder.
3. gather observation embeddings and pool them by `series_id`.
4. Compute one KG-constrained RAO distribution/evidential opinion per track.
5. Broadcast the track embedding (or soft RAO distribution) back to its
   observations and compute per-observation mode outputs.
6. Optimize the track-balanced joint loss. Select checkpoints using a
   track-level validation score, not the former sum of observation-level task
   losses.
7. Calibrate RAO and mode outputs separately on validation tracks (temperature
   scaling for categorical probabilities; validate evidential strength and
   uncertainty separately).
8. At inference, emit a track record containing one RAO assessment and an
   ordered list of observation mode assessments. Never decode RAO independently
   inside the observation loop.

Full-graph training remains possible, but supervised indices now consist of
track IDs for RAO and observation IDs for mode. If batching is needed, use
whole-track subgraphs so temporal context and pooling are never truncated
accidentally. For very long tracks, use windows for the mode path while
maintaining a track memory/token for identity.

## Evaluation and acceptance criteria

Observation accuracy for aircraft/radar/operator would hide the intended
invariant and overweight long series. Report at least:

* **track exact-match RAO accuracy** over the joint KG-valid class;
* track-level aircraft, radar, and operator marginal accuracy;
* macro-F1 and per-class recall for imbalanced RAO worlds;
* **RAO violation count**, which must be exactly zero by construction;
* per-observation mode accuracy/macro-F1 and per-track macro-averaged mode
  accuracy;
* mode-change metrics: transition precision/recall/F1 and segment boundary F1,
  so a model cannot score well by predicting one dominant mode throughout;
* negative log-likelihood, Brier score, expected calibration error, and
  risk-coverage/abstention curves separately for track RAO and observation mode;
* seen-world versus unseen/OOD results, including an explicit unknown decision;
* performance versus track length and versus observation prefix length.

Acceptance tests should assert that every emitted track has exactly one RAO,
every predicted RAO belongs to the KG allow-list (unless unknown), every mode is
compatible with at least one supported RAO world, and changing one
observation's mode cannot change the stored RAO value for another observation.

## Suggested implementation sequence

1. **Baseline:** pool observation embeddings by mean, add a joint track RAO
   head, concatenate the pooled vector into the existing mode head, and replace
   the losses/metrics at their correct granularities.
2. **Hard constraints:** derive joint RAO and radar-to-mode masks from the KG,
   add unknown handling, and change serialized predictions/LLM prompts to the
   track-with-mode-sequence schema.
3. **Improve pooling:** compare mean/max, gated attention, and a causal sequence
   aggregator. Inspect attention weights for dependence on track length and
   missing observations.
4. **Improve uncertainty:** add the evidential regularizer, calibrate both
   levels, and test synthetic contradictory or OOD tracks.
5. **Optional dynamics:** add a CRF/transition layer only after the hierarchical
   baseline is sound and evaluate it specifically on mode changes.

This sequence delivers the core invariant in the first step while keeping later
complexity measurable through ablations.
