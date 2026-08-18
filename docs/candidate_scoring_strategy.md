# Candidate scoring and selection strategy

Candidate weights determine which hypotheses enter the evidence graph, so they
must be treated as calibrated model parameters rather than presentation-friendly
constants. This document describes the current effective weights, a defensible
process for improving them, and alternatives to a single weighted sum.

## Understand the current effective weights

The sensor score is currently:

```text
sensor = 0.75 * ESM mode match
       + 0.15 * aircraft compatibility
       + 0.10 * operator/context prior

aircraft compatibility = 0.80 * kinematic consistency
                       + 0.20 * aircraft/radar KG compatibility
```

The final scalar score blends intelligence after applying an independent-source
coverage factor:

```text
effective intelligence weight = configured weight * min(source count / 3, 1)
final = (1 - effective intelligence weight) * sensor
      + effective intelligence weight * intelligence score
```

At the default intelligence weight of 0.15 and full source coverage, the
effective contributions are 63.75% ESM, 10.20% kinematics, 2.55% aircraft/radar
KG compatibility, 8.50% operator context, and 15% intelligence. With fewer than
three sources, the sensor components receive proportionally more weight.

These percentages describe the scalar blend, not information content. Correlated
ESM fields, correlated reports, missing fields, score calibration, and input
variance can make the practical influence of a component very different from
its nominal coefficient.

## Recommended weight-refinement process

### 1. Define the operational objective first

Choose metrics that reflect the cost of candidate-node omission, not only top-1
accuracy. Recommended primary metrics are:

- **Recall@K:** whether the correct candidate survives node creation;
- **mean reciprocal rank** or **NDCG:** whether useful candidates rank early;
- **negative log-likelihood and Brier score:** whether scores are probabilistically
  meaningful;
- **expected calibration error:** whether a score of 0.8 succeeds about 80% of
  the time;
- **cost-weighted error:** explicitly penalise omission of dangerous or rare
  classes more than carrying one additional candidate.

For candidate production, optimise Recall@K subject to a graph-size or average
candidate-count constraint. Top-1 accuracy alone encourages an overly narrow
shortlist.

### 2. Build leakage-safe validation splits

Fit weights only on representative historical or high-fidelity simulated data.
Split by track, emitter, time window, geography, sensor, and preferably aircraft
family so adjacent observations and duplicated reports cannot appear in both
training and validation. Maintain separate stress sets for missing ESM fields,
novel radar modes, stale reports, contradictory reports, and degraded kinematic
estimates.

Synthetic truth is useful for controlled ablations, but final coefficients
should be checked against operationally realistic data because generator priors
can otherwise be learned as if they were real-world frequencies.

### 3. Calibrate each component before combining it

Raw interval overlap, kinematic consistency, context priors, and intelligence
quality need not share a common probability scale. Calibrate each component on
held-out data using one of:

- isotonic regression when ample data supports a flexible monotonic mapping;
- Platt/logistic scaling for smaller datasets;
- beta calibration for bounded scores concentrated near zero or one;
- hierarchical calibration by sensor, radar band, or report-source class when
  enough samples exist.

Combine calibrated likelihoods or log-odds rather than assuming that every raw
0-to-1 score is directly comparable.

### 4. Estimate coefficients under constraints

A strong transparent baseline is non-negative logistic regression over the
component scores and missingness indicators. Constrain weights to be non-negative
where domain knowledge requires monotonicity, regularise them toward the current
expert values, and fit an intercept rather than forcing all weights to sum to
one. Alternatives include grid or Bayesian optimisation over the simplex using
grouped cross-validation.

Report the distribution of fitted weights across folds or bootstrap samples.
If a coefficient changes sign or varies widely, it is not sufficiently
identified and should not be presented as a stable operational value.

### 5. Make weights quality-adaptive

Fixed coefficients are unlikely to be optimal for every observation. Gate each
source using evidence quality:

- reduce ESM influence as missing-feature count and interval width increase;
- reduce kinematic influence as speed/altitude error bounds grow;
- reduce intelligence influence for low credibility, low recency, low total
  claim strength, low source independence, or high conflict;
- reduce context-prior influence under distribution shift or when a prior is
  absent, rather than treating absence as substantive evidence;
- renormalise only the reliable, available components.

The current intelligence coverage gate uses source count. It should be extended
with total quality-weighted evidence strength and a conflict penalty so three
weak or mutually contradictory sources cannot receive the same maximum weight
as three strong corroborating sources.

### 6. Run sensitivity and ablation studies

For each component, compare the full system with that component removed, sweep
its coefficient over a useful range, and plot Recall@K, calibration, and average
shortlist size. Also perturb every fitted coefficient jointly through Monte
Carlo or bootstrap sampling. Prefer a broad stable plateau over a narrow optimum
that is sensitive to small dataset changes.

Weights, calibration artifacts, dataset version, split definition, and metric
trade-offs should be versioned together. Monitor them after deployment for
population and calibration drift.

## Alternative candidate-selection methods

### Calibrated probabilistic ranking

Model each evidence component as a likelihood ratio and add log-likelihoods to
a prior log-odds. This is interpretable, handles strong negative evidence more
naturally than an arithmetic mean, and makes explicit which independence
assumptions are being made. Correlated fields or reports must be grouped or
discounted to avoid double counting.

### Learned ranking

Train a pointwise classifier or pairwise/listwise ranker over the existing
residual, match, missingness, kinematic, report-quality, conflict, and provenance
features. Gradient-boosted trees are a useful first choice for tabular data;
pairwise logistic ranking or LambdaMART directly optimises ordering. Use grouped
splits and monotonic constraints where required. Keep the current formula as a
fallback and audit baseline.

### Bayesian or Dempster-Shafer decision rules

Instead of selecting by `final_score`, rank by posterior probability, pignistic
probability, belief, or plausibility. A conservative policy can retain every
candidate whose plausibility exceeds a threshold, thereby preserving hypotheses
with high unresolved uncertainty. Conflict and ignorance should remain separate
from low match probability rather than being collapsed into one scalar.

### Conformal candidate sets

Use a held-out calibration set to choose a nonconformity threshold and output a
set of candidates with a target empirical coverage, such as 95%. This provides
a measurable omission-rate guarantee under exchangeability and naturally
produces larger sets for ambiguous observations. Monitor distribution shift,
because the guarantee weakens when deployment data differs from calibration
data.

### Dynamic threshold plus top-K safety bounds

Retain candidates within a calibrated score or log-odds margin of the leader,
subject to minimum and maximum counts. This is preferable to an unconditional
top five: clear observations create fewer nodes, while ambiguous observations
retain more alternatives. Use a minimum K as a safety guard until calibration is
trusted.

### Diversity-aware shortlist

After relevance scoring, apply maximal marginal relevance or constrained
selection so the shortlist covers distinct radar modes, aircraft families, or
operators rather than filling all slots with near-duplicates. This improves
hypothesis coverage but should not force diversity when one family is
overwhelmingly supported.

### Cascaded retrieval and reranking

Use permissive physical constraints to retrieve a high-recall pool, then apply
intelligence-aware probabilistic or learned reranking, and finally use a
coverage-aware set-selection rule. Hard filters should be reserved for physical
impossibilities with reliable bounds; uncertain or missing measurements should
degrade a score rather than eliminate a candidate.

## Suggested implementation sequence

1. Add component-level telemetry and evaluate the current formula with grouped
   Recall@K, NDCG, Brier score, calibration error, and shortlist size.
2. Calibrate each component, then fit a constrained logistic baseline with
   bootstrap confidence intervals.
3. Replace the source-count-only intelligence gate with a gate incorporating
   source independence, total claim strength, recency, and conflict.
4. Replace fixed top-K truncation with a calibrated margin or conformal set plus
   configurable minimum and maximum candidate counts.
5. Compare the transparent baseline with a monotonic boosted-tree or pairwise
   ranker; promote a learned method only if gains persist across stress sets and
   temporal holdouts.
6. Version the scoring policy and retain component scores so every candidate
   inclusion or exclusion remains explainable.
