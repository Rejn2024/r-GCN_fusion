# GNN Performance Evaluation in a Subjective Logic Paradigm

## Purpose

This note summarises useful summary statistics, diagnostics, and plots for evaluating a graph neural network (GNN) that produces **subjective-logic / evidential outputs** rather than ordinary class probabilities.

The evaluation should measure not only classification quality, but also:

- calibration;
- usefulness of uncertainty estimates;
- distinction between **vacuity / epistemic uncertainty** and **dissonance / aleatoric ambiguity**;
- behaviour on incorrect predictions;
- behaviour on out-of-distribution (OOD) inputs;
- sensitivity to graph structure.

---

## 1. Subjective-logic representation

For a \(K\)-class subjective-logic opinion:

\[
\omega = (\mathbf{b}, u, \mathbf{a})
\]

where:

- \(\mathbf{b} = (b_1,\ldots,b_K)\): belief masses;
- \(u\): uncertainty / vacuity;
- \(\mathbf{a}\): base-rate vector.

The masses satisfy:

\[
\sum_k b_k + u = 1
\]

Projected class probabilities are:

\[
p_k = b_k + a_k u
\]

For evidential / Dirichlet models:

\[
S = \sum_k \alpha_k
\]

and commonly:

\[
u = \frac{W}{S}
\]

where \(W\) is the prior weight.

Predicted class:

\[
\hat{y} = \arg\max_k p_k
\]

---

## 2. Core predictive metrics

Retain standard classification metrics so the GNN can be compared with conventional models.

Recommended:

- Accuracy
- Balanced accuracy
- Macro-F1
- Per-class precision
- Per-class recall
- Per-class F1
- Confusion matrix

These measure whether the model predicts the correct class, but do not by themselves assess uncertainty quality.

---

## 3. Probabilistic scoring and calibration

Evaluate projected probabilities \(\mathbf{p}\).

### Negative log likelihood

Useful for measuring probabilistic sharpness and correctness.

### Brier score

For multiclass classification:

\[
BS = \frac{1}{N}\sum_i\sum_k(p_{ik}-y_{ik})^2
\]

Lower is better.

### Expected calibration error

Report:

- ECE
- optionally classwise ECE

### Reliability diagram

Plot:

- x-axis: predicted confidence;
- y-axis: empirical accuracy;
- diagonal: perfect calibration.

A good subjective-logic model should be both accurate and calibrated.

---

## 4. Vacuity / epistemic uncertainty statistics

Useful summary statistics for vacuity \(u\):

- mean;
- median;
- standard deviation;
- minimum / maximum;
- 5th, 25th, 75th, 95th percentiles.

More useful are conditional summaries:

\[
E[u \mid \text{correct}]
\]

\[
E[u \mid \text{incorrect}]
\]

\[
E[u \mid \text{ID}]
\]

\[
E[u \mid \text{OOD}]
\]

Desirable behaviour generally includes:

\[
E[u \mid \text{incorrect}] >
E[u \mid \text{correct}]
\]

and:

\[
E[u \mid \text{OOD}] \gg
E[u \mid \text{ID}]
\]

### Recommended plots

- box plot of vacuity for correct vs incorrect predictions;
- violin plot of vacuity for correct / incorrect / OOD;
- histogram or ECDF of vacuity;
- per-class vacuity distribution.

---

## 5. Uncertainty versus prediction error

Test whether uncertainty rises as predictions become less reliable.

Possible error quantities:

\[
1 - p_y
\]

or:

\[
-\log p_y
\]

where \(p_y\) is the projected probability assigned to the true class.

### Recommended plot

Scatter plot:

- x-axis: vacuity or other uncertainty score;
- y-axis: prediction error.

### Summary statistic

Spearman rank correlation:

\[
\rho(u, 1-p_y)
\]

A positive correlation indicates that higher uncertainty tends to accompany poorer predictions.

---

## 6. Error-detection AUROC

Treat prediction failure as a binary target:

\[
z_i =
\begin{cases}
1 & \hat y_i \neq y_i\\
0 & \hat y_i = y_i
\end{cases}
\]

Use uncertainty \(u_i\) as the detection score.

Calculate:

\[
AUROC(u, \text{prediction error})
\]

Interpretation:

- 0.5: uncertainty provides little information about errors;
- greater than 0.5: errors tend to receive greater uncertainty;
- close to 1.0: uncertainty is very effective at identifying failures.

AUPRC may also be useful, particularly when errors are rare.

---

## 7. Risk-coverage / selective classification

Sort samples from lowest to highest uncertainty and progressively reject the most uncertain predictions.

For each retained coverage fraction, calculate error rate or accuracy.

Example quantities:

- coverage;
- selective accuracy;
- selective risk.

### Recommended plot

Risk-coverage curve:

- x-axis: fraction of predictions retained;
- y-axis: risk / error rate.

A useful uncertainty estimator should produce falling risk as uncertain predictions are rejected.

### Summary statistic

Area under the risk-coverage curve:

\[
AURC
\]

Lower is generally better when plotting risk against retained coverage.

---

## 8. Separate vacuity from dissonance / aleatoric uncertainty

A major advantage of subjective logic is the ability to distinguish:

### Vacuity / lack of evidence

Example:

\[
b=(0.05,0.03,0.02), \qquad u=0.90
\]

Interpretation:

> There is insufficient evidence.

This is predominantly epistemic uncertainty.

### Dissonance / conflicting evidence

Example:

\[
b=(0.48,0.47,0.03), \qquad u=0.02
\]

Interpretation:

> There is substantial evidence, but it supports conflicting hypotheses.

This is closer to aleatoric ambiguity.

### Recommended plot

Two-dimensional scatter:

- x-axis: vacuity / epistemic uncertainty;
- y-axis: dissonance / aleatoric uncertainty.

Colour points by one of:

- correct vs incorrect;
- true class;
- predicted class;
- ID vs OOD;
- graph structural category.

This plot is especially valuable because it shows whether the model meaningfully distinguishes **"I do not know"** from **"the evidence is ambiguous."**

---

## 9. Evidence strength

For Dirichlet models:

\[
S_i = \sum_k \alpha_{ik}
\]

Useful summaries:

- mean / median evidence strength;
- evidence on correct vs incorrect predictions;
- evidence on ID vs OOD samples;
- per-class evidence strength.

### Recommended plots

- evidence-strength histogram;
- box plot by correct / incorrect;
- box plot by ID / OOD;
- evidence vs prediction confidence;
- evidence vs node degree.

Excessively high evidence on incorrect predictions is a particularly important failure mode.

---

## 10. Class-conditional subjective-logic statistics

For each true class \(c\), calculate:

\[
E[u \mid y=c]
\]

\[
E[b_c \mid y=c]
\]

\[
E[S \mid y=c]
\]

Potential additional statistics:

- mean dissonance by class;
- mean projected probability assigned to true class;
- classwise ECE;
- classwise error-detection AUROC.

### Recommended plot

Per-class grouped bar or point plot containing:

- recall;
- mean true-class belief;
- mean vacuity;
- mean dissonance.

This can reveal classes that achieve similar classification performance but very different uncertainty behaviour.

---

## 11. Subjective-logic confusion analysis

Extend the standard confusion matrix.

### Matrix A: confusion frequency

\[
C_{ij} = N(y=i,\hat y=j)
\]

### Matrix B: mean uncertainty by confusion pair

\[
U_{ij} = E[u \mid y=i,\hat y=j]
\]

Additional matrices could contain:

- mean evidence strength;
- mean dissonance;
- mean projected confidence.

This distinguishes:

- frequent errors that the model recognises as uncertain;
- frequent errors made with unjustified confidence.

The latter are more concerning.

---

## 12. Out-of-distribution evaluation

If possible, create explicit OOD test sets.

Potential OOD graph cases include:

- unseen node feature distributions;
- novel graph topologies;
- sparse neighbourhoods;
- missing relationships;
- unseen relation patterns;
- corrupted observations;
- novel classes;
- subgraphs far outside the training distribution.

Useful metrics:

- OOD AUROC using vacuity;
- OOD AUPRC;
- mean vacuity ID vs OOD;
- evidence-strength ID vs OOD;
- risk-coverage under OOD contamination.

A well-behaved subjective-logic model should generally show elevated vacuity and reduced evidence on unfamiliar inputs.

---

## 13. GNN-specific structural diagnostics

Subjective-logic behaviour should also be stratified by graph properties.

Potential variables:

- node degree;
- neighbourhood size;
- number of observed neighbours;
- fraction of missing neighbours;
- hop distance from labelled nodes;
- relation type;
- graph density;
- component size;
- subgraph novelty;
- centrality;
- homophily / heterophily;
- amount of contradictory neighbouring evidence.

Useful plots:

- vacuity vs node degree;
- dissonance vs neighbourhood disagreement;
- evidence vs number of supporting neighbours;
- error rate vs graph distance from labelled nodes;
- uncertainty distributions grouped by relation type.

These help determine whether the uncertainty mechanism responds sensibly to the quantity and quality of graph evidence.

---

## 14. Suggested compact evaluation dashboard

Recommended headline statistics:

1. Macro-F1
2. Balanced accuracy
3. Negative log likelihood
4. Brier score
5. Expected calibration error
6. Mean vacuity
7. Error-detection AUROC
8. OOD AUROC
9. Optional AURC

Recommended core figures:

1. Confusion matrix
2. Reliability diagram
3. Vacuity: correct vs incorrect
4. Vacuity: ID vs OOD
5. Risk-coverage curve
6. Vacuity vs dissonance scatter
7. Evidence-strength distribution
8. Uncertainty vs graph structural property

---

## 15. Suggested implementation structure

A useful evaluation module could expose functions such as:

```python
evaluate_classification(...)
evaluate_calibration(...)
evaluate_vacuity(...)
evaluate_dissonance(...)
evaluate_evidence_strength(...)
evaluate_error_detection(...)
evaluate_ood_detection(...)
evaluate_risk_coverage(...)
evaluate_graph_structure_dependence(...)
plot_confusion_matrix(...)
plot_reliability_diagram(...)
plot_vacuity_distributions(...)
plot_vacuity_vs_dissonance(...)
plot_risk_coverage(...)
plot_graph_uncertainty_relationships(...)
```

A per-sample evaluation dataframe should ideally contain at least:

```text
sample_id
node_id / graph_id
true_class
predicted_class
correct
projected_probability_true_class
max_projected_probability
belief_vector
vacuity
dissonance
dirichlet_alpha
evidence_strength
is_ood
node_degree
neighbourhood_size
relation/context metadata
```

This makes later statistical analysis and plotting straightforward.

---

## 16. Key design principle

The central evaluation question is not simply:

> Is the GNN accurate?

It is:

> Is the GNN accurate, calibrated, and appropriately uncertain—and does its subjective-logic representation distinguish insufficient evidence from conflicting evidence in a useful way?

The most informative subjective-logic-specific diagnostics are likely to be:

- error-detection AUROC;
- OOD-detection AUROC;
- risk-coverage curves;
- vacuity distributions;
- vacuity-versus-dissonance plots;
- evidence strength on correct vs incorrect predictions;
- uncertainty stratified by graph structure.
