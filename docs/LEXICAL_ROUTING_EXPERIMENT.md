# Learned Lexical Routing Experiment

## Research Basis

[Cheng, Yang, and Lin, Signed Lexical Confidence for Risk-Calibrated Intent
Routing (September 2026 preprint)](https://arxiv.org/abs/2610.00262) combines a
semantic margin and signed word/character-model support in a learned correctness
gate while preserving the base prediction. We adapt that construction to frozen
DistilBERT; this is not an exact reproduction or a peer-reviewed safety guarantee.
The paper's independent risk-calibration procedure is not implemented here.

[Liang, Peng, and Sun, Selective Classification Under Distribution Shifts
(TMLR 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12470254/) motivates including
a logit-margin control. Its findings do not establish performance on our banking
cost scenarios or robustness to banking distribution shift.

## Our Experiment

No new transformer was trained. The saved DistilBERT labels, calibration,
workflow definitions, cost matrices, and development split remain unchanged.
The auxiliary lexical model advises acceptance only; it never replaces an intent.

The evaluation contains six policies fixed before this run: confidence, original
AURA, logit margin, a margin-only learned gate, a signed lexical gate, and our
cost-aware extension of that gate. No hyperparameter or threshold search was run.
This is nevertheless exploratory, because earlier development results informed
the choice of experiment. The official test set remains untouched.

### Data Boundaries

- Fit the lexical witness on 7,002 model-training rows only.
- Fit correctness gates on 1,500 calibration rows, containing 169 classifier
  errors. Standardization is fitted on those rows only.
- Compare on the same 1,501 policy-validation rows used by original AURA.
- Calibration rows were already used to fit the temperature. They are now also
  used to fit a gate; there is no independent risk-certification partition here.
  This reuse does not make policy-validation labels gate-training labels, but it
  precludes claiming an independent calibration guarantee.

The manifest enforces disjoint row IDs. This experiment retains the existing
split; it does not establish independence of duplicate or semantically related
messages across partitions.

### How the New Cost Score Works

Let `j` be the unchanged DistilBERT prediction. The gate supplies an estimated
error probability `q(x)` using its two standardized inputs. Its coefficients are
learned from held-out correctness, not manually chosen scenario weights.

Original AURA can be decomposed as:

```text
r(x) = sum over i != j of p(i | x) * C[i, j]
a(x) = sum over i != j of p(i | x)
s(x) = r(x) / a(x)
r(x) = a(x) * s(x)
```

Our experimental extension replaces `a(x)` with the learned error estimate:

```text
new_score(x) = q(x) * s(x)
```

Accept the lowest scores up to the existing coverage budget. `s(x)` is the
model's conditional consequence severity given that its prediction is wrong.
The extension retains the relative probabilities among alternative intents.
That assumption may be wrong, and the learned gate is not independently
probability-calibrated; therefore this is a **ranking proxy**, not a validated
estimate of banking harm. It is our adaptation, not the cited paper's cost rule.
With flat unit costs, it reduces to `q(x)`; if `q(x) = a(x)`, it reduces to
original AURA. Both identities have unit tests.

For evaluation, raw semantic logit differences are reconstructed as differences
of `T * log(p_calibrated)`. Calibration-partition logits come from fresh inference
on the same frozen local checkpoint. Input hashes and prediction alignment are
verified, and the checkpoint hash is checked again after the experiment.

### Learned Parameters

The fitted two-feature gate predicts correctness. On standardized inputs its
semantic-margin coefficient is 1.0561, signed-support coefficient 1.6470, and
intercept 3.9296. These values are fitted parameters, not cost weights, and are
specific to this run. Saved pipelines contain the corresponding scaling values.
The positive signed-support coefficient means lexical agreement increases the
estimated correctness, with all else fixed.

## Development Results

All entries below are moderate-scenario accepted-case cost. Every policy accepts
the same number of messages at each target. Lower is better; units are research
scenario units, not money.

| Target | Confidence | Original AURA | Margin | Margin-only gate | Signed gate | Signed cost gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50% | 4 | 4 | 4 | 4 | 2 | 3 |
| 70% | 29 | 21 | 35 | 35 | 13 | 14 |
| 80% | 56 | 59 | 57 | 57 | 46 | 33 |
| 90% | 128 | 120 | 142 | 142 | 104 | 96 |
| 100% | 300 | 300 | 300 | 300 | 300 | 300 |

At 80%, signed cost gating reduces accepted errors from 35 to 20 and scenario
cost from 59 to 33 (44.1% observed reduction versus original AURA). At 90%, errors
fall from 73 to 54, and cost from 120 to 96. All policies must coincide at 100%
because the underlying classifier's predictions are unchanged.

The cost extension is not uniformly best: the plain signed gate has lower
moderate cost at 50% and 70%. Margin alone does not explain the improvement.
The positive margin-only logistic coefficient preserves margin ranking, so its
acceptance sets match the margin control. This experiment does not isolate signed
support from a binary agreement feature or unsigned lexical margin.

Across unchanged scenarios, original AURA versus signed cost gate:

| Target | Flat | Moderate | High protection |
| --- | --- | --- | --- |
| 50% | 2 -> 2 | 4 -> 3 | 6 -> 4 |
| 70% | 14 -> 10 | 21 -> 14 | 28 -> 18 |
| 80% | 35 -> 20 | 59 -> 33 | 84 -> 46 |
| 90% | 73 -> 54 | 120 -> 96 | 168 -> 139 |
| 100% | 160 -> 160 | 300 -> 300 | 448 -> 448 |

## Uncertainty and Interpretation

Using the existing paired reranking bootstrap (10,000 replicates, seed 42), the
moderate-cost difference, signed cost gate minus original AURA, per 1,000 inputs:

| Target | Difference | Pointwise 95% interval |
| --- | ---: | --- |
| 50% | -0.67 | [-4.00, 2.00] |
| 70% | -4.66 | [-12.66, 0.00] |
| 80% | -17.32 | [-29.31, -6.00] |
| 90% | -15.99 | [-27.98, -2.00] |
| 100% | 0.00 | [0.00, 0.00] |

The 80% and 90% intervals favor the new variant under this procedure. They are
conditional on fitted models and the empirical message distribution, not adjusted
for multiple comparisons or repeated development inspection. They do not capture
gate-training variation, uncertain costs, annotation ambiguity, human review
costs/errors, or distribution shift. They are not certificates of deployed risk.
No comparative interval against confidence or the plain signed gate is claimed.

Retain all variants and failed experiments. This is a promising candidate for a
frozen final evaluation protocol, not permission to report validation as test
performance or to claim a new classifier contribution.

## Reproduction and Artifacts

From the AURA root, with transformer dependencies already installed:

```bash
python scripts/evaluate_lexical_routing.py
python -m pytest tests/unit/policies/test_lexical.py -q
```

The implementation is in `src/aura/policies/lexical.py`. Defaults follow the
referenced gate construction (SVM and logistic regularization C=1); pipeline
parameters and fitting details are recorded in the script and artifact metadata.
There is no new dependency or download required for this local run.

Ignored outputs live under `artifacts/policies/lexical_distilbert/`:

- `comparison.csv`: all six policies, five coverage levels, and three scenarios.
- `decisions.csv` and `scores.csv`: row-level decisions, scores, and gate features.
- `intervals.csv`: signed cost gate versus original AURA for every scenario.
- `gates.joblib`: lexical witness and fitted gates; load only trusted local files.
- `gate_fit.npz`: fitting row IDs, features, correctness, and frozen-model logits.
- `summary.json`: source provenance, output hashes, coefficients, and limitations.
