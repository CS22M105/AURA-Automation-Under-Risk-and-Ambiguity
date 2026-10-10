# Development Routing Uncertainty

## Question and Method

How stable is the moderate-AURA versus confidence cost difference under resampling
of the current policy-validation messages? This is exploratory development
analysis, not a final test or proof of banking safety.

The classifier, calibration, scores, and moderate cost matrix remain frozen.
For each of 10,000 replicates (seed 42), sample 1,501 message indices with
replacement. Use the same sampled messages for both policies, then rerank each
policy and accept its lowest-scoring `floor(coverage * 1501)` messages. Confidence
uses negative maximum probability; AURA uses expected moderate cost. Ties use
original row IDs; copies of a sampled message have identical scores and costs.
Ground-truth labels determine realized costs only, never selection.

For acceptance indicators A (AURA) and B (confidence), the statistic is:

```text
delta = sum_i cost(true_i, predicted_i) * (A_i - B_i) / N
```

Negative values favor AURA. Both policies have equal acceptance counts in every
replicate. The denominator includes every input, including deferred messages.
Deferrals contribute no *accepted-case* cost; human review is not assumed free.

The implementation uses SciPy's paired percentile bootstrap with 95% pointwise
intervals (2.5th and 97.5th bootstrap percentiles). See the
[SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
Resampling row indices keeps scores and observed costs paired automatically.
Percentile intervals are an approximate diagnostic, especially with sparse errors;
they are not exact-coverage guarantees. No hypothesis-test p-values are produced.

## Results

For readability these are cost differences **per 1,000 input messages**; machine
artifacts retain per-input units. Units come from the moderate research scenario,
not measured monetary or customer harm.

| Target coverage | AURA minus confidence | Pointwise 95% interval |
| --- | ---: | ---: |
| 50% | 0.00 | [-2.66, 0.00] |
| 70% | -5.33 | [-15.32, 2.00] |
| 80% | 2.00 | [-9.33, 8.66] |
| 90% | -5.33 | [-14.66, 6.00] |
| 100% | 0.00 | [0.00, 0.00] |

Both the 70% improvement and the 80% regression have intervals spanning zero.
The current development sample does not clearly establish their direction beyond
sampling variability under this procedure. This is not evidence of equivalence.
At 100%, equality is structural: both policies accept every message from the same
classifier. At 50%, the original totals tie but resampled rankings can differ.

## Limits

- These are individual intervals, not simultaneous intervals across coverages.
  They do not adjust for prior examination or selection of models and policies.
- Resampling assumes independent messages and permits class proportions to vary.
  It does not reproduce fixed stratified class counts or model customer/text clusters.
- Model-training and calibration variation, cost uncertainty, dataset-label
  ambiguity, reviewer outcomes, and distribution shift are not included.
- This evaluates offline fixed-budget ranking with cutoffs reselected on each
  sample. It is not an evaluation of a frozen deployment threshold.
- More bootstrap replicates do not add real observations or validate the cost
  assumptions. The held-out test remains necessary after freezing the protocol.

## Files and Reproduction

```bash
python scripts/evaluate_routing_uncertainty.py
python -m pytest tests/unit/evaluation/test_routing_uncertainty.py -q
```

The reusable calculation is in `src/aura/evaluation/routing_uncertainty.py`.
The script verifies input hashes and reproduces the saved comparison point
estimates before writing `artifacts/audits/routing_uncertainty/intervals.csv` and
`summary.json`. Metadata records the seed, replicate count, package versions,
input provenance, output hash, and limitations. Generated artifacts remain
ignored by Git; rerunning requires the saved scoring and comparison inputs.
