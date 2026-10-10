# Direct-Cost Learning Pilot

## Status and Design

This is a **post-test exploratory development experiment**, not a new final-test
result. No official test data or final-test predictions were used. Original models,
cost scenarios, frozen protocol, and completed final results remain unchanged.

Following the data audit, remove the one normalized-text training overlap from
each development partition for this pilot only. Fit on 1,499 calibration rows and
compare on 1,500 already-inspected policy-validation rows. There are still only
five maximum-moderate-cost examples in the fitting data.

Two matched pipelines use exactly the same semantic-margin and signed-lexical-
support features, StandardScaler preprocessing, and ridge regression (`alpha=1`).
The error model targets `1[predicted != true]`; the cost model targets frozen
moderate cost divided by its maximum (5). Correct predictions remain zero-target
training examples. Standardization is fitted on calibration rows only.

Alpha 1 is the conventional library default, fixed before this run, not an
optimized value or a banking-derived parameter. Dividing the cost target by its
fixed maximum changes its scale, not its ordering or relative costs. Outputs are
unclipped ranking scores, which may lie outside [0,1]; they are not calibrated
probabilities or guaranteed expected losses. Lower scores are accepted first.

For a stronger reference, refit the existing two-feature logistic correctness gate
on the same 1,499 rows, with its unchanged C=1 setting. Also report original AURA
and confidence on the same evaluation rows. No feature, regularization, threshold,
or model-family search was performed. All five existing coverages and three cost
scenarios are retained in the outputs.

## Observed Moderate Costs

| Coverage | Error-trained ridge | Cost-trained ridge | Refitted lexical gate | Confidence | Original AURA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50% | 3 | 3 | 2 | 4 | 4 |
| 70% | 13 | 19 | 13 | 29 | 21 |
| 80% | 43 | 43 | 46 | 56 | 59 |
| 90% | 103 | 106 | 104 | 129 | 121 |
| 100% | 300 | 300 | 300 | 300 | 300 |

The cost-trained model does not reduce observed moderate cost relative to its
matched error-trained counterpart at any reported coverage. At 80%, both accept
21 errors. Their equal totals do not establish identical acceptance sets.

The evaluation population differs from the older 1,501-row development results.
Do not compare totals across those runs without matching rows and budgets. These
tables already align all pilot methods to the same 1,500 messages.

## Descriptive Uncertainty

Predeclared pilot contrasts at 80%, using 10,000 paired reranking bootstrap
replicates and seed 42. Differences below are per 1,000 input messages:

| Candidate minus reference | Difference | Pointwise 95% interval |
| --- | ---: | --- |
| Cost ridge minus error ridge | 0.00 | [-4.00, 2.00] |
| Cost ridge minus refitted lexical gate | -2.00 | [-6.00, 6.00] |

Both intervals include zero. Neither establishes an improvement. These are
conditional descriptive intervals on reused development data, not independent
confirmation or a power calculation. They do not include fitting uncertainty,
consequence-definition uncertainty, or reviewer outcomes.

## Conclusion

This particular two-feature linear pilot does not support an added benefit from
training directly on scenario cost. It does not prove that all learned-cost
methods fail, nor identify whether feature limitations, model capacity, scarce
costly examples, or the cost assumptions caused the outcome.

Do not promote it over the existing lexical method or reuse the inspected test
to seek a favorable result. Preserve this experiment. The next substantive
research decision is whether independent consequence review and additional
out-of-sample data are available; another tuning round does not replace them.

## Reproduction

```bash
python scripts/pilot_learned_cost.py
python -m pytest tests/unit/policies/test_learned_cost.py -q
```

The script refuses to overwrite `artifacts/exploratory/learned_cost_v1/`.
That directory contains comparison tables, row-level scores/decisions, intervals,
fitted selectors, and a summary recording row IDs, provenance, coefficients,
out-of-range prediction counts, and output hashes. Original baseline artifacts
are not overwritten. Only trusted local joblib files should be loaded.
