# Learned-Cost Selector: Feasibility and Data Plan

## Status

This is a **post-test exploratory extension**, not an amendment to the completed
AURA final evaluation. The original final result remains inconclusive for the
incremental benefit of cost awareness over the plain lexical gate. No model was
trained, no cost definition changed, and no official test data or final-test
predictions were read in this audit.

The audit uses saved classifier outputs and lexical features from the original
calibration and policy-validation partitions. Their provenance hashes, row
alignment, and calibration correctness labels were checked against the frozen
training-pool manifest. These are predictions on rows excluded from classifier
training, not newly obtained or independently confirmed observations.

## Available Supervision

| Moderate scenario cost | Calibration | Policy validation |
| --- | ---: | ---: |
| 0 | 1,331 | 1,341 |
| 1 | 73 | 66 |
| 2 | 65 | 64 |
| 3 | 26 | 22 |
| 5 | 5 | 8 |
| Total messages | 1,500 | 1,501 |
| Total errors | 169 | 160 |

These are relative scenario costs, not measured financial losses or verified
real-world severity. Cost 5 is the maximum of the frozen moderate matrix; it is
not a new threshold introduced for this audit. Flat and high-protection counts
are also saved in the generated artifacts.

Calibration has 132 distinct erroneous true/predicted intent pairs; 107 occur
only once. Nine true intents have no errors at all. The five maximum-cost errors
cover only three true intents. Validation has 121 distinct error pairs, 96 seen
once, 17 true intents without errors, and eight maximum-cost errors spanning four
intents. This does not support treating all 5,929 intent pairs as separately
learnable consequences.

Omission counts are descriptive and can overlap within a single error:

| Omitted workflow flag | Calibration | Policy validation |
| --- | ---: | ---: |
| Compliance-sensitive | 3 | 3 |
| Protective action | 5 | 8 |
| Transaction correction | 24 | 21 |
| Unauthorized transaction | 4 | 3 |

There is one normalized-text match with model-training data in each partition;
both are correct predictions. Normalization lowercases and collapses whitespace.
Each development partition has unique normalized texts internally, and no
normalized text is shared between the two development partitions. This does not
rule out paraphrases or other dependence.

## What We Can and Cannot Claim

The saved data are sufficient to attempt a small numerical pilot. Counts alone
do not establish statistical power, adequate supervision, or likely improvement.
In particular, five maximum-cost training cases provide very limited coverage
of that phenomenon. Resampling them or increasing their weights does not create
additional independent evidence.

Both the temperature and existing correctness gates have already used calibration
labels. Policy validation has been inspected repeatedly. The 329 combined errors
are **not** a fresh training-and-confirmation resource. Pooling these partitions
for fitting would remove the current development comparison boundary.

## Proposed Pilot Boundaries

1. Preserve all original artifacts and results. Put any new experiment in a
   separate post-test namespace with explicit exploratory labels.
2. For the new pilot only, exclude exact normalized-text overlaps with classifier
   training. That leaves 1,499 calibration and 1,500 validation rows; error counts
   remain unchanged. Do not alter the original published/snapshotted comparisons.
3. Use calibration rows for selector fitting and any limited, predefined internal
   tuning. Keep policy-validation rows for descriptive comparisons only; do not
   call them a new holdout. Fit feature preprocessing within each internal fold.
4. Retain frozen DistilBERT and the lexical witness. Use only their raw semantic
   margin and signed lexical support as the initial features. Neither true intent,
   true cost, nor row ID is an input feature; the latter is only a tie-break key.
5. Compare a correctness-target model and a cost-target model with the same feature
   set, model family/capacity, fitting rows, tuning budget, and preprocessing. A
   small regularized regressor is a possible pilot, not a selected final method.
   Include the existing plain lexical gate as a strong reference as well.
6. The correctness target is `1[predicted != true]`; the cost target is the frozen
   `C[true, predicted]`. Both include all correct, zero-cost rows. Do not fit only
   on errors, which would estimate a different conditional quantity. Keep the
   matrix fixed and report performance under all existing scenarios.
7. Do not use the inspected official test to tune or validate this extension.
   Record any future comparison on it as post-test descriptive analysis, not
   independent confirmation. No such comparison is authorized by this plan.

For a direct-cost predictor `g(x)`, the experimental target is
`E[C(Y, predicted) | available selector features]`, which is not automatically
the full conditional expectation given the entire message. Omitted information,
sparse costly errors, and mistaken consequence assumptions can all limit it.

The pilot must retain unweighted error metrics and examine whether gains depend
on one or two expensive examples. Do not interpret an improved weighted training
loss as improved routing. Specify ranking behavior for any out-of-range regressor
outputs before running the comparison.

## Evidence Needed for a Stronger Claim

Independent workflow review comes first: validate actions and consequence
assumptions without showing which policy wins. Keep revised mappings as separate
versions and do not silently replace the original evaluation costs.

For broader out-of-sample supervision, cross-fitting is an option, not something
already achieved by this audit. Each fold's classifier and lexical witness must
exclude its held-out messages and normalized-text groups; any fold-specific
calibration must also avoid those rows. That requires new training and creates
predictions from different base models. Transfer to the original frozen classifier
would then be an assumption to test, not a free increase in its error sample size.

A confirmatory extension needs newly collected, independently labeled data or an
additional suitable task with separately justified consequences. Model choices,
primary endpoint, and outcome handling must be fixed before inspecting those
outcomes. Sampling should represent the intended message distribution. If rare
consequences are deliberately enriched, report the sampling scheme and separate
stratum-level results; do not treat enriched averages as operational prevalence.

Choose an additional-data budget from a stated practically meaningful effect,
sampling design, and uncertainty/power analysis. This audit cannot justify a
universal minimum number of costly errors or promise a breakthrough.

## Recommendation

Do not train a high-capacity cost predictor or announce a new contribution from
these counts. A tightly scoped two-feature pilot is reasonable for exploration,
but the main constraints are sparse consequence-specific supervision and the lack
of an untouched evaluation resource. Keep the completed capstone result intact
while deciding whether independent review and new data are feasible.

## Reproduction

`python scripts/audit_learned_cost_data.py` creates the write-once directory
`artifacts/audits/learned_cost_data_v1/`. It refuses to overwrite an existing audit.
Outputs: `summary.json` (including per-intent support), `cost_counts.csv`,
`error_pairs.csv`, and `row_support.csv` (labels, features, costs, overlap flags).
The audit does not load `datasets/test.csv` or `artifacts/final_evaluation/`.
