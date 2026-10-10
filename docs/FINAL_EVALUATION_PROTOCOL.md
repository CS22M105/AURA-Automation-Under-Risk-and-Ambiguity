# Final Evaluation Protocol v1

## Purpose and Status

Freeze the analysis before running the final routing comparison on the official
BANKING77 test partition. This plan follows exploratory development experiments;
it is not a preregistration of the entire project. No test data are opened by the
protocol-freezing or verification step. Historical test exposure has not been
independently established: disclose any earlier test-driven model or design choices
before claiming that the test provides independent confirmation.

The machine-readable specification is `config/final_evaluation.json`. The lock
record in `reports/protocols/aura-final-evaluation-v1.json` fingerprints that file,
this document, policy/evaluation code, and fitted artifacts. Commit the lock and
specification before final scoring. The check script detects drift; it does not
make files immutable or prevent someone from editing the lock itself.

## Frozen Inputs

- Same trained DistilBERT checkpoint and argmax predictions for every policy.
- Same fitted temperature scaler and intent/workflow definitions.
- Same flat, moderate, and high-protection consequence scenarios.
- Same trained lexical witness, semantic-only gate, and signed gate.
- No retraining, refitting, threshold search, new cost weights, or policy selection
  using test labels. The lexical witness remains fitted on model-training rows;
  gates remain fitted on calibration rows, also used for temperature fitting.

The cost-aware lexical variant uses the moderate matrix to rank messages in every
evaluation scenario. High-protection AURA uses the high-protection matrix to rank.
Changing the *evaluation* scenario does not change a policy's selection rule.
Saved local joblib artifacts are trusted inputs, not files to accept from strangers.

## Primary Comparison

Candidate: **cost-aware lexical gate** (`signed_cost_gate`).
Reference: **plain signed lexical gate** (`signed_gate`).
Operating point: **80% target coverage**, with equal floor-rounded acceptance counts.
Evaluation costs: **moderate scenario**, unchanged.

The primary endpoint is:

```text
delta = [sum accepted candidate cost - sum accepted reference cost] / N
```

Negative is favorable. The denominator includes all test messages. This comparison
asks whether cost weighting adds value beyond the learned lexical evidence, rather
than attributing every gain from the auxiliary model to consequence awareness.

The 80% target is a development-informed experimental choice, not a bank-provided
automation requirement. The moderate scenario is a research assumption, not measured
harm. Choosing these now does not erase the influence of previous development runs.

## Secondary Results

Report all nine policies from the configuration at 50%, 70%, 80%, 90%, and 100%
coverage under all three scenarios. Retain failed variants and repeated controls.
Always include confidence, original AURA, and the plain lexical gate in tables.
Report accepted/deferred counts, actual coverage, error counts, accuracy, total cost,
cost per input, and cost per accepted message. Empty accepted sets have undefined
accuracy and per-accepted cost, represented as null, not zero.

The majority safeguard may miss its target. Show the shortfall and compare it with
confidence and original AURA at its actual acceptance count. Never put its 96%-style
result in a 100% column without clearly marking the shortfall. Do not infer a
cost-aware contribution solely from beating original AURA or confidence.

## Selection and Uncertainty

This is **offline matched-budget ranking**, not a deployed fixed-threshold policy.
Score test messages without their labels; choose the lowest scores up to the
floor-rounded budget, breaking ties by original test row ID. Test score ranks
determine the offline cutoff, but test labels never determine it.

For the primary comparison, use the existing paired message bootstrap: 10,000
replicates, seed 42, two-sided 95% percentile interval. Resample the same messages
for both methods and rerank at the same coverage in each replicate. Repeated
copies have identical scores and costs. Report the interval and point estimate;
an upper endpoint below zero supports a benefit under this procedure. An interval
containing zero is inconclusive, not evidence of equivalence. An interval wholly
above zero favors the reference. Do not replace the primary endpoint after seeing
which secondary comparison wins.

Secondary intervals are pointwise descriptive only, not family-wise significance
claims. Intervals condition on fitted models and research costs, assume independent
messages, and do not include training variability or workflow-annotation uncertainty.
No human-review cost, reviewer mistakes, monetary loss, or deployment risk certificate
is inferred. Final evaluation should disclose exact normalized-text overlap between
test and training-pool data and duplicated test texts, without relabeling or silently
removing rows from the primary analysis. Any overlap-excluded analysis is secondary.

## Execution Contract

1. Run `python scripts/check_final_protocol.py` before final scoring.
2. Validate the official test schema and label mapping; record its checksum on the
   first final run. Preserve original row IDs. Do not use the test to repair models.
3. Infer and save frozen classifier probabilities, gate features, and policy scores.
4. Fix acceptance masks before computing labeled outcomes; save all decisions.
5. Evaluate the primary endpoint and complete secondary tables, including failures.
6. Save configuration, lock hash, test checksum, package/device metadata, scores,
   comparisons, intervals, and run provenance in the configured final directory.
7. Refuse to overwrite a completed final run. If a genuine implementation bug is
   discovered, document it, retain the prior output, version the protocol/code,
   and disclose that the test has already been inspected.

The final-evaluation runner is the next implementation step. This protocol freeze
does not run inference, fit a threshold, or produce final-test results.
The runner must have its own code fingerprint recorded before it opens the test
set; adding that runner must not silently alter the decisions specified here.
