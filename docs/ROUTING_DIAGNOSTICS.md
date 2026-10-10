# Routing Probability and Consequence Diagnostics

## Scope

This retrospective report covers all 1,501 policy-validation messages, including
all 160 classification errors. No model, calibration parameter, consequence cost,
or policy was changed. The test set was not used.

The report separates measurable probability behavior from assumptions requiring
human review. It does **not** automatically classify an error as a calibration
failure, classifier failure, incorrect dataset label, or invalid cost mapping.
These explanations can overlap and cannot be identified from one labeled message.

## Calculation

The saved probabilities are temperature-scaled, with T = 0.7764522793956082.
Original logits were not saved with the risk artifact. For strictly positive
probabilities, scalar temperature scaling can be inverted:

```text
p_calibrated = softmax(logits / T)
p_raw_reconstructed = softmax(T * log(p_calibrated))
```

The additive normalization constant cancels in softmax. Reconstruction is
conditional on the saved temperature-scaling artifact and floating-point
precision, not a fresh model inference. Zero probabilities are rejected rather
than clipped or invented. Input hashes and row/class alignment are checked.

For each message the report records:

- True-intent probability before and after calibration, and its change.
- True-intent rank, defined as one plus the number of strictly larger probabilities
  (equal probabilities share rank).
- Negative log likelihood (NLL): `-log(probability of true intent)`.
- Reconstructed raw and calibrated expected routing cost.
- Observed scenario cost and true-intent contribution:
  `P(true intent | message) * C[true intent, predicted route]`.
- Required actions, omitted/shared flags, and each applicable cost floor.
- Confidence and AURA acceptance at all five existing coverage levels.

The observed cost is the **maximum**, not the sum, of applicable floors.
`error_contributions.csv` includes all 77 alternative-intent contributions for
each error (12,320 records), sorted by contribution within each message. These
explain a score, not a causal explanation of model behavior.

## Findings

| Group | Messages | Mean raw NLL (reconstructed) | Mean calibrated NLL |
| --- | ---: | ---: | ---: |
| All messages | 1,501 | 0.43423 | 0.38341 |
| Correct predictions | 1,341 | 0.16655 | 0.07158 |
| Incorrect predictions | 160 | 2.67774 | 2.99699 |

Calibration improves overall log loss, while true-label probability decreases on
110 of the 160 errors. Of the errors, 49 have the true intent outside the top
three probability ranks. All nine errors accepted only by AURA at 80% have lower
true-label probability after calibration; three have the true intent outside
the top three.

Conditioning on errors selects cases the model already got wrong. Worsening NLL
on that subset does not establish miscalibration, justify removing calibration,
or show that calibration caused the 80% routing regression. In particular, the
report does not compare raw-versus-calibrated acceptance sets. Positive scalar
temperature scaling preserves each message's intent ordering, so it cannot repair
a wrong predicted intent.

### A Concrete Example

Row 4488 explicitly requests freezing a compromised card, but the prediction is
`card_payment_not_recognised` rather than `compromised_card`:

- True-intent probability: about 2.02% reconstructed raw, 0.91% calibrated.
- True-intent rank: fourth.
- Observed moderate cost: 2, from the action-mismatch floor.
- True-intent contribution to expected cost: about 0.0182.
- Both workflows share `protective_action`, so it is not counted as omitted.

The low probability limits this intent's contribution to expected risk, and the
shared-flag assumption limits the assigned consequence cost. Neither proves that
the receiving route would actually freeze the card. This remains a workflow
validation question, not permission to increase a weight until AURA wins.

## What This Supports Next

Retain the current calibrated baseline. Evaluate any proposed confidence safeguard
or alternative calibration as a separately specified development variant, with
the same coverage budgets and unchanged evaluation scenarios. Calibration fitting
must remain restricted to calibration data. Independently validate required
actions for shared-flag routes; document mapping revisions separately from policy
improvements. Do not select changes using the held-out test set.

## Reproduction

From the AURA directory with its environment active:

```bash
python scripts/diagnose_routing_errors.py
python -m pytest tests/unit/evaluation/test_routing_diagnostics.py -q
```

Outputs under `artifacts/audits/routing_diagnostics/` are `messages.csv`,
`error_contributions.csv`, and `summary.json`. The summary records aggregates,
the source summary hash, and output hashes. Artifacts remain ignored by Git.
