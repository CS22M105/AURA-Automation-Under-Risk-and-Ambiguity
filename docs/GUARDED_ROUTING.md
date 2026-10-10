# Majority-Confidence Safeguard Experiment

## Rule and Rationale

This development variant leaves DistilBERT, temperature calibration, and all
consequence scenarios unchanged. It was specified before executing this variant,
but after inspecting earlier development results; it is not a preregistered or
independent confirmatory experiment.

```text
eligible(x) = max_y P(y | x) > 0.5
budget = floor(target_coverage * number_of_messages)
accept up to budget eligible messages, ordered by moderate expected cost
break risk ties by ascending original row ID
```

The 0.5 boundary means the predicted intent has more probability mass than all
alternatives combined. It is a transparent structural constraint, not an estimated
banking safety threshold. No threshold sweep was performed. Exactly 0.5 is
ineligible. The rule never fills a shortfall with ineligible messages.

## Fair Comparison

The report includes confidence and original moderate AURA at both the requested
budget and the safeguard's realized acceptance count. Matched controls select
exactly that count, avoiding floating-point coverage rounding. All three existing
evaluation scenarios are reported, not just the scenario used to rank AURA.
Ground-truth labels are used only for evaluating already-selected messages.

## Observed Moderate-Scenario Results

| Target | Guard accepted | Guard cost | Original AURA cost at same count | Confidence cost at same count |
| --- | ---: | ---: | ---: | ---: |
| 50% | 750 | 4 | 4 | 4 |
| 70% | 1,050 | 21 | 21 | 29 |
| 80% | 1,200 | 59 | 59 | 56 |
| 90% | 1,350 | 120 | 120 | 128 |
| 100% | 1,445 | 219 | 207 | 219 |

At 50-90%, acceptance sets are identical to original AURA: the safeguard does not
bind. It therefore does not address the 80% regression.

At the 100% target, only 1,445 of 1,501 messages are eligible (96.27% coverage).
The 56 additional deferrals reduce cost relative to accepting everything, but
this is **not** a matched-coverage improvement. At the same count, original AURA
costs 207 versus the guard's 219. Original AURA accepts 125 errors, the guard 118:
fewer classification errors need not imply lower consequence-weighted cost.
Because every eligible message is accepted here, the guard's acceptance set is
also exactly the confidence baseline's set at that count.

## Conclusion and Limits

Retain original AURA as the research reference. This fixed majority safeguard
shows no improvement on the audited moderate-scenario comparisons. Do not promote
it or raise its threshold solely to improve these development results.

The result is descriptive, not a significance claim. Existing bootstrap intervals
do not apply to changed acceptance sets. Scenario units are research assumptions,
not observed banking losses. Review costs and reviewer errors are excluded.
The safeguard is an offline batch-selection experiment, not a fully specified
deployment policy or guarantee of safety.

## Reproduction

```bash
python scripts/evaluate_guarded_routing.py
python -m pytest tests/unit/policies/test_guarded.py -q
```

The reusable selector is `src/aura/policies/guarded.py`. The script verifies input
hashes and writes comparison rows for every scenario, per-message decisions,
and provenance metadata under `artifacts/policies/guarded_distilbert/`. These
ignored outputs are separate from the original baseline artifacts.
