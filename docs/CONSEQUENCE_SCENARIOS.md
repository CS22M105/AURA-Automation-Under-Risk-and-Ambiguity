# AURA Consequence Scenarios

## Purpose

AURA cannot infer real monetary loss from BANKING77. The dataset has no transaction
amounts, customer vulnerability, case outcomes, or bank loss records. Version 1 therefore
uses three explicit relative-cost scenarios to test whether policy conclusions are
sensitive to consequence assumptions.

## Construction

Correct routing has cost `0`. Every incorrect intent starts with base cost `1`. A
scenario may impose a higher floor when the predicted route omits the true intent's
required action or an evidence-backed consequence dimension:

```text
cost(i, j) = 0                                      when i = j
cost(i, j) = max(applicable scenario floors)       otherwise
```

The maximum is used instead of addition so that one unauthorized-payment error is not
counted repeatedly for its overlapping protection, investigation, and correction flags.

## Scenarios

| Component | Flat | Moderate | High protection |
|---|---:|---:|---:|
| Any incorrect intent | 1 | 1 | 1 |
| Required-action mismatch | 1 | 2 | 3 |
| Transaction-correction omission | 1 | 3 | 5 |
| Compliance omission | 1 | 3 | 5 |
| Protective-action omission | 1 | 5 | 10 |
| Unauthorized-transaction omission | 1 | 5 | 10 |

`flat` is the control: it contains no consequence preference and reduces to ordinary
misclassification cost. `moderate` introduces an ordinal distinction. `high_protection`
stress-tests whether conclusions survive a stronger preference against missing security
and unauthorized-payment workflows.

These values are not measured losses, probabilities, currency, or claims about a real
bank. They are preregistered sensitivity assumptions in relative research units.

## Interpretation

The matrix is directed. For example, under `moderate`:

- `compromised_card` routed as `age_limit` costs `5` because protection is omitted.
- `age_limit` routed as `compromised_card` costs `2` because the required informational
  action is missed, but no protective consequence is omitted.
- `card_payment_not_recognised` routed as `cash_withdrawal_not_recognised` costs `1`
  because both intents reach the same unauthorized-transaction action and carry the
  same consequence dimensions.

The project should report a conclusion as robust only when its direction holds across
the declared scenarios and matched automation coverage levels.
