# AURA Final Routing Results

## Evaluation Status

The saved `aura-final-evaluation-v1` run completed on October 10, 2026, using
3,080 official BANKING77 test messages. Its recorded output checksums were
verified during this review. No models, gates, costs, decisions, or protocol
settings were changed, and the test was not rerun.

This is a test-partition evaluation following a development-informed protocol,
not a preregistration of the entire project. The recorded exposure note remains:
"Historical test exposure not independently verified."

## Primary Result: Inconclusive

The frozen primary comparison is **cost-aware lexical gate versus plain signed
lexical gate**, at 80% automated coverage under the moderate cost scenario.
Each accepts 2,464 messages and defers 616.

| Quantity | Plain lexical gate | Cost-aware lexical gate |
| --- | ---: | ---: |
| Accepted errors | 39 | 40 |
| Accepted accuracy | 98.42% | 98.38% |
| Accepted scenario cost | 62 | 60 |
| Scenario cost per input | 0.020130 | 0.019481 |

The primary difference (cost-aware minus plain) is -0.000649 per input, or
**-0.65 scenario-cost units per 1,000 inputs**. Its paired 95% percentile bootstrap
interval is **[-5.19, +1.62] per 1,000 inputs**, with 10,000 resamples and seed 42.

Because the interval includes zero, the predeclared support criterion is not met.
The experiment does not establish that consequence weighting improves on the plain
lexical gate. This is also not evidence that the policies are equivalent. A small
observed cost decrease alongside an additional classification error illustrates
the difference between weighted cost and ordinary error count.

## Secondary Comparisons

All table entries are observed accepted-case cost under the **same moderate
evaluation scenario**. Headers are target coverage. Numeric entries within each
column have equal acceptance counts. Costs are research units, not money or
measured customer harm.

| Routing method | 50% | 70% | 80% | 90% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confidence baseline | 13 | 53 | 95 | 233 | 581 |
| Flat-cost routing | 13 | 53 | 95 | 233 | 581 |
| Original AURA (moderate) | 15 | 51 | 108 | 234 | 581 |
| AURA (high-protection ranking) | 15 | 52 | 110 | 241 | 581 |
| Majority safeguard | 15 | 51 | 108 | 236 | Not reached |
| Logit margin | 11 | 51 | 110 | 253 | 581 |
| Semantic-only gate | 11 | 51 | 110 | 253 | 581 |
| Signed lexical gate | 9 | 27 | 62 | 169 | 581 |
| Cost-aware lexical gate | 9 | 29 | 60 | 174 | 581 |

At 80%, confidence routing accepts 63 errors and original AURA 76, versus 39 for
the plain lexical gate and 40 for the cost-aware gate. The cost-aware gate's
observed cost is 36.8% below confidence (95 to 60) and 44.4% below original AURA
(108 to 60). These are descriptive secondary differences, not substitutes for
the primary comparison; no secondary uncertainty intervals were computed.

The plain lexical gate has lower moderate cost than the cost-aware gate at 70%
and 90%, and ties at 50%. Original AURA is not uniformly better than confidence.
Do not hide these outcomes or claim one method is uniformly best.

At the 100% target, the majority safeguard accepts 2,929 messages (95.10%) and
costs 382. Confidence and original AURA also cost 382 at that same acceptance
count. The guard's lower cost relative to full acceptance is not an improvement
at equal coverage. Every unguarded policy accepts all 315 classifier errors at
100%, with identical scenario cost, as expected from frozen predictions.

## Scenario Sensitivity

At the primary 80% coverage, the plain versus cost-aware lexical gates have:

| Evaluation scenario | Plain lexical | Cost-aware lexical |
| --- | ---: | ---: |
| Flat | 39 | 40 |
| Moderate | 62 | 60 |
| High protection | 86 | 81 |

The preferred ordering depends on the assumed consequences. At 90%, moderate
cost favors the plain gate (169 versus 174), while high-protection cost favors
the cost-aware gate (239 versus 235). This supports reporting sensitivity, not
claiming that the chosen costs represent verified banking harm.

## Data and Statistical Limitations

- Seven test rows match training-pool text after lowercase/whitespace normalization.
- Two test rows belong to a duplicated normalized-text group (one excess copy).
- No rows were excluded from the primary analysis, consistent with the protocol.
- These overlaps weaken a strict independence interpretation and must be disclosed.
  An overlap-excluded analysis would be secondary, not a replacement primary result.
- The bootstrap conditions on the fitted models and costs and assumes independent
  messages. It does not model training variation, uncertain workflow definitions,
  human-review cost/errors, or deployment distribution shift.
- The learned gate is not independently probability-calibrated, and the cost
  extension retains DistilBERT's relative probabilities among alternative intents.
- Selection is offline matched-budget ranking, not a frozen deployment threshold.

## Presentation Conclusion

> On the BANKING77 test partition, lexical evidence improved the observed routing
> trade-off relative to confidence and original AURA at the examined partial
> coverage levels. However, the predeclared comparison did not establish an
> additional benefit from consequence weighting over the plain lexical gate.
> Its value remains conditional on the assumed costs and operating point.

This is a meaningful research result, including a limitation of the proposed
cost-aware contribution. Do not tune further on these test outcomes and then
reuse them as independent evidence. Keep further experiments explicitly post-test.

## Evidence Files

Source directory: `artifacts/final_evaluation/aura-final-evaluation-v1/`.
The primary result is in `primary_result.json`, all 165 comparison rows in
`comparison.csv`, data-overlap counts in `overlap_summary.json`, and provenance
and output hashes in `summary.json`. `scores.csv`, `decisions.csv`, and
`predictions.npz` retain the per-message evidence.
