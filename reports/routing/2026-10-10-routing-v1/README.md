# AURA Routing Results: 2026-10-10-routing-v1

Development / policy-validation only; 1,501 messages, frozen DistilBERT.
These are research-scenario costs, not measured banking losses or final-test results.

## Moderate-Scenario Cost

| Routing method | 50% | 70% | 80% | 90% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Confidence baseline | 4 | 29 | 56 | 128 | 300 |
| Flat-cost routing | 4 | 29 | 56 | 128 | 300 |
| Original AURA (moderate) | 4 | 21 | 59 | 120 | 300 |
| AURA (high-protection ranking) | 4 | 22 | 52 | 125 | 300 |
| Majority safeguard | 4 | 21 | 59 | 120 | Not reached |
| Logit margin | 4 | 35 | 57 | 142 | 300 |
| Semantic-only gate | 4 | 35 | 57 | 142 | 300 |
| Signed lexical gate | 2 | 13 | 46 | 104 | 300 |
| Cost-aware lexical gate | 3 | 14 | 33 | 96 | 300 |

Coverage headers are targets with floor-rounded budgets. Every numeric cell in a column uses the same acceptance count. At the 100% target, the guard actually accepts 1,445 messages (96.27%), costing 219; at that count original AURA costs 207 and confidence costs 219.

Compare every coverage level; do not infer uniform superiority from a selected result.

## Included Evidence

`comparisons.csv` retains all experiment rows, matched-budget controls, scenarios, coverage levels, counts, and error totals. Repeated baselines are intentional. `intervals.csv` names the candidate and reference for each available interval; these pointwise intervals do not cover every pair of variants and are not adjusted for repeated development experimentation.

## Preservation

Local archive: `artifacts/snapshots/2026-10-10-routing-v1.zip`. Its checksum and per-file hashes are in `manifest.json`. The archive includes saved routing outputs, learned gates, diagnostic messages, source code, tests, configuration, and documents. It excludes the raw dataset, base DistilBERT weights, virtual environment, and classifier comparison artifacts. This is a routing-results snapshot, not a complete training backup.

This directory is Git-trackable. The ZIP is Git-ignored and remains local; no remote backup or upload has been performed. The snapshot command refuses to overwrite an existing ID. Hashes detect modification but do not prevent manual editing.
