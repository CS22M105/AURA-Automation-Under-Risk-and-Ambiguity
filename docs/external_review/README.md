# Independent Consequence Review

Status: prepared 2026-10-10; no external review has been completed by this packet.

## Why This Is Next

The completed final evaluation did not establish an incremental benefit from
cost-awareness. The exploratory direct-cost pilot did not improve on its matched
error-trained comparator. More tuning on those inspected datasets cannot supply
independent confirmation. See [the pilot](../LEARNED_COST_PILOT.md).

This step tests the assumptions behind the routing consequences, not whether a
reviewer likes AURA's results. It does not change any frozen specification,
model, dataset, scenario, or published result.

## Who Can Review

Seek someone with relevant banking servicing, payment disputes, fraud operations,
or compliance experience. Record their scope of expertise, jurisdiction, and any
connection to the project. A supervisor without that experience may review the
methodology but cannot substitute for banking-domain validation. AI-generated
answers are not independent expert review.

One reviewer provides expert feedback, not inter-reviewer agreement. If two
qualified reviewers are available, collect their responses separately before
discussion; retain both originals and the subsequent adjudication.

## Coordinator Procedure

1. Agree on the institution or hypothetical operating setting, jurisdiction,
   customer channel, and meaning of AUTO and DEFER. AUTO currently means routing
   to a workflow, not executing a payment. Record actual actions a route can
   perform; do not assume a human or queue performs every action it might support.
2. Create a versioned inventory covering all 77 intent IDs and names. For each,
   select examples from the development-training partition only, using a fixed
   documented rule before review. Record row IDs and the source file hash. Do not
   select examples because a policy failed on them. Do not share test messages,
   predictions, policy names, comparative results, or proposed numerical weights.
3. Give the reviewer that inventory and one copy of Stage A of
   [REVIEW_FORM.md](REVIEW_FORM.md) per intent. Share neither the proposed workflow
   mappings nor Stage B until Stage A responses are returned and preserved.
   Label-only judgments must be recorded as such if examples are unavailable.
4. In Stage B, share the existing mapping and evidence references for comparison.
   These references are leads from the internal review, not freshly verified
   legal conclusions. Ask the reviewer to check applicability and cite sources
   or operational evidence for each proposed change. Record what they reviewed.
5. Assess directed route pairs. Start with every distinct pair whose proposed
   consequence flags overlap but whose required actions differ. This structural
   rule is fixed independently of model outcomes. Log the full selected pair
   inventory, its count, and any extra pairs with their selection rationale.
   Assess both directions separately. A shared flag does not prove the wrong
   route preserves the required action. This screen is not exhaustive validation
   of all 5,852 off-diagonal entries.
6. Only after action judgments are recorded, discuss consequence ordering and
   numerical assumptions using Stage B. Do not ask reviewers to choose weights
   that improve AURA. Preserve disagreement, missing evidence, and unknowns.
7. Archive dated responses with permission, redact identifying details for any
   public repository, and record completion and unresolved-item counts. Report
   prior exposure to results; do not call the review blinded when it was not.

## What Counts as Evidence

- An applicable source can support an obligation or action, not automatically a
  numerical ratio between two harms.
- Expert judgments can support a scoped workflow interpretation or ordering.
  They remain judgments, not measured monetary losses.
- Loss, handling-time, or outcome records can support specific measured quantities
  only with defined units, population, observation period, and uncertainty.
- Unknown is missing information. Never replace it with zero cost, an invented
  number, or an automatic approval. Keep it unresolved and disclose its scope.

If evidence supports only an ordering, retain scenario/sensitivity analysis;
do not describe an expected scenario score as expected real banking loss.
Deferral also has potentially nonzero delay, effort, and error costs. Without
observations, use matched review budgets rather than claiming optimal total cost
or perfect human resolution.

## Decision After Review

Record accepted, challenged, and unresolved judgments separately. A materially
revised mapping requires a new specification ID, new matrices, and a new
prospectively frozen experiment; preserve v1 and its results. Never silently
re-score the old test and present that as independent confirmation.

Fresh evidence requires [a separate collection protocol](NEW_DATA_PLAN.md).
No model replacement or additional tuning is authorized by this packet.

If no qualified reviewer or new data is available, the defensible capstone remains
the completed scenario-based study with its inconclusive incremental-cost result.
Independent review can strengthen construct validity; it cannot guarantee better
performance or a breakthrough.
