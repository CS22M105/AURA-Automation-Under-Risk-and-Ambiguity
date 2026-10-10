# Prospective Data Collection Plan

Status: planning template, not an approved protocol or a collected dataset.
Do not collect private banking records without appropriate authorization.

## Separate Two Questions

1. Representative evaluation: independently obtained messages sampled from a
   declared population can estimate routing performance for that population.
2. Targeted challenge evaluation: deliberately enriched action-sensitive cases
   can reveal failure modes, but cannot estimate natural frequency or aggregate
   real-world risk without justified sampling weights.

Keep these cohorts separate in storage and reporting. Independently authored
fictional messages are useful challenge data, not real customer observations.
Do not paraphrase BANKING77 test cases and call the result an independent test.

## Record Before Collection

- Population, institution/jurisdiction/channel, time window, permissions, consent
  where applicable, retention policy, and public-sharing restrictions.
- Sampling frame, inclusion/exclusion rules, recruitment or authoring method,
  cohort definitions, and whether authors have seen BANKING77 or model outputs.
- Primary question, candidate/reference policy, outcome definition and units,
  review budget, treatment of ties, uncertainty method, and stopping rule.
- Sample-size rationale using a desired precision or detectable effect and
  explicit assumptions. Use sensitivity calculations when rare-event frequency
  is uncertain; there is no justified universal sample count in this packet.
- Roles allowed to see training/development data versus the sealed evaluation
  set. Set the split rule before fitting, grouping related conversations,
  authors, sources, and duplicates to avoid cross-partition leakage.

DistilBERT remains frozen. Decide explicitly whether any new data will fit a
selector/calibrator. Such data must be separated from the new final evaluation.
Do not reuse inspected BANKING77 test results for selection or threshold tuning.

## Minimum Private Record

Use a structured record with these fields; keep identifying metadata outside Git:

| Field group | Required information |
| --- | --- |
| Provenance | record ID, cohort, collection date, source type, permission reference, source/author/conversation group IDs |
| Input | redacted message, redaction status, language/channel, collection context |
| Independence | benchmark exposure, generation method, duplicate/group audit status |
| Annotation | separate annotator IDs and labels, required actions, evidence, uncertainty, disagreement and adjudication |
| Label scope | supported intent / ambiguous / multiple intents / outside 77; reason |
| Consequences | supported omitted-action consequences, units and sources for measured quantities; unknown otherwise |
| Outcomes | observed handling time, delay, resolution and errors only if actually observed; missing otherwise |
| Partition | assigned split, group assignment rule, manifest version |

Never store credentials, account numbers, transaction identifiers, or unredacted
personal data in the repository. Pseudonymous IDs can still be sensitive.

## Annotation and Independence Checks

Annotate without model predictions or policy scores. Prefer two independent
annotators followed by documented adjudication; disclose when resources permit
only one. Use qualified domain review for consequence claims, not merely intent
labeling. Preserve uncertain or unsupported cases rather than forcing 77 labels.

Predeclare treatment of unsupported/ambiguous cases and report their counts.
Excluding them limits conclusions to supported in-scope messages; it is not
evidence that AURA handles unknown intents safely.

Audit exact and normalized duplicates against all existing BANKING77 partitions
and new records. Check near-duplicate/source-group relationships as well; exact
matching alone cannot prove independence. Use any legacy-test overlap check only
for contamination detection, not to expose its outcomes to policy selection.
Record exclusions before scoring. Hash the final inputs and split manifest.

## Release Gate

Proceed to a new confirmatory run only after permissions, annotation scope,
consequence specification, sampling rationale, fixed policies/thresholds, primary
contrast, and evaluation manifest are recorded. Fix deployed thresholds on
development data; distinguish that evaluation from retrospective matched-budget
ranking. Freeze all inputs and evaluate the sealed set once.

Preserve negative results. Report annotation and cost uncertainty alongside
sampling uncertainty. Compare cost-aware routing with the same classifier's
confidence and matched error-aware controls at the declared budgets. A claim of
lower total operational cost additionally requires measured deferral costs and
reviewer outcomes; automatic-routing scenario cost alone does not establish it.
