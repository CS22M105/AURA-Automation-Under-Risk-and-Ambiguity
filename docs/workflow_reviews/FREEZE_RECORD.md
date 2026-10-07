# Workflow Specification Freeze Record

**Specification:** `aura-banking77-workflows-v1`
**Freeze date:** October 7, 2026
**Intended use:** Research experiments only
**External banking-domain review:** No

## Decision

Version 1 of the BANKING77 workflow specification is frozen after an internal,
source-informed review of all 77 intents. The review used local BANKING77 examples,
primary regulatory and standards sources, explicit rationales, and automated consistency
checks.

The freeze allows AURA's cost-aware policy experiments to proceed without continuing to
change workflow assumptions in response to model results. It is not expert validation
and must not be described as a production banking policy.

## What Is Frozen

- The mapping from 77 intent labels to operational families
- The required action for each intent
- The four qualitative consequence flags
- The evidence references supporting those flags
- The written rationale for each assignment

No numerical cost values, deferral thresholds, or AUTO/DEFER policy parameters are part
of this freeze. Those will be defined as multiple research scenarios and tested through
sensitivity analysis.

## Known Limitations

- No bank workflow, case-management, or loss dataset was available.
- No banking-domain expert independently validated the assignments.
- BANKING77 contains intent text, not transaction amounts or customer vulnerability.
- Some labels contain semantic overlap or noisy examples.
- The resulting consequence model is source-informed and semi-quantitative, not a
  monetary loss model.

## Change Control

Any later change requires:

1. A written reason and affected intent labels.
2. A new specification identifier.
3. Revalidation of all 77 mappings and evidence links.
4. Regeneration of downstream consequence matrices.
5. Separate reporting from results produced with version 1.
