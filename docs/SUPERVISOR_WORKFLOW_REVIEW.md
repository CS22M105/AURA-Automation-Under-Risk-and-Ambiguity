# AURA Supervisor Workflow Review

**Prepared:** October 7, 2026
**Project:** Cost-aware selective intent classification on BANKING77
**Current status:** Internally frozen for research; external validation remains optional

## The Project in One Paragraph

AURA uses a frozen DistilBERT classifier to predict one of 77 BANKING77 customer-support
intents. It will later decide whether to accept that prediction automatically or defer
the message for human review. The decision should account for the fact that confusing
two routine questions is different from missing a compromised card or unauthorized
payment. Before assigning any costs, the project therefore maps every intent to the
support action it requires and records which consequences have external evidence.

## Where This Review Fits

```text
Customer message
      |
      v
Frozen DistilBERT classifier
      |
      v
Calibrated probabilities for 77 intents
      |
      +-------------------------------+
      |                               |
      v                               v
Predicted intent             Frozen workflow map v1  <-- INTERNALLY REVIEWED
      |                               |
      +---------------+---------------+
                      v
             Consequence scenarios    <-- NOT CREATED YET
                      |
                      v
              Expected routing risk
                      |
                      v
                 AUTO / DEFER
```

The classifier and calibration are complete. The workflow map has received an internal
dataset-and-evidence review and is frozen for research experiments. It has not been
validated by a banking-domain expert, so this packet remains available for optional
external review and later improvement.

## What Has Been Defined

Each BANKING77 intent has five relevant fields:

1. **Operational family:** the type of support workflow it should reach.
2. **Required action:** what that workflow must do.
3. **Consequence flags:** evidence-backed consequences that may be omitted by a wrong route.
4. **Evidence sources:** sources supporting each consequence flag.
5. **Rationale:** a short explanation of the assignment.

No monetary losses, pairwise costs, or auto/defer thresholds have been assigned.

## Workflow Summary

| Operational family | Intents | Consequence-bearing intents | Purpose |
|---|---:|---:|---|
| Account administration | 2 | 0 | Update or close an account |
| Card servicing | 15 | 0 | Fulfil, manage, or restore a card feature |
| Funding and top-up | 11 | 2 | Configure or resolve account funding |
| Identity and compliance | 4 | 3 | Perform or explain customer due diligence |
| Routine information | 15 | 0 | Answer eligibility, product, fee, rate, or timing questions |
| Security protection | 4 | 3 | Secure access or a payment instrument |
| Transaction correction | 21 | 19 | Trace, cancel, refund, or correct a transaction |
| Transfer servicing | 2 | 1 | Configure a beneficiary or investigate a transfer charge |
| Unauthorized transaction | 3 | 3 | Investigate authorization, protect access, and handle correction |
| **Total** | **77** | **31 unique intents** | |

An intent may carry multiple consequence flags. The four flags are:

- `protective_action`
- `unauthorized_transaction`
- `transaction_correction`
- `compliance_sensitive`

## Decisions Requiring Particular Attention

| Intent | Current decision | Question for reviewer |
|---|---|---|
| `lost_or_stolen_phone` | Security protection with protective-action consequence | Should a lost phone enter the same immediate protection workflow as a compromised card? |
| `passcode_forgotten` | Security workflow without a consequence flag | Is authenticated access restoration the correct route, despite no reported compromise? |
| `card_swallowed` | Card restoration without a security flag | Does an ATM-retained card require an institution-specific security escalation? |
| `verify_source_of_funds` | Compliance-sensitive workflow | Do the broad dataset examples justify compliance treatment, or should some be ordinary transaction-origin questions? |
| `balance_not_updated_after_cheque_or_cash_deposit` | Correction workflow without a consequence flag | Is this the correct operational family even though the cited payment regulations do not cover it cleanly? |
| `request_refund` | Refund workflow without a consequence flag | Is it appropriate to avoid assuming refund entitlement from the label alone? |
| `topping_up_by_card` | Funding support without a consequence flag | The examples overlap with pending top-ups. Should this remain an instructional intent? |
| `get_physical_card` | Card management for PIN retrieval | The label is misleading, but the dataset examples consistently ask where to find a PIN. Is this action acceptable? |
| `apple_pay_or_google_pay` | Funding support | The examples concern digital-wallet top-ups rather than general wallet/card management. Is funding the right route? |
| `transfer_into_account` | Funding support | The examples concern adding money to the customer's account. Should this remain separate from outgoing-transfer servicing? |

## Evidence Boundary

The evidence supports workflow type and consequence dimensions, not numerical costs.
The principal sources are:

- UK Payment Services Regulations 2017
- FCA fraudulent-payments guidance, updated May 2026
- FATF Recommendation 10, amended June 2026
- Local BANKING77 examples for task semantics

The specification validator rejects consequence flags with missing, unknown, or
unrelated evidence. It also rejects routing use unless every intent is marked `frozen`.

## Optional External Review

An external reviewer may review the workflow summary and the ten highlighted decisions
above. The complete
machine-readable specification is in `config/intent_workflows.yaml`. Detailed reasoning
is available in `docs/workflow_reviews/`.

Select one outcome:

- [ ] **Approve:** The workflow assignments and consequence flags may be frozen for the experiment.
- [ ] **Approve with changes:** Apply the revisions listed below, then return for confirmation.
- [ ] **Do not approve:** The proposed workflow framework needs substantial revision.

**Required revisions:**

```text


```

**Reviewer name:** ______________________________________________

**Role or expertise:** __________________________________________

**Review date:** _________________________________________________

**Signature or written confirmation reference:** __________________

## What Happens After External Review

1. Record the reviewer and approval reference.
2. Apply any requested revisions and rerun validation.
3. Publish a new frozen specification version if assignments change.
4. Generate qualitative pairwise routing-consequence profiles.
5. Define multiple explicit cost scenarios and sensitivity tests.
6. Compare confidence-only and cost-aware AUTO/DEFER policies at matched coverage.

Approval does not validate AURA for real banking deployment. It only freezes the
research assumptions before policy experiments, preventing test-driven or arbitrary
changes to the workflow map.
