# Account and Transfer-Servicing Workflow Review

**Review date:** October 7, 2026
**Scope:** The final five provisional BANKING77 intents
**Status:** Dataset and evidence reviewed; not domain-approved or frozen

## Review Standard

This batch completes the first-pass review of all 77 intents. Account-administration
intents must request a change to the customer's account record or lifecycle. Transfer
servicing must concern an outgoing transfer recipient or a completed-transfer charge.
Instructions for funding the customer's own account belong to funding support instead.

## Decisions

| Label | Intent | Decision | Reason |
|---:|---|---|---|
| 7 | `beneficiary_not_allowed` | Retain transfer servicing without flag | The request concerns configuring or resolving eligibility for an outgoing-transfer beneficiary; it does not establish that funds moved. |
| 30 | `edit_personal_details` | Retain account administration without flag | The user requests a routine update to name, address, or other account details. |
| 55 | `terminate_account` | Retain account administration without flag | The user explicitly requests account closure. No monetary or compliance consequence is inferred from that request alone. |
| 64 | `transfer_fee_charged` | Retain transfer servicing and correction flag; change evidence | The examples report a fee already charged on a transfer, unlike prospective fee-information intents. Charge and disclosure provisions are more precise evidence than general execution provisions. |
| 65 | `transfer_into_account` | Move to `funding_top_up` without flag | The examples ask how to add money to the customer's own account by bank transfer, not how to service an outgoing transfer. |

## Evidence Used

- [UK Payment Services Regulations 2017, current contents](https://www.legislation.gov.uk/uksi/2017/752/contents)
- Local BANKING77 training examples for labels 7, 30, 55, 64, and 65

The regulations support investigation and disclosure of an assessed transfer charge.
The remaining decisions are operational interpretations of the dataset and do not carry
consequence flags.

## Review Completion

All 77 intents have now received a first-pass dataset and evidence review. Their status
is `reviewed`, not `frozen`. This means:

- The label-to-workflow mappings are internally consistent and tested.
- Consequence flags are attached only to declared supporting evidence.
- No numerical costs have been introduced.
- Routing code must still reject the specification.

## Remaining Gate

A supervisor or banking-domain reviewer must inspect the review documents and approve
or revise the mappings. Only that external review should change entries to `frozen`.
Until then, AURA cannot generate or evaluate a cost-aware auto/defer policy from this
specification.
