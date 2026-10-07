# Transaction-Correction Workflow Review

**Review date:** October 7, 2026  
**Scope:** 21 BANKING77 intents assigned to `transaction_correction`  
**Status:** Evidence reviewed; not domain-approved or frozen

## Review Standard

This review asks whether the message requires investigation, cancellation, tracing,
refund handling, or correction of a particular transaction. It also checks whether a
consequence flag has source support. A flag indicates that omission of the workflow can
leave a financial transaction unresolved; it does not assign a monetary value or state
that every example is legally entitled to reimbursement.

The review uses the current consolidated UK Payment Services Regulations. Regulations
74 and 81-94 cover notification, refusal, revocation, execution, tracing, and liability.
Regulations 43-61, 66, and 84 separately address information about charges and exchange
rates and deductions from transferred amounts.

## Decisions

| Labels | Intents | Decision | Evidence treatment |
|---|---|---|---|
| 5 | `balance_not_updated_after_bank_transfer` | Retain family, action, and flag | A missing bank transfer requires tracing or correction. |
| 6 | `balance_not_updated_after_cheque_or_cash_deposit` | Retain family and action; remove flag | The cited payment-execution provisions do not establish consequences for cheque or cash deposits. The operational assignment follows the dataset meaning only. |
| 8 | `cancel_transfer` | Retain | Revocation is addressed separately from failed or incorrectly executed payments, so the evidence scope now includes regulation 83. |
| 15, 19, 34 | Card-payment, withdrawal, and statement charges | Retain; use charge evidence | These are disputed transaction charges, but the evidence establishes information and charge rules rather than a guaranteed refund. |
| 17, 76 | Wrong exchange rate for card payment or withdrawal | Retain; use exchange-rate evidence | The workflow must investigate the applied transaction rate. Correction remains conditional on the investigation. |
| 25, 26, 27 | Declined card payment, cash withdrawal, or transfer | Retain | Refusal of a payment order is part of transaction execution handling. |
| 35 | `failed_transfer` | Retain | A failed transfer requires execution-state investigation even when no completed payment needs reversal. |
| 45, 46, 48 | Pending card payment, cash withdrawal, or transfer | Retain | The unresolved execution state requires tracing and potentially correction. |
| 51 | `Refund_not_showing_up` | Retain | The intent concerns tracing an expected financial correction that is not reflected. |
| 52 | `request_refund` | Retain family and action; remove flag | The BANKING77 label alone does not establish refund entitlement or urgency, so no consequence flag is asserted. |
| 53 | `reverted_card_payment?` | Retain | The apparent reversal requires investigation of payment state and merchant receipt. |
| 63 | `transaction_charged_twice` | Retain | A possible duplicate charge requires investigation and correction if confirmed. |
| 66 | `transfer_not_received_by_recipient` | Retain | A completed-looking but missing transfer requires tracing. |
| 75 | `wrong_amount_of_cash_received` | Retain | The discrepancy between the withdrawal and cash dispensed requires investigation and correction. |

## Evidence Used

- [UK Payment Services Regulations 2017, current contents](https://www.legislation.gov.uk/uksi/2017/752/contents)
- [UK Payment Services Regulations 2017, consolidated text](https://www.legislation.gov.uk/uksi/2017/752/2026-04-28/data.htm)

The legislation was listed as current with changes known to be in force through
September 15, 2026 when reviewed. Its requirements support the workflow categories,
but they do not provide numerical misrouting costs.

## Remaining Gate

These entries are `reviewed`, not `frozen`. A supervisor or banking-domain reviewer
must still confirm whether the proposed queues match realistic bank operations. The
cheque/cash-deposit and generic refund-request entries deserve particular attention
because their assignments are based on task semantics rather than the cited payment
execution provisions.
