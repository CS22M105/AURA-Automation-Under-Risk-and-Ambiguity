# Security and Unauthorized Workflow Review

**Review date:** October 7, 2026  
**Scope:** BANKING77 labels 16, 20, 22, 28, 41, 42, and 44  
**Status:** Evidence reviewed; not domain-approved or frozen

## Review Standard

This review checks whether each intent's operational family, required action, and
consequence flags follow from the meaning of the BANKING77 label and the cited UK
payments evidence. It does not estimate monetary loss, assign numerical weights, or
claim that a bank would use exactly these queues.

An entry can be marked `reviewed` when:

1. Its operational family describes the downstream handling it needs.
2. Its required action is compatible with that family.
3. Every consequence flag is supported by a cited source.
4. The rationale does not claim more than the evidence establishes.

Only supervisor or domain review can move an entry from `reviewed` to `frozen`.

## Decisions

| Label | Intent | Review decision | Reason |
|---:|---|---|---|
| 16 | `card_payment_not_recognised` | Retain | An unrecognized card payment requires authorization investigation, possible account or instrument protection, and correction or refund handling. |
| 20 | `cash_withdrawal_not_recognised` | Retain | An unrecognized cash withdrawal requires authorization investigation and may require prompt protection and financial correction. |
| 22 | `compromised_card` | Retain | The immediate operational need is to secure the payment instrument and prevent further use. No transaction-correction flag is added because the label does not state that a payment occurred. |
| 28 | `direct_debit_payment_not_recognised` | Retain | An unrecognized direct debit requires authorization investigation and dispute or refund handling. |
| 41 | `lost_or_stolen_card` | Retain | Loss or theft requires prompt prevention of further card use and replacement support. No unauthorized-transaction flag is added unless an unrecognized payment is also reported. |
| 42 | `lost_or_stolen_phone` | Retain with caveat | A lost phone may expose mobile-banking credentials, so access protection is appropriate. Institution-specific controls still require domain confirmation. |
| 44 | `passcode_forgotten` | Retain | This is authenticated access restoration, not evidence of compromise or an unauthorized transaction. It therefore remains in the security workflow but carries no consequence flag. |

## Evidence Used

- [UK Payment Services Regulations 2017, regulations 72-77](https://www.legislation.gov.uk/uksi/2017/752/part/7/crossheading/authorisation-of-payment-transactions/made?view=extent)
- [FCA fraudulent payments guidance, updated May 15, 2026](https://www.fca.org.uk/consumers/fraudulent-payments)

The FCA guidance says consumers should contact their provider immediately about an
unauthorized payment, describes investigation and next-business-day refund handling,
and directs consumers to report lost cards or exposed passwords promptly. These claims
support urgency and workflow type, not numerical error costs.

## Remaining Gate

All seven entries remain unusable by routing code because `reviewed` is not `frozen`.
Before freezing, a supervisor or banking-domain reviewer should confirm the family and
action assignments, especially the treatment of `lost_or_stolen_phone` and
`passcode_forgotten`.
