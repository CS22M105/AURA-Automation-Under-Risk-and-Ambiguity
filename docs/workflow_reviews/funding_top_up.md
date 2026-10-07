# Funding and Top-Up Workflow Review

**Review date:** October 7, 2026
**Scope:** Eleven BANKING77 intents assigned to `funding_top_up`
**Status:** Evidence reviewed; not domain-approved or frozen

## Review Standard

This review separates routine funding setup and information from unresolved movement of
money. A top-up receives the `transaction_correction` flag only when the labeled intent
states that funds are pending or have been reversed. A failed attempt alone does not
establish that money was taken, and a question about fees does not establish a disputed
charge.

This is deliberately conservative. BANKING77 contains no transaction amount, account
ledger, or downstream case outcome from which actual harm could be inferred.

## Decisions

| Label | Intent | Decision | Reason |
|---:|---|---|---|
| 2 | `apple_pay_or_google_pay` | Move from card servicing without flag | The local examples ask whether or how to fund an account through Apple Pay or Google Pay, rather than how to manage a linked bank card. |
| 4 | `automatic_top_up` | Retain without flag | This configures an optional funding feature. |
| 47 | `pending_top_up` | Retain action and flag | The examples explicitly describe funds remaining pending or absent from the available balance. |
| 56 | `top_up_by_bank_transfer_charge` | Change to information; remove flag | The examples primarily ask prospectively whether a fee applies rather than disputing an assessed charge. |
| 57 | `top_up_by_card_charge` | Change to information; remove flag | The examples ask what card-funding fees apply, so automatic financial-correction treatment would overstate the request. |
| 58 | `top_up_by_cash_or_cheque` | Retain without flag | This asks whether a funding method is supported. |
| 59 | `top_up_failed` | Change to funding support; remove flag | A declined or failed attempt needs diagnosis or an alternative method, but the label does not prove that funds require correction. |
| 61 | `top_up_reverted` | Retain action and flag | The examples describe funding that appeared and was then reversed or disappeared, leaving transaction state to investigate. |
| 62 | `topping_up_by_card` | Retain without flag, with caveat | The intended class is card-funding instructions, although some training examples appear to describe missing top-ups and overlap with label 47. |
| 65 | `transfer_into_account` | Move from transfer servicing without flag | The examples ask how to fund the customer's own account by bank transfer rather than how to route an outgoing transfer. |
| 71 | `verify_top_up` | Retain corrected funding assignment | The examples ask about a card top-up verification code, not customer identity due diligence. |

## Evidence Used

- [UK Payment Services Regulations 2017, current contents](https://www.legislation.gov.uk/uksi/2017/752/contents)
- Local BANKING77 training examples for the nine reviewed labels

The regulations support disclosure of payment charges and handling of payment execution
states. They do not establish that every failed top-up caused financial loss. The local
examples are therefore used to determine whether an intent is informational, failed,
pending, or reverted.

## Dataset Limitation

Label 62 contains examples that semantically resemble `pending_top_up`. This is a source
of label noise or intent overlap, not a reason to invent a higher cost. It should be
reported later when analyzing DistilBERT confusions between funding intents.

## Remaining Gate

All eleven entries are `reviewed`, not `frozen`. A banking-domain reviewer should confirm
whether pending and reverted card top-ups use the same operational queue and whether
failed top-ups ever require a separate financial-correction path.
