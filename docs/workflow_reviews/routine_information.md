# Routine-Information Workflow Review

**Review date:** October 7, 2026
**Scope:** Fifteen BANKING77 intents assigned to `routine_information`
**Status:** Dataset reviewed; not domain-approved or frozen

## Review Standard

An intent remains in `routine_information` when its labeled request can be satisfied by
explaining eligibility, availability, limits, fees, rates, timing, or instructions. The
topic may mention cards, transfers, or top-ups, but it does not report a transaction that
must be traced or corrected and does not request a protected operational action.

No consequence flag is assigned merely because the subject is financial. BANKING77
does not provide transaction amounts, customer circumstances, or evidence that an
informational response caused harm.

## Decisions

| Labels | Intents | Decision | Reason |
|---:|---|---|---|
| 1, 24 | `age_limit`, `country_support` | Retain | These explain account or product eligibility. |
| 3, 10 | `atm_support`, `card_acceptance` | Retain | These explain where a card can be used. |
| 12 | `card_delivery_estimate` | Retain | This asks for a general delivery timeframe, unlike `card_arrival`, which concerns an already missing or trackable delivery. |
| 29 | `disposable_card_limits` | Retain | This explains feature restrictions; occasional failure wording is label overlap rather than proof of a transaction problem. |
| 31, 32 | `exchange_charge`, `exchange_rate` | Retain | These ask prospectively about exchange pricing rather than disputing a rate already applied to a transaction. |
| 33, 36 | `exchange_via_app`, `fiat_currency_support` | Retain | These explain exchange instructions and supported currencies. |
| 50 | `receiving_money` | Retain | The examples ask how salary or another person can send funds; they do not report a missing transfer. |
| 54, 60 | `supported_cards_and_currencies`, `top_up_limits` | Retain | These explain supported funding instruments, currencies, and limits rather than a failed or unresolved top-up. |
| 67 | `transfer_timing` | Retain | This asks for expected transfer duration, unlike `pending_transfer` or `transfer_not_received_by_recipient`. |
| 73 | `visa_or_mastercard` | Retain | This explains the payment network offered for a card product. |

## Evidence Used

- Local BANKING77 training examples for all fifteen reviewed labels
- [Original BANKING77 dataset description](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)

This review is based on task semantics. External regulation may establish disclosure
requirements for charges and exchange rates, but it does not turn every informational
question into a high-consequence routing event.

## Important Overlaps

- `card_delivery_estimate` versus `card_arrival`
- `exchange_rate` versus transaction-specific wrong-exchange-rate intents
- `receiving_money` versus `transfer_into_account`
- `supported_cards_and_currencies` versus card top-up instructions
- `transfer_timing` versus `pending_transfer`

These pairs should be highlighted in the later confusion and accepted-error analysis.
Their semantic similarity does not justify assigning them equal consequences when their
required actions differ.

## Remaining Gate

All fifteen entries are `reviewed`, not `frozen`. Domain review should confirm that a
generic information route is a realistic abstraction and whether any institution would
send product-specific information to separate specialist queues.
