# Card-Servicing Workflow Review

**Review date:** October 7, 2026
**Scope:** Sixteen BANKING77 intents initially assigned to `card_servicing`
**Status:** Dataset reviewed; not domain-approved or frozen

## Review Standard

This review separates card fulfilment, routine instrument management, and restoration
of a card feature. It uses local BANKING77 examples to determine operational meaning,
because two labels are not reliable descriptions of their utterances. No reviewed card
intent receives a consequence flag solely because it involves a payment instrument.

## Decisions

| Labels | Intents | Decision | Reason |
|---:|---|---|---|
| 0, 13, 21 | `activate_my_card`, `card_linking`, `change_pin` | Retain card management | These requests configure or update an existing payment instrument. |
| 2 | `apple_pay_or_google_pay` | Move to `funding_top_up` | The examples overwhelmingly ask about topping up through Apple Pay, Google Pay, or an Apple Watch. |
| 9 | `card_about_to_expire` | Retain card management | The request concerns renewal or replacement of an expiring card. |
| 11 | `card_arrival` | Retain fulfilment | The examples ask for delivery status or tracking of an already ordered card. |
| 14, 23, 72 | `card_not_working`, `contactless_not_working`, `virtual_card_not_working` | Retain restoration | These troubleshoot ordinary card or card-feature functionality. |
| 18 | `card_swallowed` | Retain restoration without security flag | An ATM-retained card needs recovery or replacement, but the label alone does not report compromise or unauthorized use. |
| 37, 39, 40, 43 | Disposable, spare, virtual, and physical-card ordering intents | Retain fulfilment | These request or locate a card product. |
| 38 | `get_physical_card` | Change to card management | Despite its label, the local examples consistently ask where to retrieve the PIN for an existing physical card. |
| 49 | `pin_blocked` | Retain restoration without security flag | The examples describe failed PIN attempts and unlocking, not credential theft or compromise. |

## Evidence Used

- Local BANKING77 training examples for all sixteen reviewed labels
- [Original BANKING77 dataset description](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data)

This batch is a task-semantics review rather than a regulatory one. None of these labels
asserts unauthorized payment, financial correction, or customer due diligence, so no
external source is used to manufacture a consequence flag.

## Dataset Limitations

- `get_physical_card` is an opaque label for PIN-retrieval examples and overlaps
  conceptually with `change_pin`.
- `card_not_working` contains some declined-payment wording and can overlap with
  `declined_card_payment`.
- `card_arrival` overlaps with the informational `card_delivery_estimate` intent.

These overlaps should be checked later in the DistilBERT confusion analysis. They do
not justify arbitrary costs between the labels.

## Remaining Gate

All sixteen reviewed entries remain non-frozen. A banking-domain reviewer should confirm
whether the fulfilment, management, and restoration actions correspond to realistic
queues and whether ATM-retained cards require an institution-specific security route.
