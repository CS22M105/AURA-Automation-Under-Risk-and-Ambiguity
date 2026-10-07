# Identity-Compliance Workflow Review

**Review date:** October 7, 2026  
**Scope:** BANKING77 labels 68, 69, 70, 71, and 74  
**Status:** Evidence reviewed; not domain-approved or frozen

## Review Standard

This review distinguishes customer due diligence from ordinary use of the word
"verification." An intent receives the `compliance_sensitive` flag only when the
correct workflow performs identity or source-of-funds due diligence. Merely explaining
why verification exists, or verifying a card top-up, is not enough.

FATF Recommendation 10 supports risk-based customer identification and verification,
ongoing due diligence, and source-of-funds scrutiny where necessary. It does not imply
that every request about funds requires enhanced due diligence, and it does not provide
a numerical cost for misrouting.

## Decisions

| Label | Intent | Decision | Reason |
|---:|---|---|---|
| 68 | `unable_to_verify_identity` | Retain family, action, and flag | The user is blocked in a customer identity-verification process and needs that process resolved. |
| 69 | `verify_my_identity` | Retain family, action, and flag | The requested workflow performs customer identity verification. |
| 70 | `verify_source_of_funds` | Retain with caveat | The label includes source-of-funds verification, but several utterances only ask where money originated. The compliance route is retained for the labeled task, subject to domain confirmation. |
| 71 | `verify_top_up` | Move to `funding_top_up`; remove compliance flag | Dataset examples ask for a card top-up verification code. This is funding authentication or support, not customer due diligence. |
| 74 | `why_verify_identity` | Retain family; remove compliance flag | The appropriate route explains identity checks but does not itself perform verification. FATF remains explanatory evidence, not evidence of heightened routing consequence. |

## Evidence Used

- [FATF Recommendations, amended June 2026](https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html)
- Local BANKING77 training examples for labels 68, 69, 70, 71, and 74

The FATF source supports the existence and purpose of due-diligence workflows. The
dataset examples determine what each BANKING77 label actually requests. This avoids
assigning compliance sensitivity from a label name alone.

## Remaining Gate

All five entries are `reviewed`, not `frozen`. A supervisor or banking-domain reviewer
should confirm the treatment of `verify_source_of_funds`, because its examples mix
formal verification language with general questions about transaction origin.
