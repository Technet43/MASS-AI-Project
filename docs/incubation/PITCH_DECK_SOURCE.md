# MASS-AI — Pitch Deck Source

This source text is designed for an incubation presentation. It intentionally separates demonstrated prototype capabilities from assumptions, plans, and pricing hypotheses.

## Slide 1 — Cover

**MASS-AI**

Explainable smart-meter anomaly analysis for inspection prioritization

Subtitle: A pilot-ready prototype for utility decision support

Presenter and contact: add verified founder and contact details before submission

## Slide 2 — Problem

Utilities need to decide which suspicious meters to inspect first. Raw consumption anomalies can be caused by legitimate behavior, data quality issues, or operational events, so a simple alert list can create costly false positives.

The problem statement should be supported by cited, current sector evidence if market-size or loss figures are shown. Do not present uncited national percentages, monetary losses, or rollout figures as project evidence.

## Slide 3 — Solution

MASS-AI converts smart-meter inputs into a prioritized review queue with explanation summaries. The prototype combines:

- statistical and temporal consumption signals;
- contextual and peer features when those inputs are available;
- multiple model families and a stacking ensemble; and
- a case-review workflow for analyst feedback.

The intended role is decision support for inspection prioritization, not an autonomous fraud verdict.

## Slide 4 — Technical approach

The current feature schema includes consumption statistics, temporal patterns, anomaly signals, peer/network context, and operational fields. Availability depends on the source data.

For daily-column datasets, the integration layer can map compatible inputs into the existing schema. Some fields are compatibility values when the source lacks hourly, network, or operational metadata. These values must be disclosed and cannot support final academic or business claims.

## Slide 5 — Prototype demonstration

Show only screens that can be demonstrated from the current repository:

- dashboard risk-score views;
- prioritized case queue;
- case detail and explanation summary;
- synthetic scenario controls; and
- available export or review screens.

Describe the product as a pilot-ready prototype. Do not describe it as production-ready.

## Slide 6 — Validation status

The default committed evaluation is a controlled comparison between synthetic data and an SGCC-style realistic proxy:

| Evaluation input | AUC | F1 |
| --- | ---: | ---: |
| Synthetic Turkey Urban, 900 customers | 0.9994 | 0.9697 |
| SGCC-style realistic proxy, 700 customers | 0.9119 | 0.8000 |
| AUC difference | 0.0875 | — |

Suggested speaking note: “The proxy is generated data. It helps us rehearse a domain shift, but it is not a real SGCC benchmark or field validation. We need a documented public dataset and a partner pilot before making operational claims.”

## Slide 7 — Pilot design

Propose a scoped, privacy-safe pilot:

1. Agree data access, anonymization, and governance.
2. Assess feature availability and the validity of the input mapping.
3. Evaluate a jointly agreed inspection queue.
4. Review error cases and explanations with operators.
5. Decide on expansion only after the evidence is reviewed.

The pilot must define its own review capacity, success criteria, and data-retention rules with the partner.

## Slide 8 — Commercial model

The commercial model is a **pricing hypothesis**, not an accepted price list or revenue forecast:

- possible pilot or onboarding scope;
- possible recurring platform access;
- a result-based component only if it is mutually agreed, measurable, and legally feasible.

Do not show revenue, savings, or price figures unless they are sourced, dated, and clearly labelled as assumptions.

## Slide 9 — Go-to-market status

The repository does not document customer interviews, signed pilots, letters of intent, or partner commitments. Present the next milestone as outreach and discovery, not achieved traction.

Target-account selection should be based on verified public information and should be reviewed before external use.

## Slide 10 — Team

Use only verified bios, roles, relevant experience, and advisor relationships. If a role or advisor has not been confirmed, leave it out rather than using a placeholder as evidence.

## Slide 11 — Incubation ask

Ask for:

- energy-sector and regulatory mentorship;
- guidance on data governance and pilot contracting;
- introductions that may support a pilot discovery conversation; and
- feedback on positioning, team composition, and commercial validation.

## Slide 12 — Close and next evidence

MASS-AI has a working prototype and a transparent validation plan. The next evidence to earn is a reproducible public-data evaluation, a privacy-safe pilot dataset, and operator-reviewed outcomes.

Supporting documents:

- [One-pager](ONE_PAGER.md)
- [Business model](BUSINESS_MODEL.md)
- [Go-to-market and pilot plan](GO_TO_MARKET_AND_PILOT.md)
- [Product validation summary](../product/REAL_DATA_VALIDATION_SUMMARY.md)
