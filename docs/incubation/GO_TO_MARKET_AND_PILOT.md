# MASS-AI — Go-to-Market and Pilot Plan

## Current traction status

No customer interview, meeting, letter of intent, data-sharing agreement, pilot, or revenue is documented in this repository. This document is a plan for earning that evidence; it must not be presented as completed traction.

## Target-account criteria

Build an outreach list only from verified, current public information. Prioritize organizations that can:

- discuss smart-meter or non-technical-loss workflows;
- sponsor a limited discovery or data-quality conversation;
- involve the relevant data, privacy, and field-operations stakeholders; and
- assess a bounded pilot without requiring premature production commitments.

Do not assign loss rates, technology maturity, willingness to buy, or contact status to an organization without a source and confirmation.

## Discovery approach

1. Prepare a concise prototype demonstration and a one-page data request.
2. Ask for a discovery conversation about workflow, data availability, privacy, and inspection capacity.
3. Record only verified outcomes, permissions, and next steps.
4. Offer a scoped evaluation only after the partner agrees that the data and governance conditions are suitable.

### Outreach message template

Subject: Exploration of smart-meter inspection prioritization

Hello [name],

MASS-AI is a pilot-ready prototype for explainable smart-meter anomaly analysis and inspection prioritization. The current repository includes synthetic testing and a controlled project-generated SGCC-style proxy evaluation; it does not claim field validation.

We would value a short discovery conversation about your inspection workflow, available anonymized data, governance requirements, and whether a limited evaluation could be appropriate. We can share the prototype, validation boundaries, and a privacy-safe data request.

Kind regards,

[verified sender details]

## Pilot phases

### Phase 0 — Discovery and governance

- Identify operational owner, data owner, privacy/legal owner, and review process.
- Agree whether a pilot is appropriate and what data can be shared.
- Document data minimization, anonymization, retention, and access rules.

### Phase 1 — Data assessment and integration

- Profile data quality, coverage, labels, and feature availability.
- Map compatible fields through the integration layer.
- Identify any proxy-derived compatibility fields and exclude or ablate them for final evaluation.

### Phase 2 — Evaluation design

- Agree a leakage-resistant split, review capacity, and decision threshold.
- Produce a reproducible report with PR-AUC, precision at capacity, recall, calibration, and error analysis.
- Define how inspection outcomes will be recorded.

### Phase 3 — Operator review

- Review a bounded risk queue with the partner’s field or analyst team.
- Record confirmed, rejected, and inconclusive outcomes.
- Collect feedback on explanation usefulness and workflow fit.

### Phase 4 — Decision

- Review technical evidence, governance readiness, operational fit, and commercial viability.
- Decide whether to stop, refine, repeat, or expand the pilot.

## Commercial boundary

Any pilot fee, recurring fee, or result-based fee is a **pricing hypothesis** until validated with a partner. Do not claim expected revenue, savings, or a partner commitment before it exists.

## Near-term evidence goals

- verified discovery conversations;
- a documented data request and governance path;
- a reproducible public-data or partner-data evaluation; and
- operator-reviewed outcomes from a permitted pilot.

See the [one-pager](ONE_PAGER.md), [business model](BUSINESS_MODEL.md), and [pilot data request](../product/PILOT_DATA_REQUEST.md).
