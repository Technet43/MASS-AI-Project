# MASS-AI Startup Readiness

**Audience:** Internal planning

**Purpose:** A single, evidence-based view of product, validation, incubation, and pilot readiness.

## Executive snapshot

MASS-AI has a working prototype, synthetic scenario tooling, a shared core engine, dashboard views, and a data-integration layer. Its appropriate current description is **pilot-ready prototype with an integration layer**.

The repository does not provide evidence of a public real-data benchmark, Turkish field validation, customer discovery, signed partner commitments, revenue, or operational loss reduction. Those are the principal readiness gaps.

## Readiness by workstream

| Workstream | Evidence currently present | Main gap before a stronger claim |
| --- | --- | --- |
| Product | Shared core engine, dashboard, case-review workflow, synthetic presets, and local container tooling | Partner workflow testing, authentication, logging, upload hardening, and deployment ownership |
| Validation | Synthetic evaluation and a controlled SGCC-style realistic proxy; input adapter for compatible daily-column data | Reproducible real-data benchmark, leakage controls, field availability analysis, and operator review |
| Data governance | Privacy-safe data request and validation requirements are documented | Partner-approved access, retention, deletion, security, and audit controls |
| Go-to-market | Discovery and pilot plan exists | Verified conversations, customer feedback, agreement, and permitted pilot data |
| Commercial model | Hybrid commercial structure is documented as a pricing hypothesis | Willingness-to-pay evidence, procurement constraints, and verified economics |
| Team and narrative | Incubation source materials and technical narrative exist | Verified team bios, relevant sector support, and a concise founder narrative |

## Product and technical readiness

### Present

- Core model and adapter code is organized under shared/core.
- A Streamlit dashboard and case-management surfaces are available.
- Synthetic data supports demos and controlled regression-style evaluation.
- The engine exposes ROC-AUC, PR-AUC, precision at K, recall at K, and calibration-related outputs.
- Docker, Docker Compose, and a test workflow are present.

### Open work

- Split the dashboard into smaller modules.
- Add private-pilot authentication, structured logging, and operational audit trails.
- Define model artifact, rollback, monitoring, and retraining policies.
- Test the workflow with real data-quality conditions and a partner review process.

## Data and validation readiness

### What the current results mean

The committed comparison between synthetic data and an SGCC-style realistic proxy measures a controlled domain shift. The proxy is generated data and is not a real SGCC dataset or field validation. The reported proxy result must not be used as proof of utility performance, savings, or fraud-detection effectiveness.

### Required next evidence

1. Document a real public dataset’s source, license, label definition, preprocessing, and split policy.
2. Disclose and exclude or ablate proxy-derived fields where real inputs lack them.
3. Report operational metrics at a partner-agreed inspection capacity.
4. Conduct a privacy-safe pilot with operator-reviewed outcomes before discussing production deployment.

See [real-data requirements](../product/REAL_DATA_REQUIREMENTS.md) and [model validation](../product/MODEL_VALIDATION.md).

## Incubation readiness

### Materials available

- [One-pager](../incubation/ONE_PAGER.md)
- [Pitch deck source](../incubation/PITCH_DECK_SOURCE.md)
- [Business model](../incubation/BUSINESS_MODEL.md)
- [Go-to-market and pilot plan](../incubation/GO_TO_MARKET_AND_PILOT.md)
- Product validation and pilot-data-request documents

### Submission gaps

- Finalize only verified founder, team, advisor, and contact information.
- Add cited sources before using market-size, loss-rate, rollout, or competitive claims.
- Keep proxy results labelled as generated-data evidence.
- State pricing and revenue only as pricing hypotheses until validated.

## Pilot readiness checklist

- [ ] Identify a partner owner for operations, data, and privacy.
- [ ] Agree a privacy-safe data request and retention path.
- [ ] Run a documented data-quality and feature-availability assessment.
- [ ] Agree split policy, inspection capacity, and review criteria.
- [ ] Review a bounded queue with operators and record outcomes.
- [ ] Decide on an expanded pilot only after evidence and governance are reviewed.

## Priority order

1. Validate the team narrative and submission facts.
2. Start documented customer discovery.
3. Obtain appropriately governed data for a reproducible evaluation.
4. Conduct operator review of the prototype workflow.
5. Validate pricing, integration scope, and deployment requirements with a partner.

## Claim guide

Use “pilot-ready prototype” or “integration layer” for the current state. Do not use unqualified production, real-SGCC-benchmark, field-validation, or customer-impact language unless the corresponding evidence is documented.
