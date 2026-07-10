# Pilot Readiness and Production Path

## Current classification

MASS-AI is a pilot-ready prototype with a data-integration layer. It is not a production-ready deployment. The list below separates evidence already present in the repository from work that must precede any production discussion.

## Evidenced in the repository

- Core engine and adapters are organized under shared/core.
- A Streamlit dashboard is available under new_web/dashboard.
- Dockerfile, Docker Compose, and a GitHub Actions test workflow are present.
- The engine reports ROC-AUC, PR-AUC, precision at K, recall at K, and calibration-related metrics.
- Synthetic and project-generated SGCC-style proxy workflows exist.
- The current proxy result is explicitly not real-data or field validation.

## Product and application work

- Split the large dashboard module into smaller, testable units.
- Add authentication suitable for a private partner pilot.
- Add structured logging and operational audit trails.
- Harden large-upload, malformed-file, and missing-label handling.
- Define an accessible and reviewable analyst workflow with a partner.

## Data and model operations

- Obtain a documented public benchmark or privacy-safe pilot dataset under appropriate permission.
- Version datasets and record provenance for every evaluation.
- Exclude or ablate unavailable fields instead of silently relying on compatibility values.
- Add drift checks for scores and features.
- Define model approval, retraining, rollback, and artifact-versioning policies.

## Infrastructure and governance

- Define a staging environment and deployment ownership.
- Define retention, backup, deletion, and access-control policies for pilot data.
- Introduce secrets management; never commit partner data or credentials.
- Document incident handling and review responsibilities.

## Business and pilot readiness

- Identify and qualify potential pilot partners without claiming conversations or commitments that have not occurred.
- Agree a one-page pilot scope, data request, success criteria, and governance model.
- Validate the commercial model with a partner; current pricing material is a pricing hypothesis, not a quote or revenue record.
- Collect operator feedback from an actual pilot review.

## Production claim rule

Do not describe MASS-AI as production-ready until a real-data pilot, human-review workflow, monitoring, privacy controls, access controls, and partner governance have been tested and documented.

See [real-data requirements](REAL_DATA_REQUIREMENTS.md) and [model validation](MODEL_VALIDATION.md).
