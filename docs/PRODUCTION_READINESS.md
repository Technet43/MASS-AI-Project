# Production Readiness Checklist

This checklist tracks the gap between the current prototype and a field-ready deployment.

## Application

- [x] Core engine separated under `shared/core/`.
- [x] Dashboard can be run through Streamlit.
- [x] Dockerfile and Compose workflow added for repeatable local deployment.
- [ ] Dashboard app should be split further; `new_web/dashboard/app.py` is still too large.
- [ ] Add authentication suitable for partner demos or private pilot use.
- [ ] Add structured logging instead of UI-only logs.
- [ ] Add error handling for large uploads, malformed CSV files, and missing labels.

## Data

- [x] Synthetic and SGCC-style proxy workflows exist.
- [ ] Public SGCC real-data benchmark should be run and committed as a report.
- [ ] Turkish DSO pilot dataset should be obtained under NDA or formal permission.
- [ ] Add dataset versioning and provenance metadata for every benchmark.
- [ ] Define privacy rules for customer IDs, location, and inspection labels.

## Model Operations

- [x] Multiple model families are available in the engine.
- [ ] Add PR-AUC, precision@K, recall@K, calibration, and threshold reports.
- [ ] Add drift checks for score distribution and feature distribution.
- [ ] Add retraining policy and approval flow.
- [ ] Store trained model artifacts outside Git with version metadata.

## Infrastructure

- [x] GitHub Actions test workflow exists.
- [x] Docker runtime is defined.
- [ ] Add staging deployment target for the Streamlit dashboard.
- [ ] Add backup and retention policy for uploaded pilot data.
- [ ] Add secrets management; never commit partner data or credentials.

## Business And Pilot

- [ ] Identify 3 target pilot partners.
- [ ] Prepare one-page pilot scope and NDA-ready data request.
- [ ] Define success metrics: recovered loss, inspection hit rate, false-positive tolerance, and time saved.
- [ ] Collect operator feedback from at least one demo session.
- [ ] Add team slide with technical, energy-sector, and business ownership.

## Production Claim Rule

Do not describe the project as production-ready until all of these are true:

- A real dataset benchmark exists.
- A pilot partner has reviewed the workflow.
- Monitoring, privacy, and access-control policies are documented.
- The model has been evaluated with operational thresholds, not only AUC/F1.
