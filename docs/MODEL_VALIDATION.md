# Model Validation Plan

MASS-AI should be evaluated as an inspection-prioritization system, not only as a high-AUC classifier.

## Current Known Limits

- Synthetic AUC near 1.0 is not enough evidence for field performance.
- SGCC-style proxy data measures domain shift but is still generated data.
- Some features in real-data adapters are proxy-filled when the source data lacks operational fields.
- The current default tests focus on core behavior, not full dashboard or large-file integration.

## Required Metrics

Report these metrics for every benchmark:

| Metric | Why it matters |
|---|---|
| ROC-AUC | General ranking quality. |
| PR-AUC | More informative when theft labels are rare. |
| Precision@K | Matches limited field inspection capacity. |
| Recall@K | Shows how many true theft cases the inspection queue catches. |
| F1 | Useful summary, but not enough alone. |
| Confusion matrix | Makes false positives and false negatives visible. |
| Calibration curve | Shows whether risk scores can be interpreted as probabilities. |

## Split Rules

Use stricter splits before making strong claims:

- Customer-level split for static datasets.
- Time-based split when timestamps are available.
- Transformer or feeder group split when network hierarchy exists.
- No customer should appear in both train and test for final metrics.

## Baselines

At minimum, compare against:

- Logistic regression.
- Random forest.
- XGBoost or gradient boosting.
- Simple rules such as high zero-reading percentage and sharp consumption drop.

The model is only convincing if it beats simple rules under the same split.

## Threshold Policy

Operational deployment needs a threshold tied to field capacity.

Example:

- If a region can inspect 200 meters per week, report precision and recall for the top 200 risk scores.
- If false positives are costly, optimize for precision@K instead of raw F1.
- If safety or revenue leakage risk is high, report recall at a minimum precision target.

## Regression Gates

Suggested CI gates:

- Unit tests must pass.
- Coverage should not decrease below the existing threshold.
- Benchmark scripts must run on small synthetic/proxy samples.
- Any change to feature generation should include at least one test that checks schema stability.
