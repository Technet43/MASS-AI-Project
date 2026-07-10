# MASS-AI Real-Data Validation Summary

## Purpose

This is the concise, shareable statement of the project’s current validation evidence for incubator and technical-review audiences.

## Current evidence status

The repository currently contains:

- L0: synthetic-data evaluation for demos, regression tests, and controlled experiments;
- L1: a controlled project-generated SGCC-style proxy evaluation; and
- an integration layer that can map compatible daily-column input files into the current feature schema.

The repository does not contain a committed public SGCC dataset benchmark, Turkish distribution-company data, field inspections, customer interviews, or operational loss-reduction results.

## Controlled proxy result

The recorded May 2026 run compares a synthetic input with a project-generated SGCC-style proxy:

| Dataset | AUC | F1 | What it demonstrates |
| --- | ---: | ---: | --- |
| Synthetic Turkey Urban, 900 customers | 0.9994 | 0.9697 | In-distribution behavior on generated data |
| Project-generated SGCC-style proxy, 700 customers | 0.9119 | 0.8000 | Behavior under a different generated distribution |
| AUC difference | 0.0875 | — | A measured proxy domain-shift gap |

The proxy was designed as a realistic control, not as a substitute for real data. These values are not a real SGCC benchmark, field validation, customer result, or production-performance claim.

## What exists today

- shared/core/real_data.py contains the input adapter, proxy generator, and benchmark/report helpers.
- scripts/benchmark_real_vs_synthetic.py runs the controlled comparison or accepts a compatible CSV path.
- shared/tests/test_real_data.py covers the adapter and benchmark utilities.
- reports/real_data_validation_report.md records the controlled proxy run.

The implementation is an integration layer and a pilot-ready prototype capability. It still requires dataset-specific validation before it can support operational use.

## What a real benchmark must document

Before calling an evaluation a public real-data benchmark, document:

- dataset source, license, date range, row count, label definition, and preprocessing;
- train/test split policy and leakage controls;
- unavailable operational fields and every proxy-derived compatibility field;
- ROC-AUC, PR-AUC, precision at the planned inspection capacity, recall, calibration, and error analysis.

The feature adapter may fill compatibility fields when a source omits them. Those values are suitable for pipeline checks, not final performance or business claims.

## Next evidence required

1. A reproducible benchmark on a properly documented public dataset.
2. A privacy-safe pilot dataset supplied under appropriate permission.
3. Operator review of a bounded inspection queue and documented outcomes.
4. A joint decision on integration, monitoring, privacy, and access controls before any production discussion.

See [real-data requirements](REAL_DATA_REQUIREMENTS.md) for the validation levels and [model validation](MODEL_VALIDATION.md) for the metric and split policy.
