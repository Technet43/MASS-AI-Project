# MASS-AI Feature Catalog

## Purpose

This catalog describes the feature schema used by the current MASS-AI prototype. It is intended to help technical reviewers understand the inputs available to the scoring workflow and the limits of the current validation evidence.

## Evidence boundary

The feature schema is exercised with synthetic data and a controlled SGCC-style realistic proxy. The proxy is generated data; it is not a public SGCC dataset, partner data, or field validation. Reported proxy metrics therefore describe a controlled domain-shift exercise only and must not be presented as utility performance.

## Feature groups

### Profile and identifiers

- customer_id
- profile
- region
- contract_type
- meter_health
- premise_density
- synthetic_preset
- transformer_id and feeder_id
- latitude and longitude
- tariff_group
- label and theft_type

### Consumption statistics

- mean_consumption, std_consumption, min_consumption, max_consumption, and median_consumption
- skewness, kurtosis, and iqr
- mean_daily_total, std_daily_total, cv_daily, and load_factor
- event_spike_ratio and billing_volatility

### Temporal patterns

- night_day_ratio
- weekend_weekday_ratio
- peak_offpeak_ratio
- morning_noon_ratio
- baseload_ratio
- peak_hour
- solar_relief_factor

### Anomaly and change signals

- zero_measurement_pct
- sudden_change_ratio
- trend_slope
- rolling_weekly_volatility
- anomaly_burst_ratio
- temperature_sensitivity

### Peer and network context

- transformer_loss_pct
- customer_share_of_loss
- transformer_peer_count
- peer_consumption_ratio
- peer_zero_pct_deviation
- peer_rank_in_transformer

### Operational and domain context

- meter_age_years
- contract_demand_kw
- outage_event_count
- tamper_event_count
- days_since_last_tamper
- tamper_density
- is_summer_peak
- is_ramadan_period
- is_bayram_week
- seasonal_anomaly_flag

## Interpretation guidance

The schema combines consumption, temporal, contextual, and operational fields. Whether a field is available and trustworthy depends on the source dataset:

- Synthetic presets can provide the complete schema for demos and regression tests.
- SGCC-style daily-column inputs provide consumption history and labels when available, but may lack hourly, network, and operational metadata.
- The current adapter creates compatibility values for missing fields so the prototype pipeline can run. Those proxy-derived fields must not support final academic, commercial, or operational claims.

Before comparing this schema with a published baseline, cite the selected reference dataset and use the same availability assumptions. Feature-count comparisons alone are not evidence of superior performance.

## Data-integration layer

The integration layer in shared/core/real_data.py provides:

- an adapter for compatible SGCC-style daily-column inputs;
- a controlled SGCC-style realistic proxy generator;
- benchmark and Markdown-report helpers for synthetic/proxy comparisons.

The adapter is an integration layer for evaluation work. It does not turn the default proxy result into real-world validation.

## Recorded controlled proxy run

The committed May 2026 report records the following controlled run:

| Evaluation input | AUC | F1 | Interpretation |
| --- | ---: | ---: | --- |
| Synthetic Turkey Urban, 900 customers | 0.9994 | 0.9697 | In-distribution synthetic result |
| SGCC-style realistic proxy, 700 customers | 0.9119 | 0.8000 | Generated domain-shift control |
| Difference | 0.0875 | — | Gap between the two controlled inputs |

This comparison is useful for testing how the prototype behaves under a different generated distribution. It is not a real SGCC benchmark, a field-validation result, or evidence of loss reduction.

Run the controlled comparison with:

    python scripts/benchmark_real_vs_synthetic.py --synthetic-n 1000 --sample 750

Run a compatible, properly licensed real input only when its provenance, preprocessing, missing fields, and split policy can be documented:

    python scripts/benchmark_real_vs_synthetic.py --real /path/to/input.csv --sample 800

## Next validation steps

1. Document the source, license, date range, labels, and preprocessing for a public real dataset.
2. Exclude or ablate proxy-derived fields when they are unavailable in that source.
3. Report PR-AUC, precision at the inspection capacity, recall, calibration, and error analysis under leakage-resistant splits.
4. Treat Turkish distribution-company data and operator review as separate pilot evidence, not as a consequence of proxy results.

See [real-data requirements](REAL_DATA_REQUIREMENTS.md) and the [validation summary](REAL_DATA_VALIDATION_SUMMARY.md) for the claim boundaries.
