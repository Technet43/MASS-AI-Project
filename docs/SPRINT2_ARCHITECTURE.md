# MASS-AI Sprint 2 Architecture Notes

## Scope
- Synthetic dataset presets for demo realism
- Row-level explainability for scored customers
- Executive brief enrichment with preset context and risk drivers

## Synthetic Presets
- `Turkey Urban`
  - Metro-heavy customer mix
  - Strong night tamper behavior
  - Mixed residential and mixed-use feeders
- `Industrial Theft Sweep`
  - Industrial and commercial concentration
  - Larger contract demand and higher exposure
  - More bypass and peak clipping patterns
- `Mixed Retail Anomalies`
  - Retail corridor behavior
  - Weekend masking and intermittent zero patterns
  - More time-of-use contracts
- `Rural Meter Drift`
  - Rural feeder distribution
  - Aging meter drift and outage masking
  - Slower low-visibility anomalies

## Explainability Flow
1. Generate or load features.
2. Train the model stack.
3. Score each customer.
4. Build row-level drivers from:
   - theft pattern
   - meter health
   - exposure pressure
   - robust z-score deviations on anomaly features
5. Persist the scored dataframe for UI, charts, and report export.

## Main Outputs
- `risk_reason_1`
- `risk_reason_2`
- `risk_reason_3`
- `risk_drivers`
- `risk_summary`

## UI Wiring
- Sidebar synthetic preset selector feeds `MassAIEngine.generate_synthetic(..., preset_name=...)`
- Customer detail appends `Why flagged`
- Ops detail appends `Why flagged` when the current run contains the selected customer
- Executive brief includes:
  - preset name
  - preset context
  - explainability snapshot
  - selected case rationale

## Next Cleanup
- Split tab controllers out of `mass_ai_desktop.py`
- Remove remaining dead helper code now shadowed by extracted modules
- Add preset-specific screenshots or sample CSV fixtures for demo QA
