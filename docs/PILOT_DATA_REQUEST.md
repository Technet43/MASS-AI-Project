# Pilot Data Request

Use this as the starting point when contacting a distribution company, municipality, or energy-sector partner.

## Purpose

MASS-AI needs a limited, privacy-safe dataset to validate whether smart-meter anomaly scores can improve inspection prioritization.

## Requested Data

Minimum useful sample:

- 5,000 to 50,000 meters or customer accounts.
- 6 to 24 months of daily consumption readings.
- Confirmed inspection outcome or theft/anomaly label where available.
- Region, tariff group, feeder, transformer, and meter type if available.
- Inspection date, tamper event date, or case status if available.

## Privacy-Safe Format

The partner should remove or hash:

- Customer name.
- Phone number.
- National ID or tax ID.
- Full address.
- Contract number if it can identify a person directly.

Stable anonymous IDs are enough for modeling.

## Evaluation Deliverables

The pilot report should include:

- Data quality summary.
- Model performance with operational thresholds.
- Top-risk inspection queue sample.
- False-positive and false-negative review.
- Recommended next pilot scope.

## Success Criteria

A pilot is useful if it can answer:

- Does the model rank confirmed theft/anomaly cases above normal meters?
- How many true cases appear in the top inspection queue?
- Which features are trusted by field teams?
- Which data fields are missing or unreliable?
- What false-positive rate is acceptable for weekly inspections?
