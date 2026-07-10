# Pilot Data Request

Use this document as a starting point for an exploratory conversation with a distribution company, municipality, or energy-sector partner.

## Purpose

MASS-AI is a pilot-ready prototype with a data-integration layer. A limited, privacy-safe dataset is needed to assess whether its risk scores can help prioritize inspections. The repository does not claim a partner dataset, field validation, or recovery outcome today.

## Requested data

A useful initial sample may include:

- 5,000 to 50,000 meters or customer accounts;
- 6 to 24 months of daily consumption readings;
- confirmed inspection outcome or theft/anomaly label where available;
- region, tariff group, feeder, transformer, and meter type when available;
- inspection date, tamper-event date, or case status when available.

The final scope should be agreed with the partner’s legal, privacy, data-governance, and operational teams.

## Privacy-safe format

The partner should remove or hash:

- customer names;
- direct contact details;
- national ID or tax ID;
- full addresses; and
- contract numbers when they can directly identify a person.

Stable anonymous IDs are sufficient for modeling and evaluation.

## Evaluation deliverables

A pilot report should include:

- a data-quality and feature-availability summary;
- the split policy and validation-level statement;
- model performance at an agreed operational threshold;
- a bounded top-risk inspection queue for review;
- false-positive and false-negative analysis; and
- a jointly agreed recommendation for the next pilot scope.

## Questions the pilot should answer

- Do risk scores rank confirmed cases above normal meters under the agreed split?
- How many confirmed cases appear in the agreed inspection queue?
- Which inputs and explanations are trusted by field teams?
- Which data fields are missing, unreliable, or unsuitable for use?
- What false-positive rate is acceptable for the partner’s weekly inspection capacity?

See [real-data requirements](REAL_DATA_REQUIREMENTS.md) for the evidence and claim boundaries.
