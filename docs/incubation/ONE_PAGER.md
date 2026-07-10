# MASS-AI — One-Pager

**Project:** MASS-AI

**Focus:** Explainable, inspection-prioritization support for smart-meter anomaly analysis

**Current status:** Pilot-ready prototype; not a production deployment

## The problem

Non-technical losses and suspicious meter behavior consume utility investigation capacity. Smart-meter data can help, but analysts need more than a raw anomaly score: they need a prioritized queue, understandable risk drivers, and a feedback loop from inspection outcomes.

## The solution

MASS-AI is a prototype decision-support workflow that:

- transforms meter data into statistical, temporal, peer, and operational features;
- combines Isolation Forest, XGBoost, Random Forest, Gradient Boosting, and a stacking ensemble;
- presents prioritized cases with explanation summaries; and
- supports case-review and pilot-measurement workflows.

The repository includes a Streamlit dashboard, a shared core engine, synthetic scenario generation, and an integration layer for compatible daily-column inputs.

## Validation evidence and limits

The current committed evidence is synthetic evaluation plus a controlled project-generated SGCC-style proxy. The proxy is generated data and is not a public SGCC dataset, Turkish utility data, or field validation.

| Evaluation input | AUC | F1 | Interpretation |
| --- | ---: | ---: | --- |
| Synthetic Turkey Urban, 900 customers | 0.9994 | 0.9697 | In-distribution synthetic result |
| Project-generated SGCC-style proxy, 700 customers | 0.9119 | 0.8000 | Generated domain-shift control |
| AUC difference | 0.0875 | — | Proxy-domain-shift signal |

These results show a controlled gap between two generated inputs. They do not demonstrate utility performance, customer impact, or field accuracy. A properly documented public dataset and a privacy-safe partner pilot are still required.

## Pilot proposition

The next step is a bounded, privacy-safe evaluation with a distribution-company or municipality partner:

1. Agree the data, privacy, and review scope.
2. Map available fields through the integration layer and disclose missing or proxy-derived fields.
3. Evaluate a jointly agreed inspection queue under a leakage-resistant split.
4. Review false positives, false negatives, and explanation usefulness with operators.
5. Decide whether a broader pilot is justified.

## Commercial direction

The proposed commercial approach is a **pricing hypothesis**, not a quote, contract, pipeline, or revenue record: a pilot/onboarding component, a recurring platform component, and—only if mutually agreed and legally feasible—a result-based component. Pricing and success criteria must be validated with a partner.

## Incubation ask

MASS-AI is seeking:

- energy-sector and data-governance mentorship;
- introductions that can support an appropriately scoped pilot conversation;
- feedback on the team narrative, pilot design, and commercial hypothesis.

## Main gaps to close

- No public real-data benchmark or partner pilot outcome is committed.
- No customer interview, letter of intent, or signed partner agreement is claimed.
- Team, governance, monitoring, access control, and deployment plans require further definition.

For detail, see the [product validation summary](../product/REAL_DATA_VALIDATION_SUMMARY.md), [pilot data request](../product/PILOT_DATA_REQUEST.md), and [go-to-market plan](GO_TO_MARKET_AND_PILOT.md).
