# MASS-AI Architecture (2026 Target State)

## Guiding Principle
**shared/core is the single source of truth.**

Everything else (dashboards, desktop apps, future APIs) are thin consumers of the engine.

## Current Recommended Structure

```
MASS-AI-Project/
├── shared/
│   ├── core/                    # ← ONLY PLACE WITH REAL ML + BUSINESS LOGIC
│   │   ├── mass_ai_engine.py    # MassAIEngine (4 base models plus a stacking ensemble, 5 model outputs in total; explainability, synthetic generator)
│   │   ├── dashboard_adapters.py # Helpers for UIs
│   │   ├── ops_store.py
│   │   └── ...
│   ├── data/
│   ├── tests/
│   └── requirements.txt         # The only requirements file that matters
│
├── new_web/
│   ├── dashboard/               # Thin Streamlit UI (being migrated)
│   └── site/                    # Marketing / demo site
│
├── old_desktop/                 # Legacy (maintenance mode only)
│
├── legacy/                      # Archived old code (do not use)
│
├── docs/
│   ├── product/                 # Product and validation documentation
│   ├── incubation/              # Incubation-facing source material
│   ├── planning/                # Project planning
│   ├── research/                # Research drafts
│   └── archive/                 # Historical notes
│
├── apps/launcher/MASS_AI_LAUNCHER.py
└── README.md
```

## Key Decisions

- **Why shared/core?**
  - Avoid duplication of complex synthetic data logic and model training.
  - Easier to test, easier to improve.

- **Dashboard Strategy**
  - Long term: `new_web/dashboard/app.py` should be < 1200 lines and mostly call adapters + engine.

- **Legacy Policy**
  - Old code is moved to `legacy/` instead of deleted, for historical reference.

## How to Run (Recommended)

```bash
# Install once
pip install -r shared/requirements.txt

# Best experience
python apps/launcher/MASS_AI_LAUNCHER.py

# Or directly
streamlit run new_web/dashboard/app.py
```

## Data Model and Validation Scope

The current feature schema combines consumption statistics, temporal patterns, peer/network context, and operational fields. Input availability varies by dataset; daily-column public inputs often do not contain the full operational or network context.

See the feature definitions and claim boundaries in [docs/product/FEATURE_CATALOG.md](docs/product/FEATURE_CATALOG.md).

**Key points**:
- The prototype contains roughly 40 engineered features.
- Peer and network context is used only when source data provides meaningful hierarchy information.
- Domain-oriented fields require observed source data; compatibility values must not be treated as observations.
- Relative comparisons with published work require a cited, like-for-like evaluation.

## Status (as of latest work)
- shared/core is the primary engine layer.
- Dashboard migration and modularization remain in progress.
- Automated tests and local container tooling are present.
- The prototype still needs real-data validation, partner governance, and operational testing.

---

## Real Data Integration Layer (Added May 2026)

**Location**: `shared/core/real_data.py`

This module provides an integration layer for compatible daily-column inputs and a controlled project-generated SGCC-style proxy. The proxy is generated data and is not a public SGCC benchmark or field validation.

Key components:
- `generate_realistic_sgcc_proxy()` — Generates a project-generated SGCC-style proxy for controlled domain-shift testing.
- `extract_sgcc_style_features()` — Robust mapper that converts classic SGCC daily-column format into the engine's ~40 feature schema.
- `run_real_data_benchmark()` — Runs synthetic vs proxy comparison and returns clear gap metrics.
- `generate_real_data_validation_report()` — Produces investor/incubator-ready markdown reports.

**Current status (May 2026)**:
- A controlled synthetic-versus-proxy run recorded an AUC difference of roughly 0.08–0.09.
- Automated report generation and dedicated adapter tests are present.
- A documented public-data evaluation and partner pilot are still required.

This integration layer can support future real-data work, but its proxy result must not be presented as operational or field evidence.
