# MASS-AI Architecture (2026 Target State)

## Guiding Principle
**shared/core is the single source of truth.**

Everything else (dashboards, desktop apps, future APIs) are thin consumers of the engine.

## Current Recommended Structure

```
MASS-AI-Project/
├── shared/
│   ├── core/                    # ← ONLY PLACE WITH REAL ML + BUSINESS LOGIC
│   │   ├── mass_ai_engine.py    # MassAIEngine (6 models, stacking, explainability, synthetic generator)
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
│   └── ADR/                     # Architecture Decision Records
│
├── MASS_AI_LAUNCHER.py          # Primary way to launch apps
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
python MASS_AI_LAUNCHER.py

# Or directly
streamlit run new_web/dashboard/app.py
```

## Data Model & Global Standards Alignment

The project deliberately uses a richer feature set than most academic baselines (SGCC, CER Ireland, London Smart Meter).

See the full comparison and feature list in [docs/Feature_Catalog.md](docs/Feature_Catalog.md).

**Key points**:
- ~40 features vs typical 10–20 in literature
- Strong peer/network context (transformer & feeder level) — rarely present in public papers
- Domain-specific signals tailored to Turkish utility operations (seasonal flags, tamper events, outages)
- This richness is both a technical strength and an important credibility point for incubation applications.

## Status (as of latest work)
- Engine is the clear heart
- Dashboard migration in progress
- Testing and structure being professionalized to reach 9+ rating in all categories
- Feature catalog and global standards narrative documented for both technical reviewers and incubation juries

---

## Real Data Integration Layer (Added May 2026)

**Location**: `shared/core/real_data.py`

This module was created to directly address the #1 incubation risk: "only synthetic data validation".

Key components:
- `generate_realistic_sgcc_proxy()` — Produces statistically realistic SGCC-style data for controlled domain-shift testing.
- `extract_sgcc_style_features()` — Robust mapper that converts classic SGCC daily-column format into the engine's ~40 feature schema.
- `run_real_data_benchmark()` — Runs synthetic vs proxy comparison and returns clear gap metrics.
- `generate_real_data_validation_report()` — Produces investor/incubator-ready markdown reports.

**Current Status (27 May 2026)**:
- First controlled benchmark completed: Synthetic AUC ~0.99 → SGCC-proxy AUC ~0.91 (gap ~0.08-0.09).
- Automated report generation working.
- 5+ unit tests added (`shared/tests/test_real_data.py`).

This layer is the foundation for future real utility data integration (Turkish DSOs or public SGCC files).

