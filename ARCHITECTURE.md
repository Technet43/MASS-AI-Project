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

## Status (as of latest work)
- Engine is the clear heart
- Dashboard migration in progress
- Testing and structure being professionalized to reach 9+ rating in all categories
