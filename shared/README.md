# Shared Core (Current Source of Truth)

This is the **recommended and actively maintained** Python core for MASS-AI.

## For Dashboard / UI Developers

Prefer using helpers from `shared/core/dashboard_adapters.py` instead of duplicating logic inside `new_web/dashboard/app.py`.

Example:
```python
from shared.core.dashboard_adapters import load_synthetic_data_via_engine
data = load_synthetic_data_via_engine(preset="Industrial Theft Sweep")
```

The long-term goal is to make `new_web/dashboard/app.py` a thin presentation layer.

## Current Recommended Structure (2026)

- `shared/core/`
  - `mass_ai_engine.py` — Main `MassAIEngine` (4 base models plus a stacking ensemble, 5 model outputs in total; explainability and a strong synthetic generator with Turkish regional presets)
  - `ops_store.py` — SQLite-backed Ops Center
  - Supporting modules (domain, metadata, prefs, support bundle)

- `shared/data/` — Sample / processed datasets
- `shared/tests/` — Tests for the core (run via launcher or unittest)

## How the UIs should use it

- Desktop analyst app (`old_desktop/mass_ai_desktop.py`) → already correctly imports from `shared/core`
- Streamlit dashboard (`new_web/dashboard/app.py`) → being migrated to use the engine (no more massive in-file duplication)
- Root launcher (`MASS_AI_LAUNCHER.py`) → orchestrates the above

## Goal

Eliminate duplication. The 2300+ line Streamlit file should become a thin UI over the shared engine.

Old duplicated logic in `project/`, `new_web/dashboard` (legacy sections), and `old_desktop` copies should be deprecated over time.
