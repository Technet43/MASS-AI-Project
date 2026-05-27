# Shared Core (Current Source of Truth)

This is the **recommended and actively maintained** Python core for MASS-AI.

## Current Recommended Structure (2026)

- `shared/core/`
  - `mass_ai_engine.py` — Main `MassAIEngine` (6 models, stacking, explainability, strong synthetic generator with Turkish regional presets)
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
