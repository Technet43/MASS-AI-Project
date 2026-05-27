# Project Layout (DEPRECATED - Historical Only)

**WARNING (2026):** 
This entire `project/` directory is legacy and should not be used for new development.

**Current single source of truth:**
→ `../shared/core/mass_ai_engine.py` (MassAIEngine + full 6-model stack + Turkish synthetic generator)

**What used to live here:**
- Old duplicated engine implementations
- Legacy desktop code (now in `../old_desktop/`)
- Old tests and pipelines

**Recommendation:** 
Do not run code from this folder. Use the root launcher or `shared/` directly.

This folder is kept only for git history and reference during the ongoing de-duplication effort.

## Main Folders

- `core/`
  Shared engine, domain logic, metadata, preferences, persistence, and support helpers.
- `old_desktop/`
  First version of the app.
  Tkinter desktop UI, desktop-only UI kit, packaging files, and desktop requirements.
- `new_web/`
  Newer version of the app.
  Streamlit dashboard and web-oriented requirements.
- `data/`
  Shared datasets used by both versions.
- `tests/`
  Unit tests for the shared core.
- `archive/`
  Older research code removed from the main runtime layout.

## Entry Points

- Old desktop:
  `START_MASS_AI_DESKTOP.bat`
- Launcher:
  `START_MASS_AI.bat`
- New web directly:
  `START_MASS_AI_WEB.bat`

## Install

- Both versions:
  `python -m pip install -r project/requirements.txt`
- Desktop only:
  `python -m pip install -r project/old_desktop/requirements.txt`
- Web only:
  `python -m pip install -r project/new_web/requirements.txt`
