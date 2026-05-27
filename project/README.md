# Project Layout (DEPRECATED - Moving to Legacy)

**WARNING (2026):** 
This directory is being phased out. Most of its content has already been moved to `../legacy/`.

**Current single source of truth:**
`../shared/core/` (especially `mass_ai_engine.py`)

**What remains here (temporary):**
- Old README and requirements (for reference only)
- `tests/` — deprecated (active tests are in `../shared/tests/`)

**Next step:** Remaining items will be moved to `legacy/` or deleted.

**Do not develop here.** Use `shared/core/` and the main launchers instead.

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
