# Project Layout

This folder is organized around the two MASS-AI app versions plus a shared core.

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
