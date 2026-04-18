# Project Layout

This branch is organized for the web path only.

## Main Folders

- `core/`
  Shared domain, engine, metadata, persistence, and helper modules.
- `web/`
  Current Streamlit web application and web-specific dependency file.
- `data/`
  Demo datasets and processed CSV assets used by the app.
- `tests/`
  Unit tests for shared runtime behavior.
- `archive/`
  Older research and experiment code removed from the active runtime path.

## Entry Points

- Primary launch:
  `START_MASS_AI.bat`
- Direct web launch:
  `START_MASS_AI_WEB.bat`
- Smoke checks:
  `RUN_SMOKE_TESTS.bat`

## Install

- Web runtime:
  `python -m pip install -r project/requirements.txt`
- Direct web requirements:
  `python -m pip install -r project/web/requirements.txt`

## Desktop History

The old local desktop app is no longer part of `main`. Use the `desktop-local` branch if you need that version.
