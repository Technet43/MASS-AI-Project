# MASS-AI Project Layout

This workspace is organized around a shared core plus two app versions: the old desktop version and the new web version.

## Top Level

- `MASS_AI_LAUNCHER.py`: unified launcher
- `START_MASS_AI.bat`: launcher entry point
- `START_MASS_AI_DESKTOP.bat`: direct desktop entry point
- `BUILD_DESKTOP_EXE.bat`: packaging script
- `RUN_SMOKE_TESTS.bat`: compile + unit test check
- `docs/`: product and project notes
- `images/`: screenshots and visual assets
- `business_docs/`: business-facing files and archived papers

## Project Folder

- `project/core/`: shared engine, metadata, persistence, and helper modules
- `project/old_desktop/`: first Tkinter desktop version
- `project/new_web/`: newer Streamlit web version
- `project/archive/`: archived research code outside the main app flow

## Organized Subfolders

- `project/tests/`: unit and regression tests
- `project/data/processed/`: processed CSV outputs and demo datasets
- `project/data/maps/`: map-specific sample data
- `project/old_desktop/artifacts/`: desktop build outputs and logs
- `project/old_desktop/packaging/`: PyInstaller spec and packaging helpers
- `project/archive/legacy_pipeline/`: older research and experiment pipeline scripts

## Quick Usage

- Desktop app: `START_MASS_AI_DESKTOP.bat`
- Web dashboard: `START_MASS_AI_WEB.bat`
- Launcher: `START_MASS_AI.bat`
- Dashboard: launcher button or `python -m streamlit run project/new_web/dashboard/app.py`
- Smoke tests: `RUN_SMOKE_TESTS.bat`
- Legacy experiments: `python project/archive/legacy_pipeline/run_pipeline.py --quick`
