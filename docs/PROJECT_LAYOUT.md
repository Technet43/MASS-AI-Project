# MASS-AI Project Layout

This workspace was reorganized to keep the shipping desktop app easy to find while moving legacy and generated files out of the way.

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

- `project/mass_ai_desktop.py`: main desktop app
- `project/mass_ai_engine.py`: scoring engine and synthetic presets
- `project/mass_ai_domain.py`: shared formatting and report helpers
- `project/ops_store.py`: persistent Ops Center SQLite store
- `project/ui_kit.py`: shared desktop UI system
- `project/app_prefs.py`: saved appearance preferences
- `project/app_metadata.py`: version/build metadata
- `project/support_bundle.py`: support ZIP export

## Organized Subfolders

- `project/dashboard/`: Streamlit dashboard
- `project/legacy_pipeline/`: older research and pipeline scripts
- `project/tests/`: unit and regression tests
- `project/data/processed/`: processed CSV outputs and demo datasets
- `project/data/maps/`: map-specific sample data
- `project/artifacts/`: build outputs and logs
- `project/packaging/`: PyInstaller spec and packaging helpers

## Quick Usage

- Desktop app: `START_MASS_AI_DESKTOP.bat`
- Launcher: `START_MASS_AI.bat`
- Dashboard: launcher button or `python -m streamlit run project/dashboard/app.py`
- Smoke tests: `RUN_SMOKE_TESTS.bat`
- Legacy experiments: `python project/legacy_pipeline/run_pipeline.py --quick`
