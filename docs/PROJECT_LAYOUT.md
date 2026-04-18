# MASS-AI Project Layout

`main` is organized around a web-first product story.

## Top Level

- `index.html`: public landing page
- `styles.css`: landing page styles
- `script.js`: landing page behavior
- `START_MASS_AI.bat`: primary web launch
- `START_MASS_AI_WEB.bat`: direct web launch alias
- `INSTALL_REQUIREMENTS.bat`: install the active web dependency set
- `RUN_SMOKE_TESTS.bat`: compile plus unit checks
- `docs/`: product and rollout notes
- `images/`: approved web visual intake area

## Project Folder

- `project/core/`: shared engine and helper modules
- `project/web/`: current Streamlit web runtime
- `project/data/`: demo and processed datasets
- `project/tests/`: unit tests
- `project/archive/`: archived research code

## Branch Contract

- `main`: web-facing branch
- `desktop-local`: old local desktop version

## Quick Usage

- Install:
  `INSTALL_REQUIREMENTS.bat`
- Run web app:
  `START_MASS_AI.bat`
- Run smoke checks:
  `RUN_SMOKE_TESTS.bat`
