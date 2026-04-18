# MASS-AI Web Pilot Checklist

## Preflight

- Install requirements with `INSTALL_REQUIREMENTS.bat`.
- Run `RUN_SMOKE_TESTS.bat`.
- Launch the web app with `START_MASS_AI.bat`.

## Web Flow

1. Open the dashboard in the browser.
2. Confirm the app loads without import or dataset errors.
3. Navigate across the main tabs.
4. Confirm feature data and charts render.
5. Open at least one customer detail or drill-down flow.
6. Confirm the app still supports synthetic demo usage.

## Data Flow

1. Confirm processed demo files are readable from `project/data/processed`.
2. Confirm fallback raw-series generation still works when the raw sample is missing.
3. Confirm scoring visuals render after data load.

## Branch Clarity

1. Confirm `README.md` describes `main` as web-only.
2. Confirm no desktop screenshots or desktop launch steps remain in `main`.
3. Confirm `desktop-local` is the only documented path for the old desktop app.
