# MASS-AI Pilot Checklist

## Preflight
- Install requirements.
- Run `RUN_SMOKE_TESTS.bat`.
- Confirm the desktop app opens.
- Confirm the launcher opens.

## Analyst Flow
1. Open the desktop app.
2. Select a synthetic preset.
3. Run a synthetic dataset.
4. Confirm the `Operations` queue is populated.
5. Open one case.
6. Change status to `Escalated`.
7. Add a note.
8. Confirm the history timeline shows:
   - case creation
   - analysis sync
   - case update
   - note entry

## Data Ingestion Flow
1. Load a CSV or Excel file with reordered columns.
2. Confirm scoring completes.
3. Confirm missing-label datasets fall back safely.
4. Confirm the customer detail panel shows `Why flagged`.

## Export Flow
1. Export results CSV.
2. Export charts PNG.
3. Export executive brief HTML.
4. Export support bundle ZIP.

## Persistence Flow
1. Close the desktop app.
2. Reopen the desktop app.
3. Confirm the same case status and notes are still visible.

## Packaging Flow
1. Run `BUILD_DESKTOP_EXE.bat`.
2. Confirm `project\artifacts\dist\build_manifest.json` exists.
3. Confirm the built executable launches on Windows.
