# MASS-AI Sprint Progress

Running log of Day 1–7 execution against `PROJE_ICIN_YAPILACAKLAR_RAPORU.md`.

## Day 1 — Decisions locked (by Gemini, 2026-04-16)

Captured in `docs/DAY1_ALIGNMENT.md` and `docs/CANONICAL_TELEMETRY_SCHEMA.md`:

- Product position: "utility pilot / proof-of-value decision-support platform".
- Model set: batch = IF/XGB/RF/GB/Stacking; realtime = 3-feature IF; research = LSTM (legacy).
- Canonical ingest contract: long-format telemetry with `meter_id, voltage, current, active_power, [timestamp]`.

## Day 2 — Data flow unification (2026-04-16)

Implementation of the telemetry bridge between synthetic generation and realtime ingest.

**Changed**

- `project/mass_ai_engine.py` — `generate_synthetic()` gained `emit_telemetry`, `telemetry_start_ts`, `telemetry_freq`, `seed` kwargs; emits per-customer long-format rows; new `to_long_telemetry()` accessor.
- `project/synthetic/__init__.py` — new package.
- `project/synthetic/export_telemetry.py` — CLI that runs the generator once and writes feature-table, long-telemetry, and a JSON manifest (seed, preset, known theft cases).
- `project/realtime_ingest/simulate_ingest.py` — replay a telemetry CSV through `data_loader`, with dry-run fallback when psycopg2/Postgres is unavailable.
- `project/dashboard/app.py` — Telemetry tab gained a "Load demo CSV" button that works without Postgres by reading the bundled demo CSV.
- `project/tests/test_telemetry_export.py` — 3 new tests covering schema, `data_loader.validate()` round-trip, and CLI.

**Generated**

- `project/data/demo/demo_telemetry.csv` — 16,800 rows × 100 meters × 7 days, canonical schema.
- `project/data/demo/demo_feature_table.csv` — batch-training companion for the same customers.
- `project/data/demo/demo_manifest.json` — seed `20260416`, preset `Turkey Urban`, 15 known theft cases.

**Validation**

- `python -m unittest discover project/tests` → 16 tests OK.
- `python -m project.realtime_ingest.simulate_ingest --file project/data/demo/demo_telemetry.csv --dry-run --limit 3` → exits 0.
- Telemetry CSV passes `data_loader.validate()` with no missing columns.

## Day 3 — Decision-first overview + error UX (2026-04-16)

Dashboard is now ordered around decisions, not data dumps.

**Changed**

- `project/dashboard/ui_helpers.py` — new module with `friendly_error()` (one-line error + hint + technical expander) and `action_card()` (tone-colored decision card).
- `project/dashboard/app.py`
  - Overview restructured: Row 1 → 3 action cards (high-risk customers / hottest region / pending investigations). KPI row + donut + histogram collapsed into "Istatistik detayi" expander. Alerts section replaced with triage list (top 10 sorted, `Open`→customer detail jump).
  - Telemetry tab: `Load demo CSV` button via `_load_demo_telemetry_csv()` — works without Postgres.
  - `friendly_error()` applied at 4 error sites (evidence fetch, DB alerts, telemetry DB, demo CSV load, sample fallback).
  - Live alerts empty state: now points user to Telemetry tab CTA.

**Validation**

- `python -m unittest discover project/tests` → 16 tests OK.
- `python -c "import ast; ast.parse(open('project/dashboard/app.py').read())"` → parses cleanly.
