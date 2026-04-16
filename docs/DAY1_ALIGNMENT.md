# Day 1 Alignment (2026-04-16)

This document locks the Day 1 decisions for MASS-AI and serves as the source of truth for positioning, model narrative, and ingest contract.

## 1) Official Product Position

MASS-AI is positioned as a **utility pilot / proof-of-value decision-support platform**.

- It is not presented as full enterprise production deployment yet.
- Primary value: prioritize suspicious meters, explain risk drivers, and support analyst case workflow.
- Target outcome: pilot readiness with clear next-phase roadmap.

## 2) Official Model Storyline

Current runtime paths are:

| Path | File | Active Models | Purpose |
|---|---|---|---|
| Batch training + scoring | `project/mass_ai_engine.py` | Isolation Forest, XGBoost, Random Forest, Gradient Boosting, Stacking Ensemble | Main scoring path for dashboard/desktop workflows |
| Realtime ingest scoring | `project/realtime_ingest/inference.py` | Isolation Forest on 3 realtime features (`p_avg_1h`, `v_std_1h`, `i_peak_1h`) | Fast anomaly signal for stream/near-realtime flow |
| Research / legacy | `project/legacy_pipeline/*` | LSTM Autoencoder and historical explainability experiments | Reference and experimentation, not the primary runtime path |

Important clarification:
- The current `Stacking Ensemble` meta-learner in `mass_ai_engine.py` combines XGBoost + Random Forest + Gradient Boosting.
- LSTM content remains in the repo but is not the default production/pilot scoring runtime.

## 3) Canonical Data Schema Decision

Canonical ingest format is **long-format telemetry** with one event per row.

Required fields:
- `meter_id`
- `voltage`
- `current`
- `active_power`

Optional field:
- `timestamp` (UTC ISO-8601 recommended)

Detailed contract: [CANONICAL_TELEMETRY_SCHEMA.md](CANONICAL_TELEMETRY_SCHEMA.md)

Examples:
- `project/realtime_ingest/examples/canonical_telemetry_template.csv`
- `project/realtime_ingest/examples/mqtt_payload_example.json`

## 4) Reference Lock

The following files are locked as Day 1 reference set:

- This file: `docs/DAY1_ALIGNMENT.md`
- Schema contract: `docs/CANONICAL_TELEMETRY_SCHEMA.md`
- Sprint/task report: `PROJE_ICIN_YAPILACAKLAR_RAPORU.md`

If any doc conflicts with this set, this set wins until a new dated alignment update is published.

