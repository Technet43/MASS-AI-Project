# Canonical Telemetry Schema (Day 1)

This schema defines the official ingest contract for telemetry into MASS-AI.

## Format

- Record format: **long telemetry**
- Granularity: one meter reading event per row/message
- Time zone: UTC recommended

## Required Fields

| Field | Type | Example | Rules |
|---|---|---|---|
| `meter_id` | string | `METER-00001` | Non-empty ID |
| `voltage` | float | `230.4` | Typical range check (soft): 150-300 |
| `current` | float | `4.2` | Non-negative |
| `active_power` | float | `965.0` | Non-negative |

## Optional Fields

| Field | Type | Example | Rules |
|---|---|---|---|
| `timestamp` | ISO-8601 datetime string | `2026-04-16T08:30:00Z` | If missing, ingest time may be used |

## Column Alias Normalization

The loader maps common aliases into canonical names:

- `sayac_id`, `meter`, `id` -> `meter_id`
- `gerilim`, `v` -> `voltage`
- `akim`, `i`, `a` -> `current`
- `guc`, `power`, `p` -> `active_power`
- `zaman`, `time`, `tarih` -> `timestamp`

## Validation Rules

- Missing required columns: reject input.
- Timestamp parsing failure: fallback to current UTC timestamp.
- Numeric coercion failures: reject row or file according to loader mode.
- Recommended operational checks:
  - duplicate `(meter_id, timestamp)` detection
  - impossible values (negative power/current, extreme voltage)
  - missing interval density monitoring

## CSV Example

```csv
meter_id,timestamp,voltage,current,active_power
METER-00001,2026-04-16T08:30:00Z,230.4,4.2,965.0
METER-00001,2026-04-16T08:31:00Z,230.1,4.3,989.0
METER-00002,2026-04-16T08:30:00Z,228.7,0.2,41.5
```

## MQTT Payload Example

```json
{
  "meter_id": "METER-00001",
  "timestamp": "2026-04-16T08:30:00Z",
  "voltage": 230.4,
  "current": 4.2,
  "active_power": 965.0
}
```

