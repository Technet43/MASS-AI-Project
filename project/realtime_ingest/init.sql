-- MASS-AI Sprint 2 + 3: full schema
-- Runs automatically when Postgres container first starts.

-- ── Sprint 2: raw ingestion ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS raw_telemetry (
    id           SERIAL PRIMARY KEY,
    meter_id     VARCHAR(64)  NOT NULL,
    voltage      FLOAT        NOT NULL,
    current      FLOAT        NOT NULL,
    active_power FLOAT        NOT NULL,
    received_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_telemetry_received_at ON raw_telemetry (received_at DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_meter_id    ON raw_telemetry (meter_id);

-- ── Sprint 3: feature engineering output ─────────────────────────────────
CREATE TABLE IF NOT EXISTS processed_features (
    id           SERIAL PRIMARY KEY,
    meter_id     VARCHAR(64) NOT NULL,
    p_avg_1h     FLOAT,          -- 1-hour rolling mean active power (W)
    v_std_1h     FLOAT,          -- 1-hour voltage standard deviation
    i_peak_1h    FLOAT,          -- 1-hour current peak value
    sample_count INTEGER,        -- number of raw readings in the window
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_features_meter_id   ON processed_features (meter_id);
CREATE INDEX IF NOT EXISTS idx_features_computed_at ON processed_features (computed_at DESC);

-- ── Sprint 3: anomaly alerts ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    id            SERIAL PRIMARY KEY,
    meter_id      VARCHAR(64) NOT NULL,
    anomaly_score FLOAT       NOT NULL,
    severity      VARCHAR(20) NOT NULL DEFAULT 'KRITIK',
    features_id   INTEGER REFERENCES processed_features(id) ON DELETE SET NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged  BOOLEAN     NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_meter_id   ON alerts (meter_id);

