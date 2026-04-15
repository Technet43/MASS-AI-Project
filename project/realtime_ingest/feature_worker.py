"""
MASS-AI Feature Worker — Sprint 3
===================================
Runs as a standalone process alongside the Go gateway.

Every POLL_INTERVAL seconds it:
  1. Reads the last WINDOW_MINUTES of raw_telemetry per meter.
  2. Computes: p_avg_1h, v_std_1h, i_peak_1h.
  3. Writes a row to processed_features.
  4. Scores with AnomalyDetector → anomaly_score in [0, 1].
  5. If score > ALERT_THRESHOLD → inserts into alerts (severity = KRITIK).
  6. Periodically re-trains the model on the full processed_features history.

Run:
    python feature_worker.py

Environment variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD  (same as gateway)
    POLL_INTERVAL   seconds between feature passes      (default 30)
    WINDOW_MINUTES  rolling window for features         (default 60)
    ALERT_THRESHOLD anomaly_score threshold for alerts  (default 0.80)
    RETRAIN_EVERY   how many passes before model re-fit (default 10)
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time

import numpy as np
import psycopg2
import psycopg2.extras

# Local inference module (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from inference import AnomalyDetector, extract_features, extract_batch_features

# ── Config ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("feature_worker")

POLL_INTERVAL   = int(os.getenv("POLL_INTERVAL",   "30"))
WINDOW_MINUTES  = int(os.getenv("WINDOW_MINUTES",  "60"))
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.80"))
RETRAIN_EVERY   = int(os.getenv("RETRAIN_EVERY",   "10"))

DB_CFG = dict(
    host=os.getenv("DB_HOST",     "localhost"),
    port=int(os.getenv("DB_PORT", "5433")),
    dbname=os.getenv("DB_NAME",   "mass_ai"),
    user=os.getenv("DB_USER",     "mass_ai"),
    password=os.getenv("DB_PASSWORD", "mass_ai_secret"),
)

# ── SQL ───────────────────────────────────────────────────────────────────────
SQL_FETCH_WINDOW = """
    SELECT meter_id,
           array_agg(voltage      ORDER BY received_at) AS voltages,
           array_agg(current      ORDER BY received_at) AS currents,
           array_agg(active_power ORDER BY received_at) AS powers,
           COUNT(*) AS n
    FROM   raw_telemetry
    WHERE  received_at >= NOW() - INTERVAL '%s minutes'
    GROUP  BY meter_id
    HAVING COUNT(*) >= 2
"""

SQL_INSERT_FEATURES = """
    INSERT INTO processed_features (meter_id, p_avg_1h, v_std_1h, i_peak_1h, sample_count)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING id
"""

SQL_INSERT_ALERT = """
    INSERT INTO alerts (meter_id, anomaly_score, severity, features_id)
    VALUES (%s, %s, %s, %s)
"""

SQL_FETCH_HISTORY = """
    SELECT p_avg_1h, v_std_1h, i_peak_1h
    FROM   processed_features
    WHERE  p_avg_1h IS NOT NULL
      AND  v_std_1h IS NOT NULL
      AND  i_peak_1h IS NOT NULL
    ORDER  BY computed_at DESC
    LIMIT  5000
"""


# ── Worker ────────────────────────────────────────────────────────────────────

class FeatureWorker:
    def __init__(self) -> None:
        self.running = True
        self.detector = AnomalyDetector()
        self._pass_count = 0
        self._conn: psycopg2.extensions.connection | None = None

    def stop(self, *_: object) -> None:
        log.info("Shutdown requested.")
        self.running = False

    # ── DB helpers ──────────────────────────────────────────────────────────

    def _connect(self) -> None:
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                self._conn = psycopg2.connect(**DB_CFG)
                self._conn.autocommit = False
                log.info("Connected to Postgres at %s:%s/%s",
                         DB_CFG["host"], DB_CFG["port"], DB_CFG["dbname"])
                return
            except psycopg2.OperationalError as exc:
                log.warning("Waiting for Postgres: %s", exc)
                time.sleep(3)
        raise RuntimeError("Could not connect to Postgres after 60 s")

    def _ensure_conn(self) -> None:
        try:
            if self._conn is None or self._conn.closed:
                self._connect()
            else:
                self._conn.reset()
        except Exception:
            self._connect()

    # ── Core pass ───────────────────────────────────────────────────────────

    def _run_pass(self) -> None:
        self._ensure_conn()
        assert self._conn is not None

        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SQL_FETCH_WINDOW, (WINDOW_MINUTES,))
            rows = cur.fetchall()

        if not rows:
            log.info("No telemetry data in the last %d minutes. Waiting…", WINDOW_MINUTES)
            return

        log.info("Processing %d meters via %s backend…",
                 len(rows), self.detector.backend)
        alerts_fired = 0

        # ── Rust-accelerated batch feature extraction ──────────────────────
        batch = [
            (list(row["voltages"]), list(row["currents"]), list(row["powers"]))
            for row in rows
        ]
        features = extract_batch_features(batch, window=WINDOW_MINUTES * 2)

        with self._conn.cursor() as cur:
            for row, (p_avg, v_std, i_peak) in zip(rows, features):
                meter_id     = row["meter_id"]
                sample_count = int(row["n"])

                # Persist features
                cur.execute(SQL_INSERT_FEATURES,
                            (meter_id, p_avg, v_std, i_peak, sample_count))
                feature_id = cur.fetchone()[0]

                # Score
                score = self.detector.score(p_avg, v_std, i_peak)

                if score > ALERT_THRESHOLD:
                    severity = "KRITIK" if score > 0.90 else "YUKSEK"
                    cur.execute(SQL_INSERT_ALERT,
                                (meter_id, score, severity, feature_id))
                    alerts_fired += 1
                    log.warning("ALERT  meter=%-15s score=%.3f severity=%s",
                                meter_id, score, severity)
                else:
                    log.debug("OK     meter=%-15s score=%.3f", meter_id, score)

        self._conn.commit()
        log.info("Pass complete — features=%d alerts=%d", len(rows), alerts_fired)

    # ── Model re-training ────────────────────────────────────────────────────

    def _maybe_retrain(self) -> None:
        if self._pass_count % RETRAIN_EVERY != 0:
            return
        self._ensure_conn()
        assert self._conn is not None

        with self._conn.cursor() as cur:
            cur.execute(SQL_FETCH_HISTORY)
            history = cur.fetchall()

        if not history:
            return

        X = np.array([[r[0], r[1], r[2]] for r in history], dtype=float)
        log.info("Re-training model on %d historical feature rows…", len(X))
        self.detector.fit(X)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._connect()
        log.info("Feature worker started — poll=%ds window=%dmin threshold=%.2f",
                 POLL_INTERVAL, WINDOW_MINUTES, ALERT_THRESHOLD)

        while self.running:
            try:
                self._maybe_retrain()
                self._run_pass()
            except Exception as exc:
                log.error("Pass failed: %s", exc, exc_info=True)
                # Reset connection on error so next pass gets a fresh one.
                if self._conn and not self._conn.closed:
                    try:
                        self._conn.rollback()
                    except Exception:
                        pass

            self._pass_count += 1
            time.sleep(POLL_INTERVAL)

        if self._conn and not self._conn.closed:
            self._conn.close()
        log.info("Feature worker stopped.")


def main() -> int:
    worker = FeatureWorker()
    signal.signal(signal.SIGINT,  worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
