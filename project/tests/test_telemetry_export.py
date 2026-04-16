import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd

from mass_ai_engine import MassAIEngine
from realtime_ingest.data_loader import REQUIRED_COLS, normalize_columns, validate
from synthetic.export_telemetry import run as export_run


class TelemetryEmissionTests(unittest.TestCase):
    def test_emit_telemetry_produces_canonical_schema(self):
        engine = MassAIEngine()
        features = engine.generate_synthetic(
            n_customers=4, n_days=2, emit_telemetry=True, seed=20260416
        )
        telemetry = engine.to_long_telemetry()

        self.assertEqual(len(features), 4)
        self.assertEqual(len(telemetry), 4 * 2 * 24)
        self.assertEqual(
            list(telemetry.columns),
            ["meter_id", "timestamp", "voltage", "current", "active_power"],
        )
        self.assertEqual(telemetry["meter_id"].nunique(), 4)
        self.assertTrue((telemetry["voltage"] >= 210).all())
        self.assertTrue((telemetry["voltage"] <= 250).all())
        self.assertTrue((telemetry["current"] >= 0).all())
        self.assertTrue((telemetry["active_power"] >= 0).all())

    def test_telemetry_passes_data_loader_validation(self):
        engine = MassAIEngine()
        engine.generate_synthetic(
            n_customers=3, n_days=3, emit_telemetry=True, seed=7
        )
        telemetry = engine.to_long_telemetry().copy()
        telemetry["timestamp"] = telemetry["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "telemetry.csv"
            telemetry.to_csv(path, index=False)
            loaded = pd.read_csv(path)

        loaded = normalize_columns(loaded)
        self.assertEqual(validate(loaded), [])
        self.assertTrue(REQUIRED_COLS.issubset(set(loaded.columns)))

    def test_export_cli_round_trip(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            tdir = Path(tmpdir)
            args = argparse.Namespace(
                customers=5,
                days=3,
                preset="Turkey Urban",
                start_ts="2026-04-09T00:00:00Z",
                seed=42,
                telemetry_output=tdir / "telemetry.csv",
                feature_output=tdir / "features.csv",
                manifest=tdir / "manifest.json",
            )
            manifest = export_run(args)

            self.assertTrue(args.telemetry_output.exists())
            self.assertTrue(args.feature_output.exists())
            self.assertTrue(args.manifest.exists())
            self.assertEqual(manifest["customers"], 5)
            self.assertEqual(manifest["days"], 3)

            loaded = normalize_columns(pd.read_csv(args.telemetry_output))
            self.assertEqual(validate(loaded), [])
            self.assertEqual(len(loaded), 5 * 3 * 24)


if __name__ == "__main__":
    unittest.main()
