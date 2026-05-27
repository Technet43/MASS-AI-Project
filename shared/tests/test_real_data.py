"""
Tests for the real_data module (SGCC mapper + realistic proxy + benchmark).

These tests validate the "real data" story that is critical for incubation.
"""

import unittest
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Make shared/core importable
SHARED_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = SHARED_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from real_data import (
    generate_realistic_sgcc_proxy,
    extract_sgcc_style_features,
    run_real_data_benchmark,
    generate_real_data_validation_report,
)


class TestRealisticSGCCProxy(unittest.TestCase):

    def test_proxy_generates_correct_shape_and_labels(self):
        df = generate_realistic_sgcc_proxy(n_customers=200, theft_rate=0.08, random_state=42)
        self.assertEqual(len(df), 200)
        self.assertIn("label", df.columns)
        self.assertIn("mean_consumption", df.columns)
        self.assertIn("zero_measurement_pct", df.columns)
        # Theft rate should be roughly respected
        actual_theft = df["label"].mean()
        self.assertGreater(actual_theft, 0.04)
        self.assertLess(actual_theft, 0.15)

    def test_proxy_has_required_engine_features(self):
        df = generate_realistic_sgcc_proxy(n_customers=50, random_state=123)
        required = ["mean_consumption", "std_consumption", "zero_measurement_pct",
                    "cv_daily", "peer_consumption_ratio", "transformer_id"]
        for col in required:
            self.assertIn(col, df.columns)


class TestSGCCStyleFeatureExtraction(unittest.TestCase):

    def test_extract_from_minimal_fake_sgcc_format(self):
        # Simulate a realistic number of daily columns (SGCC often has hundreds)
        cols = {"CONS_NO": ["C001", "C002", "C003"], "FLAG": [0, 1, 0]}
        for i in range(40):  # enough columns to pass the min_daily_cols check
            cols[f"2014/1/{i+1}"] = [1.2 + i*0.01, 0.3 + i*0.005, 2.1 - i*0.01]

        fake_df = pd.DataFrame(cols)
        features = extract_sgcc_style_features(fake_df, label_col="FLAG", sample_size=None, min_daily_cols=30)
        self.assertGreater(len(features), 0)
        self.assertIn("label", features.columns)
        self.assertIn("mean_consumption", features.columns)
        self.assertEqual(features["label"].tolist(), [0, 1, 0])


class TestRealDataBenchmark(unittest.TestCase):

    def test_benchmark_runs_without_crashing_and_returns_expected_keys(self):
        # Keep it small and fast for CI
        result = run_real_data_benchmark(
            n_synthetic=120,
            n_real_proxy=80,
            preset="Turkey Urban",
            random_state=42,
        )

        self.assertIn("synthetic_in_dist_auc", result)
        self.assertIn("sgcc_proxy_in_dist_auc", result)
        self.assertIn("auc_gap", result)
        self.assertIn("best_model_synthetic", result)
        self.assertGreaterEqual(result["synthetic_in_dist_auc"], 0.5)
        self.assertGreaterEqual(result["sgcc_proxy_in_dist_auc"], 0.5)

    def test_validation_report_generator_produces_readable_markdown(self):
        fake_result = {
            "synthetic_in_dist_auc": 0.99,
            "sgcc_proxy_in_dist_auc": 0.91,
            "auc_gap": 0.08,
            "synthetic_in_dist_f1": 0.95,
            "sgcc_proxy_in_dist_f1": 0.78,
            "best_model_synthetic": "XGBoost",
            "best_model_on_proxy": "Random Forest",
            "n_synthetic": 300,
            "n_real_proxy": 200,
            "synthetic_preset": "Turkey Urban",
            "timestamp": "2026-05-27T15:40:00",
            "note": "Test note for report generation.",
        }

        # Should not raise and should return a path
        path = generate_real_data_validation_report(fake_result, output_path="/tmp/test_real_validation.md")
        self.assertTrue(Path(path).exists())

        content = Path(path).read_text(encoding="utf-8")
        self.assertIn("Gerçek Veri Validasyon Raporu", content)
        self.assertIn("0.99", content)
        self.assertIn("0.91", content)


class TestRealDataEdgeCases(unittest.TestCase):

    def test_proxy_respects_custom_theft_rate(self):
        for rate in [0.05, 0.12, 0.20]:
            df = generate_realistic_sgcc_proxy(n_customers=300, theft_rate=rate, random_state=99)
            actual = df["label"].mean()
            # Allow reasonable statistical tolerance
            self.assertGreaterEqual(actual, rate * 0.6)
            self.assertLessEqual(actual, rate * 1.6)

    def test_sgcc_extractor_handles_turkish_column_names(self):
        # Simulate possible real utility data with Turkish column names
        rows = 25
        data = {
            "ABONE_NO": [f"A{i:04d}" for i in range(rows)],
            "KACAK_FLAG": [0] * 20 + [1] * 5,
        }
        for day in range(35):
            data[f"GUN_{day}"] = [round(1.5 + (day % 7) * 0.1, 2) for _ in range(rows)]

        df = pd.DataFrame(data)
        features = extract_sgcc_style_features(df, label_col="KACAK_FLAG", min_daily_cols=20)
        self.assertEqual(len(features), rows)
        self.assertIn("label", features.columns)
        self.assertTrue((features["label"].iloc[-5:] == 1).all())

    def test_report_generator_handles_minimal_input(self):
        minimal = {
            "synthetic_in_dist_auc": 0.85,
            "sgcc_proxy_in_dist_auc": 0.79,
            "auc_gap": 0.06,
        }
        path = generate_real_data_validation_report(minimal, output_path="/tmp/minimal_report.md")
        self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()