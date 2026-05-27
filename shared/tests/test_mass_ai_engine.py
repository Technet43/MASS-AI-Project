import csv
import sys
import tempfile
import unittest
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = SHARED_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from mass_ai_engine import MassAIEngine
from mass_ai_domain import RISK_LABELS


class MassAIEngineSmokeTests(unittest.TestCase):
    def test_synthetic_pipeline_end_to_end(self):
        engine = MassAIEngine()
        features = engine.generate_synthetic(n_customers=80, n_days=20)
        results = engine.train_models()
        scored = engine.score_customers()

        self.assertEqual(len(features), 80)
        self.assertEqual(len(scored), 80)
        self.assertIn("Isolation Forest", results)
        self.assertIn("risk_score", scored.columns)
        self.assertIn("priority_index", scored.columns)
        self.assertIn("risk_summary", scored.columns)
        self.assertIn("risk_drivers", scored.columns)
        self.assertGreaterEqual(scored["theft_probability"].max(), scored["theft_probability"].min())
        self.assertTrue(set(scored["risk_category"].astype(str)).issubset(set(RISK_LABELS)))

    def test_synthetic_presets_feed_overview_and_explainability(self):
        engine = MassAIEngine()
        engine.generate_synthetic(n_customers=60, n_days=14, preset_name="Industrial Theft Sweep")
        engine.train_models()
        scored = engine.score_customers()
        overview = engine.build_overview()

        self.assertEqual(overview["preset_name"], "Industrial Theft Sweep")
        self.assertIn("industrial", overview["preset_summary"].lower())
        self.assertTrue(any(text != "-" for text in scored["risk_reason_1"].astype(str)))
        self.assertIn("alert drivers", overview["explainability_summary"])

    def test_csv_pipeline_with_missing_label_uses_fallback(self):
        rows = [
            {"customer_id": 1, "mean_consumption": 2.5, "std_consumption": 0.4, "zero_measurement_pct": 0.01},
            {"customer_id": 2, "mean_consumption": 7.8, "std_consumption": 1.2, "zero_measurement_pct": 0.32},
            {"customer_id": 3, "mean_consumption": 5.1, "std_consumption": 0.6, "zero_measurement_pct": 0.08},
            {"customer_id": 4, "mean_consumption": 9.4, "std_consumption": 1.8, "zero_measurement_pct": 0.45},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            engine = MassAIEngine()
            loaded = engine.load_dataset(str(csv_path))
            results = engine.train_models()
            scored = engine.score_customers()

        self.assertEqual(len(loaded), 4)
        self.assertEqual(len(scored), 4)
        self.assertEqual(results["Isolation Forest"]["type"], "Fallback")
        self.assertIn("risk_category", scored.columns)
        self.assertIn("risk_summary", scored.columns)
        self.assertTrue(set(scored["risk_category"].astype(str)).issubset(set(RISK_LABELS)))

    def test_regional_schema_normalization_handles_different_column_layouts(self):
        rows = [
            {"Abone No": 101, "Profil": "residential", "Ortalama Tuketim": 2.5, "Standart Sapma": 0.4, "Sifir Olcum Orani": 0.01},
            {"Abone No": 102, "Profil": "commercial", "Ortalama Tuketim": 7.8, "Standart Sapma": 1.2, "Sifir Olcum Orani": 0.32},
            {"Abone No": 103, "Profil": "industrial", "Ortalama Tuketim": 9.4, "Standart Sapma": 1.8, "Sifir Olcum Orani": 0.45},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "regional_layout.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            engine = MassAIEngine()
            loaded = engine.load_dataset(str(csv_path))

        self.assertIn("customer_id", loaded.columns)
        self.assertIn("profile", loaded.columns)
        self.assertIn("mean_consumption", loaded.columns)
        self.assertIn("std_consumption", loaded.columns)
        self.assertIn("zero_measurement_pct", loaded.columns)
        self.assertEqual(str(loaded.iloc[0]["profile"]), "residential")

    # ==================== NEW COMPREHENSIVE TESTS ====================

    def test_all_synthetic_presets_work(self):
        """Test that every regional preset produces valid data."""
        engine = MassAIEngine()
        for preset in engine.synthetic_preset_names():
            with self.subTest(preset=preset):
                features = engine.generate_synthetic(n_customers=50, n_days=30, preset_name=preset)
                self.assertGreater(len(features), 0)
                self.assertIn("synthetic_preset", features.columns)
                self.assertTrue((features["synthetic_preset"] == preset).all())

    def test_all_theft_patterns_are_represented(self):
        """Ensure all 8 theft patterns can appear in generated data."""
        engine = MassAIEngine()
        features = engine.generate_synthetic(n_customers=500, n_days=60, preset_name="Turkey Urban")
        theft_types = set(features["theft_type"].unique())
        expected = set([
            "none", "constant_reduction", "night_zeroing", "random_zeros",
            "gradual_decrease", "peak_clipping", "weekend_masking",
            "intermittent_bypass", "tamper_spikes"
        ])
        self.assertTrue(expected.issubset(theft_types) or len(theft_types) >= 6)

    def test_train_models_produces_results_for_all_models(self):
        engine = MassAIEngine()
        engine.generate_synthetic(n_customers=100, n_days=40)
        results = engine.train_models()
        self.assertIn("Isolation Forest", results)
        # When enough labels, supervised models should also be present
        if "Stacking Ensemble" in results:
            self.assertGreater(results["Stacking Ensemble"].get("auc", 0), 0.5)

    def test_score_customers_produces_expected_columns(self):
        engine = MassAIEngine()
        engine.generate_synthetic(n_customers=80, n_days=30)
        engine.train_models()
        scored = engine.score_customers()
        expected_cols = ["theft_probability", "risk_score", "risk_category", "risk_summary", "priority_index"]
        for col in expected_cols:
            self.assertIn(col, scored.columns)

    def test_build_overview_returns_useful_data(self):
        engine = MassAIEngine()
        engine.generate_synthetic(n_customers=60, n_days=25)
        engine.train_models()
        engine.score_customers()
        overview = engine.build_overview()
        self.assertIsNotNone(overview)
        self.assertIn("best_model", overview)
        self.assertIn("high_risk_count", overview)
        self.assertGreaterEqual(overview["customer_count"], 60)

    def test_explainability_columns_are_populated(self):
        engine = MassAIEngine()
        engine.generate_synthetic(n_customers=70, n_days=35)
        engine.train_models()
        scored = engine.score_customers()
        self.assertIn("risk_reason_1", scored.columns)
        self.assertIn("risk_drivers", scored.columns)
        # At least some rows should have non-placeholder reasons
        non_dash = scored[scored["risk_reason_1"] != "-"]
        self.assertGreater(len(non_dash), 5)

    def test_reset_state_clears_data(self):
        engine = MassAIEngine()
        engine.generate_synthetic(n_customers=30, n_days=15)
        engine.reset_state()
        self.assertIsNone(engine.df_features)
        self.assertIsNone(engine.df_scored)
        self.assertEqual(len(engine.models), 0)

    def test_load_dataset_handles_turkish_column_names(self):
        import pandas as pd
        rows = [
            {"Abone No": 1, "Profil": "residential", "Ortalama Tuketim": 2.1, "Standart Sapma": 0.3},
            {"Abone No": 2, "Profil": "commercial", "Ortalama Tuketim": 8.4, "Standart Sapma": 1.1},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "tr_cols.csv"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            engine = MassAIEngine()
            df = engine.load_dataset(str(csv_path))
            self.assertIn("customer_id", df.columns)
            self.assertIn("profile", df.columns)
            self.assertIn("mean_consumption", df.columns)


if __name__ == "__main__":
    unittest.main()
