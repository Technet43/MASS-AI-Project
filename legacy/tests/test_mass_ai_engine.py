import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_DIR / "core"
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


if __name__ == "__main__":
    unittest.main()
