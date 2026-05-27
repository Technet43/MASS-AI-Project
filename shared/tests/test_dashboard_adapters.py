import unittest
from pathlib import Path
import sys
import pandas as pd

# Make shared/core importable (same pattern as other tests)
SHARED_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = SHARED_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from dashboard_adapters import (
    get_engine,
    load_synthetic_data_via_engine,
    prepare_for_simulation,
)


class TestDashboardAdapters(unittest.TestCase):

    def test_get_engine_returns_engine_or_none(self):
        engine = get_engine()
        # It should either return a working engine or None (if dependencies missing)
        if engine is not None:
            self.assertTrue(hasattr(engine, "generate_synthetic"))

    def test_load_synthetic_data_via_engine_returns_valid_structure(self):
        result = load_synthetic_data_via_engine(n_customers=40, n_days=20)
        
        # Either we get data or a clear error
        if "error" in result:
            self.assertIsInstance(result["error"], str)
        else:
            self.assertIn("scored", result)
            self.assertIn("overview", result)
            self.assertIn("best_model", result)
            scored = result["scored"]
            self.assertGreater(len(scored), 0)
            self.assertIn("theft_probability", scored.columns)

    def test_prepare_for_simulation_normalizes_columns(self):
        # Simulate engine output that might have different column names
        fake_engine_output = pd.DataFrame({
            "customer_id": [1, 2],
            "risk_score": [85, 30],
            "risk_category": ["high", "low"],
        })

        result = prepare_for_simulation(fake_engine_output)

        self.assertIn("theft_probability", result.columns)
        self.assertIn("risk_level", result.columns)
        self.assertAlmostEqual(result.iloc[0]["theft_probability"], 0.85)

    def test_prepare_for_simulation_handles_empty(self):
        empty = pd.DataFrame()
        result = prepare_for_simulation(empty)
        self.assertTrue(result.empty)

    def test_load_synthetic_respects_preset(self):
        result = load_synthetic_data_via_engine(
            n_customers=30, 
            n_days=15, 
            preset="Rural Meter Drift"
        )
        if "scored" in result:
            # The preset should be reflected in the data
            self.assertIn("synthetic_preset", result["scored"].columns)

    def test_extract_metrics_from_engine_returns_expected_shape(self):
        # We can't easily create a full engine result here without running heavy code,
        # so we test the shape with a minimal fake dict
        fake_result = {
            "best_model": "Stacking Ensemble",
            "overview": {
                "high_risk_count": 12,
                "customer_count": 100
            }
        }
        from dashboard_adapters import extract_metrics_from_engine
        metrics = extract_metrics_from_engine(fake_result)
        self.assertTrue(metrics.get("engine_mode"))
        self.assertEqual(metrics["best_model"], "Stacking Ensemble")

    def test_get_scored_data_and_metrics_combines_helpers(self):
        from dashboard_adapters import get_scored_data_and_metrics
        fake_result = {
            "scored": pd.DataFrame({
                "customer_id": [1],
                "risk_score": [75],
                "risk_category": ["high"]
            }),
            "best_model": "Random Forest",
            "overview": {"customer_count": 50}
        }
        scored, metrics = get_scored_data_and_metrics(fake_result)
        self.assertFalse(scored.empty)
        self.assertIn("risk_level", scored.columns)
        self.assertTrue(metrics.get("engine_mode"))

    def test_run_engine_based_scoring_produces_expected_columns(self):
        from dashboard_adapters import run_engine_based_scoring
        fake_result = {
            "scored": pd.DataFrame({
                "customer_id": [1, 2],
                "risk_score": [85, 30],
                "risk_category": ["high", "low"]
            }),
            "best_model": "Stacking Ensemble",
            "overview": {"high_risk_count": 5}
        }
        scored, metrics = run_engine_based_scoring(fake_result)
        self.assertIn("theft_probability", scored.columns)
        self.assertIn("risk_level", scored.columns)
        self.assertTrue(metrics.get("engine_mode"))


if __name__ == "__main__":
    unittest.main()
