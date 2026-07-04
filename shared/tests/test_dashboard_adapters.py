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
                "customer_count": 100,
                "best_model_performance": {
                    "auc": 0.91,
                    "pr_auc": 0.84,
                    "precision_at_k": 0.61,
                    "recall_at_k": 0.55,
                },
            }
        }
        from dashboard_adapters import extract_metrics_from_engine
        metrics = extract_metrics_from_engine(fake_result)
        self.assertTrue(metrics.get("engine_mode"))
        self.assertEqual(metrics["best_model"], "Stacking Ensemble")
        self.assertIn("best_model_performance", metrics)
        self.assertEqual(metrics["best_model_performance"]["pr_auc"], 0.84)

    def test_get_scored_data_and_metrics_combines_helpers(self):
        from dashboard_adapters import get_scored_data_and_metrics
        fake_result = {
            "scored": pd.DataFrame({
                "customer_id": [1],
                "risk_score": [75],
                "risk_category": ["high"]
            }),
            "best_model": "Random Forest",
            "overview": {
                "customer_count": 50,
                "best_model_performance": {"auc": 0.9, "pr_auc": 0.85},
            }
        }
        scored, metrics = get_scored_data_and_metrics(fake_result)
        self.assertFalse(scored.empty)
        self.assertIn("risk_level", scored.columns)
        self.assertTrue(metrics.get("engine_mode"))
        self.assertIn("best_model_performance", metrics)

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

    def test_build_simulation_customer_pool_selects_correctly(self):
        from dashboard_adapters import build_simulation_customer_pool

        fake_df = pd.DataFrame({
            "customer_id": [10, 20, 30, 40],
            "theft_probability": [0.2, 0.9, 0.1, 0.75],
            "profile": ["residential"] * 4
        })

        pool = build_simulation_customer_pool(fake_df, selected_customer_id=30, n_customers=3)

        self.assertEqual(len(pool), 3)
        self.assertIn(30, pool["customer_id"].values)  # selected must be included
        # Should prefer high risk for the others
        self.assertIn(20, pool["customer_id"].values)  # 0.9 risk should be picked

    def test_initialize_live_simulation_state_builds_buffers(self):
        from dashboard_adapters import initialize_live_simulation_state

        customers = pd.DataFrame({
            "customer_id": [1, 2],
            "predicted_theft": [0, 1],
            "profile": ["residential", "commercial"]
        })

        raw = pd.DataFrame({
            "customer_id": [1, 1, 1, 2, 2],
            "consumption_kw": [1.1, 1.2, 0.0, 3.5, 3.6],
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h").tolist() * 1  # simplified
        })

        state = initialize_live_simulation_state(customers, raw, sim_points=3)

        self.assertIn("customer_data", state)
        self.assertIn(1, state["customer_data"])
        self.assertIn(2, state["customer_data"])
        self.assertEqual(len(state["customer_data"][1]["values"]), 3)
        self.assertEqual(state["customer_data"][2]["label"], 1)


if __name__ == "__main__":
    unittest.main()
