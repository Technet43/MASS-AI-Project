"""
Dashboard Adapters

Thin helpers that the Streamlit dashboard (and future UIs) can use
to interact with MassAIEngine without duplicating logic.

Goal: Keep the dashboard as a pure presentation layer.
"""

from typing import Any, Dict, Optional

import pandas as pd

# Robust import that works both when run as part of package and in tests
try:
    from .mass_ai_engine import MassAIEngine
except ImportError:
    import sys
    from pathlib import Path
    CORE_DIR = Path(__file__).resolve().parent
    if str(CORE_DIR) not in sys.path:
        sys.path.insert(0, str(CORE_DIR))
    from mass_ai_engine import MassAIEngine


def get_engine() -> Optional[MassAIEngine]:
    """Cached engine instance for dashboard use."""
    # In a real app you might use Streamlit session_state here.
    # For now we return a fresh one (callers can cache).
    try:
        return MassAIEngine()
    except Exception:
        return None


def load_synthetic_data_via_engine(
    n_customers: int = 1500,
    n_days: int = 120,
    preset: Optional[str] = None
) -> Dict[str, Any]:
    """Preferred way for the dashboard to get rich scored data."""
    engine = get_engine()
    if engine is None:
        return {"error": "Engine unavailable"}

    try:
        features = engine.generate_synthetic(
            n_customers=n_customers,
            n_days=n_days,
            preset_name=preset
        )
        engine.train_models()
        scored = engine.score_customers()
        overview = engine.build_overview() or {}

        return {
            "features": features,
            "scored": scored,
            "overview": overview,
            "best_model": overview.get("best_model", "Stacking Ensemble"),
            "engine": engine,
        }
    except Exception as e:
        return {"error": str(e)}


def prepare_for_simulation(engine_scored_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize engine output so the simulation code can consume it easily."""
    if engine_scored_df is None or engine_scored_df.empty:
        return pd.DataFrame()

    df = engine_scored_df.copy()

    # Ensure columns the old simulation code expects
    if "theft_probability" not in df.columns and "risk_score" in df.columns:
        df["theft_probability"] = df["risk_score"] / 100.0

    if "risk_level" not in df.columns and "risk_category" in df.columns:
        df["risk_level"] = df["risk_category"].astype(str)

    return df


def get_engine_performance_summary(engine_result: dict) -> dict:
    """Return a clean summary of model performance when using the engine."""
    if not engine_result or "overview" not in engine_result:
        return {"error": "No engine data"}

    overview = engine_result.get("overview", {})
    best_model = engine_result.get("best_model", "Unknown")

    return {
        "best_model": best_model,
        "high_risk_count": overview.get("high_risk_count", 0),
        "critical_count": overview.get("critical_count", 0),
        "total_customers": overview.get("customer_count", 0),
        "estimated_monthly_loss": overview.get("total_loss", 0),
        "average_risk": overview.get("average_probability", 0),
    }


def get_rich_scored_data_for_ui(engine_result: dict) -> pd.DataFrame:
    """Return the scored dataframe optimized for dashboard display."""
    if not engine_result or "scored" not in engine_result:
        return pd.DataFrame()

    df = engine_result["scored"].copy()

    # Standardize columns the UI likes
    if "risk_category" in df.columns:
        df["risk_level"] = df["risk_category"].astype(str)

    if "risk_score" in df.columns and "theft_probability" not in df.columns:
        df["theft_probability"] = df["risk_score"] / 100.0

    return df


def prepare_simulation_from_engine(engine_result: dict, n_customers: int = 5) -> dict:
    """
    High-level helper that returns everything the live simulation needs
    when running on top of the engine.
    """
    if not engine_result or "scored" not in engine_result:
        return {"error": "No engine scored data"}

    scored = get_rich_scored_data_for_ui(engine_result)

    # Pick interesting customers (highest risk first)
    if "theft_probability" in scored.columns:
        top_customers = scored.nlargest(n_customers, "theft_probability")
    else:
        top_customers = scored.head(n_customers)

    return {
        "simulation_df": top_customers,
        "best_model": engine_result.get("best_model"),
        "overview": engine_result.get("overview", {}),
    }


def get_engine_scored_for_performance(engine_result: dict) -> dict:
    """Prepare data specifically for the model performance tab."""
    if not engine_result:
        return {}

    summary = get_engine_performance_summary(engine_result)
    scored = get_rich_scored_data_for_ui(engine_result)

    return {
        "summary": summary,
        "scored_sample": scored.head(100) if not scored.empty else pd.DataFrame(),
        "engine_results": engine_result.get("engine_results", {}),
    }


def prepare_dashboard_data(engine_result: dict) -> dict:
    """
    High-level function that returns a ready-to-use dictionary for the main
    dashboard tabs. This is the preferred way going forward.
    """
    if not engine_result:
        return {"error": "No engine data"}

    scored = get_rich_scored_data_for_ui(engine_result)
    metrics = extract_metrics_from_engine(engine_result)
    performance = get_engine_scored_for_performance(engine_result)

    return {
        "scored_df": scored,
        "metrics": metrics,
        "performance": performance,
        "overview": engine_result.get("overview", {}),
        "best_model": engine_result.get("best_model"),
    }


def get_scored_data_and_metrics(engine_result: dict) -> tuple:
    """
    One-stop helper that returns both the scored dataframe and metrics dict
    ready for the dashboard. Reduces duplication in app.py.
    """
    if not engine_result:
        return pd.DataFrame(), {}

    scored = get_rich_scored_data_for_ui(engine_result)
    metrics = extract_metrics_from_engine(engine_result)

    return scored, metrics


def run_engine_based_scoring(engine_result: dict) -> tuple:
    """
    Modern replacement for the old duplicated `run_models` logic.
    When we have engine data, we use the real models and results instead of
    re-training simple RF + Isolation Forest inside the dashboard.
    """
    if not engine_result or "scored" not in engine_result:
        return pd.DataFrame(), {}

    scored = get_rich_scored_data_for_ui(engine_result)
    metrics = extract_metrics_from_engine(engine_result)

    # Ensure columns that old dashboard code expects
    if "theft_probability" not in scored.columns and "risk_score" in scored.columns:
        scored["theft_probability"] = scored["risk_score"] / 100.0

    if "risk_level" not in scored.columns and "risk_category" in scored.columns:
        scored["risk_level"] = scored["risk_category"].astype(str)

    return scored, metrics


def extract_metrics_from_engine(engine_result: dict) -> dict:
    """
    Converts engine result into the metrics format expected by the current
    (still partially legacy) dashboard performance rendering.
    This is a transition helper to reduce duplication.
    """
    if not engine_result:
        return {}

    overview = engine_result.get("overview", {})
    best_model = engine_result.get("best_model", "Stacking Ensemble")

    metrics = {
        "engine_mode": True,
        "best_model": best_model,
        "overview": overview,
    }

    # If the engine has detailed per-model results, expose them
    engine_obj = engine_result.get("engine")
    if engine_obj and hasattr(engine_obj, "results"):
        metrics["engine_results"] = engine_obj.results

    return metrics


def initialize_live_simulation_state(
    sim_customers: pd.DataFrame,
    simulation_raw: pd.DataFrame,
    sim_points: int
) -> dict:
    """
    Pure data preparation helper for the live simulation animation.
    Builds the per-customer buffer structures needed for the step-by-step
    Plotly animation without any UI or Streamlit code.

    This replaces the duplicated inline dict building that was inside app.py.
    """
    if sim_customers is None or sim_customers.empty or simulation_raw is None:
        return {"customer_data": {}, "error": "Insufficient data for simulation"}

    customer_data = {}
    for _, customer in sim_customers.iterrows():
        customer_id = customer["customer_id"]
        customer_raw = simulation_raw[
            simulation_raw["customer_id"] == customer_id
        ].head(sim_points)

        customer_data[customer_id] = {
            "values": customer_raw["consumption_kw"].values if not customer_raw.empty else np.array([]),
            "label": int(customer.get("predicted_theft", customer.get("label", 0))),
            "profile": customer.get("profile", "residential"),
            "buffer_x": [],
            "buffer_y": [],
        }

    return {"customer_data": customer_data}


def build_simulation_customer_pool(
    simulation_df: pd.DataFrame,
    selected_customer_id,
    n_customers: int = 5
) -> pd.DataFrame:
    """
    Clean replacement for the old deprecated build_simulation_customer_pool.
    Selects a pool of customers for the live simulation view, prioritizing
    the selected customer and then highest-risk or other interesting ones.

    This lives in the adapter layer so the dashboard stays thin.
    """
    if simulation_df is None or simulation_df.empty:
        return pd.DataFrame()

    df = simulation_df.copy()

    # Ensure we have the selected customer
    selected = df[df["customer_id"] == selected_customer_id]
    others = df[df["customer_id"] != selected_customer_id]

    # Sort others by risk if available
    if "theft_probability" in others.columns:
        others = others.sort_values("theft_probability", ascending=False)
    elif "risk_score" in others.columns:
        others = others.sort_values("risk_score", ascending=False)

    pool = pd.concat([selected, others.head(max(n_customers - 1, 0))])

    # If still too small, just take head
    if len(pool) < n_customers:
        pool = df.head(n_customers)

    return pool.reset_index(drop=True)
