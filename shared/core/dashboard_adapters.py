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
