"""
Example script: Testing MASS-AI engine with real/public smart meter data.

This script demonstrates how to move beyond synthetic data.

Note: The full SGCC dataset is large. For quick testing, you can download
a processed sample from Kaggle (search "SGCC electricity theft") or use
the Irish CER dataset.

Usage:
    python scripts/test_with_real_data.py --path path/to/your_sgcc.csv
"""

import argparse
from pathlib import Path

from shared.core.real_data import load_sgcc_dataset, run_on_sgcc_sample
from shared.core.mass_ai_engine import MassAIEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to SGCC-style CSV")
    parser.add_argument("--sample", type=int, default=300, help="Number of customers to sample")
    args = parser.parse_args()

    print(f"Loading data from: {args.path}")
    df = load_sgcc_dataset(args.path, sample_size=args.sample)

    print(f"Loaded {len(df)} customers")

    engine = MassAIEngine()

    # For real SGCC data, we would normally do custom feature engineering.
    # Here we show the loading path. In practice you would map daily columns
    # to the features expected by the engine (or extend the engine).

    try:
        features = engine.load_dataset(args.path)
        print(f"Features loaded: {features.shape}")

        engine.train_models()
        scored = engine.score_customers()

        overview = engine.build_overview()
        print("\n=== Engine Overview on Real-style Data ===")
        print(f"Best model: {overview.get('best_model')}")
        print(f"High risk customers: {overview.get('high_risk_count')}")
        print(f"Total customers: {overview.get('customer_count')}")

    except Exception as e:
        print(f"\nDirect loading failed: {e}")
        print("This is expected for raw SGCC format. A custom feature mapper is needed.")
        print("See shared/core/real_data.py for the starting point.")


if __name__ == "__main__":
    main()
