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

from shared.core.mass_ai_engine import MassAIEngine
from shared.core.real_data import load_and_convert_sgcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to SGCC-style CSV")
    parser.add_argument("--sample", type=int, default=300, help="Number of customers to sample")
    parser.add_argument("--label-col", type=str, default="FLAG", help="Name of the theft label column")
    args = parser.parse_args()

    print(f"Loading data from: {args.path}")
    features = load_and_convert_sgcc(
        args.path,
        label_col=args.label_col,
        sample_size=args.sample,
    )

    print(f"Converted {len(features)} customers into {len(features.columns)} MASS-AI features")

    engine = MassAIEngine()
    engine.df_features = features

    engine.train_models()
    scored = engine.score_customers()

    overview = engine.build_overview() or {}
    print("\n=== Engine Overview on SGCC-style Data ===")
    print(f"Best model: {overview.get('best_model')}")
    print(f"High risk customers: {overview.get('high_risk_count')}")
    print(f"Total customers: {overview.get('customer_count')}")

    if features["label"].nunique() > 1:
        from sklearn.metrics import f1_score, roc_auc_score
        y_true = features["label"].astype(int)
        y_score = scored["theft_probability"].astype(float)
        print(f"ROC-AUC: {roc_auc_score(y_true, y_score):.4f}")
        print(f"F1: {f1_score(y_true, (y_score >= 0.5).astype(int)):.4f}")


if __name__ == "__main__":
    main()
