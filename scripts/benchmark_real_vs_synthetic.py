#!/usr/bin/env python3
"""
MASS-AI — Real vs Synthetic Benchmark Runner

Bu script SGCC benzeri gerçek (veya gerçekçi proxy) veri ile sentetik veriyi karşılaştırır.

Kullanım:
    python scripts/benchmark_real_vs_synthetic.py
    python scripts/benchmark_real_vs_synthetic.py --real /path/to/your_sgcc.csv --sample 800

Not: Gerçek SGCC dosyanız yoksa script kontrollü "realistic proxy" ile çalışır.
     Bu proxy, literatürde yayınlanmış SGCC istatistiklerine (düşük ortalama tüketim,
     farklı varyans, ~8.5% theft rate) göre tasarlanmıştır.

Amaç: İnkübatörlere "sadece sentetik veriyle doğrulanmış" riskini somut sayı ile göstermek.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make shared.core importable when running the script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHARED_CORE = PROJECT_ROOT / "shared" / "core"
if str(SHARED_CORE) not in sys.path:
    sys.path.insert(0, str(SHARED_CORE))

from real_data import (
    generate_real_data_validation_report,
    generate_realistic_sgcc_proxy,
    load_and_convert_sgcc,
    run_real_data_benchmark,
)
from mass_ai_engine import MassAIEngine


def main():
    parser = argparse.ArgumentParser(description="MASS-AI real vs synthetic benchmark")
    parser.add_argument("--real", type=str, default=None, help="Path to real SGCC-style CSV (optional)")
    parser.add_argument("--sample", type=int, default=900, help="Sample size for real data")
    parser.add_argument("--synthetic-n", type=int, default=1500, help="Synthetic customers for baseline")
    parser.add_argument("--preset", type=str, default="Turkey Urban", help="Synthetic preset")
    parser.add_argument("--output", type=str, default="reports/real_vs_synthetic_report.json",
                        help="Where to write the JSON report")
    args = parser.parse_args()

    print("=" * 70)
    print("MASS-AI — ELECTRICITY THEFT DETECTION")
    print("Real Data Generalization Benchmark (SGCC-style)")
    print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)

    Path("reports").mkdir(exist_ok=True)

    if args.real and Path(args.real).exists():
        print(f"\n[1] Loading REAL SGCC data from: {args.real}")
        real_features = load_and_convert_sgcc(args.real, sample_size=args.sample)
        print(f"    Converted {len(real_features)} customers with {len(real_features.columns)} columns")

        eng = MassAIEngine()
        eng.df_features = real_features
        eng.train_models()
        scored = eng.score_customers()

        from sklearn.metrics import roc_auc_score, f1_score
        y = real_features["label"].values
        p = scored["theft_probability"].values
        auc = roc_auc_score(y, p)
        f1 = f1_score(y, (p >= 0.5).astype(int))

        result = {
            "source": "real_sgcc",
            "path": str(args.real),
            "n_customers": len(real_features),
            "theft_rate": float(y.mean()),
            "best_model": eng.best_model_name(),
            "auc_on_real": round(float(auc), 4),
            "f1_on_real": round(float(f1), 4),
            "timestamp": datetime.now().isoformat(),
        }
        print("\n=== REAL DATA RESULT ===")
        print(f"Best model : {result['best_model']}")
        print(f"AUC on real: {result['auc_on_real']}")
        print(f"F1 on real : {result['f1_on_real']}")

    else:
        print("\n[1] No real SGCC file found — running controlled domain-shift benchmark")
        print("    (proxy mimics published SGCC statistics: lower usage, ~8.5% theft, different shape)")

        result = run_real_data_benchmark(
            n_synthetic=args.synthetic_n,
            n_real_proxy=args.sample,
            preset=args.preset,
        )

        print("\n=== BENCHMARK RESULTS (Controlled Domain Shift) ===")
        print(f"Synthetic (in-distribution) AUC      : {result['synthetic_in_dist_auc']}")
        print(f"Synthetic F1                           : {result['synthetic_in_dist_f1']}")
        print(f"SGCC-proxy (in-distribution) AUC     : {result['sgcc_proxy_in_dist_auc']}")
        print(f"SGCC-proxy F1                          : {result['sgcc_proxy_in_dist_f1']}")
        print(f"AUC Gap (synthetic vs proxy)           : {result['auc_gap']}")
        print(f"Best model on synthetic                : {result['best_model_synthetic']}")
        print(f"Best model on proxy                    : {result['best_model_on_proxy']}")
        print("\n>>> " + result["note"])

    # Write JSON report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Generate beautiful markdown validation report (new)
    try:
        md_path = generate_real_data_validation_report(result)
        print(f"Detaylı Markdown validasyon raporu: {md_path}")
    except Exception as e:
        print(f"Markdown rapor üretilemedi: {e}")

    print(f"\nJSON rapor: {out_path}")
    print("=" * 70)
    print("Bu çıktı inkübatör sunumlarında 'Gerçek veriyle test' maddesi için kullanılabilir.")
    print("Gerçek SGCC dosyanız olduğunda --real parametresiyle tekrar çalıştırın.")
    print("=" * 70)


if __name__ == "__main__":
    main()