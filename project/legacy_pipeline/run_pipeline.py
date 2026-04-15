"""
MASS-AI: Pipeline Orchestrator v2.0
====================================
Tek komutla tum sistemi calistir.

Kullanim:
    python run_pipeline.py          # Tam pipeline
    python run_pipeline.py --quick  # Sadece veri + temel modeller
    python run_pipeline.py --all    # Her sey (LSTM + 1D-CNN + GAN + Federated dahil)
    python run_pipeline.py --production  # Production modulleri (trafo + registry + drift)
    python run_pipeline.py --sgcc   # SGCC veri seti testi

Yazar: Omer Burak Kocak
"""

import subprocess, sys, time, os, argparse

BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, '..'))
DASHBOARD_APP = os.path.join(PROJECT_ROOT, 'dashboard', 'app.py')

STAGES = {
    'data':       ('generate_synthetic_data.py',  'Sentetik veri uretimi (2000 musteri, 180 gun)'),
    'models':     ('theft_detection_model.py',     'Temel modeller (IF, XGBoost, RF)'),
    'advanced':   ('advanced_pipeline.py',         'Gelismis pipeline (Stacking + SHAP + Threshold)'),
    'context':    ('context_aware.py',             'Context-Aware modul (tatil/komsu/standby)'),
    'lstm':       ('lstm_autoencoder.py',          'LSTM Autoencoder'),
    'modules':    ('advanced_modules.py',          'Gelismis moduller (1D-CNN + GAN + Federated + PDF)'),
    'production': ('production_modules.py',        'Production (Trafo + Registry + Validation + Drift)'),
    'sgcc':       ('sgcc_pipeline.py',             'SGCC veri seti testi'),
}

def run_stage(script, desc):
    path = os.path.join(BASE, script)
    if not os.path.exists(path):
        print(f"\n  [SKIP] {script} bulunamadi")
        return True
    print(f"\n{'='*65}\n  {desc}\n  Dosya: {script}\n{'='*65}")
    start = time.time()
    result = subprocess.run([sys.executable, path], capture_output=False, cwd=BASE)
    elapsed = time.time() - start
    ok = result.returncode == 0
    print(f"\n  {'OK' if ok else 'HATA'} ({elapsed:.1f}s)")
    return ok

def main():
    parser = argparse.ArgumentParser(description='MASS-AI Pipeline v2.0')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--sgcc', action='store_true')
    parser.add_argument('--dashboard', action='store_true')
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  MASS-AI Pipeline Orchestrator v2.0")
    print("  8 model | 34+ ozellik | Production-ready")
    print("="*65)

    t0 = time.time()
    res = {}

    if args.sgcc:
        res['sgcc'] = run_stage(*STAGES['sgcc'][:2])
    elif args.production:
        res['data'] = run_stage(*STAGES['data'][:2])
        res['production'] = run_stage(*STAGES['production'][:2])
    elif args.quick:
        res['data'] = run_stage(*STAGES['data'][:2])
        res['models'] = run_stage(*STAGES['models'][:2])
    else:
        for k in ['data','models','advanced','context']:
            res[k] = run_stage(*STAGES[k][:2])
        if args.all:
            for k in ['lstm','modules','production']:
                res[k] = run_stage(*STAGES[k][:2])

    elapsed = time.time() - t0
    ok = sum(1 for v in res.values() if v)
    fail = sum(1 for v in res.values() if not v)

    print(f"\n{'='*65}")
    print(f"  PIPELINE TAMAMLANDI — {elapsed:.1f}s, {ok}/{ok+fail} basarili")
    print(f"{'='*65}")
    print(f"  Ciktilar: data/processed/*.csv, docs/*.png, models/registry/")
    print(f"  Dashboard: streamlit run ..\\dashboard\\app.py")
    print(f"  Tam pipeline: python run_pipeline.py --all")
    print(f"{'='*65}")

    if args.dashboard:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', DASHBOARD_APP], cwd=PROJECT_ROOT)

if __name__ == "__main__":
    main()
