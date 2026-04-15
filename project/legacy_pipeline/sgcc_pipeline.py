"""
MASS-AI: SGCC Veri Seti Uyumluluk Modulu
==========================================
SGCC (State Grid Corporation of China) veri seti yapisinda
gercekci veri ureterek pipeline'i test eder.

SGCC Format:
- 42,372 musteri (biz 5,000 ile test ediyoruz)
- 1,035 gun gunluk tuketim (2014-01-01 — 2016-10-31)
- Son sutun: FLAG (0=normal, 1=kacak)
- ~%10 kacak orani (dengesiz veri)
- Eksik degerler var (NaN)

Gercek SGCC verisini Kaggle'dan indirmek icin:
  kaggle datasets download -d bensalem14/sgcc-dataset
  veya: https://www.kaggle.com/datasets/bensalem14/sgcc-dataset

Yazar: Omer Burak Kocak
"""

import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                            confusion_matrix, average_precision_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_or_generate_sgcc(data_dir, n_customers=5000, n_days=1035):
    """
    Gercek SGCC CSV varsa yukle, yoksa ayni formatta uret.
    """
    sgcc_path = os.path.join(data_dir, 'sgcc_data.csv')

    if os.path.exists(sgcc_path):
        print(f"[SGCC] Mevcut veri yukleniyor: {sgcc_path}")
        df = pd.read_csv(sgcc_path)
        print(f"    {df.shape[0]} musteri, {df.shape[1]} sutun")
        return df

    print(f"[SGCC] Gercekci SGCC-format veri uretiliyor ({n_customers} musteri, {n_days} gun)...")

    # Tarihler: 2014-01-01 — 2016-10-31
    dates = pd.date_range('2014-01-01', periods=n_days, freq='D')
    date_cols = [d.strftime('%Y-%m-%d') for d in dates]

    # Kacak orani: ~%10 (SGCC gercek oran)
    n_theft = int(n_customers * 0.10)
    n_normal = n_customers - n_theft
    labels = np.array([0] * n_normal + [1] * n_theft)
    np.random.shuffle(labels)

    data = np.zeros((n_customers, n_days))

    for i in range(n_customers):
        # Temel tuketim profili
        base = np.random.lognormal(mean=1.5, sigma=0.8)  # kWh/gun
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        weekly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        noise = np.random.normal(0, base * 0.15, n_days)

        consumption = base * seasonal * weekly + noise
        consumption = np.maximum(consumption, 0)

        # Kacak senaryolari
        if labels[i] == 1:
            theft_type = np.random.choice(['reduce', 'zero', 'gradual', 'peak_clip', 'night'])
            theft_start = np.random.randint(200, 800)

            if theft_type == 'reduce':
                factor = np.random.uniform(0.3, 0.7)
                consumption[theft_start:] *= factor
            elif theft_type == 'zero':
                zero_days = np.random.choice(range(theft_start, n_days),
                                            size=int((n_days - theft_start) * 0.3), replace=False)
                consumption[zero_days] = 0
            elif theft_type == 'gradual':
                decay = np.linspace(1.0, np.random.uniform(0.2, 0.5), n_days - theft_start)
                consumption[theft_start:] *= decay
            elif theft_type == 'peak_clip':
                threshold = np.percentile(consumption, 60)
                consumption[theft_start:] = np.minimum(consumption[theft_start:], threshold)
            elif theft_type == 'night':
                # Rastgele gunlerde cok dusuk
                low_days = np.random.choice(range(theft_start, n_days),
                                           size=int((n_days - theft_start) * 0.4), replace=False)
                consumption[low_days] *= np.random.uniform(0.05, 0.15)

        # Eksik degerler ekle (SGCC'de %5-15 NaN var)
        nan_ratio = np.random.uniform(0.05, 0.15)
        nan_indices = np.random.choice(n_days, size=int(n_days * nan_ratio), replace=False)
        consumption[nan_indices] = np.nan

        data[i] = consumption

    # DataFrame olustur (SGCC formati)
    customer_ids = [f'C_{i:05d}' for i in range(n_customers)]
    df = pd.DataFrame(data, columns=date_cols)
    df.insert(0, 'CONS_NO', customer_ids)
    df['FLAG'] = labels

    # Kaydet
    df.to_csv(sgcc_path, index=False)
    print(f"    Kaydedildi: {sgcc_path}")
    print(f"    Normal: {n_normal}, Kacak: {n_theft} (%{n_theft/n_customers*100:.0f})")
    print(f"    Boyut: {os.path.getsize(sgcc_path)/1024/1024:.1f} MB")

    return df


def preprocess_sgcc(df):
    """SGCC verisini on-isle: eksik deger doldurma, ozellik cikarimi"""
    print("\n[ON-ISLEME] SGCC verisi hazirlaniyor...")

    date_cols = [c for c in df.columns if c not in ['CONS_NO', 'FLAG']]
    consumption = df[date_cols].values.copy()  # writable copy
    labels = df['FLAG'].values

    # 1. Eksik deger doldurma (lineer interpolasyon)
    print(f"    Eksik deger orani: %{np.isnan(consumption).mean()*100:.1f}")
    for i in range(len(consumption)):
        series = pd.Series(consumption[i])
        series = series.interpolate(method='linear', limit_direction='both')
        consumption[i] = series.fillna(0).values

    print(f"    Interpolasyon sonrasi NaN: {np.isnan(consumption).sum()}")

    # 2. Ozellik cikarimi
    print("    Ozellik cikarimi yapiliyor...")
    features = []

    for i in range(len(consumption)):
        c = consumption[i]
        n_days = len(c)

        # Temel istatistikler
        mean_c = np.nanmean(c)
        std_c = np.nanstd(c)
        min_c = np.nanmin(c)
        max_c = np.nanmax(c)
        median_c = np.nanmedian(c)

        # Carpiklik ve basiklik
        if std_c > 0:
            skew = float(pd.Series(c).skew())
            kurt = float(pd.Series(c).kurtosis())
        else:
            skew = 0
            kurt = 0

        # Sifir gun orani
        zero_days = np.sum(c < 0.01) / n_days

        # Ani degisim
        diffs = np.abs(np.diff(c))
        mean_diff = np.mean(diffs) if len(diffs) > 0 else 0
        sudden = np.sum(diffs > 3 * mean_diff) / len(diffs) if mean_diff > 0 else 0

        # Trend
        x = np.arange(n_days)
        if n_days > 1:
            slope = np.polyfit(x, c, 1)[0]
        else:
            slope = 0

        # Haftalik patern
        weekly_means = [np.mean(c[d::7]) for d in range(7)]
        weekly_cv = np.std(weekly_means) / (np.mean(weekly_means) + 1e-8)

        # Aylik ortalamalar
        monthly = [np.mean(c[m*30:min((m+1)*30, n_days)]) for m in range(n_days // 30)]
        monthly_cv = np.std(monthly) / (np.mean(monthly) + 1e-8) if monthly else 0

        # Ilk yari vs son yari
        half = n_days // 2
        first_half = np.mean(c[:half])
        second_half = np.mean(c[half:])
        half_ratio = second_half / (first_half + 1e-8)

        # Quantile ozellikleri
        q10 = np.percentile(c, 10)
        q25 = np.percentile(c, 25)
        q75 = np.percentile(c, 75)
        q90 = np.percentile(c, 90)
        iqr = q75 - q25

        # Entropi (tuketim dagilimi)
        hist, _ = np.histogram(c, bins=20, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        features.append({
            'mean': mean_c, 'std': std_c, 'min': min_c, 'max': max_c,
            'median': median_c, 'skewness': skew, 'kurtosis': kurt,
            'cv': std_c / (mean_c + 1e-8),
            'zero_day_ratio': zero_days,
            'sudden_change_ratio': sudden,
            'trend_slope': slope,
            'weekly_cv': weekly_cv,
            'monthly_cv': monthly_cv,
            'half_ratio': half_ratio,
            'q10': q10, 'q25': q25, 'q75': q75, 'q90': q90,
            'iqr': iqr,
            'entropy': entropy,
            'max_min_ratio': max_c / (min_c + 1e-8),
            'range': max_c - min_c,
        })

    feat_df = pd.DataFrame(features)
    feat_df['FLAG'] = labels
    print(f"    {len(feat_df)} musteri x {len(feat_df.columns)-1} ozellik")

    return feat_df


def train_and_evaluate(feat_df):
    """Tum modelleri egit ve degerlendir"""
    print("\n" + "=" * 65)
    print("MODEL EGITIMI VE DEGERLENDIRME (SGCC Format)")
    print("=" * 65)

    feature_cols = [c for c in feat_df.columns if c != 'FLAG']
    X = feat_df[feature_cols].values
    y = feat_df['FLAG'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"\nEgitim: {len(X_train)} ({(y_train==1).sum()} kacak)")
    print(f"Test:   {len(X_test)} ({(y_test==1).sum()} kacak)")

    results = {}

    # 1. Isolation Forest
    print("\n--- Isolation Forest ---")
    iso = IsolationForest(n_estimators=200, contamination=0.10, random_state=42)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_test)
    iso_pred = (iso.predict(X_test) == -1).astype(int)
    iso_auc = roc_auc_score(y_test, iso_scores)
    iso_f1 = f1_score(y_test, iso_pred)
    print(f"ROC-AUC: {iso_auc:.4f}, F1: {iso_f1:.4f}")
    results['Isolation Forest'] = {'auc': iso_auc, 'f1': iso_f1, 'type': 'Unsupervised'}

    # 2. XGBoost
    print("\n--- XGBoost ---")
    n_neg, n_pos = (y_train==0).sum(), (y_train==1).sum()
    xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                        scale_pos_weight=n_neg/n_pos, subsample=0.8,
                        colsample_bytree=0.8, eval_metric='logloss',
                        random_state=42, use_label_encoder=False, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict(X_test)
    xgb_auc = roc_auc_score(y_test, xgb_prob)
    xgb_f1 = f1_score(y_test, xgb_pred)
    print(f"ROC-AUC: {xgb_auc:.4f}, F1: {xgb_f1:.4f}")
    results['XGBoost'] = {'auc': xgb_auc, 'f1': xgb_f1, 'type': 'Supervised'}

    # 3. Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                               class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf_prob)
    rf_f1 = f1_score(y_test, rf_pred)
    print(f"ROC-AUC: {rf_auc:.4f}, F1: {rf_f1:.4f}")
    results['Random Forest'] = {'auc': rf_auc, 'f1': rf_f1, 'type': 'Supervised'}

    # 4. Gradient Boosting
    print("\n--- Gradient Boosting ---")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                    learning_rate=0.05, subsample=0.8, random_state=42)
    gb.fit(X_train, y_train)
    gb_prob = gb.predict_proba(X_test)[:, 1]
    gb_pred = gb.predict(X_test)
    gb_auc = roc_auc_score(y_test, gb_prob)
    gb_f1 = f1_score(y_test, gb_pred)
    print(f"ROC-AUC: {gb_auc:.4f}, F1: {gb_f1:.4f}")
    results['Gradient Boosting'] = {'auc': gb_auc, 'f1': gb_f1, 'type': 'Supervised'}

    # 5. Stacking Ensemble
    print("\n--- Stacking Ensemble ---")
    stack = StackingClassifier(
        estimators=[('xgb', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          scale_pos_weight=n_neg/n_pos, eval_metric='logloss',
                                          random_state=42, use_label_encoder=False, verbosity=0)),
                    ('rf', RandomForestClassifier(n_estimators=150, max_depth=7,
                                                  class_weight='balanced', random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                      learning_rate=0.05, random_state=42))],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5, stack_method='predict_proba', n_jobs=-1
    )
    stack.fit(X_train, y_train)
    stack_prob = stack.predict_proba(X_test)[:, 1]
    stack_pred = stack.predict(X_test)
    stack_auc = roc_auc_score(y_test, stack_prob)
    stack_f1 = f1_score(y_test, stack_pred)
    stack_ap = average_precision_score(y_test, stack_prob)
    print(f"ROC-AUC: {stack_auc:.4f}, F1: {stack_f1:.4f}, AP: {stack_ap:.4f}")
    results['Stacking Ensemble'] = {'auc': stack_auc, 'f1': stack_f1, 'ap': stack_ap, 'type': 'Meta-Learning'}

    # Detayli rapor (Stacking)
    print("\n" + "=" * 65)
    print("STACKING ENSEMBLE — Detayli Rapor")
    print("=" * 65)
    print(classification_report(y_test, stack_pred, target_names=['Normal', 'Kacak']))

    # Feature importance
    print("En Onemli 10 Ozellik (RF):")
    imp = sorted(zip(feature_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, val in imp[:10]:
        bar = "█" * int(val * 100)
        print(f"    {feat:<25} {val:.4f} {bar}")

    return results, y_test, stack_prob, rf, feature_cols


def plot_sgcc_results(results, y_test, stack_prob, output_dir):
    """Sonuc grafikleri"""
    print("\n[GRAFIK] Sonuclar olusturuluyor...")
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('MASS-AI: SGCC Format Veri Seti Sonuclari', fontsize=16, fontweight='bold')

    # 1. Model karsilastirma
    ax = axes[0]
    names = list(results.keys())
    aucs = [results[n]['auc'] for n in names]
    f1s = [results[n]['f1'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, aucs, w, label='ROC-AUC', color='#2E86C1')
    ax.bar(x + w/2, f1s, w, label='F1 Score', color='#E74C3C')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    ax.set_ylabel('Skor')
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Karsilastirmasi')
    ax.legend()
    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - w/2, a + 0.02, f'{a:.3f}', ha='center', fontsize=7)
        ax.text(i + w/2, f + 0.02, f'{f:.3f}', ha='center', fontsize=7)

    # 2. Confusion Matrix (Stacking)
    ax = axes[1]
    cm = confusion_matrix(y_test, (stack_prob >= 0.5).astype(int))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=20, fontweight='bold',
                   color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Normal', 'Kacak'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Normal', 'Kacak'])
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gercek')
    ax.set_title('Stacking Ensemble — Confusion Matrix')

    # 3. Skor dagilimi
    ax = axes[2]
    ax.hist(stack_prob[y_test==0], bins=40, alpha=0.7, color='#27AE60', label='Normal', density=True)
    ax.hist(stack_prob[y_test==1], bins=40, alpha=0.7, color='#E74C3C', label='Kacak', density=True)
    ax.axvline(x=0.5, color='black', ls='--', lw=1.5, label='Esik (0.5)')
    ax.set_xlabel('Kacak Olasiligi')
    ax.set_ylabel('Yogunluk')
    ax.set_title('Skor Dagilimi')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'sgcc_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Kaydedildi: {path}")
    return path


def main():
    print("=" * 65)
    print("MASS-AI: SGCC Veri Seti ile Test")
    print("  42,372 musteri formatinda gercekci veri ile dogrulama")
    print("=" * 65)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(data_dir, exist_ok=True)

    # SGCC verisi yukle veya uret
    df = load_or_generate_sgcc(data_dir, n_customers=5000, n_days=1035)

    # On-isleme ve ozellik cikarimi
    feat_df = preprocess_sgcc(df)

    # Model egitimi ve degerlendirme
    results, y_test, stack_prob, rf, feature_cols = train_and_evaluate(feat_df)

    # Grafikler
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    plot_sgcc_results(results, y_test, stack_prob, output_dir)

    # Final ozet
    print("\n" + "=" * 65)
    print("SGCC FORMAT — FINAL SONUCLAR")
    print("=" * 65)
    print(f"\n{'Model':<25} {'ROC-AUC':>10} {'F1':>10} {'Tip':<15}")
    print("-" * 62)
    for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
        print(f"{name:<25} {res['auc']:>10.4f} {res['f1']:>10.4f} {res['type']:<15}")
    print("=" * 65)

    # Sentetik vs SGCC karsilastirma
    print("\n" + "=" * 65)
    print("SENTETIK vs SGCC FORMAT KARSILASTIRMA")
    print("=" * 65)
    print(f"\n{'Model':<25} {'Sentetik':>10} {'SGCC':>10} {'Fark':>10}")
    print("-" * 57)
    synthetic_aucs = {
        'Stacking Ensemble': 0.9428,
        'Random Forest': 0.9461,
        'XGBoost': 0.9322,
        'Gradient Boosting': 0.9380,
        'Isolation Forest': 0.8208,
    }
    for name in results:
        syn = synthetic_aucs.get(name, 0)
        sgcc = results[name]['auc']
        diff = sgcc - syn
        print(f"{name:<25} {syn:>10.4f} {sgcc:>10.4f} {diff:>+10.4f}")
    print("=" * 65)

    print(f"\nNOT: Gercek SGCC verisini Kaggle'dan indirmek icin:")
    print(f"  kaggle datasets download -d bensalem14/sgcc-dataset")
    print(f"  Indirilen CSV'yi data/raw/sgcc_data.csv olarak kaydedin.")
    print(f"  Script otomatik olarak gercek veriyi kullanacaktir.")
    print("=" * 65)


if __name__ == "__main__":
    main()
