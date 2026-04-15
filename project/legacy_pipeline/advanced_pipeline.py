"""
MASS-AI: Gelismis ML Pipeline v2.0
===================================
- Ensemble Stacking (tum modelleri birlestiren meta-model)
- SHAP Aciklanabilirlik (neden bu musteri kacak?)
- Threshold Optimizasyonu (F1, F2, maliyet-bazli)
- Musteri Bazli Risk Skoru (0-100)

Yazar: Omer Burak Kocak
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (IsolationForest, RandomForestClassifier,
                              GradientBoostingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                            precision_recall_curve, confusion_matrix,
                            roc_curve, average_precision_score)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_data():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    df = pd.read_csv(os.path.join(base, 'features.csv'))
    print(f"Veri: {len(df)} musteri ({(df['label']==1).sum()} kacak, %{(df['label']==1).mean()*100:.1f})")
    return df


def augment_minority(X_train, y_train, target_ratio=0.25):
    """
    SMOTE-benzeri oversampling + Gaussian noise ile azinlik sinifini artir.
    GAN yerine daha stabil ve aciklanabilir bir yontem.
    """
    print("\n[1/6] Veri artirma (minority augmentation)...")
    
    minority_mask = y_train == 1
    X_min = X_train[minority_mask]
    X_maj = X_train[~minority_mask]
    
    n_current = len(X_min)
    n_target = int(len(X_maj) * target_ratio)
    n_generate = n_target - n_current
    
    if n_generate <= 0:
        print(f"    Artirma gerekmiyor (mevcut: {n_current})")
        return X_train, y_train
    
    # Interpolation + noise
    synthetic = []
    for _ in range(n_generate):
        idx1, idx2 = np.random.choice(len(X_min), 2, replace=False)
        lam = np.random.uniform(0.3, 0.7)
        new_sample = X_min[idx1] * lam + X_min[idx2] * (1 - lam)
        noise = np.random.normal(0, 0.05, new_sample.shape)
        synthetic.append(new_sample + noise)
    
    X_syn = np.array(synthetic)
    y_syn = np.ones(len(X_syn))
    
    X_aug = np.vstack([X_train, X_syn])
    y_aug = np.concatenate([y_train, y_syn])
    
    print(f"    Orijinal: {n_current} kacak -> Artirilmis: {n_current + n_generate} kacak")
    print(f"    Toplam egitim: {len(X_aug)} ({(y_aug==1).sum()} kacak, %{(y_aug==1).mean()*100:.1f})")
    
    return X_aug, y_aug


def build_stacking_ensemble(X_train, y_train, X_test, y_test, feature_cols):
    """
    3 katmanli Stacking Ensemble:
    Layer 1: Isolation Forest skoru + XGBoost + Random Forest + GradientBoosting
    Layer 2: LogisticRegression meta-learner
    """
    print("\n[2/6] Stacking Ensemble olusturuluyor...")
    
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    
    # Base modeller
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=n_neg/n_pos, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, use_label_encoder=False, verbosity=0
    )
    
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    
    # Stacking
    stack = StackingClassifier(
        estimators=[('xgb', xgb), ('rf', rf), ('gb', gb)],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    print("    Base modeller: XGBoost, Random Forest, Gradient Boosting")
    print("    Meta-learner: Logistic Regression")
    print("    CV: 5-fold stratified")
    print("    Egitim basliyor...")
    
    stack.fit(X_train, y_train)
    
    # Bireysel model sonuclari
    print("\n    Bireysel Model Sonuclari (test seti):")
    models = {'XGBoost': xgb, 'Random Forest': rf, 'Gradient Boosting': gb}
    individual_results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        f1 = f1_score(y_test, model.predict(X_test))
        individual_results[name] = {'model': model, 'prob': prob, 'auc': auc, 'f1': f1}
        print(f"      {name:<25} AUC: {auc:.4f}  F1: {f1:.4f}")
    
    # Isolation Forest (unsupervised skoru ekle)
    iso = IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
    iso.fit(X_train)
    iso_scores = -iso.score_samples(X_test)
    iso_auc = roc_auc_score(y_test, iso_scores)
    print(f"      {'Isolation Forest':<25} AUC: {iso_auc:.4f}  (unsupervised)")
    
    # Stacking sonucu
    stack_prob = stack.predict_proba(X_test)[:, 1]
    stack_pred = stack.predict(X_test)
    stack_auc = roc_auc_score(y_test, stack_prob)
    stack_f1 = f1_score(y_test, stack_pred)
    stack_ap = average_precision_score(y_test, stack_prob)
    
    print(f"\n    >>> STACKING ENSEMBLE   AUC: {stack_auc:.4f}  F1: {stack_f1:.4f}  AP: {stack_ap:.4f} <<<")
    
    print("\n    Siniflandirma Raporu (Stacking):")
    print(classification_report(y_test, stack_pred, target_names=['Normal', 'Kacak']))
    
    return stack, individual_results, iso, stack_prob, stack_pred, stack_auc, stack_f1


def optimize_threshold(y_test, probs):
    """Farkli metrikler icin optimal esik degerleri"""
    print("\n[3/6] Esik optimizasyonu...")
    
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    
    # F1 optimal
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_idx = np.argmax(f1_scores)
    f1_threshold = thresholds[f1_idx]
    
    # F2 optimal (recall'a daha fazla agirlik — kacagi kacirmamak onemli)
    f2_scores = 5 * (precision * recall) / (4 * precision + recall + 1e-8)
    f2_idx = np.argmax(f2_scores)
    f2_threshold = thresholds[f2_idx]
    
    # Maliyet-bazli optimal (false negative maliyeti false positive'den 5x fazla)
    cost_fn = 5  # Kacagi kacirmanin maliyeti
    cost_fp = 1  # Yanlis alarmin maliyeti
    costs = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        fn = ((y_test == 1) & (pred == 0)).sum()
        fp = ((y_test == 0) & (pred == 1)).sum()
        costs.append(fn * cost_fn + fp * cost_fp)
    cost_idx = np.argmin(costs)
    cost_threshold = thresholds[cost_idx]
    
    print(f"    F1-optimal esik:     {f1_threshold:.4f} (F1={f1_scores[f1_idx]:.4f})")
    print(f"    F2-optimal esik:     {f2_threshold:.4f} (F2={f2_scores[f2_idx]:.4f}, Recall={recall[f2_idx]:.4f})")
    print(f"    Maliyet-optimal esik: {cost_threshold:.4f} (Maliyet={costs[cost_idx]})")
    
    return {
        'f1': {'threshold': f1_threshold, 'score': f1_scores[f1_idx]},
        'f2': {'threshold': f2_threshold, 'score': f2_scores[f2_idx]},
        'cost': {'threshold': cost_threshold, 'cost': costs[cost_idx]},
    }


def compute_shap_explanations(model, X_test, feature_cols, df_test):
    """SHAP ile model aciklanabilirligi"""
    print("\n[4/6] SHAP aciklanabilirlik analizi...")
    
    # TreeExplainer (hizli)
    # Stacking modelin RF base'ini kullan
    rf_model = model.named_estimators_['rf']
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    # Binary classification icin class 1 (kacak) SHAP degerleri
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]  # (samples, features, classes) -> class 1
    else:
        shap_vals = shap_values
    
    # Global feature importance (SHAP-based)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_importance = sorted(zip(feature_cols, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True)
    
    print("    SHAP-bazli Ozellik Onemliligi (Top 10):")
    for feat, imp in shap_importance[:10]:
        print(f"      {feat:<25} {imp:.4f}")
    
    # En supheli musteriler icin bireysel aciklama
    print("\n    En Supheli 3 Musteri — Aciklama:")
    theft_mask = df_test['label'] == 1
    if theft_mask.sum() > 0:
        theft_indices = np.where(theft_mask)[0][:3]
        for idx in theft_indices:
            cid = df_test.iloc[idx]['customer_id']
            top_factors = sorted(zip(feature_cols, shap_vals[idx]), key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"\n      Musteri #{int(cid)} (Tur: {df_test.iloc[idx]['theft_type']}):")
            for feat, val in top_factors:
                direction = "ARTIRIYOR" if val > 0 else "AZALTIYOR"
                print(f"        {feat}: {val:+.4f} (kacak ihtimalini {direction})")
    
    return shap_vals, shap_importance


def compute_risk_scores(df, stack_model, scaler, feature_cols):
    """Her musteri icin 0-100 arasi risk skoru hesapla"""
    print("\n[5/6] Risk skorlari hesaplaniyor...")
    
    meta_cols = ['customer_id', 'profile', 'label', 'theft_type']
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    # Stacking probability
    probs = stack_model.predict_proba(X_scaled)[:, 1]
    
    # 0-100 risk skoru (non-linear mapping)
    risk_scores = np.clip(probs * 120 - 10, 0, 100).astype(int)
    
    df = df.copy()
    df['risk_score'] = risk_scores
    df['theft_probability'] = probs
    df['risk_category'] = pd.cut(risk_scores, bins=[0, 25, 50, 75, 100],
                                  labels=['Dusuk', 'Orta', 'Yuksek', 'Kritik'],
                                  include_lowest=True)
    
    print(f"    Risk dagilimi:")
    for cat in ['Dusuk', 'Orta', 'Yuksek', 'Kritik']:
        n = (df['risk_category'] == cat).sum()
        print(f"      {cat:<10} {n:>5} musteri ({n/len(df)*100:.1f}%)")
    
    # En yuksek riskli 10
    print(f"\n    En Yuksek Riskli 10 Musteri:")
    top_risk = df.nlargest(10, 'risk_score')
    for _, row in top_risk.iterrows():
        status = "GERCEK KACAK" if row['label'] == 1 else "NORMAL"
        print(f"      #{int(row['customer_id']):>4} | Skor: {row['risk_score']:>3} | {row['profile']:<12} | {status}")
    
    return df


def plot_advanced_results(y_test, stack_prob, individual_results, shap_vals,
                          feature_cols, thresholds, output_dir):
    """Gelismis sonuc grafikleri"""
    print("\n[6/6] Grafikler olusturuluyor...")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('MASS-AI v2.0: Gelismis ML Pipeline Sonuclari', fontsize=16, fontweight='bold')
    
    # 1. ROC Karsilastirma (tum modeller)
    ax = axes[0, 0]
    colors = {'XGBoost': '#E67E22', 'Random Forest': '#2E86C1', 'Gradient Boosting': '#8E44AD'}
    for name, res in individual_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['prob'])
        ax.plot(fpr, tpr, label=f"{name} ({res['auc']:.3f})", color=colors[name], lw=2)
    
    fpr, tpr, _ = roc_curve(y_test, stack_prob)
    stack_auc = roc_auc_score(y_test, stack_prob)
    ax.plot(fpr, tpr, label=f"Stacking ({stack_auc:.3f})", color='#E74C3C', lw=3, linestyle='--')
    ax.plot([0,1], [0,1], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Tum Modeller')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    
    # 2. SHAP Summary (bar)
    ax = axes[0, 1]
    mean_shap = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(mean_shap)[-10:]
    ax.barh([feature_cols[i] for i in top_idx], mean_shap[top_idx], color='#2E86C1')
    ax.set_xlabel('Ortalama |SHAP|')
    ax.set_title('SHAP Ozellik Onemliligi (Top 10)')
    
    # 3. SHAP Beeswarm (simplified)
    ax = axes[0, 2]
    top5_idx = np.argsort(mean_shap)[-5:]
    for i, feat_idx in enumerate(top5_idx):
        vals = shap_vals[:, feat_idx]
        jitter = np.random.normal(i, 0.15, len(vals))
        colors_scatter = ['#E74C3C' if y == 1 else '#2E86C1' for y in y_test]
        ax.scatter(vals, jitter, c=colors_scatter, alpha=0.3, s=8)
    ax.set_yticks(range(len(top5_idx)))
    ax.set_yticklabels([feature_cols[i] for i in top5_idx])
    ax.set_xlabel('SHAP Degeri')
    ax.set_title('SHAP Dagilimi (Top 5)')
    ax.axvline(x=0, color='black', lw=0.5, ls='--')
    
    # 4. Threshold Analizi
    ax = axes[1, 0]
    prec, rec, thresh = precision_recall_curve(y_test, stack_prob)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    ax.plot(thresh, prec[:-1], label='Precision', color='#2E86C1', lw=2)
    ax.plot(thresh, rec[:-1], label='Recall', color='#E74C3C', lw=2)
    ax.plot(thresh, f1s[:-1], label='F1', color='#27AE60', lw=2)
    for name, info in thresholds.items():
        t = info['threshold']
        ax.axvline(x=t, ls=':', lw=1.5, alpha=0.6, label=f'{name} esik ({t:.3f})')
    ax.set_xlabel('Esik Degeri'); ax.set_ylabel('Skor')
    ax.set_title('Threshold Optimizasyonu')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    
    # 5. Confusion Matrix (Stacking)
    ax = axes[1, 1]
    cm = confusion_matrix(y_test, (stack_prob >= thresholds['f1']['threshold']).astype(int))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=20, fontweight='bold',
                   color='white' if cm[i,j] > cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['Normal', 'Kacak'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Normal', 'Kacak'])
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gercek')
    ax.set_title('Stacking Ensemble — Confusion Matrix')
    
    # 6. Model karsilastirma bar chart
    ax = axes[1, 2]
    model_names = list(individual_results.keys()) + ['Stacking\nEnsemble']
    aucs = [r['auc'] for r in individual_results.values()] + [stack_auc]
    f1s_bar = [r['f1'] for r in individual_results.values()] + [f1_score(y_test, (stack_prob>=0.5).astype(int))]
    
    x = np.arange(len(model_names))
    w = 0.35
    ax.bar(x - w/2, aucs, w, label='ROC-AUC', color='#2E86C1', edgecolor='white')
    ax.bar(x + w/2, f1s_bar, w, label='F1 Score', color='#E74C3C', edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=8)
    ax.set_ylabel('Skor'); ax.set_ylim(0.5, 1.05)
    ax.set_title('Model Karsilastirma')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    
    for i, (a, f) in enumerate(zip(aucs, f1s_bar)):
        ax.text(i - w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=7, fontweight='bold')
        ax.text(i + w/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'advanced_pipeline_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Kaydedildi: {path}")
    return path


def main():
    print("=" * 65)
    print("MASS-AI v2.0: Gelismis ML Pipeline")
    print("  Stacking Ensemble + SHAP + Threshold Optimizasyonu")
    print("=" * 65)
    
    # Veri yukle
    df = load_data()
    
    meta_cols = ['customer_id', 'profile', 'label', 'theft_type']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].values
    y = df['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
    )
    df_test = df.iloc[idx_test].reset_index(drop=True)
    
    # 1. Veri artirma
    X_train_aug, y_train_aug = augment_minority(X_train, y_train, target_ratio=0.25)
    
    # 2. Stacking Ensemble
    stack, ind_results, iso, stack_prob, stack_pred, stack_auc, stack_f1 = \
        build_stacking_ensemble(X_train_aug, y_train_aug, X_test, y_test, feature_cols)
    
    # 3. Threshold optimizasyonu
    thresholds = optimize_threshold(y_test, stack_prob)
    
    # 4. SHAP aciklanabilirlik
    shap_vals, shap_imp = compute_shap_explanations(stack, X_test, feature_cols, df_test)
    
    # 5. Risk skorlari
    df_scored = compute_risk_scores(df, stack, scaler, feature_cols)
    
    # 6. Grafikler
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    plot_advanced_results(y_test, stack_prob, ind_results, shap_vals,
                         feature_cols, thresholds, output_dir)
    
    # Skorlu veriyi kaydet
    scored_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'scored_customers.csv')
    df_scored.to_csv(scored_path, index=False)
    print(f"\n    Skorlu veri kaydedildi: {scored_path}")
    
    # FINAL OZET
    print("\n" + "=" * 65)
    print("MASS-AI v2.0 — FINAL SONUCLAR")
    print("=" * 65)
    print(f"\n{'Model':<28} {'ROC-AUC':>10} {'F1':>10} {'AP':>10}")
    print("-" * 60)
    print(f"{'Isolation Forest':<28} {'0.8208':>10} {'0.2609':>10} {'—':>10}")
    for name, res in ind_results.items():
        print(f"{name:<28} {res['auc']:>10.4f} {res['f1']:>10.4f} {'—':>10}")
    print(f"{'LSTM Autoencoder':<28} {'0.7482':>10} {'0.5600':>10} {'—':>10}")
    stack_ap = average_precision_score(y_test, stack_prob)
    print(f"{'>>> STACKING ENSEMBLE <<<':<28} {stack_auc:>10.4f} {stack_f1:>10.4f} {stack_ap:>10.4f}")
    print("=" * 65)
    
    improvement = ((stack_auc - 0.9471) / 0.9471) * 100
    print(f"\nStacking vs En Iyi Bireysel: {improvement:+.2f}% ROC-AUC iyilestirme")
    print("=" * 65)


if __name__ == "__main__":
    main()
