"""
MASS-AI: Kacak Elektrik Tespit Modeli
=====================================
Isolation Forest (unsupervised) + XGBoost (supervised) + LSTM Autoencoder
karsilastirmali analiz.

Yazar: Omer Burak Kocak
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_curve, 
                            average_precision_score, f1_score)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

def load_data():
    """Ozellik matrisini yukle"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')
    df = pd.read_csv(data_path)
    print(f"Veri yuklendi: {df.shape[0]} musteri, {df.shape[1]} sutun")
    print(f"Normal: {(df['label']==0).sum()}, Kacak: {(df['label']==1).sum()}")
    return df


def prepare_features(df):
    """Ozellik ve hedef degisken ayirma"""
    meta_cols = ['customer_id', 'profile', 'label', 'theft_type']
    feature_cols = [c for c in df.columns if c not in meta_cols]
    
    X = df[feature_cols].values
    y = df['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler


def train_isolation_forest(X_train, y_train, X_test, y_test):
    """Unsupervised anomali tespiti"""
    print("\n" + "="*50)
    print("MODEL 1: Isolation Forest (Unsupervised)")
    print("="*50)
    
    # Contamination = kacak orani tahmini
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=0.12,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    
    # Sadece normal verilerle egit (gercek senaryoda etiket yok)
    iso_forest.fit(X_train)
    
    # Tahmin (-1 = anomali, 1 = normal)
    y_pred_raw = iso_forest.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)
    
    # Anomali skoru
    scores = -iso_forest.score_samples(X_test)
    
    print("\nSiniflandirma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Kacak']))
    
    auc = roc_auc_score(y_test, scores)
    print(f"ROC-AUC Skoru: {auc:.4f}")
    
    return iso_forest, y_pred, scores, auc


def train_xgboost(X_train, y_train, X_test, y_test, feature_cols):
    """Supervised siniflandirma"""
    print("\n" + "="*50)
    print("MODEL 2: XGBoost (Supervised)")
    print("="*50)
    
    # Dengesiz veri icin scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=n_neg / n_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    
    print("\nSiniflandirma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Kacak']))
    
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC Skoru: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb, np.vstack([X_train, X_test]), 
                                np.concatenate([y_train, y_test]), 
                                cv=cv, scoring='f1')
    print(f"5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Ozellik onemliligi
    importance = dict(zip(feature_cols, xgb.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nEn Onemli 10 Ozellik:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")
    
    return xgb, y_pred, y_prob, auc, top_features


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest - karsilastirma icin"""
    print("\n" + "="*50)
    print("MODEL 3: Random Forest (Supervised)")
    print("="*50)
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print("\nSiniflandirma Raporu:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Kacak']))
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Skoru: {auc:.4f}")
    
    return rf, y_pred, y_prob, auc


def plot_results(y_test, results, df_test, output_dir):
    """Sonuc grafikleri"""
    print("\n[Grafikler olusturuluyor...]")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('MASS-AI: Kacak Elektrik Tespit Sonuclari', fontsize=16, fontweight='bold')
    
    # 1. Model karsilastirma
    ax = axes[0, 0]
    models = list(results.keys())
    aucs = [results[m]['auc'] for m in models]
    colors = ['#1B4F72', '#2E86C1', '#85C1E9']
    bars = ax.bar(models, aucs, color=colors, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('ROC-AUC Skoru')
    ax.set_title('Model Karsilastirmasi')
    ax.set_ylim(0.5, 1.05)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Hedef: 0.90')
    ax.legend()
    
    # 2. Confusion Matrix (XGBoost)
    ax = axes[0, 1]
    cm = confusion_matrix(y_test, results['XGBoost']['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Kacak'], yticklabels=['Normal', 'Kacak'])
    ax.set_ylabel('Gercek')
    ax.set_xlabel('Tahmin')
    ax.set_title('XGBoost - Confusion Matrix')
    
    # 3. Feature Importance (XGBoost)
    ax = axes[1, 0]
    top_feats = results['XGBoost']['top_features']
    feat_names = [f[0] for f in top_feats][::-1]
    feat_vals = [f[1] for f in top_feats][::-1]
    ax.barh(feat_names, feat_vals, color='#2E86C1')
    ax.set_xlabel('Onem Skoru')
    ax.set_title('XGBoost - Ozellik Onemliligi (Top 10)')
    
    # 4. Kacak turune gore tespit orani
    ax = axes[1, 1]
    if 'theft_type' in df_test.columns:
        theft_types = df_test[df_test['label'] == 1]['theft_type'].unique()
        detection_rates = []
        type_names = []
        for tt in theft_types:
            mask = (df_test['theft_type'] == tt).values
            if mask.sum() > 0:
                correct = (results['XGBoost']['y_pred'][mask] == 1).sum()
                total = mask.sum()
                detection_rates.append(correct / total * 100)
                type_names.append(tt.replace('_', '\n'))
        
        bars = ax.bar(type_names, detection_rates, color=['#1B4F72', '#2E86C1', '#5DADE2', '#85C1E9', '#AED6F1'])
        ax.set_ylabel('Tespit Orani (%)')
        ax.set_title('Kacak Turune Gore Tespit Basarisi')
        ax.set_ylim(0, 110)
        for bar, rate in zip(bars, detection_rates):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'model_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Kaydedildi: {fig_path}")
    
    return fig_path


def main():
    print("="*60)
    print("MASS-AI: Kacak Elektrik Tespit Modeli Egitimi")
    print("="*60)
    
    # Veri yukle
    df = load_data()
    X, y, feature_cols, scaler = prepare_features(df)
    
    # Train/test ayir
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
    )
    df_test = df.iloc[idx_test].reset_index(drop=True)
    
    print(f"\nEgitim seti: {len(X_train)} ({(y_train==1).sum()} kacak)")
    print(f"Test seti:   {len(X_test)} ({(y_test==1).sum()} kacak)")
    
    # Model 1: Isolation Forest
    iso_model, iso_pred, iso_scores, iso_auc = train_isolation_forest(X_train, y_train, X_test, y_test)
    
    # Model 2: XGBoost
    xgb_model, xgb_pred, xgb_prob, xgb_auc, xgb_top = train_xgboost(X_train, y_train, X_test, y_test, feature_cols)
    
    # Model 3: Random Forest
    rf_model, rf_pred, rf_prob, rf_auc = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Sonuclari topla
    results = {
        'Isolation\nForest': {'auc': iso_auc, 'y_pred': iso_pred},
        'XGBoost': {'auc': xgb_auc, 'y_pred': xgb_pred, 'top_features': xgb_top},
        'Random\nForest': {'auc': rf_auc, 'y_pred': rf_pred},
    }
    
    # Grafikler
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
    fig_path = plot_results(y_test, results, df_test, output_dir)
    
    # Ozet tablo
    print("\n" + "="*60)
    print("FINAL SONUCLAR")
    print("="*60)
    print(f"{'Model':<22} {'ROC-AUC':>10} {'F1 (Kacak)':>12}")
    print("-"*46)
    
    for name, y_p in [('Isolation Forest', iso_pred), ('XGBoost', xgb_pred), ('Random Forest', rf_pred)]:
        f1 = f1_score(y_test, y_p)
        auc_val = {'Isolation Forest': iso_auc, 'XGBoost': xgb_auc, 'Random Forest': rf_auc}[name]
        print(f"{name:<22} {auc_val:>10.4f} {f1:>12.4f}")
    
    print("="*60)


if __name__ == "__main__":
    main()
