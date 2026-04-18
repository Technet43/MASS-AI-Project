"""
MASS-AI: Production ML Modülleri
=================================
1. Transformer Enerji Dengesi — Trafodan çekme tespiti (Doğu bölgeleri)
2. Model Registry — Eğitilmiş modelleri kaydet/yükle/versiyonla
3. Data Validation — Veri kalite kontrolü ve temizleme
4. Batch Prediction — Büyük veri setlerini parça parça skorlama
5. Model Drift Detection — Zaman içinde model performans takibi

Yazar: Ömer Burak Koçak
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)


# ================================================================
# 1. TRANSFORMER ENERJİ DENGESİ ANALİZİ
# ================================================================

class TransformerBalanceAnalyzer:
    """
    Trafo bazlı enerji dengesi analizi.
    Trafo sayacı ölçümü vs müşteri sayaçları toplamı karşılaştırması.
    Fark belirli bir eşiği aşarsa → o bölgede kaçak var.
    
    Bu özellikle Doğu'da yaygın olan "trafodan direkt çekme" 
    yöntemini tespit eder — müşteri sayacında görünmez ama
    trafo sayacında görünür.
    """
    
    def __init__(self, technical_loss_pct=0.04, alert_threshold=0.12):
        """
        technical_loss_pct: Beklenen teknik kayıp oranı (%4 tipik)
        alert_threshold: Bu oranın üstü kaçak şüphesi (%12)
        """
        self.technical_loss_pct = technical_loss_pct
        self.alert_threshold = alert_threshold
    
    def generate_transformer_data(self, customer_data, n_transformers=40, customers_per_trafo=50):
        """Trafo bazlı veri üret (simülasyon)"""
        print("\n[TRAFO] Trafo bazlı enerji dengesi analizi...")
        
        trafo_records = []
        
        for t_id in range(n_transformers):
            # Bu trafoya bağlı müşteriler
            start_idx = t_id * customers_per_trafo
            end_idx = min(start_idx + customers_per_trafo, len(customer_data))
            
            if start_idx >= len(customer_data):
                break
            
            trafo_customers = customer_data[start_idx:end_idx]
            
            # Müşteri sayaçları toplamı
            customer_total = sum(c.get('mean_consumption', 0) * 24 * 30 for c in trafo_customers)
            
            # Kaçak müşteri sayısı
            n_theft = sum(1 for c in trafo_customers if c.get('label', 0) == 1)
            
            # Gerçek tüketim (kaçak dahil — trafo sayacı bunu görür)
            theft_consumption = sum(
                c.get('mean_consumption', 0) * 24 * 30 * (1 / 0.4)  # Kaçakçının gerçek tüketimi
                for c in trafo_customers if c.get('label', 0) == 1
            )
            
            # Trafo sayacı = müşteri toplamı + teknik kayıp + kaçak tüketim
            technical_loss = customer_total * self.technical_loss_pct
            transformer_reading = customer_total + technical_loss + theft_consumption * 0.6
            
            # Enerji dengesi farkı
            balance_gap = transformer_reading - customer_total
            balance_ratio = balance_gap / (transformer_reading + 1e-6)
            
            # Bölge (doğu bölgeleri daha yüksek kaçak)
            region = "East" if t_id % 3 == 0 else "West" if t_id % 3 == 1 else "Central"
            
            trafo_records.append({
                'transformer_id': f'TR-{t_id:04d}',
                'region': region,
                'n_customers': len(trafo_customers),
                'n_theft_customers': n_theft,
                'customer_total_kwh': round(customer_total, 2),
                'transformer_reading_kwh': round(transformer_reading, 2),
                'balance_gap_kwh': round(balance_gap, 2),
                'balance_ratio': round(balance_ratio, 4),
                'technical_loss_kwh': round(technical_loss, 2),
                'estimated_theft_kwh': round(balance_gap - technical_loss, 2),
                'alert': balance_ratio > self.alert_threshold,
                'risk_level': 'Critical' if balance_ratio > 0.25 else 'High' if balance_ratio > 0.15 else 'Medium' if balance_ratio > self.alert_threshold else 'Low',
            })
        
        df = pd.DataFrame(trafo_records)
        
        # Özet
        alert_count = df['alert'].sum()
        total_theft_kwh = df['estimated_theft_kwh'].sum()
        
        print(f"    Toplam trafo: {len(df)}")
        print(f"    Alarm veren: {alert_count} (%{alert_count/len(df)*100:.0f})")
        print(f"    Tahmini toplam kaçak: {total_theft_kwh:,.0f} kWh/ay")
        print(f"    Risk dağılımı:")
        for level in ['Critical', 'High', 'Medium', 'Low']:
            n = (df['risk_level'] == level).sum()
            print(f"      {level}: {n}")
        
        return df
    
    def identify_suspicious_transformers(self, trafo_df, top_n=10):
        """En şüpheli trafoları listele"""
        suspicious = trafo_df[trafo_df['alert']].sort_values('balance_ratio', ascending=False).head(top_n)
        
        print(f"\n    En şüpheli {min(top_n, len(suspicious))} trafo:")
        for _, row in suspicious.iterrows():
            print(f"      {row['transformer_id']} ({row['region']}): "
                  f"Fark %{row['balance_ratio']*100:.1f}, "
                  f"Tahmini kaçak: {row['estimated_theft_kwh']:,.0f} kWh, "
                  f"Müşteri: {row['n_customers']} ({row['n_theft_customers']} şüpheli)")
        
        return suspicious


# ================================================================
# 2. MODEL REGISTRY
# ================================================================

class ModelRegistry:
    """
    Eğitilmiş modelleri kaydet, yükle, versiyonla.
    Her model: model dosyası + metadata + performans metrikleri
    """
    
    def __init__(self, registry_dir="models/registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.registry_dir / "catalog.json"
        self.catalog = self._load_catalog()
    
    def _load_catalog(self):
        if self.catalog_path.exists():
            with open(self.catalog_path) as f:
                return json.load(f)
        return {"models": []}
    
    def _save_catalog(self):
        with open(self.catalog_path, 'w') as f:
            json.dump(self.catalog, f, indent=2, default=str)
    
    def register(self, model, name, version, scaler=None, feature_names=None, metrics=None, description=""):
        """Model kaydet"""
        model_id = f"{name}_v{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        model_dir = self.registry_dir / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Model kaydet
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Scaler kaydet
        if scaler:
            joblib.dump(scaler, model_dir / "scaler.joblib")
        
        # Metadata
        metadata = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "feature_names": feature_names or [],
            "metrics": metrics or {},
            "model_path": str(model_path),
            "file_size_kb": round(os.path.getsize(model_path) / 1024, 1),
            "checksum": hashlib.md5(open(model_path, 'rb').read()).hexdigest(),
        }
        
        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.catalog["models"].append(metadata)
        self._save_catalog()
        
        print(f"    [REGISTRY] Kaydedildi: {model_id} ({metadata['file_size_kb']} KB)")
        return model_id
    
    def load(self, model_id=None, name=None, version=None):
        """Model yükle"""
        if model_id:
            entry = next((m for m in self.catalog["models"] if m["model_id"] == model_id), None)
        elif name:
            candidates = [m for m in self.catalog["models"] if m["name"] == name]
            if version:
                entry = next((m for m in candidates if m["version"] == version), None)
            else:
                entry = candidates[-1] if candidates else None
        else:
            return None
        
        if not entry:
            print(f"    [REGISTRY] Model bulunamadı")
            return None
        
        model_dir = self.registry_dir / entry["model_id"]
        model = joblib.load(model_dir / "model.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib") if (model_dir / "scaler.joblib").exists() else None
        
        return {"model": model, "scaler": scaler, "metadata": entry}
    
    def list_models(self):
        """Tüm modelleri listele"""
        print(f"\n    [REGISTRY] {len(self.catalog['models'])} model kayıtlı:")
        for m in self.catalog["models"]:
            print(f"      {m['model_id']}: {m['name']} v{m['version']} — "
                  f"AUC: {m['metrics'].get('roc_auc', 'N/A')}, "
                  f"{m['file_size_kb']} KB")
        return self.catalog["models"]


# ================================================================
# 3. DATA VALIDATION
# ================================================================

class DataValidator:
    """
    Veri kalite kontrolü ve temizleme.
    CSV yüklenmeden önce ve sonra kontroller.
    """
    
    REQUIRED_COLUMNS = {'customer_id'}
    NUMERIC_COLUMNS = {'mean_consumption', 'std_consumption', 'zero_measurement_pct',
                       'sudden_change_ratio', 'night_day_ratio', 'trend_slope'}
    
    def validate(self, df):
        """Tam veri doğrulama raporu üret"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'issues': [],
            'warnings': [],
            'stats': {},
            'is_valid': True,
        }
        
        # 1. Zorunlu sütun kontrolü
        missing_required = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_required:
            report['issues'].append(f"Zorunlu sütunlar eksik: {missing_required}")
            report['is_valid'] = False
        
        # 2. Boş satır kontrolü
        empty_rows = df.isna().all(axis=1).sum()
        if empty_rows > 0:
            report['warnings'].append(f"{empty_rows} tamamen boş satır bulundu")
        
        # 3. Duplike customer_id kontrolü
        if 'customer_id' in df.columns:
            dupes = df['customer_id'].duplicated().sum()
            if dupes > 0:
                report['warnings'].append(f"{dupes} tekrarlayan customer_id bulundu")
        
        # 4. Eksik değer oranları
        null_pcts = df.isnull().mean()
        high_null = null_pcts[null_pcts > 0.3]
        if len(high_null) > 0:
            report['warnings'].append(f"%30'dan fazla eksik değer: {list(high_null.index)}")
        
        # 5. Sayısal sütun kontrolü
        for col in self.NUMERIC_COLUMNS & set(df.columns):
            non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum() - df[col].isna().sum()
            if non_numeric > 0:
                report['warnings'].append(f"'{col}' sütununda {non_numeric} sayısal olmayan değer")
        
        # 6. Negatif tüketim kontrolü
        if 'mean_consumption' in df.columns:
            negatives = (pd.to_numeric(df['mean_consumption'], errors='coerce') < 0).sum()
            if negatives > 0:
                report['issues'].append(f"{negatives} negatif tüketim değeri — fiziksel olarak imkansız")
        
        # 7. Aykırı değer tespiti (IQR)
        outlier_cols = []
        for col in self.NUMERIC_COLUMNS & set(df.columns):
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(values) > 10:
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                iqr = q3 - q1
                outliers = ((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum()
                if outliers > len(values) * 0.05:
                    outlier_cols.append(f"{col} ({outliers} aykırı)")
        if outlier_cols:
            report['warnings'].append(f"Aykırı değer yoğunluğu: {outlier_cols}")
        
        # İstatistikler
        report['stats'] = {
            'null_percentage': round(df.isnull().mean().mean() * 100, 2),
            'numeric_columns': len([c for c in df.columns if df[c].dtype in ['float64', 'int64']]),
            'categorical_columns': len([c for c in df.columns if df[c].dtype == 'object']),
            'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }
        
        return report
    
    def clean(self, df):
        """Veriyi temizle ve standartlaştır"""
        cleaned = df.copy()
        log = []
        
        # Boş satırları kaldır
        before = len(cleaned)
        cleaned = cleaned.dropna(how='all')
        if len(cleaned) < before:
            log.append(f"{before - len(cleaned)} boş satır silindi")
        
        # Sayısal sütunları dönüştür
        for col in self.NUMERIC_COLUMNS & set(cleaned.columns):
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
        
        # Eksik değerleri doldur
        for col in cleaned.columns:
            if cleaned[col].dtype in ['float64', 'int64']:
                null_count = cleaned[col].isna().sum()
                if null_count > 0:
                    median_val = cleaned[col].median()
                    cleaned[col] = cleaned[col].fillna(median_val)
                    log.append(f"'{col}': {null_count} eksik → medyan ({median_val:.3f})")
        
        # Negatif tüketim düzelt
        if 'mean_consumption' in cleaned.columns:
            negatives = (cleaned['mean_consumption'] < 0).sum()
            if negatives > 0:
                cleaned.loc[cleaned['mean_consumption'] < 0, 'mean_consumption'] = 0
                log.append(f"mean_consumption: {negatives} negatif → 0")
        
        return cleaned, log


# ================================================================
# 4. BATCH PREDICTION
# ================================================================

class BatchPredictor:
    """
    Büyük veri setlerini parça parça işleme.
    Memory-efficient, progress tracking, error handling.
    """
    
    def __init__(self, model, scaler, feature_cols, batch_size=5000):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.batch_size = batch_size
    
    def predict(self, df, progress_callback=None):
        """Batch prediction with progress"""
        n_total = len(df)
        n_batches = (n_total + self.batch_size - 1) // self.batch_size
        
        all_probs = []
        all_preds = []
        errors = []
        
        print(f"\n    [BATCH] {n_total} müşteri, {n_batches} batch ({self.batch_size}/batch)")
        
        for i in range(n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, n_total)
            batch = df.iloc[start:end]
            
            try:
                # Feature extraction
                X = batch[self.feature_cols].fillna(0).values
                X_scaled = self.scaler.transform(X)
                
                # Prediction
                probs = self.model.predict_proba(X_scaled)[:, 1]
                preds = self.model.predict(X_scaled)
                
                all_probs.extend(probs.tolist())
                all_preds.extend(preds.tolist())
                
            except Exception as e:
                errors.append({"batch": i, "error": str(e), "rows": f"{start}-{end}"})
                all_probs.extend([0.0] * (end - start))
                all_preds.extend([0] * (end - start))
            
            if progress_callback:
                progress_callback(i + 1, n_batches)
            
            if (i + 1) % max(1, n_batches // 5) == 0:
                print(f"      Batch {i+1}/{n_batches} tamamlandı ({end}/{n_total})")
        
        if errors:
            print(f"    ⚠️ {len(errors)} batch'te hata oluştu")
        
        return np.array(all_probs), np.array(all_preds), errors


# ================================================================
# 5. MODEL DRIFT DETECTION
# ================================================================

class DriftDetector:
    """
    Model performansının zaman içinde değişimini takip eder.
    Feature distribution shift ve performance degradation tespiti.
    """
    
    def __init__(self, baseline_metrics=None):
        self.baseline = baseline_metrics or {}
        self.history = []
    
    def record_performance(self, y_true, y_pred, y_prob, timestamp=None):
        """Performans kaydı ekle"""
        ts = timestamp or datetime.now().isoformat()
        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        
        entry = {
            'timestamp': ts,
            'roc_auc': round(auc, 4),
            'f1_score': round(f1, 4),
            'n_samples': len(y_true),
            'positive_rate': round(y_true.mean(), 4),
            'prediction_rate': round(y_pred.mean(), 4),
        }
        
        self.history.append(entry)
        return entry
    
    def check_drift(self, current_metrics, threshold=0.05):
        """Baseline'a göre drift kontrolü"""
        if not self.baseline:
            return {"drift_detected": False, "message": "Baseline tanımlı değil"}
        
        alerts = []
        
        for metric in ['roc_auc', 'f1_score']:
            baseline_val = self.baseline.get(metric, 0)
            current_val = current_metrics.get(metric, 0)
            
            if baseline_val > 0:
                change = (current_val - baseline_val) / baseline_val
                if abs(change) > threshold:
                    direction = "düştü" if change < 0 else "yükseldi"
                    alerts.append(f"{metric}: {baseline_val:.4f} → {current_val:.4f} ({change:+.2%} {direction})")
        
        # Positive rate drift (veri dağılımı değişimi)
        if 'positive_rate' in current_metrics and 'positive_rate' in self.baseline:
            base_rate = self.baseline['positive_rate']
            curr_rate = current_metrics['positive_rate']
            if base_rate > 0:
                rate_change = abs(curr_rate - base_rate) / base_rate
                if rate_change > 0.3:
                    alerts.append(f"Kaçak oranı değişti: {base_rate:.2%} → {curr_rate:.2%}")
        
        return {
            "drift_detected": len(alerts) > 0,
            "alerts": alerts,
            "message": f"{len(alerts)} drift tespit edildi" if alerts else "Drift yok, model stabil"
        }
    
    def plot_history(self, output_path="docs/drift_history.png"):
        """Performans geçmişi grafiği"""
        if len(self.history) < 2:
            return
        
        fig, ax = plt.subplots(figsize=(10, 4))
        df = pd.DataFrame(self.history)
        ax.plot(range(len(df)), df['roc_auc'], 'o-', color='#2563eb', label='ROC-AUC', lw=2)
        ax.plot(range(len(df)), df['f1_score'], 's-', color='#dc2626', label='F1 Score', lw=2)
        
        if self.baseline:
            ax.axhline(y=self.baseline.get('roc_auc', 0), ls='--', color='#2563eb', alpha=0.5, label='Baseline AUC')
            ax.axhline(y=self.baseline.get('f1_score', 0), ls='--', color='#dc2626', alpha=0.5, label='Baseline F1')
        
        ax.set_xlabel('Run'); ax.set_ylabel('Skor'); ax.set_title('Model Performans Takibi')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150); plt.close()


# ================================================================
# ANA ÇALIŞTIRICI
# ================================================================

def main():
    print("=" * 65)
    print("  MASS-AI: Production ML Modülleri")
    print("  Trafo Analizi + Registry + Validation + Batch + Drift")
    print("=" * 65)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Veri yükle
    features_path = os.path.join(data_dir, 'features.csv')
    if not os.path.exists(features_path):
        print(f"  UYARI: {features_path} bulunamadı")
        print(f"  Önce generate_synthetic_data.py çalıştırın")
        # Demo veri üret
        print("  Demo veri üretiliyor...")
        n = 500
        df = pd.DataFrame({
            'customer_id': range(n),
            'profile': np.random.choice(['residential', 'commercial', 'industrial'], n, p=[0.7, 0.2, 0.1]),
            'label': np.random.choice([0, 1], n, p=[0.88, 0.12]),
            'theft_type': 'none',
            'mean_consumption': np.random.lognormal(1, 0.8, n),
            'std_consumption': np.random.exponential(0.5, n),
            'zero_measurement_pct': np.random.beta(1, 20, n),
            'sudden_change_ratio': np.random.beta(1, 15, n),
            'night_day_ratio': np.random.beta(3, 10, n),
            'trend_slope': np.random.normal(0, 0.01, n),
            'mean_daily_total': np.random.lognormal(2, 0.8, n),
            'cv_daily': np.random.exponential(0.3, n),
        })
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(features_path, index=False)
    
    df = pd.read_csv(features_path)
    print(f"\n  Veri yüklendi: {len(df)} müşteri")
    
    # ===== 1. DATA VALIDATION =====
    print("\n" + "=" * 65)
    print("  [1/5] Data Validation")
    print("=" * 65)
    
    validator = DataValidator()
    report = validator.validate(df)
    
    print(f"    Geçerli: {'✅ Evet' if report['is_valid'] else '❌ Hayır'}")
    print(f"    Satır: {report['total_rows']}, Sütun: {report['total_columns']}")
    print(f"    Eksik: %{report['stats']['null_percentage']}")
    print(f"    Bellek: {report['stats']['memory_mb']} MB")
    if report['issues']:
        for issue in report['issues']:
            print(f"    ❌ {issue}")
    if report['warnings']:
        for warn in report['warnings'][:5]:
            print(f"    ⚠️ {warn}")
    
    cleaned_df, clean_log = validator.clean(df)
    if clean_log:
        print(f"\n    Temizleme:")
        for log in clean_log[:5]:
            print(f"      {log}")
    
    # ===== 2. MODEL TRAINING + REGISTRY =====
    print("\n" + "=" * 65)
    print("  [2/5] Model Training & Registry")
    print("=" * 65)
    
    meta_cols = ['customer_id', 'profile', 'label', 'theft_type']
    extra_meta = ['vacation_type', 'has_vacation', 'anomaly_score', 'theft_probability',
                  'predicted_theft', 'risk_level', 'risk_score', 'risk_category']
    feature_cols = [c for c in cleaned_df.columns if c not in meta_cols + extra_meta and cleaned_df[c].dtype in ['float64', 'int64']]
    
    X = cleaned_df[feature_cols].fillna(0).values
    y = cleaned_df['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    
    # RF model eğit
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    rf_auc = roc_auc_score(y_test, rf_prob)
    rf_f1 = f1_score(y_test, rf_pred)
    
    print(f"    RF AUC: {rf_auc:.4f}, F1: {rf_f1:.4f}")
    
    # Registry'ye kaydet
    registry = ModelRegistry(os.path.join(base_dir, "models", "registry"))
    model_id = registry.register(
        model=rf, name="random_forest", version="1.0",
        scaler=scaler, feature_names=feature_cols,
        metrics={"roc_auc": round(rf_auc, 4), "f1_score": round(rf_f1, 4)},
        description="Production RF model with balanced class weights"
    )
    
    # IF model
    iso = IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
    iso.fit(X_train)
    
    registry.register(
        model=iso, name="isolation_forest", version="1.0",
        scaler=scaler, feature_names=feature_cols,
        metrics={"type": "unsupervised"},
        description="Unsupervised anomaly detection"
    )
    
    registry.list_models()
    
    # ===== 3. BATCH PREDICTION =====
    print("\n" + "=" * 65)
    print("  [3/5] Batch Prediction")
    print("=" * 65)
    
    predictor = BatchPredictor(rf, scaler, feature_cols, batch_size=200)
    probs, preds, errors = predictor.predict(cleaned_df)
    
    cleaned_df = cleaned_df.copy()
    cleaned_df['theft_probability'] = probs
    cleaned_df['predicted_theft'] = preds
    cleaned_df['risk_score'] = (probs * 100).round(1)
    
    print(f"    Toplam skorlanan: {len(probs)}")
    print(f"    Yüksek risk (>%70): {(probs > 0.7).sum()}")
    print(f"    Hata: {len(errors)}")
    
    # ===== 4. TRANSFORMER ANALYSIS =====
    print("\n" + "=" * 65)
    print("  [4/5] Transformer Enerji Dengesi")
    print("=" * 65)
    
    trafo = TransformerBalanceAnalyzer(technical_loss_pct=0.04, alert_threshold=0.12)
    customer_data = cleaned_df.to_dict('records')
    trafo_df = trafo.generate_transformer_data(customer_data, n_transformers=10)
    trafo.identify_suspicious_transformers(trafo_df, top_n=5)
    
    # Trafo sonuçlarını kaydet
    trafo_path = os.path.join(data_dir, 'transformer_analysis.csv')
    trafo_df.to_csv(trafo_path, index=False)
    print(f"\n    Kaydedildi: {trafo_path}")
    
    # ===== 5. DRIFT DETECTION =====
    print("\n" + "=" * 65)
    print("  [5/5] Drift Detection")
    print("=" * 65)
    
    drift = DriftDetector(baseline_metrics={
        'roc_auc': rf_auc,
        'f1_score': rf_f1,
        'positive_rate': y.mean(),
    })
    
    # Baseline kayıt
    entry = drift.record_performance(y_test, rf_pred, rf_prob, "2026-03-01")
    print(f"    Baseline: AUC={entry['roc_auc']}, F1={entry['f1_score']}")
    
    # Simüle edilmiş gelecek performanslar
    for month in range(2, 7):
        noise = np.random.normal(0, 0.02)
        sim_prob = np.clip(rf_prob + np.random.normal(0, 0.03, len(rf_prob)), 0, 1)
        sim_pred = (sim_prob >= 0.5).astype(int)
        entry = drift.record_performance(y_test, sim_pred, sim_prob, f"2026-{month:02d}-01")
    
    # Drift kontrolü
    current = drift.history[-1]
    drift_result = drift.check_drift(current)
    print(f"\n    Drift durumu: {drift_result['message']}")
    if drift_result['alerts']:
        for alert in drift_result['alerts']:
            print(f"      ⚠️ {alert}")
    
    # Grafik
    drift.plot_history(os.path.join(base_dir, "docs", "drift_history.png"))
    
    # ===== FINAL ÖZET =====
    print("\n" + "=" * 65)
    print("  PRODUCTION MODÜLLER — ÖZET")
    print("=" * 65)
    print(f"""
    ✅ Data Validation     — {report['total_rows']} satır doğrulandı, {len(report['issues'])} hata, {len(report['warnings'])} uyarı
    ✅ Model Registry      — {len(registry.catalog['models'])} model kaydedildi
    ✅ Batch Prediction     — {len(probs)} müşteri skorlandı, {len(errors)} hata
    ✅ Transformer Balance  — {len(trafo_df)} trafo analiz edildi, {trafo_df['alert'].sum()} alarm
    ✅ Drift Detection      — {len(drift.history)} kayıt, {'DRIFT VAR' if drift_result['drift_detected'] else 'Stabil'}
    """)
    print("=" * 65)


if __name__ == "__main__":
    main()
