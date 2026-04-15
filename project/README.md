<div align="center">

# ⚡ MASS-AI

### Milli Akıllı Sayaç Sistemleri için Yapay Zeka Tabanlı Anomali Tespit ve Kaçak Elektrik Sınıflandırma Sistemi

**AI-Powered Anomaly Detection & Electricity Theft Classification for Turkey's National Smart Meter Systems**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

[Türkçe](#-proje-özeti) • [English](#-project-summary) • [Kurulum](#-kurulum--installation) • [Sonuçlar](#-sonuçlar--results) • [Dashboard](#-dashboard)

</div>

---

## 🇹🇷 Proje Özeti

Türkiye, **1 Mart 2026** tarihinde **MASS (Milli Akıllı Sayaç Sistemleri)** projesini başlatarak 2028'e kadar **50 milyon** elektrik sayacını akıllı sayaçlarla değiştirmeyi hedeflemektedir. Bu sayaçlar 15 dakikalık aralıklarla tüketim verisi üretecek, ancak bu devasa veriyi analiz edecek bir AI altyapısı henüz mevcut değildir.

**MASS-AI**, bu boşluğu doldurmak için geliştirilmiş bir yapay zeka sistemidir:

- 🔍 **Kaçak Elektrik Tespiti** — ML ve deep learning ile anormal tüketim paternlerini tespit eder
- 📊 **Gerçek Zamanlı İzleme** — Streamlit tabanlı interaktif dashboard ile anomali takibi
- 🗺️ **Bölgesel Analiz** — Türkiye haritası üzerinde risk dağılımı görselleştirmesi
- 🚨 **Alarm Sistemi** — Yüksek riskli müşteriler için otomatik uyarı mekanizması

### Neden Şimdi?

| Problem | Boyut |
|---------|-------|
| Türkiye'de kayıp/kaçak oranı | Bazı bölgelerde **%28'e** kadar (Dicle EDAŞ) |
| Yıllık ekonomik kayıp | **Milyarlarca TL** |
| MASS akıllı sayaç hedefi | 2028'e kadar **50 milyon** sayaç |
| Yenilenebilir enerji hedefi | **%47.8** (2025) |

---

## 🇬🇧 Project Summary

Turkey launched the **MASS (National Smart Meter Systems)** project on March 1, 2026, aiming to replace 50 million electricity meters with smart meters by 2028. MASS-AI is an AI-powered system that detects electricity theft and consumption anomalies from smart meter data using multiple ML approaches.

---

## 🏗️ Mimari / Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MASS-AI Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Veri Üretimi │───▶│   Özellik    │───▶│    Model     │  │
│  │              │    │  Çıkarımı    │    │   Eğitimi    │  │
│  │ • 2000 müşteri│    │ • 20+ özellik│    │ • 4 model   │  │
│  │ • 180 gün    │    │ • İstatistik │    │ • Supervised │  │
│  │ • 5 kaçak türü│    │ • Zamansal   │    │ • Unsuperv.  │  │
│  │ • 15dk aralık│    │ • Anomali    │    │ • Deep Learn.│  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │        │
│                                                   ▼        │
│                                          ┌──────────────┐  │
│                                          │  Streamlit   │  │
│                                          │  Dashboard   │  │
│                                          │ • Harita     │  │
│                                          │ • Grafikler  │  │
│                                          │ • Alarmlar   │  │
│                                          │ • Detay      │  │
│                                          └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Proje Yapısı / Project Structure

```
mass-ai/
├── 📂 src/
│   ├── generate_synthetic_data.py   # Sentetik veri üretici
│   ├── theft_detection_model.py     # ML modelleri (IF, XGBoost, RF)
│   ├── lstm_autoencoder.py          # LSTM Autoencoder (Deep Learning)
│   └── advanced_pipeline.py         # Stacking Ensemble + SHAP + Risk Skoru
├── 📂 dashboard/
│   └── app.py                       # Streamlit dashboard v2.0 (~870 satır, 5 sekme)
├── 📂 data/
│   ├── raw/                         # Ham veri setleri
│   └── processed/                   # İşlenmiş veriler + risk skorları
├── 📂 models/                       # Eğitilmiş model dosyaları
├── 📂 docs/                         # Sonuç grafikleri, IEEE paper
├── 📂 notebooks/                    # Jupyter notebook'lar
├── run_pipeline.py                  # Tek komutla tüm sistemi çalıştır
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Kurulum / Installation

### Gereksinimler
- Python 3.10+
- pip

### Adımlar

```bash
# 1. Repo'yu klonla
git clone https://github.com/YOUR_USERNAME/mass-ai.git
cd mass-ai

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Tek komutla tüm pipeline'ı çalıştır
python run_pipeline.py --all        # Her şey (LSTM dahil)
python run_pipeline.py              # Hızlı (LSTM hariç)
python run_pipeline.py --quick      # En hızlı (sadece temel modeller)

# 4. Dashboard'u başlat
streamlit run dashboard/app.py

# Veya pipeline + dashboard birlikte
python run_pipeline.py --dashboard
```

---

## 🧠 Modeller / Models

### 4 Farklı Yaklaşım

| # | Model | Tip | Açıklama |
|---|-------|-----|----------|
| 1 | **Isolation Forest** | Unsupervised | Etiket gerektirmez, anomali izolasyonu |
| 2 | **XGBoost** | Supervised | Gradient boosting, özellik önemliliği |
| 3 | **Random Forest** | Supervised | Ensemble öğrenme, dengesiz veri desteği |
| 4 | **Gradient Boosting** | Supervised | Sekansiyel ağaç güçlendirme |
| 5 | **LSTM Autoencoder** | Deep Learning | Zaman serisi yeniden üretim hatası |
| 6 | **Stacking Ensemble** | Meta-Learning | Tüm modelleri birleştiren üst model |

### Kaçak Elektrik Senaryoları (5 Tür)

| Tür | Açıklama | Gerçek Dünya Karşılığı |
|-----|----------|------------------------|
| `constant_reduction` | Tüm tüketim %30-70 düşük | Sayaç manipülasyonu |
| `night_zeroing` | Gece 00-06 tüketimi sıfır | Kablo bypass |
| `random_zeros` | Rastgele günlerde sıfır | Sayaç durdurma |
| `gradual_decrease` | Aylık %5-10 azalma | Kademeli hırsızlık |
| `peak_clipping` | Pik tüketim kesilir | Akım sınırlayıcı |

### Özellik Mühendisliği (20+ Özellik)

- **İstatistiksel:** ortalama, std, min, max, medyan, çarpıklık, basıklık
- **Zamansal:** gece/gündüz oranı, hafta içi/sonu farkı, pik saati
- **Anomali göstergeleri:** sıfır ölçüm yüzdesi, ani değişim oranı, trend eğimi

---

## 📊 Sonuçlar / Results

### Model Karşılaştırması

| Model | ROC-AUC | F1 Score | AP | Tip |
|-------|---------|----------|----|-----|
| **Stacking Ensemble** | **0.9428** | **0.8727** | **0.9004** | Ensemble |
| Random Forest | 0.9461 | 0.8704 | — | Supervised |
| Gradient Boosting | 0.9380 | 0.8411 | — | Supervised |
| XGBoost | 0.9322 | 0.8440 | — | Supervised |
| Isolation Forest | 0.8208 | 0.2609 | — | Unsupervised |
| LSTM Autoencoder | 0.7482* | 0.5600* | — | Deep Learning |

*\*Müşteri bazlı değerlendirme. LSTM Autoencoder unsupervised çalışır — etiket gerektirmeden kaçak tespiti yapabilmesi en büyük avantajıdır.*

### En Önemli Özellikler (XGBoost Feature Importance)

1. `sudden_change_ratio` — Ani tüketim değişim oranı (0.240)
2. `mean_daily_total` — Günlük ortalama tüketim (0.115)
3. `kurtosis` — Tüketim dağılımı basıklığı (0.093)
4. `mean_consumption` — Ortalama tüketim (0.083)
5. `peak_hour` — Pik tüketim saati (0.054)

### Model Sonuç Grafikleri

<div align="center">
<img src="docs/model_results.png" alt="Model Results" width="800">
</div>

### LSTM Autoencoder Sonuçları

<div align="center">
<img src="docs/lstm_autoencoder_results.png" alt="LSTM Results" width="800">
</div>

---

## 📺 Dashboard

Streamlit tabanlı interaktif izleme paneli:

- 🗺️ Türkiye haritası üzerinde bölgesel anomali görselleştirmesi
- 📊 Risk dağılımı ve kaçak olasılık histogramları
- 🔎 Müşteri bazlı detaylı tüketim analizi
- 🚨 Otomatik alarm tablosu
- 📋 Kaçak türüne göre tespit başarısı analizi

<div align="center">
<img src="docs/dashboard_preview.png" alt="Dashboard Preview" width="800">
</div>

```bash
streamlit run dashboard/app.py
```

---

## 🔬 Teknik Detaylar

### Veri Seti
- **2000 müşteri**, 180 gün, 15 dakikalık aralıklarla tüketim verisi
- **%12 kaçak oranı** (Türkiye gerçeğine yakın)
- 3 müşteri profili: Konut (%70), Ticari (%20), Sanayi (%10)
- Gerçekçi günlük/haftalık/mevsimsel paternler

### LSTM Autoencoder Mimarisi
```
Encoder: Input(96,1) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(32)
Decoder: RepeatVector(96) → LSTM(32) → Dropout(0.2) → LSTM(64) → Dense(1)
Loss: MSE | Optimizer: Adam(lr=0.001) | EarlyStopping + ReduceLROnPlateau
```

---

## 🆕 v2.0 Yeni Özellikler

- **Stacking Ensemble** — XGBoost + Random Forest + Gradient Boosting → LogisticRegression meta-learner (F1: 0.873, AP: 0.900)
- **SHAP Açıklanabilirlik** — "Bu müşteri neden kaçak?" sorusuna cevap, özellik bazlı açıklama
- **Threshold Optimizasyonu** — F1-optimal, F2-optimal (recall ağırlıklı), maliyet-bazlı eşik değerleri
- **Veri Artırma** — SMOTE-benzeri minority oversampling ile dengesiz veri problemi çözümü
- **Risk Skoru (0-100)** — Her müşteri için sayısal risk skoru ve kategori
- **Pipeline Orchestrator** — `python run_pipeline.py --all` ile tek komutla tüm sistem

## 🗺️ Yol Haritası / Roadmap

- [x] Sentetik veri üretici
- [x] Isolation Forest (unsupervised)
- [x] XGBoost & Random Forest (supervised)
- [x] LSTM Autoencoder (deep learning)
- [x] Stacking Ensemble + SHAP
- [x] Streamlit Dashboard v2.0 (5 sekme)
- [x] IEEE formatında teknik makale taslağı
- [x] Pipeline orchestrator
- [ ] Gerçek veri seti entegrasyonu (SGCC / London Smart Meter)
- [ ] 1D-CNN gerilim anomali sınıflandırması
- [ ] ESP32 + CT sensör fiziksel prototipi

---

## 📄 Lisans / License

Bu proje MIT lisansı altında lisanslanmıştır.

---

## 👤 Yazar / Author

**Ömer Burak Koçak**

🎓 Marmara Üniversitesi, Elektrik-Elektronik Mühendisliği — 2026

---

## 📚 Referanslar / References

1. EPDK, "Akıllı Sayaç Sistemleri Yönetmeliği (MASS)," 2026.
2. T.C. Enerji ve Tabii Kaynaklar Bakanlığı, "2025 Bütçe Sunumu."
3. U.S. Department of Commerce, "Turkey Smart Grid Market Report," 2024.
4. PNNL, "AI Use Cases for Smart Grid," PNNL-38003, 2024.
5. Zheng et al., "Wide & Deep CNN for Electricity Theft Detection," IEEE Access, 2018.

---

<div align="center">

**MASS-AI** — Türkiye'nin Akıllı Enerji Geleceği İçin 🇹🇷⚡

</div>
