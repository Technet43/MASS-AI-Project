"""
MASS-AI Dashboard v2.1
======================

STATUS: Under active migration to use shared/core/dashboard_adapters.py
Old duplicated ML logic is being progressively removed.

This Streamlit dashboard has been updated to support a full bilingual user
interface without changing the underlying analytics workflow.

What this file now does:
- Manages the active UI language with ``st.session_state``.
- Defaults the application language to Turkish.
- Keeps every visible label inside a central ``TRANSLATIONS`` dictionary.
- Uses language-agnostic internal keys for filters and risk levels so the
  original functionality remains stable.
- Localizes tabs, filters, KPIs, map labels, alerts, buttons, chart titles,
  table columns, warnings, and helper texts.

Implementation note:
- The data pipeline, model execution flow, and chart logic are intentionally
  preserved. Only presentation and text rendering were refactored to add
  bilingual support and improve readability.
"""

from pathlib import Path
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT_DIR = DASHBOARD_DIR.parent.parent
CORE_DIR = ROOT_DIR / "shared" / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# Use the real shared engine instead of duplicating ML logic
try:
    from mass_ai_engine import MassAIEngine
    from mass_ai_domain import RISK_LABELS as ENGINE_RISK_LABELS
    ENGINE_AVAILABLE = True
except Exception as import_exc:
    MassAIEngine = None
    ENGINE_RISK_LABELS = ["low", "medium", "high", "critical"]
    ENGINE_AVAILABLE = False

PROFILE_OPTIONS = ["residential", "commercial", "industrial", "mixed_use"]
RISK_OPTIONS = ["low", "medium", "high", "critical"]
RISK_COLORS = {
    "low": "#27AE60",
    "medium": "#F39C12",
    "high": "#E67E22",
    "critical": "#E74C3C",
}
THEFT_TYPE_OPTIONS = [
    "constant_reduction",
    "night_zeroing",
    "random_zeros",
    "gradual_decrease",
    "peak_clipping",
    "weekend_masking",
    "intermittent_bypass",
    "tamper_spikes",
]

# Keep a minimal feature list for legacy upload paths; engine handles the real heavy lifting
FEATURE_COLUMNS = [
    "mean_consumption",
    "std_consumption",
    "min_consumption",
    "max_consumption",
    "median_consumption",
    "skewness",
    "kurtosis",
    "mean_daily_total",
    "std_daily_total",
    "cv_daily",
    "night_day_ratio",
    "weekend_weekday_ratio",
    "peak_hour",
    "zero_measurement_pct",
    "sudden_change_ratio",
    "trend_slope",
    "iqr",
]


TRANSLATIONS = {
    "tr": {
        "page_title": "MASS-AI | Akıllı Sayaç Anomali Tespiti",
        "sidebar_app_title": "MASS-AI v2.0",
        "sidebar_app_subtitle": "Akıllı Sayaç Anomali Tespiti",
        "language_toggle": "Türkçe / English",
        "language_help": "Kapalı: Türkçe, Açık: English",
        "filters_header": "Filtreler",
        "profile_filter": "Müşteri Profili",
        "risk_filter": "Risk Seviyesi",
        "threshold_filter": "Kaçak Olasılık Eşiği",
        "displayed_count": "Gösterilen",
        "project_info_header": "Proje Bilgisi",
        "project_author": "Ömer Burak Koçak",
        "project_school": "Marmara Üniversitesi EEE - 2026",
        "main_header": "MASS-AI Dashboard",
        "main_subheader": "Milli Akıllı Sayaç Sistemleri - Yapay Zeka Tabanlı Anomali Tespit ve Kaçak Elektrik Sınıflandırma v2.0",
        "tabs": {
            "overview": "Genel Bakış",
            "timeseries": "Zaman Serisi Karşılaştırma",
            "performance": "Model Performansı",
            "customer": "Müşteri Detay",
            "simulation": "Canlı Simülasyon",
        },
        "profiles": {
            "residential": "Konut",
            "commercial": "Ticari",
            "industrial": "Sanayi",
        },
        "risk_levels": {
            "low": "Düşük",
            "medium": "Orta",
            "high": "Yüksek",
            "critical": "Kritik",
        },
        "theft_types": {
            "none": {"short": "Normal", "chart": "Normal Desen", "detail": "Normal"},
            "constant_reduction": {
                "short": "Sabit Azaltma",
                "chart": "Sabit Azaltma<br>(Sayaç Manipülasyonu)",
                "detail": "Sabit Azaltma",
            },
            "night_zeroing": {
                "short": "Gece Sıfırlaması",
                "chart": "Gece Sıfırlaması<br>(Kablo Bypass)",
                "detail": "Gece Sıfırlaması",
            },
            "random_zeros": {
                "short": "Rastgele Sıfırlar",
                "chart": "Rastgele Sıfırlar<br>(Sayaç Durdurma)",
                "detail": "Rastgele Sıfırlar",
            },
            "gradual_decrease": {
                "short": "Kademeli Azalma",
                "chart": "Kademeli Azalma<br>(Yavaş Hırsızlık)",
                "detail": "Kademeli Azalma",
            },
            "peak_clipping": {
                "short": "Pik Kırpma",
                "chart": "Pik Kırpma<br>(Akım Sınırlandırma)",
                "detail": "Pik Kırpma",
            },
        },
        "feature_labels": {
            "mean_consumption": "Ortalama Tüketim",
            "std_consumption": "Standart Sapma",
            "min_consumption": "Minimum Tüketim",
            "max_consumption": "Maksimum Tüketim",
            "median_consumption": "Medyan Tüketim",
            "skewness": "Çarpıklık",
            "kurtosis": "Basıklık",
            "mean_daily_total": "Ortalama Günlük Toplam",
            "std_daily_total": "Günlük Toplam Std",
            "cv_daily": "Günlük Varyasyon Katsayısı",
            "night_day_ratio": "Gece/Gündüz Oranı",
            "weekend_weekday_ratio": "Hafta Sonu/Hafta İçi Oranı",
            "peak_hour": "Pik Saat",
            "zero_measurement_pct": "Sıfır Ölçüm Oranı",
            "zero_day_pct": "Sıfır Gün Oranı",
            "sudden_change_ratio": "Ani Değişim Oranı",
            "trend_slope": "Trend Eğimi",
            "q25": "25. Yüzdelik",
            "q75": "75. Yüzdelik",
            "iqr": "Çeyrekler Arası Aralık",
        },
        "overview": {
            "kpi_total_customers": "Toplam Müşteri",
            "kpi_detected_anomalies": "Tespit Edilen Anomali",
            "kpi_avg_score": "Ort. Anomali Skoru",
            "kpi_critical_alerts": "Kritik Uyarı",
            "delta_urgent": "Acil",
            "delta_clear": "Temiz",
            "empty_state": "Seçili filtrelerde gösterilecek kayıt bulunmuyor.",
            "map_title": "Bölgesel Anomali Haritası",
            "map_colorbar": "Kaçak Olasılığı",
            "risk_distribution_title": "Risk Seviyesi Dağılımı",
            "probability_distribution_title": "Kaçak Olasılık Dağılımı",
            "probability_axis": "Kaçak Olasılığı",
            "customer_count_axis": "Müşteri Sayısı",
            "alerts_title": "Anomali Alarmları",
            "no_alerts": "Seçilen eşik değerinde alarm bulunmuyor.",
            "alerts_summary": "{count} müşteri için kaçak şüphe alarmı (eşik >= {threshold})",
            "alert_columns": {
                "customer_id": "ID",
                "profile": "Profil",
                "theft_probability": "Kaçak Olasılığı",
                "risk_level": "Risk",
                "mean_consumption": "Ort. Tüketim",
                "zero_measurement_pct": "Sıfır %",
                "sudden_change_ratio": "Ani Değişim",
            },
        },
        "timeseries": {
            "title": "Zaman Serisi Karşılaştırma: Normal ve Kaçak",
            "subtitle": "Aynı profildeki normal ve kaçak müşterilerin tüketim desenlerini yan yana karşılaştırın.",
            "profile_select": "Müşteri Profili",
            "duration_select": "Gösterilecek Süre",
            "normal_customer_select": "Normal Müşteri Seç",
            "theft_customer_select": "Kaçak Müşteri Seç",
            "days_template": "{days} Gün",
            "insufficient_data": "Bu profil için yeterli veri yok (ilk 200 müşteri içinde).",
            "normal_subplot": "Normal Müşteri #{customer_id} (Risk: {risk})",
            "theft_subplot": "Kaçak Müşteri #{customer_id} - {theft_type} (Risk: {risk})",
            "legend_normal": "Normal",
            "legend_theft": "Kaçak",
            "legend_zero": "Sıfır Tüketim",
            "stats_title": "İstatistik Karşılaştırma",
            "stats_metric_column": "Metrik",
            "stats_normal_column": "Normal #{customer_id}",
            "stats_theft_column": "Kaçak #{customer_id}",
            "stats_metrics": [
                "Ort. Tüketim (kW)",
                "Std Tüketim",
                "Min Tüketim",
                "Max Tüketim",
                "Sıfır Ölçüm %",
                "Ani Değişim Oranı",
                "Gece/Gündüz Oranı",
                "Kaçak Olasılığı",
            ],
            "all_patterns_title": "Tüm Kaçak Türleri - Örnek Tüketim Desenleri",
        },
        "performance": {
            "title": "Model Performans Analizi",
            "roc_title": "ROC Eğrisi",
            "pr_title": "Precision-Recall Eğrisi",
            "roc_random": "Rastgele (AUC=0.500)",
            "pr_baseline": "Temel Seviye ({value})",
            "x_false_positive": "Yanlış Pozitif Oranı",
            "y_true_positive": "Doğru Pozitif Oranı",
            "x_recall": "Recall",
            "y_precision": "Precision",
            "rf_confusion_title": "Random Forest - Karmaşıklık Matrisi",
            "if_confusion_title": "Isolation Forest - Karmaşıklık Matrisi",
            "axis_predicted": "Tahmin",
            "axis_actual": "Gerçek",
            "class_normal": "Normal",
            "class_theft": "Kaçak",
            "feature_importance_title": "Özellik Önemliliği (Random Forest)",
            "importance_axis": "Önem Skoru",
            "detection_by_type_title": "Kaçak Türüne Göre Tespit Başarısı",
            "detection_rate_axis": "Tespit Oranı (%)",
            "model_table_title": "Model Karşılaştırma Tablosu",
            "model_table_caption": "*Müşteri bazlı değerlendirme",
            "model_table_columns": {
                "model": "Model",
                "type": "Tip",
                "roc_auc": "ROC-AUC",
                "f1": "F1 Score",
                "advantage": "Avantaj",
            },
            "model_table_rows": [
                {
                    "model": "Random Forest",
                    "type": "Denetimli",
                    "roc_auc": "0.9471",
                    "f1": "0.8704",
                    "advantage": "En yüksek doğruluk",
                },
                {
                    "model": "XGBoost",
                    "type": "Denetimli",
                    "roc_auc": "0.9373",
                    "f1": "0.8468",
                    "advantage": "Güçlü karar sınırları",
                },
                {
                    "model": "Isolation Forest",
                    "type": "Denetimsiz",
                    "roc_auc": "0.8208",
                    "f1": "0.2609",
                    "advantage": "Etiket gerektirmez",
                },
                {
                    "model": "LSTM Autoencoder",
                    "type": "Derin Öğrenme",
                    "roc_auc": "0.7482*",
                    "f1": "0.5600*",
                    "advantage": "Zaman serisi modelleme",
                },
            ],
        },
        "customer": {
            "title": "Müşteri Detay İnceleme",
            "view_label": "Gösterim",
            "view_modes": {
                "high_risk": "Yüksek Riskli",
                "all_customers": "Tüm Müşteriler",
            },
            "no_customers": "Bu filtrede müşteri yok.",
            "customer_select": "Müşteri Seç",
            "customer_option": "#{customer_id} ({risk})",
            "profile_label": "Profil",
            "risk_label": "Risk",
            "probability_label": "Kaçak Olasılığı",
            "score_label": "Anomali Skoru",
            "theft_type_label": "Kaçak Türü",
            "high_risk_box": "<strong>YÜKSEK RİSK</strong>",
            "normal_box": "Normal",
            "consumption_profile": "Tüketim Profili",
            "daily_total": "Günlük Toplam Tüketim",
            "legend_consumption": "Tüketim",
            "legend_zero": "Sıfır",
            "legend_daily": "Günlük",
            "no_raw_data": "Ham tüketim verisi mevcut değil (ilk 200 müşteri).",
            "feature_profile_title": "Müşteri Özellik Profili",
            "population_average": "Popülasyon Ort.",
            "customer_series": "Müşteri #{customer_id}",
            "radar_labels": [
                "Ort. Tüketim",
                "Std Sapma",
                "Sıfır %",
                "Ani Değişim",
                "Gece/Gündüz",
                "Günlük CV",
            ],
        },
        "simulation": {
            "title": "Canlı Simülasyon Modu",
            "subtitle": "Gerçek zamanlı akan veri simülasyonu - MASS sayaçlarından gelen veriyi canlı izleyin.",
            "speed_label": "Hız",
            "customers_label": "Müşteri Sayısı",
            "points_label": "Veri Noktası",
            "customer_select": "Simülasyon için Müşteri Seç",
            "speed_option": "{value}s aralık",
            "start_button": "Simülasyonu Başlat",
            "upload_section_title": "Dışarıdan Veri Yükle",
            "upload_section_subtitle": "CSV dosyası yükleyerek anomali analizi çalıştırın ve isterseniz bu veriyi simülasyonda kullanın.",
            "upload_label": "CSV dosyası yükle",
            "upload_help": "Beklenen sütunlar: timestamp, consumption veya consumption_kw, customer_id",
            "upload_missing_timestamp": "CSV içinde zaman bilgisi sütunu bulunamadı.",
            "upload_missing_consumption": "CSV içinde tüketim sütunu bulunamadı.",
            "upload_empty": "Yüklenen dosyada analiz edilebilecek veri bulunamadı.",
            "upload_error": "CSV işlenirken hata oluştu: {error}",
            "upload_results_title": "Yüklenen Veri Analizi",
            "upload_chart_title": "Yüklenen Veri Risk Skorları",
            "upload_table_title": "Yüklenen Veri Sonuç Tablosu",
            "upload_kpi_customers": "Yüklenen Müşteri",
            "upload_kpi_points": "Yüklenen Ölçüm",
            "upload_kpi_anomalies": "Riskli Müşteri",
            "upload_kpi_avg_risk": "Ort. Risk",
            "use_uploaded_button": "Bu veriyi simülasyonda kullan",
            "reset_source_button": "Varsayılan veriye dön",
            "active_uploaded_info": "Simülasyon şu anda yüklenen CSV verisini kullanıyor.",
            "active_default_info": "Simülasyon şu anda varsayılan örnek veriyi kullanıyor.",
            "uploaded_customer_axis": "Müşteri ID",
            "uploaded_risk_axis": "Kaçak Olasılığı",
            "uploaded_predicted_yes": "Evet",
            "uploaded_predicted_no": "Hayır",
            "uploaded_table_columns": {
                "customer_id": "Müşteri ID",
                "theft_probability": "Kaçak Olasılığı",
                "anomaly_score": "Anomali Skoru",
                "risk_level": "Risk Seviyesi",
                "predicted_theft": "Model Alarmı",
            },
            "metric_streaming": "Akan Ölçüm",
            "metric_anomalies": "Anomali Tespit",
            "metric_average": "Ort. Tüketim",
            "metric_alarms": "Alarm Sayısı",
            "legend_customer": "#{customer_id} - {status}",
            "status_normal": "Normal",
            "status_theft": "Kaçak",
            "alarm_delta": "Uyarı",
            "zero_alarm_warning": "{count} sıfır tüketim alarmı tespit edildi!",
            "completed": "Simülasyon tamamlandı! {points} ölçüm işlendi, {anomalies} anomali tespit edildi.",
            "preview_info": "Yukarıdaki butona basarak canlı simülasyonu başlatın. Simülasyon sırasında akan veriyi ve anomali tespitini gerçek zamanlı izleyebilirsiniz.",
            "preview_legend": "Önizleme",
            "preview_title": "Önizleme - Müşteri #{customer_id} (Risk: {risk})",
        },
        "units": {
            "kw": "kW",
            "kwh_day": "kWh/gün",
        },
        "footer": "MASS-AI v2.0 | Ömer Burak Koçak | Marmara Üniversitesi EEE | Mart 2026",
    },
    "en": {
        "page_title": "MASS-AI | Smart Meter Anomaly Detection",
        "sidebar_app_title": "MASS-AI v2.0",
        "sidebar_app_subtitle": "Smart Meter Anomaly Detection",
        "language_toggle": "Türkçe / English",
        "language_help": "Off: Türkçe, On: English",
        "filters_header": "Filters",
        "profile_filter": "Customer Profile",
        "risk_filter": "Risk Level",
        "threshold_filter": "Theft Probability Threshold",
        "displayed_count": "Displayed",
        "project_info_header": "Project Information",
        "project_author": "Omer Burak Kocak",
        "project_school": "Marmara University EEE - 2026",
        "main_header": "MASS-AI Dashboard",
        "main_subheader": "National Smart Metering Systems - AI-Powered Anomaly Detection and Electricity Theft Classification v2.0",
        "tabs": {
            "overview": "Overview",
            "timeseries": "Time-Series Comparison",
            "performance": "Model Performance",
            "customer": "Customer Detail",
            "simulation": "Live Simulation",
        },
        "profiles": {
            "residential": "Residential",
            "commercial": "Commercial",
            "industrial": "Industrial",
        },
        "risk_levels": {
            "low": "Low",
            "medium": "Medium",
            "high": "High",
            "critical": "Critical",
        },
        "theft_types": {
            "none": {"short": "Normal", "chart": "Normal Pattern", "detail": "Normal"},
            "constant_reduction": {
                "short": "Constant Reduction",
                "chart": "Constant Reduction<br>(Meter Manipulation)",
                "detail": "Constant Reduction",
            },
            "night_zeroing": {
                "short": "Night Zeroing",
                "chart": "Night Zeroing<br>(Cable Bypass)",
                "detail": "Night Zeroing",
            },
            "random_zeros": {
                "short": "Random Zeros",
                "chart": "Random Zeros<br>(Meter Interruption)",
                "detail": "Random Zeros",
            },
            "gradual_decrease": {
                "short": "Gradual Decrease",
                "chart": "Gradual Decrease<br>(Slow Theft)",
                "detail": "Gradual Decrease",
            },
            "peak_clipping": {
                "short": "Peak Clipping",
                "chart": "Peak Clipping<br>(Current Limiting)",
                "detail": "Peak Clipping",
            },
        },
        "feature_labels": {
            "mean_consumption": "Average Consumption",
            "std_consumption": "Standard Deviation",
            "min_consumption": "Minimum Consumption",
            "max_consumption": "Maximum Consumption",
            "median_consumption": "Median Consumption",
            "skewness": "Skewness",
            "kurtosis": "Kurtosis",
            "mean_daily_total": "Average Daily Total",
            "std_daily_total": "Daily Total Std",
            "cv_daily": "Daily Coefficient of Variation",
            "night_day_ratio": "Night/Day Ratio",
            "weekend_weekday_ratio": "Weekend/Weekday Ratio",
            "peak_hour": "Peak Hour",
            "zero_measurement_pct": "Zero Measurement Share",
            "zero_day_pct": "Zero-Day Share",
            "sudden_change_ratio": "Sudden Change Ratio",
            "trend_slope": "Trend Slope",
            "q25": "25th Percentile",
            "q75": "75th Percentile",
            "iqr": "Interquartile Range",
        },
        "overview": {
            "kpi_total_customers": "Total Customers",
            "kpi_detected_anomalies": "Detected Anomalies",
            "kpi_avg_score": "Avg. Anomaly Score",
            "kpi_critical_alerts": "Critical Alerts",
            "delta_urgent": "Urgent",
            "delta_clear": "Clear",
            "empty_state": "No records match the selected filters.",
            "map_title": "Regional Anomaly Map",
            "map_colorbar": "Theft Probability",
            "risk_distribution_title": "Risk Level Distribution",
            "probability_distribution_title": "Theft Probability Distribution",
            "probability_axis": "Theft Probability",
            "customer_count_axis": "Customer Count",
            "alerts_title": "Anomaly Alerts",
            "no_alerts": "No alerts were triggered at the selected threshold.",
            "alerts_summary": "Theft suspicion alert for {count} customers (threshold >= {threshold})",
            "alert_columns": {
                "customer_id": "ID",
                "profile": "Profile",
                "theft_probability": "Theft Probability",
                "risk_level": "Risk",
                "mean_consumption": "Avg. Consumption",
                "zero_measurement_pct": "Zero %",
                "sudden_change_ratio": "Sudden Change",
            },
        },
        "timeseries": {
            "title": "Time-Series Comparison: Normal vs Theft",
            "subtitle": "Compare the consumption patterns of normal and theft customers side by side within the same profile.",
            "profile_select": "Customer Profile",
            "duration_select": "Display Period",
            "normal_customer_select": "Select Normal Customer",
            "theft_customer_select": "Select Theft Customer",
            "days_template": "{days} Days",
            "insufficient_data": "Not enough data is available for this profile within the first 200 customers.",
            "normal_subplot": "Normal Customer #{customer_id} (Risk: {risk})",
            "theft_subplot": "Theft Customer #{customer_id} - {theft_type} (Risk: {risk})",
            "legend_normal": "Normal",
            "legend_theft": "Theft",
            "legend_zero": "Zero Consumption",
            "stats_title": "Statistical Comparison",
            "stats_metric_column": "Metric",
            "stats_normal_column": "Normal #{customer_id}",
            "stats_theft_column": "Theft #{customer_id}",
            "stats_metrics": [
                "Average Consumption (kW)",
                "Consumption Std",
                "Minimum Consumption",
                "Maximum Consumption",
                "Zero Measurement %",
                "Sudden Change Ratio",
                "Night/Day Ratio",
                "Theft Probability",
            ],
            "all_patterns_title": "All Theft Types - Sample Consumption Patterns",
        },
        "performance": {
            "title": "Model Performance Analysis",
            "roc_title": "ROC Curve",
            "pr_title": "Precision-Recall Curve",
            "roc_random": "Random (AUC=0.500)",
            "pr_baseline": "Baseline ({value})",
            "x_false_positive": "False Positive Rate",
            "y_true_positive": "True Positive Rate",
            "x_recall": "Recall",
            "y_precision": "Precision",
            "rf_confusion_title": "Random Forest - Confusion Matrix",
            "if_confusion_title": "Isolation Forest - Confusion Matrix",
            "axis_predicted": "Predicted",
            "axis_actual": "Actual",
            "class_normal": "Normal",
            "class_theft": "Theft",
            "feature_importance_title": "Feature Importance (Random Forest)",
            "importance_axis": "Importance Score",
            "detection_by_type_title": "Detection Performance by Theft Type",
            "detection_rate_axis": "Detection Rate (%)",
            "model_table_title": "Model Comparison Table",
            "model_table_caption": "*Customer-level evaluation",
            "model_table_columns": {
                "model": "Model",
                "type": "Type",
                "roc_auc": "ROC-AUC",
                "f1": "F1 Score",
                "advantage": "Advantage",
            },
            "model_table_rows": [
                {
                    "model": "Random Forest",
                    "type": "Supervised",
                    "roc_auc": "0.9471",
                    "f1": "0.8704",
                    "advantage": "Highest accuracy",
                },
                {
                    "model": "XGBoost",
                    "type": "Supervised",
                    "roc_auc": "0.9373",
                    "f1": "0.8468",
                    "advantage": "Strong decision boundaries",
                },
                {
                    "model": "Isolation Forest",
                    "type": "Unsupervised",
                    "roc_auc": "0.8208",
                    "f1": "0.2609",
                    "advantage": "No labels required",
                },
                {
                    "model": "LSTM Autoencoder",
                    "type": "Deep Learning",
                    "roc_auc": "0.7482*",
                    "f1": "0.5600*",
                    "advantage": "Time-series modeling",
                },
            ],
        },
        "customer": {
            "title": "Customer Detail Review",
            "view_label": "View",
            "view_modes": {
                "high_risk": "High Risk",
                "all_customers": "All Customers",
            },
            "no_customers": "No customers match this filter.",
            "customer_select": "Select Customer",
            "customer_option": "#{customer_id} ({risk})",
            "profile_label": "Profile",
            "risk_label": "Risk",
            "probability_label": "Theft Probability",
            "score_label": "Anomaly Score",
            "theft_type_label": "Theft Type",
            "high_risk_box": "<strong>HIGH RISK</strong>",
            "normal_box": "Normal",
            "consumption_profile": "Consumption Profile",
            "daily_total": "Daily Total Consumption",
            "legend_consumption": "Consumption",
            "legend_zero": "Zero",
            "legend_daily": "Daily",
            "no_raw_data": "Raw consumption data is not available for this customer (first 200 customers only).",
            "feature_profile_title": "Customer Feature Profile",
            "population_average": "Population Average",
            "customer_series": "Customer #{customer_id}",
            "radar_labels": [
                "Avg. Consumption",
                "Std Dev",
                "Zero %",
                "Sudden Change",
                "Night/Day",
                "Daily CV",
            ],
        },
        "simulation": {
            "title": "Live Simulation Mode",
            "subtitle": "Real-time streaming data simulation - monitor incoming MASS meter data live.",
            "speed_label": "Speed",
            "customers_label": "Customer Count",
            "points_label": "Data Points",
            "customer_select": "Select Customer for Simulation",
            "speed_option": "{value}s interval",
            "start_button": "Start Simulation",
            "upload_section_title": "Upload External Data",
            "upload_section_subtitle": "Upload a CSV file to run anomaly analysis and optionally use that data in the simulation.",
            "upload_label": "Upload CSV file",
            "upload_help": "Expected columns: timestamp, consumption or consumption_kw, customer_id",
            "upload_missing_timestamp": "No timestamp column could be found in the CSV file.",
            "upload_missing_consumption": "No consumption column could be found in the CSV file.",
            "upload_empty": "The uploaded file does not contain analyzable data.",
            "upload_error": "An error occurred while processing the CSV file: {error}",
            "upload_results_title": "Uploaded Data Analysis",
            "upload_chart_title": "Uploaded Data Risk Scores",
            "upload_table_title": "Uploaded Data Results Table",
            "upload_kpi_customers": "Uploaded Customers",
            "upload_kpi_points": "Uploaded Readings",
            "upload_kpi_anomalies": "High-Risk Customers",
            "upload_kpi_avg_risk": "Avg. Risk",
            "use_uploaded_button": "Use this data in simulation",
            "reset_source_button": "Return to default data",
            "active_uploaded_info": "The simulation is currently using the uploaded CSV data.",
            "active_default_info": "The simulation is currently using the default sample data.",
            "uploaded_customer_axis": "Customer ID",
            "uploaded_risk_axis": "Theft Probability",
            "uploaded_predicted_yes": "Yes",
            "uploaded_predicted_no": "No",
            "uploaded_table_columns": {
                "customer_id": "Customer ID",
                "theft_probability": "Theft Probability",
                "anomaly_score": "Anomaly Score",
                "risk_level": "Risk Level",
                "predicted_theft": "Model Alert",
            },
            "metric_streaming": "Streaming Readings",
            "metric_anomalies": "Detected Anomalies",
            "metric_average": "Avg. Consumption",
            "metric_alarms": "Alarm Count",
            "legend_customer": "#{customer_id} - {status}",
            "status_normal": "Normal",
            "status_theft": "Theft",
            "alarm_delta": "Alert",
            "zero_alarm_warning": "{count} zero-consumption alarms detected!",
            "completed": "Simulation completed! {points} readings processed, {anomalies} anomalies detected.",
            "preview_info": "Click the button above to start the live simulation. During the run you can monitor streaming data and anomaly detection in real time.",
            "preview_legend": "Preview",
            "preview_title": "Preview - Customer #{customer_id} (Risk: {risk})",
        },
        "units": {
            "kw": "kW",
            "kwh_day": "kWh/day",
        },
        "footer": "MASS-AI v2.0 | Omer Burak Kocak | Marmara University EEE | March 2026",
    },
}


def initialize_language_state():
    """Keep the UI language in session state and default to Turkish."""
    if "language" not in st.session_state:
        st.session_state.language = "tr"
    if "language_toggle" not in st.session_state:
        st.session_state.language_toggle = st.session_state.language == "en"
    else:
        st.session_state.language = "en" if st.session_state.language_toggle else "tr"


def get_translations():
    """Return the active translation bundle for the current session."""
    return TRANSLATIONS[st.session_state.language]


def get_profile_label(profile_key):
    return get_translations()["profiles"].get(profile_key, str(profile_key).title())


def get_risk_label(risk_key):
    return get_translations()["risk_levels"].get(risk_key, str(risk_key).title())


def get_theft_type_label(theft_type_key, variant="short"):
    theft_entry = get_translations()["theft_types"].get(theft_type_key)
    if isinstance(theft_entry, dict):
        return theft_entry.get(variant, theft_entry.get("short", str(theft_type_key)))
    return str(theft_entry or theft_type_key).replace("_", " ").title()


def get_feature_label(feature_key):
    return get_translations()["feature_labels"].get(
        feature_key,
        str(feature_key).replace("_", " ").title(),
    )


def hex_to_rgba(hex_color, alpha):
    """Convert a hex color like '#27AE60' into a Plotly-compatible rgba string."""
    hex_color = str(hex_color).lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(0, 0, 0, {alpha})"
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def find_first_matching_column(columns, aliases):
    """Return the first matching column name using case-insensitive aliases."""
    normalized_lookup = {str(column).strip().lower(): column for column in columns}
    for alias in aliases:
        if alias in normalized_lookup:
            return normalized_lookup[alias]
    return None


def normalize_uploaded_raw_data(uploaded_df):
    """
    LEGACY / DEPRECATED
    Use shared/core/dashboard_adapters or engine methods instead.
    """
    raise RuntimeError(
        "normalize_uploaded_raw_data is deprecated. "
        "Use shared/core/dashboard_adapters instead."
    )


def build_uploaded_features(raw_df):
    """
    LEGACY / DEPRECATED
    This function contained duplicated feature engineering logic.
    New development should use shared/core/dashboard_adapters.py or engine methods.
    """
    raise RuntimeError(
        "build_uploaded_features is deprecated. "
        "Use shared/core/dashboard_adapters or engine methods instead."
    )


def score_uploaded_features(reference_features_df, uploaded_features_df):
    """
    LEGACY / DEPRECATED
    This function contained duplicated model training and scoring logic.
    All new development must go through shared/core/dashboard_adapters.py.
    """
    raise RuntimeError(
        "score_uploaded_features is deprecated. "
        "Use shared/core/dashboard_adapters instead."
    )


def build_simulation_customer_pool(simulation_df, selected_customer_id, n_customers):
    """
    LEGACY / DEPRECATED
    Simulation logic should prefer the new adapter helpers.
    """
    raise RuntimeError(
        "build_simulation_customer_pool is deprecated. "
        "Use shared/core/dashboard_adapters instead."
    )

    extra_customers = remaining_customers.head(max(n_customers - 1, 0))
    return pd.concat(
        [selected_customer.to_frame().T, extra_customers],
        ignore_index=True,
    )


initialize_language_state()
t = get_translations()


def get_engine() -> "MassAIEngine | None":
    """
    LEGACY / DEPRECATED
    Use shared.core.dashboard_adapters.get_engine instead.
    """
    raise RuntimeError("get_engine is deprecated. Use shared/core/dashboard_adapters instead.")


def load_synthetic_via_engine(n_customers: int = 2000, n_days: int = 180, preset: str | None = None):
    """
    LEGACY / DEPRECATED - Use the adapter version instead.
    """
    raise RuntimeError("load_synthetic_via_engine is deprecated.")
    engine = get_engine()
    if engine is None:
        return None, "Engine not available - falling back to legacy in-file implementation"

    try:
        features = engine.generate_synthetic(n_customers=n_customers, n_days=n_days, preset_name=preset)
        engine.train_models()
        scored = engine.score_customers()
        overview = engine.build_overview() or {}
        return {
            "features": features,
            "scored": scored,
            "overview": overview,
            "engine": engine,
            "best_model": overview.get("best_model", "Stacking Ensemble"),
        }, None
    except Exception as exc:
        return None, f"Engine failed: {exc}"


def get_engine_metrics_and_df(engine_result: dict | None):
    """
    DEPRECATED - This function is no longer used in main paths.
    Use functions from shared.core.dashboard_adapters instead.
    """
    raise RuntimeError(
        "get_engine_metrics_and_df is deprecated. "
        "Use shared.core.dashboard_adapters instead."
    )


def prepare_simulation_data_from_engine(engine_scored_df):
    """
    LEGACY / DEPRECATED
    Use shared.core.dashboard_adapters.prepare_for_simulation or prepare_simulation_from_engine instead.
    """
    if engine_scored_df is None or engine_scored_df.empty:
        return None
    sim_df = engine_scored_df.copy()
    if "theft_probability" not in sim_df.columns and "risk_score" in sim_df.columns:
        sim_df["theft_probability"] = sim_df["risk_score"] / 100
    if "risk_level" not in sim_df.columns and "risk_category" in sim_df.columns:
        sim_df["risk_level"] = sim_df["risk_category"].astype(str)
    return sim_df


st.set_page_config(
    page_title=t["page_title"],
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .alert-box {
        background: #FDEDEC;
        border-left: 4px solid #E74C3C;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .safe-box {
        background: #EAFAF1;
        border-left: 4px solid #27AE60;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #E74C3C;
        border-radius: 50%;
        margin-right: 6px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


def build_fallback_raw_data(features_df):
    """
    LEGACY / DEPRECATED
    This function is no longer needed in the modern engine-first flow.
    """
    raise RuntimeError(
        "build_fallback_raw_data is deprecated."
    )
        std_value = float(getattr(row, "std_consumption", 0.2) or 0.2)
        theft_type = str(getattr(row, "theft_type", "none") or "none")
        label = int(getattr(row, "label", 0) or 0)
        rng = np.random.default_rng(int(customer_id) + 2026)

        signal = mean_value
        signal += np.sin(np.linspace(0, 6 * np.pi, periods)) * max(std_value * 0.6, 0.04)
        signal += rng.normal(0, max(std_value * 0.18, 0.03), periods)
        signal = np.clip(signal, 0, None)

        if label == 1:
            if theft_type == "night_zeroing":
                signal[::8] = 0
            elif theft_type == "random_zeros":
                zero_mask = rng.choice([0, 1], size=periods, p=[0.9, 0.1]).astype(bool)
                signal[zero_mask] = 0
            elif theft_type == "constant_reduction":
                signal *= 0.45
            elif theft_type == "gradual_decrease":
                signal *= np.linspace(1.0, 0.55, periods)
            elif theft_type == "peak_clipping":
                clip_level = np.quantile(signal, 0.72)
                signal = np.minimum(signal, clip_level)

        series_frames.append(
            pd.DataFrame(
                {
                    "customer_id": customer_id,
                    "timestamp": timestamps,
                    "consumption_kw": signal,
                }
            )
        )

    return pd.concat(series_frames, ignore_index=True)


@st.cache_data
def load_data(use_engine: bool = True, n_customers: int = 1500, n_days: int = 120, preset: str | None = None):
    """
    Main data loading entry point.
    Strongly prefers the engine via adapters. Legacy CSV fallback is kept only for compatibility.
    """
    if use_engine:
        # Strongly prefer the new clean adapter from shared/core
        try:
            from shared.core.dashboard_adapters import load_synthetic_data_via_engine
            result = load_synthetic_data_via_engine(n_customers=n_customers, n_days=n_days, preset=preset)
            if "scored" in result and "error" not in result:
                scored = result["scored"].copy()
                if "risk_level" not in scored.columns and "risk_category" in scored.columns:
                    scored["risk_level"] = scored["risk_category"].astype(str)
                if "anomaly_score" not in scored.columns and "risk_score" in scored.columns:
                    scored["anomaly_score"] = scored["risk_score"]
                # Store engine data for simulation etc.
                if "overview" in result:
                    st.session_state["engine_data"] = result
                return scored, None
        except Exception:
            pass

    # Final fallback to pre-generated CSVs (should be rare)
    base = ROOT_DIR / "shared" / "data" / "processed"
    features = pd.read_csv(base / "features.csv")
    raw_path = base / "raw_consumption_sample.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    else:
        raw = pd.DataFrame()
    return features, raw


@st.cache_data
def run_models(features_df):
    """
    LEGACY / DEPRECATED - Bu fonksiyon kullanılmamalıdır.
    Tüm yeni geliştirme için shared/core/dashboard_adapters.py kullanın.
    """
    raise RuntimeError("run_models is deprecated. Use shared/core/dashboard_adapters instead.")


def render_sidebar(features_df):
    """Render sidebar controls and return the filtered frame plus threshold."""
    st.sidebar.toggle(t["language_toggle"], key="language_toggle")
    st.session_state.language = "en" if st.session_state.language_toggle else "tr"
    local_t = get_translations()

    st.sidebar.caption(local_t["language_help"])
    st.sidebar.markdown(f"## ⚡ {local_t['sidebar_app_title']}")
    st.sidebar.markdown(f"*{local_t['sidebar_app_subtitle']}*")

    # Show when we're using the real shared engine (ongoing anti-duplication effort)
    if ENGINE_AVAILABLE and get_engine() is not None:
        st.sidebar.success("✓ Advanced Engine active (shared/core)", icon="🚀")
        # Show quick engine stats if available
        if st.session_state.get("engine_data"):
            overview = st.session_state["engine_data"].get("overview", {})
            if overview.get("high_risk_count"):
                st.sidebar.caption(f"High risk: {overview.get('high_risk_count')} | Best model: {st.session_state['engine_data'].get('best_model', 'Engine')}")
    else:
        st.sidebar.warning("Using legacy in-file models", icon="⚠️")

    st.sidebar.markdown("---")

    st.sidebar.markdown(f"### {local_t['filters_header']}")
    profile_filter = st.sidebar.multiselect(
        local_t["profile_filter"],
        options=PROFILE_OPTIONS,
        default=PROFILE_OPTIONS,
        format_func=get_profile_label,
    )

    risk_filter = st.sidebar.multiselect(
        local_t["risk_filter"],
        options=RISK_OPTIONS,
        default=RISK_OPTIONS,
        format_func=get_risk_label,
    )

    prob_threshold = st.sidebar.slider(local_t["threshold_filter"], 0.0, 1.0, 0.5, 0.05)

    # Engine preset selector (leverages the strong synthetic presets in shared/core)
    engine = get_engine()
    if engine is not None and ENGINE_AVAILABLE:
        preset_options = ["Turkey Urban", "Industrial Theft Sweep", "Mixed Retail Anomalies", "Rural Meter Drift"]
        selected_preset = st.sidebar.selectbox("Sentetik Veri Preseti (Engine)", preset_options, index=0)

        if st.sidebar.button("🔄 Bu preset ile yeniden üret", use_container_width=True):
            with st.spinner("MassAIEngine ile yeni sentetik veri üretiliyor..."):
                fresh, err = load_synthetic_via_engine(n_customers=1200, n_days=90, preset=selected_preset)
                if fresh:
                    st.session_state["engine_data"] = fresh
                    st.success("Yeni veri engine üzerinden üretildi. Sayfayı yenileyin veya simülasyonu başlatın.")
                else:
                    st.error(f"Engine üretimi başarısız: {err}")
    else:
        selected_preset = None

    filtered = features_df[
        (features_df["profile"].isin(profile_filter))
        & (features_df["risk_level"].isin(risk_filter))
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**{local_t['displayed_count']}:** {len(filtered)} / {len(features_df)}"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {local_t['project_info_header']}")
    st.sidebar.markdown(f"**{local_t['project_author']}**")
    st.sidebar.markdown(local_t["project_school"])

    return filtered, prob_threshold


def render_overview(df, threshold, raw_df):
    local_t = get_translations()
    total = len(df)
    detected = int((df["theft_probability"] >= threshold).sum()) if total else 0
    detection_rate = (detected / total * 100) if total else 0
    critical = int((df["risk_level"] == "critical").sum()) if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(local_t["overview"]["kpi_total_customers"], f"{total:,}")
    col2.metric(
        local_t["overview"]["kpi_detected_anomalies"],
        f"{detected}",
        delta=f"{detection_rate:.1f}%",
        delta_color="inverse",
    )
    col3.metric(
        local_t["overview"]["kpi_avg_score"],
        f"{df['anomaly_score'].mean():.3f}" if total else "0.000",
    )
    col4.metric(
        local_t["overview"]["kpi_critical_alerts"],
        f"{critical}",
        delta=(
            local_t["overview"]["delta_urgent"]
            if critical > 0
            else local_t["overview"]["delta_clear"]
        ),
        delta_color="inverse" if critical > 0 else "normal",
    )

    if total == 0:
        st.markdown("---")
        st.warning(local_t["overview"]["empty_state"])
        return

    st.markdown("---")
    st.markdown(f"### {local_t['overview']['map_title']}")

    np.random.seed(42)
    cities = {
        "Istanbul": (41.01, 28.97, 0.35),
        "Ankara": (39.93, 32.86, 0.15),
        "Izmir": (38.42, 27.14, 0.10),
        "Diyarbakir": (37.91, 40.22, 0.10),
        "Antalya": (36.90, 30.69, 0.08),
        "Adana": (37.00, 35.32, 0.07),
        "Bursa": (40.19, 29.06, 0.08),
        "Gaziantep": (37.06, 37.38, 0.07),
    }

    lats, lons, city_names = [], [], []
    for _ in range(len(df)):
        city = np.random.choice(list(cities.keys()), p=[value[2] for value in cities.values()])
        lat_base, lon_base, _ = cities[city]
        lats.append(lat_base + np.random.normal(0, 0.3))
        lons.append(lon_base + np.random.normal(0, 0.3))
        city_names.append(city)

    map_df = df.copy()
    map_df["lat"] = lats
    map_df["lon"] = lons
    map_df["city"] = city_names
    map_df["profile_display"] = map_df["profile"].apply(get_profile_label)
    map_df["risk_level_display"] = map_df["risk_level"].apply(get_risk_label)

    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="theft_probability",
        size="anomaly_score",
        color_continuous_scale="RdYlGn_r",
        range_color=[0, 1],
        size_max=12,
        zoom=5,
        center={"lat": 39.0, "lon": 35.0},
        mapbox_style="carto-positron",
        hover_data={
            "customer_id": True,
            "profile_display": True,
            "theft_probability": ":.2f",
            "risk_level_display": True,
            "lat": False,
            "lon": False,
        },
        labels={
            "customer_id": local_t["overview"]["alert_columns"]["customer_id"],
            "profile_display": local_t["overview"]["alert_columns"]["profile"],
            "theft_probability": local_t["overview"]["map_colorbar"],
            "risk_level_display": local_t["overview"]["alert_columns"]["risk_level"],
        },
        height=450,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {local_t['overview']['risk_distribution_title']}")
        risk_counts = (
            df["risk_level"].value_counts().reindex(RISK_OPTIONS, fill_value=0)
        )
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[get_risk_label(level) for level in risk_counts.index],
                    values=risk_counts.values,
                    hole=0.5,
                    marker_colors=[RISK_COLORS[level] for level in risk_counts.index],
                    textinfo="label+percent",
                    textfont_size=13,
                )
            ]
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"### {local_t['overview']['probability_distribution_title']}")
        fig = go.Figure()
        color_map = {
            "residential": "#1B4F72",
            "commercial": "#2E86C1",
            "industrial": "#85C1E9",
        }
        for profile in PROFILE_OPTIONS:
            subset = df[df["profile"] == profile]
            fig.add_trace(
                go.Histogram(
                    x=subset["theft_probability"],
                    name=get_profile_label(profile),
                    marker_color=color_map[profile],
                    opacity=0.7,
                    nbinsx=30,
                )
            )
        fig.update_layout(
            barmode="overlay",
            height=350,
            xaxis_title=local_t["overview"]["probability_axis"],
            yaxis_title=local_t["overview"]["customer_count_axis"],
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### {local_t['overview']['alerts_title']}")
    alerts = df[df["theft_probability"] >= threshold].sort_values(
        "theft_probability",
        ascending=False,
    )

    if len(alerts) == 0:
        st.success(local_t["overview"]["no_alerts"])
        return

    st.warning(
        local_t["overview"]["alerts_summary"].format(
            count=len(alerts),
            threshold=f"{threshold:.0%}",
        )
    )

    display_df = alerts[
        [
            "customer_id",
            "profile",
            "theft_probability",
            "risk_level",
            "mean_consumption",
            "zero_measurement_pct",
            "sudden_change_ratio",
        ]
    ].copy()
    display_df["profile"] = display_df["profile"].apply(get_profile_label)
    display_df["risk_level"] = display_df["risk_level"].apply(get_risk_label)
    display_df["theft_probability"] = display_df["theft_probability"].apply(lambda value: f"{value:.1%}")
    display_df["zero_measurement_pct"] = display_df["zero_measurement_pct"].apply(
        lambda value: f"{value:.1%}"
    )
    display_df["sudden_change_ratio"] = display_df["sudden_change_ratio"].apply(
        lambda value: f"{value:.4f}"
    )
    display_df["mean_consumption"] = display_df["mean_consumption"].apply(
        lambda value: f"{value:.2f} {local_t['units']['kw']}"
    )
    display_df = display_df.rename(columns=local_t["overview"]["alert_columns"])
    st.dataframe(display_df, use_container_width=True, height=350)


def render_timeseries_comparison(df, raw_df):
    local_t = get_translations()
    st.markdown(f"### {local_t['timeseries']['title']}")
    st.markdown(f"*{local_t['timeseries']['subtitle']}*")

    col1, col2 = st.columns([1, 1])
    with col1:
        profile_sel = st.selectbox(
            local_t["timeseries"]["profile_select"],
            PROFILE_OPTIONS,
            format_func=get_profile_label,
        )
    with col2:
        days_sel = st.selectbox(
            local_t["timeseries"]["duration_select"],
            [3, 7, 14, 30],
            index=1,
            format_func=lambda value: local_t["timeseries"]["days_template"].format(days=value),
        )

    normal_pool = df[
        (df["label"] == 0)
        & (df["profile"] == profile_sel)
        & (df["customer_id"] < 200)
    ]
    theft_pool = df[
        (df["label"] == 1)
        & (df["profile"] == profile_sel)
        & (df["customer_id"] < 200)
    ]

    if len(normal_pool) == 0 or len(theft_pool) == 0:
        st.info(local_t["timeseries"]["insufficient_data"])
        return

    normal_ids = normal_pool["customer_id"].sort_values().tolist()
    theft_ids = theft_pool["customer_id"].sort_values().tolist()

    selector_col1, selector_col2 = st.columns([1, 1])
    with selector_col1:
        selected_normal_id = st.selectbox(
            local_t["timeseries"]["normal_customer_select"],
            normal_ids,
        )
    with selector_col2:
        selected_theft_id = st.selectbox(
            local_t["timeseries"]["theft_customer_select"],
            theft_ids,
        )

    normal_cust = normal_pool[normal_pool["customer_id"] == selected_normal_id].iloc[0]
    theft_cust = theft_pool[theft_pool["customer_id"] == selected_theft_id].iloc[0]
    n_points = days_sel * 96

    normal_raw = raw_df[raw_df["customer_id"] == normal_cust["customer_id"]].head(n_points)
    theft_raw = raw_df[raw_df["customer_id"] == theft_cust["customer_id"]].head(n_points)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            local_t["timeseries"]["normal_subplot"].format(
                customer_id=int(normal_cust["customer_id"]),
                risk=f"{normal_cust['theft_probability']:.0%}",
            ),
            local_t["timeseries"]["theft_subplot"].format(
                customer_id=int(theft_cust["customer_id"]),
                theft_type=get_theft_type_label(theft_cust["theft_type"], "detail"),
                risk=f"{theft_cust['theft_probability']:.0%}",
            ),
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=normal_raw["timestamp"],
            y=normal_raw["consumption_kw"],
            mode="lines",
            line=dict(color="#27AE60", width=1),
            fill="tozeroy",
            fillcolor="rgba(39,174,96,0.1)",
            name=local_t["timeseries"]["legend_normal"],
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=theft_raw["timestamp"],
            y=theft_raw["consumption_kw"],
            mode="lines",
            line=dict(color="#E74C3C", width=1),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.1)",
            name=local_t["timeseries"]["legend_theft"],
            showlegend=True,
        ),
        row=2,
        col=1,
    )

    zero_points = theft_raw[theft_raw["consumption_kw"] < 0.01]
    if len(zero_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=zero_points["timestamp"],
                y=zero_points["consumption_kw"],
                mode="markers",
                marker=dict(color="#F39C12", size=4, symbol="x"),
                name=local_t["timeseries"]["legend_zero"],
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(height=550, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(title_text=local_t["units"]["kw"], row=1, col=1)
    fig.update_yaxes(title_text=local_t["units"]["kw"], row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"#### {local_t['timeseries']['stats_title']}")
    comp_data = {
        local_t["timeseries"]["stats_metric_column"]: local_t["timeseries"]["stats_metrics"],
        local_t["timeseries"]["stats_normal_column"].format(
            customer_id=int(normal_cust["customer_id"])
        ): [
            f"{normal_cust['mean_consumption']:.3f}",
            f"{normal_cust['std_consumption']:.3f}",
            f"{normal_cust['min_consumption']:.3f}",
            f"{normal_cust['max_consumption']:.3f}",
            f"{normal_cust['zero_measurement_pct']:.1%}",
            f"{normal_cust['sudden_change_ratio']:.4f}",
            f"{normal_cust['night_day_ratio']:.3f}",
            f"{normal_cust['theft_probability']:.1%}",
        ],
        local_t["timeseries"]["stats_theft_column"].format(
            customer_id=int(theft_cust["customer_id"])
        ): [
            f"{theft_cust['mean_consumption']:.3f}",
            f"{theft_cust['std_consumption']:.3f}",
            f"{theft_cust['min_consumption']:.3f}",
            f"{theft_cust['max_consumption']:.3f}",
            f"{theft_cust['zero_measurement_pct']:.1%}",
            f"{theft_cust['sudden_change_ratio']:.4f}",
            f"{theft_cust['night_day_ratio']:.3f}",
            f"{theft_cust['theft_probability']:.1%}",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"### {local_t['timeseries']['all_patterns_title']}")

    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=[
            get_theft_type_label(theft_type, "chart") for theft_type in THEFT_TYPE_OPTIONS
        ],
    )
    colors = ["#E74C3C", "#E67E22", "#F39C12", "#8E44AD", "#2980B9"]

    for index, theft_type in enumerate(THEFT_TYPE_OPTIONS, start=1):
        tt_customers = df[(df["theft_type"] == theft_type) & (df["customer_id"] < 200)]
        if len(tt_customers) == 0:
            continue
        customer_id = tt_customers.iloc[0]["customer_id"]
        cust_raw = raw_df[raw_df["customer_id"] == customer_id].head(96 * 3)
        color = colors[index - 1]
        fig.add_trace(
            go.Scatter(
                y=cust_raw["consumption_kw"].values,
                mode="lines",
                line=dict(color=color, width=1),
                fill="tozeroy",
                fillcolor=(
                    f"rgba({int(color[1:3], 16)},"
                    f"{int(color[3:5], 16)},"
                    f"{int(color[5:7], 16)},0.1)"
                ),
                name=get_theft_type_label(theft_type),
                showlegend=False,
            ),
            row=1,
            col=index,
        )

    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    for index in range(1, 6):
        fig.update_yaxes(
            title_text=local_t["units"]["kw"] if index == 1 else "",
            row=1,
            col=index,
        )
        fig.update_xaxes(title_text="", showticklabels=False, row=1, col=index)
    st.plotly_chart(fig, use_container_width=True)


def render_model_performance(df, metrics):
    local_t = get_translations()
    st.markdown(f"### {local_t['performance']['title']}")

    # New path: When using the real MassAIEngine, show richer info
    if metrics and metrics.get("engine_mode"):
        st.success(f"🚀 Using advanced engine — Best model: **{metrics.get('best_model', 'Stacking Ensemble')}**")

        overview = metrics.get("overview", {})
        if overview:
            c1, c2, c3 = st.columns(3)
            c1.metric("High Risk Customers", overview.get("high_risk_count", "-"))
            c2.metric("Critical Cases", overview.get("critical_count", "-"))
            c3.metric("Est. Monthly Exposure", f"{overview.get('total_loss', 0):,.0f} ₺")

        # Prefer the new adapter helper for performance data when available
        try:
            from shared.core.dashboard_adapters import get_engine_scored_for_performance
            perf_data = get_engine_scored_for_performance({"overview": metrics.get("overview", {}), "best_model": metrics.get("best_model")})
            if perf_data.get("summary"):
                st.caption("Performance summary powered by shared/core adapters")
        except Exception:
            pass

        engine_res = metrics.get("engine_results", {})
        if engine_res:
            st.markdown("**Engine Model Performance (from shared/core):**")
            cols = st.columns(min(4, len(engine_res)))
            for i, (model_name, res) in enumerate(list(engine_res.items())[:4]):
                auc = res.get("auc", 0)
                f1 = res.get("f1", 0)
                cols[i].metric(model_name, f"AUC {auc:.3f}", f"F1 {f1:.3f}")

        if "best_model" in metrics:
            st.caption(f"Primary model powering this view: **{metrics['best_model']}**")

        st.info("Detailed ROC/PR curves for the full 6-model ensemble coming in follow-up work.")
        st.markdown("---")
        if "rf_auc" not in metrics:
            return
    else:
        # Legacy path metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RF ROC-AUC", f"{metrics.get('rf_auc', 0):.4f}")
        col2.metric("RF F1 Score", f"{metrics.get('rf_f1', 0):.4f}")
        col3.metric("IF ROC-AUC", f"{metrics.get('iso_auc', 0):.4f}")
        col4.metric("IF F1 Score", f"{metrics.get('iso_f1', 0):.4f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### {local_t['performance']['roc_title']}")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics["rf_fpr"],
                y=metrics["rf_tpr"],
                mode="lines",
                name=f"Random Forest (AUC={metrics['rf_auc']:.3f})",
                line=dict(color="#2E86C1", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metrics["iso_fpr"],
                y=metrics["iso_tpr"],
                mode="lines",
                name=f"Isolation Forest (AUC={metrics['iso_auc']:.3f})",
                line=dict(color="#E67E22", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name=local_t["performance"]["roc_random"],
                line=dict(color="gray", width=1, dash="dash"),
            )
        )
        fig.update_layout(
            xaxis_title=local_t["performance"]["x_false_positive"],
            yaxis_title=local_t["performance"]["y_true_positive"],
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.35, y=0.1, bgcolor="rgba(255,255,255,0.8)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### {local_t['performance']['pr_title']}")
        baseline = (metrics["y_test"] == 1).sum() / len(metrics["y_test"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics["rf_rec"],
                y=metrics["rf_prec"],
                mode="lines",
                name="Random Forest",
                line=dict(color="#2E86C1", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metrics["iso_rec"],
                y=metrics["iso_prec"],
                mode="lines",
                name="Isolation Forest",
                line=dict(color="#E67E22", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[baseline, baseline],
                mode="lines",
                name=local_t["performance"]["pr_baseline"].format(value=f"{baseline:.2f}"),
                line=dict(color="gray", width=1, dash="dash"),
            )
        )
        fig.update_layout(
            xaxis_title=local_t["performance"]["x_recall"],
            yaxis_title=local_t["performance"]["y_precision"],
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.02, y=0.1, bgcolor="rgba(255,255,255,0.8)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    class_labels = [
        local_t["performance"]["class_normal"],
        local_t["performance"]["class_theft"],
    ]

    with col1:
        st.markdown(f"#### {local_t['performance']['rf_confusion_title']}")
        rf_cm = metrics["rf_cm"]
        fig = go.Figure(
            data=go.Heatmap(
                z=rf_cm,
                x=class_labels,
                y=class_labels,
                colorscale="Blues",
                showscale=False,
                text=rf_cm,
                texttemplate="%{text}",
                textfont={"size": 22},
            )
        )
        fig.update_layout(
            xaxis_title=local_t["performance"]["axis_predicted"],
            yaxis_title=local_t["performance"]["axis_actual"],
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### {local_t['performance']['if_confusion_title']}")
        iso_cm = metrics["iso_cm"]
        fig = go.Figure(
            data=go.Heatmap(
                z=iso_cm,
                x=class_labels,
                y=class_labels,
                colorscale="Oranges",
                showscale=False,
                text=iso_cm,
                texttemplate="%{text}",
                textfont={"size": 22},
            )
        )
        fig.update_layout(
            xaxis_title=local_t["performance"]["axis_predicted"],
            yaxis_title=local_t["performance"]["axis_actual"],
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### {local_t['performance']['feature_importance_title']}")
    importance = metrics["importance"]
    top_features = sorted(importance.items(), key=lambda item: item[1], reverse=True)[:12]
    feature_names = [get_feature_label(item[0]) for item in top_features][::-1]
    feature_values = [item[1] for item in top_features][::-1]

    fig = go.Figure(
        go.Bar(
            x=feature_values,
            y=feature_names,
            orientation="h",
            marker=dict(color=feature_values, colorscale="Blues", showscale=False),
            text=[f"{value:.3f}" for value in feature_values],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=80, t=20, b=20),
        xaxis_title=local_t["performance"]["importance_axis"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### {local_t['performance']['detection_by_type_title']}")
    col1, col2 = st.columns(2)

    with col1:
        theft_df = df[df["label"] == 1]
        type_detection = (
            theft_df.groupby("theft_type")
            .agg(total=("label", "count"), detected=("predicted_theft", "sum"))
            .reset_index()
        )
        type_detection["rate"] = type_detection["detected"] / type_detection["total"] * 100
        type_detection["theft_type_display"] = type_detection["theft_type"].apply(
            get_theft_type_label
        )

        fig = go.Figure(
            go.Bar(
                x=type_detection["theft_type_display"],
                y=type_detection["rate"],
                marker_color=["#1B4F72", "#2E86C1", "#5DADE2", "#85C1E9", "#AED6F1"],
                text=[f"{value:.0f}%" for value in type_detection["rate"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            yaxis_title=local_t["performance"]["detection_rate_axis"],
            yaxis_range=[0, 110],
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"#### {local_t['performance']['model_table_title']}")
        model_columns = local_t["performance"]["model_table_columns"]
        model_data = pd.DataFrame(local_t["performance"]["model_table_rows"]).rename(
            columns=model_columns
        )
        st.dataframe(model_data, use_container_width=True, hide_index=True)
        st.caption(local_t["performance"]["model_table_caption"])


def render_customer_detail(df, raw_df):
    local_t = get_translations()
    st.markdown(f"### {local_t['customer']['title']}")

    col1, col2 = st.columns([1, 3])

    with col1:
        view_mode = st.radio(
            local_t["customer"]["view_label"],
            ["high_risk", "all_customers"],
            format_func=lambda option: local_t["customer"]["view_modes"][option],
            label_visibility="collapsed",
        )

        if view_mode == "high_risk":
            pool = df[df["theft_probability"] > 0.5].sort_values(
                "theft_probability",
                ascending=False,
            )
        else:
            pool = df.sort_values("customer_id")

        if len(pool) == 0:
            st.info(local_t["customer"]["no_customers"])
            return

        selected_id = st.selectbox(
            local_t["customer"]["customer_select"],
            options=pool["customer_id"].tolist(),
            format_func=lambda customer_id: local_t["customer"]["customer_option"].format(
                customer_id=customer_id,
                risk=f"{df[df['customer_id'] == customer_id]['theft_probability'].values[0]:.0%}",
            ),
        )

        cust = df[df["customer_id"] == selected_id].iloc[0]
        st.markdown(f"**{local_t['customer']['profile_label']}:** {get_profile_label(cust['profile'])}")
        st.markdown(f"**{local_t['customer']['risk_label']}:** {get_risk_label(cust['risk_level'])}")
        st.markdown(
            f"**{local_t['customer']['probability_label']}:** {cust['theft_probability']:.1%}"
        )
        st.markdown(f"**{local_t['customer']['score_label']}:** {cust['anomaly_score']:.3f}")

        if cust["label"] == 1:
            st.markdown(
                f"**{local_t['customer']['theft_type_label']}:** "
                f"{get_theft_type_label(cust['theft_type'], 'detail')}"
            )

        if cust["theft_probability"] > 0.7:
            st.markdown(
                f'<div class="alert-box">{local_t["customer"]["high_risk_box"]}</div>',
                unsafe_allow_html=True,
            )
        elif cust["theft_probability"] < 0.3:
            st.markdown(
                f'<div class="safe-box">{local_t["customer"]["normal_box"]}</div>',
                unsafe_allow_html=True,
            )

    with col2:
        cust_raw = raw_df[raw_df["customer_id"] == selected_id]
        if len(cust_raw) > 0:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=[
                    local_t["customer"]["consumption_profile"],
                    local_t["customer"]["daily_total"],
                ],
                row_heights=[0.6, 0.4],
            )

            fig.add_trace(
                go.Scatter(
                    x=cust_raw["timestamp"],
                    y=cust_raw["consumption_kw"],
                    mode="lines",
                    line=dict(color="#2E86C1", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(46,134,193,0.1)",
                    name=local_t["customer"]["legend_consumption"],
                ),
                row=1,
                col=1,
            )

            zeros = cust_raw[cust_raw["consumption_kw"] < 0.01]
            if len(zeros) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=zeros["timestamp"],
                        y=zeros["consumption_kw"],
                        mode="markers",
                        marker=dict(color="red", size=3),
                        name=local_t["customer"]["legend_zero"],
                    ),
                    row=1,
                    col=1,
                )

            daily = (
                cust_raw.set_index("timestamp")
                .resample("D")["consumption_kw"]
                .sum()
                .reset_index()
            )
            bar_colors = [
                "#E74C3C" if value < daily["consumption_kw"].mean() * 0.3 else "#2E86C1"
                for value in daily["consumption_kw"]
            ]
            fig.add_trace(
                go.Bar(
                    x=daily["timestamp"],
                    y=daily["consumption_kw"],
                    marker_color=bar_colors,
                    name=local_t["customer"]["legend_daily"],
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
            )
            fig.update_yaxes(title_text=local_t["units"]["kw"], row=1, col=1)
            fig.update_yaxes(title_text=local_t["units"]["kwh_day"], row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(local_t["customer"]["no_raw_data"])

    st.markdown("---")
    st.markdown(f"#### {local_t['customer']['feature_profile_title']}")

    features_to_show = [
        "mean_consumption",
        "std_consumption",
        "zero_measurement_pct",
        "sudden_change_ratio",
        "night_day_ratio",
        "cv_daily",
    ]
    radar_labels = local_t["customer"]["radar_labels"]

    cust_vals = []
    pop_vals = []
    for feature_name in features_to_show:
        feature_min = df[feature_name].min()
        feature_max = df[feature_name].max()
        cust_vals.append((cust[feature_name] - feature_min) / (feature_max - feature_min + 1e-8))
        pop_vals.append((df[feature_name].mean() - feature_min) / (feature_max - feature_min + 1e-8))

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=pop_vals + [pop_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(46,134,193,0.1)",
            line=dict(color="#2E86C1"),
            name=local_t["customer"]["population_average"],
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=cust_vals + [cust_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="#E74C3C"),
            name=local_t["customer"]["customer_series"].format(customer_id=int(selected_id)),
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=400,
        margin=dict(l=60, r=60, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_live_simulation(df, raw_df):
    local_t = get_translations()
    st.markdown(
        (
            '<p style="font-size:1.3rem; font-weight:600;">'
            '<span class="live-indicator"></span>'
            f"{local_t['simulation']['title']}"
            "</p>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(f"*{local_t['simulation']['subtitle']}*")

    uploaded_override_active = st.session_state.get("simulation_source") == "uploaded"
    simulation_df = st.session_state.get("simulation_features_override", df) if uploaded_override_active else df
    simulation_raw = st.session_state.get("simulation_raw_override", raw_df) if uploaded_override_active else raw_df

    # Prefer the new clean adapter for simulation when engine data is present
    if not uploaded_override_active and st.session_state.get("engine_data"):
        try:
            from shared.core.dashboard_adapters import prepare_simulation_from_engine
            sim_data = prepare_simulation_from_engine(st.session_state["engine_data"], n_customers=6)
            if "simulation_df" in sim_data and len(sim_data["simulation_df"]) > 0:
                simulation_df = sim_data["simulation_df"]
        except Exception:
            pass  # silently fall back to whatever df we have

    if uploaded_override_active:
        st.success(local_t["simulation"]["active_uploaded_info"])
        if st.button(local_t["simulation"]["reset_source_button"], use_container_width=True):
            st.session_state.pop("simulation_source", None)
            st.session_state.pop("simulation_features_override", None)
            st.session_state.pop("simulation_raw_override", None)
            st.rerun()
    else:
        st.caption(local_t["simulation"]["active_default_info"])

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_speed = st.selectbox(
            local_t["simulation"]["speed_label"],
            [0.05, 0.1, 0.2, 0.5],
            index=1,
            format_func=lambda value: local_t["simulation"]["speed_option"].format(value=value),
        )
    with col2:
        max_customers = max(1, min(8, len(simulation_df)))
        customer_options = [count for count in [3, 5, 8] if count <= max_customers]
        if not customer_options:
            customer_options = [max_customers]
        n_customers = st.selectbox(
            local_t["simulation"]["customers_label"],
            customer_options,
            index=min(1, len(customer_options) - 1),
        )
    with col3:
        sim_points = st.selectbox(
            local_t["simulation"]["points_label"],
            [50, 100, 200],
            index=1,
        )

    all_customer_ids = simulation_df["customer_id"].sort_values().tolist()
    selected_customer_id = st.selectbox(
        local_t["simulation"]["customer_select"],
        all_customer_ids,
    )
    selected_customer = simulation_df[simulation_df["customer_id"] == selected_customer_id].iloc[0]

    # Use a simple safe fallback instead of the deprecated function
    try:
        sim_customers = simulation_df[simulation_df["customer_id"] != selected_customer_id].head(n_customers)
        if len(sim_customers) == 0:
            sim_customers = simulation_df.head(n_customers)
    except Exception:
        sim_customers = simulation_df.head(n_customers)

    st.markdown("---")
    st.markdown(f"#### {local_t['simulation']['upload_section_title']}")
    st.caption(local_t["simulation"]["upload_section_subtitle"])
    uploaded_file = st.file_uploader(
        local_t["simulation"]["upload_label"],
        type=["csv"],
        help=local_t["simulation"]["upload_help"],
    )

    if uploaded_file is not None:
        st.info(
            "Yüklenen dosya ile simülasyon şu anda bakım aşamasındadır. "
            "Lütfen ana dashboard üzerinden engine ile sentetik veri kullanın veya "
            "gerçek veri entegrasyonu için ekip ile iletişime geçin. "
            "(Eski duplicate feature engineering kaldırıldı — yeni adaptör ile yeniden yazılacak.)"
        )

    if st.button(local_t["simulation"]["start_button"], type="primary", use_container_width=True):
        metric_cols = st.columns(4)
        metric_cols[0].markdown(f"**{local_t['simulation']['metric_streaming']}**")
        metric_cols[1].markdown(f"**{local_t['simulation']['metric_anomalies']}**")
        metric_cols[2].markdown(f"**{local_t['simulation']['metric_average']}**")
        metric_cols[3].markdown(f"**{local_t['simulation']['metric_alarms']}**")

        metric_stream = metric_cols[0].empty()
        metric_anomaly = metric_cols[1].empty()
        metric_avg = metric_cols[2].empty()
        metric_alarm = metric_cols[3].empty()

        chart_placeholder = st.empty()
        alert_placeholder = st.empty()
        progress = st.progress(0)

        alarm_count = 0
        anomaly_count = 0
        total_consumption = 0

        customer_data = {}
        for _, customer in sim_customers.iterrows():
            customer_id = customer["customer_id"]
            customer_raw = simulation_raw[simulation_raw["customer_id"] == customer_id].head(sim_points)
            customer_data[customer_id] = {
                "values": customer_raw["consumption_kw"].values,
                "label": int(customer.get("predicted_theft", customer.get("label", 0))),
                "profile": customer.get("profile", "residential"),
                "buffer_x": [],
                "buffer_y": [],
            }

        for step in range(sim_points):
            fig = make_subplots(
                rows=len(sim_customers),
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
            )

            for row_index, (customer_id, customer_payload) in enumerate(customer_data.items(), start=1):
                if step >= len(customer_payload["values"]):
                    continue

                value = customer_payload["values"][step]
                customer_payload["buffer_x"].append(step)
                customer_payload["buffer_y"].append(value)
                total_consumption += value

                color = "#E74C3C" if customer_payload["label"] == 1 else "#27AE60"
                status_label = (
                    local_t["simulation"]["status_theft"]
                    if customer_payload["label"] == 1
                    else local_t["simulation"]["status_normal"]
                )
                trace_name = local_t["simulation"]["legend_customer"].format(
                    customer_id=int(customer_id),
                    status=status_label,
                )

                fig.add_trace(
                    go.Scatter(
                        x=customer_payload["buffer_x"],
                        y=customer_payload["buffer_y"],
                        mode="lines",
                        line=dict(color=color, width=1.5),
                        fill="tozeroy",
                        fillcolor=hex_to_rgba(color, 0.08),
                        name=trace_name,
                        showlegend=True,
                    ),
                    row=row_index,
                    col=1,
                )

                if value < 0.01 and customer_payload["label"] == 1:
                    anomaly_count += 1
                    fig.add_trace(
                        go.Scatter(
                            x=[step],
                            y=[value],
                            mode="markers",
                            marker=dict(color="red", size=8, symbol="x"),
                            showlegend=False,
                        ),
                        row=row_index,
                        col=1,
                    )

                if value < 0.01 and step > 5:
                    alarm_count += 1

            fig.update_layout(
                height=80 * len(sim_customers) + 100,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(orientation="h", y=1.02),
            )

            for row_index in range(1, len(sim_customers) + 1):
                fig.update_yaxes(title_text=local_t["units"]["kw"], row=row_index, col=1)

            chart_placeholder.plotly_chart(fig, use_container_width=True)

            metric_stream.metric(
                local_t["simulation"]["metric_streaming"],
                f"{step + 1}/{sim_points}",
            )
            metric_anomaly.metric(
                local_t["simulation"]["metric_anomalies"],
                f"{anomaly_count}",
            )
            avg_consumption = total_consumption / max((step + 1) * len(sim_customers), 1)
            metric_avg.metric(
                local_t["simulation"]["metric_average"],
                f"{avg_consumption:.2f} {local_t['units']['kw']}",
            )
            metric_alarm.metric(
                local_t["simulation"]["metric_alarms"],
                f"{alarm_count}",
                delta=local_t["simulation"]["alarm_delta"] if alarm_count > 0 else "",
            )

            if alarm_count > 0 and step % 20 == 0 and step > 0:
                alert_placeholder.warning(
                    local_t["simulation"]["zero_alarm_warning"].format(count=alarm_count)
                )

            progress.progress((step + 1) / sim_points)
            time.sleep(sim_speed)

        progress.empty()
        st.success(
            local_t["simulation"]["completed"].format(
                points=sim_points,
                anomalies=anomaly_count,
            )
        )
    else:
        st.markdown("---")
        st.info(local_t["simulation"]["preview_info"])

        if len(sim_customers) == 0:
            return

        preview_customer = selected_customer
        preview_raw = simulation_raw[
            simulation_raw["customer_id"] == preview_customer["customer_id"]
        ].head(96 * 3)
        if len(preview_raw) > 0:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=preview_raw["timestamp"],
                    y=preview_raw["consumption_kw"],
                    mode="lines",
                    line=dict(color="#2E86C1", width=1),
                    fill="tozeroy",
                    fillcolor="rgba(46,134,193,0.1)",
                    name=local_t["simulation"]["preview_legend"],
                )
            )
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                title=local_t["simulation"]["preview_title"].format(
                    customer_id=int(preview_customer["customer_id"]),
                    risk=f"{preview_customer['theft_probability']:.0%}",
                ),
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    local_t = get_translations()

    st.markdown(f'<p class="main-header">⚡ {local_t["main_header"]}</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="sub-header">{local_t["main_subheader"]}</p>',
        unsafe_allow_html=True,
    )

    # Prefer the real shared MassAIEngine (big step toward removing duplication)
    # === Preferred Path: Use MassAIEngine via clean adapters ===
    engine_data, raw_df = load_data(use_engine=True)

    used_engine = False

    if raw_df is None and isinstance(engine_data, dict):
        # Clean modern path - use the high-level adapter
        try:
            from shared.core.dashboard_adapters import prepare_dashboard_data
            prepared = prepare_dashboard_data(engine_data)
            features_df = prepared.get("scored_df", pd.DataFrame())
            metrics = prepared.get("metrics", {})
            used_engine = True
        except Exception:
            # Rare fallback
            features_df, raw_df = load_data(use_engine=False)
            features_df, metrics = run_models(features_df)
    else:
        # Legacy / non-engine data path
        if raw_df is None:
            features_df = engine_data if engine_data is not None else pd.DataFrame()
        else:
            features_df = engine_data if engine_data is not None else pd.DataFrame()

        try:
            from shared.core.dashboard_adapters import run_engine_based_scoring
            features_df, metrics = run_engine_based_scoring(engine_data if isinstance(engine_data, dict) else {})
        except Exception:
            features_df, metrics = run_models(features_df)
    filtered_df, threshold = render_sidebar(features_df)
    local_t = get_translations()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            local_t["tabs"]["overview"],
            local_t["tabs"]["timeseries"],
            local_t["tabs"]["performance"],
            local_t["tabs"]["customer"],
            local_t["tabs"]["simulation"],
        ]
    )

    with tab1:
        render_overview(filtered_df, threshold, raw_df)

    with tab2:
        render_timeseries_comparison(features_df, raw_df)

    with tab3:
        render_model_performance(features_df, metrics)

    with tab4:
        render_customer_detail(features_df, raw_df)

    with tab5:
        render_live_simulation(features_df, raw_df)

    st.markdown("---")
    st.markdown(
        (
            "<div style='text-align:center; color:#999; font-size:0.85rem;'>"
            f"{local_t['footer']}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
