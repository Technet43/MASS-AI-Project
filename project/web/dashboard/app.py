"""
MASS-AI Dashboard v2.0
======================
Akilli Sayac Verileri Icin Gercek Zamanli Anomali Izleme Paneli

Ozellikler:
- Bolgesel anomali haritasi
- Risk dagilimi grafikleri
- Musteri detay inceleme
- Alarm tablosu
- Kacak turu analizi
- [YENi] Zaman serisi karsilastirma (normal vs kacak)
- [YENi] Model performans sayfasi (ROC, confusion matrix, PR curve)
- [YENi] Canli simulasyon modu (akan veri efekti)

Yazar: Omer Burak Kocak
Calistirma: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DASHBOARD_DIR.parent.parent
CORE_DIR = PROJECT_DIR / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

# ========== SAYFA AYARLARI ==========
st.set_page_config(
    page_title="MASS-AI | Akilli Sayac Anomali Tespiti",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
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
""", unsafe_allow_html=True)


# ========== VERI YUKLEME ==========
def build_fallback_raw_data(features_df):
    periods = 96 * 3
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="15min")
    series_frames = []

    for row in features_df.itertuples(index=False):
        customer_id = getattr(row, "customer_id")
        mean_value = float(getattr(row, "mean_consumption", 1.0) or 1.0)
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
def load_data():
    base = PROJECT_DIR / 'data' / 'processed'
    features = pd.read_csv(base / 'features.csv')
    raw_path = base / 'raw_consumption_sample.csv'
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    else:
        raw = build_fallback_raw_data(features)
    return features, raw


@st.cache_data
def run_models(features_df):
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                                 roc_auc_score, f1_score, classification_report)

    meta_cols = ['customer_id', 'profile', 'label', 'theft_type']
    feature_cols = [c for c in features_df.columns if c not in meta_cols]
    X = features_df[feature_cols].values
    y = features_df['label'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
    )

    # Isolation Forest
    iso = IsolationForest(n_estimators=200, contamination=0.12, random_state=42)
    iso.fit(X_scaled)
    iso_scores = -iso.score_samples(X_scaled)
    iso_preds_all = (iso.predict(X_scaled) == -1).astype(int)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_probs_all = rf.predict_proba(X_scaled)[:, 1]
    rf_preds_all = rf.predict(X_scaled)

    # Test seti metrikleri
    rf_probs_test = rf.predict_proba(X_test)[:, 1]
    rf_preds_test = rf.predict(X_test)
    iso_scores_test = -iso.score_samples(X_test)
    iso_preds_test = (iso.predict(X_test) == -1).astype(int)

    # ROC curves
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs_test)
    iso_fpr, iso_tpr, _ = roc_curve(y_test, iso_scores_test)

    # PR curves
    rf_prec, rf_rec, _ = precision_recall_curve(y_test, rf_probs_test)
    iso_prec, iso_rec, _ = precision_recall_curve(y_test, iso_scores_test)

    # Confusion matrices
    rf_cm = confusion_matrix(y_test, rf_preds_test)
    iso_cm = confusion_matrix(y_test, iso_preds_test)

    # AUC & F1
    rf_auc = roc_auc_score(y_test, rf_probs_test)
    iso_auc = roc_auc_score(y_test, iso_scores_test)
    rf_f1 = f1_score(y_test, rf_preds_test)
    iso_f1 = f1_score(y_test, iso_preds_test)

    # Feature importance
    importance = dict(zip(feature_cols, rf.feature_importances_))

    features_df = features_df.copy()
    features_df['anomaly_score'] = iso_scores
    features_df['theft_probability'] = rf_probs_all
    features_df['predicted_theft'] = rf_preds_all
    features_df['risk_level'] = pd.cut(
        features_df['theft_probability'],
        bins=[0, 0.3, 0.6, 0.85, 1.0],
        labels=['Dusuk', 'Orta', 'Yuksek', 'Kritik']
    )

    metrics = {
        'rf_fpr': rf_fpr, 'rf_tpr': rf_tpr, 'rf_auc': rf_auc, 'rf_f1': rf_f1, 'rf_cm': rf_cm,
        'iso_fpr': iso_fpr, 'iso_tpr': iso_tpr, 'iso_auc': iso_auc, 'iso_f1': iso_f1, 'iso_cm': iso_cm,
        'rf_prec': rf_prec, 'rf_rec': rf_rec, 'iso_prec': iso_prec, 'iso_rec': iso_rec,
        'importance': importance, 'feature_cols': feature_cols,
        'y_test': y_test, 'rf_probs_test': rf_probs_test, 'iso_scores_test': iso_scores_test,
    }

    return features_df, metrics


# ========== SIDEBAR ==========
def render_sidebar(features_df):
    st.sidebar.markdown("## ⚡ MASS-AI v2.0")
    st.sidebar.markdown("*Akilli Sayac Anomali Tespiti*")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Filtreler")
    profile_filter = st.sidebar.multiselect(
        "Musteri Profili",
        options=['residential', 'commercial', 'industrial'],
        default=['residential', 'commercial', 'industrial'],
        format_func=lambda x: {'residential': '🏠 Konut', 'commercial': '🏢 Ticari', 'industrial': '🏭 Sanayi'}[x]
    )

    risk_filter = st.sidebar.multiselect(
        "Risk Seviyesi",
        options=['Dusuk', 'Orta', 'Yuksek', 'Kritik'],
        default=['Dusuk', 'Orta', 'Yuksek', 'Kritik']
    )

    prob_threshold = st.sidebar.slider("Kacak Olasilik Esigi", 0.0, 1.0, 0.5, 0.05)

    filtered = features_df[
        (features_df['profile'].isin(profile_filter)) &
        (features_df['risk_level'].isin(risk_filter))
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Gosterilen:** {len(filtered)} / {len(features_df)}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Proje Bilgisi")
    st.sidebar.markdown("**Omer Burak Kocak**")
    st.sidebar.markdown("Marmara Uni. EEE — 2026")

    return filtered, prob_threshold


# ========== TAB 1: GENEL BAKIS ==========
def render_overview(df, threshold, raw_df):
    # KPI Cards
    total = len(df)
    detected = (df['theft_probability'] >= threshold).sum()
    detection_rate = detected / total * 100
    critical = (df['risk_level'] == 'Kritik').sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Musteri", f"{total:,}")
    col2.metric("Tespit Edilen Anomali", f"{detected}", delta=f"%{detection_rate:.1f}", delta_color="inverse")
    col3.metric("Ort. Anomali Skoru", f"{df['anomaly_score'].mean():.3f}")
    col4.metric("Kritik Uyari", f"{critical}", delta="Acil" if critical > 0 else "Temiz", delta_color="inverse" if critical > 0 else "normal")

    st.markdown("---")

    # Harita
    st.markdown("### 🗺️ Bolgesel Anomali Haritasi")
    np.random.seed(42)
    cities = {
        'Istanbul': (41.01, 28.97, 0.35), 'Ankara': (39.93, 32.86, 0.15),
        'Izmir': (38.42, 27.14, 0.10), 'Diyarbakir': (37.91, 40.22, 0.10),
        'Antalya': (36.90, 30.69, 0.08), 'Adana': (37.00, 35.32, 0.07),
        'Bursa': (40.19, 29.06, 0.08), 'Gaziantep': (37.06, 37.38, 0.07),
    }
    lats, lons, city_names = [], [], []
    for _ in range(len(df)):
        city = np.random.choice(list(cities.keys()), p=[v[2] for v in cities.values()])
        lat_base, lon_base, _ = cities[city]
        lats.append(lat_base + np.random.normal(0, 0.3))
        lons.append(lon_base + np.random.normal(0, 0.3))
        city_names.append(city)

    map_df = df.copy()
    map_df['lat'] = lats
    map_df['lon'] = lons
    map_df['city'] = city_names

    fig = px.scatter_mapbox(
        map_df, lat='lat', lon='lon', color='theft_probability',
        size='anomaly_score', color_continuous_scale='RdYlGn_r', range_color=[0, 1],
        size_max=12, zoom=5, center={'lat': 39.0, 'lon': 35.0},
        mapbox_style='carto-positron',
        hover_data={'customer_id': True, 'profile': True, 'theft_probability': ':.2f', 'risk_level': True, 'lat': False, 'lon': False},
        labels={'theft_probability': 'Kacak Olasiligi'}, height=450
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Risk dagilimi
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Risk Seviyesi Dagilimi")
        risk_counts = df['risk_level'].value_counts()
        colors = {'Dusuk': '#27AE60', 'Orta': '#F39C12', 'Yuksek': '#E67E22', 'Kritik': '#E74C3C'}
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index, values=risk_counts.values, hole=0.5,
            marker_colors=[colors.get(r, '#999') for r in risk_counts.index],
            textinfo='label+percent', textfont_size=13
        )])
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📈 Kacak Olasilik Dagilimi")
        fig = go.Figure()
        for profile, color in [('residential', '#1B4F72'), ('commercial', '#2E86C1'), ('industrial', '#85C1E9')]:
            subset = df[df['profile'] == profile]
            fig.add_trace(go.Histogram(
                x=subset['theft_probability'],
                name={'residential': 'Konut', 'commercial': 'Ticari', 'industrial': 'Sanayi'}[profile],
                marker_color=color, opacity=0.7, nbinsx=30
            ))
        fig.update_layout(barmode='overlay', height=350,
                         xaxis_title='Kacak Olasiligi', yaxis_title='Musteri Sayisi',
                         margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Alarm tablosu
    st.markdown("### 🚨 Anomali Alarmlari")
    alerts = df[df['theft_probability'] >= threshold].sort_values('theft_probability', ascending=False)
    if len(alerts) == 0:
        st.success("Secilen esik degerinde alarm bulunmuyor.")
    else:
        st.warning(f"**{len(alerts)} musteri** icin kacak suphe alarmi (esik >= {threshold:.0%})")
        display_df = alerts[['customer_id', 'profile', 'theft_probability', 'risk_level',
                             'mean_consumption', 'zero_measurement_pct', 'sudden_change_ratio']].copy()
        display_df.columns = ['ID', 'Profil', 'Kacak Olasiligi', 'Risk', 'Ort. Tuketim', 'Sifir %', 'Ani Degisim']
        display_df['Kacak Olasiligi'] = display_df['Kacak Olasiligi'].apply(lambda x: f"{x:.1%}")
        display_df['Sifir %'] = display_df['Sifir %'].apply(lambda x: f"{x:.1%}")
        display_df['Ani Degisim'] = display_df['Ani Degisim'].apply(lambda x: f"{x:.4f}")
        display_df['Ort. Tuketim'] = display_df['Ort. Tuketim'].apply(lambda x: f"{x:.2f} kW")
        st.dataframe(display_df, use_container_width=True, height=350)


# ========== TAB 2: ZAMAN SERISI KARSILASTIRMA ==========
def render_timeseries_comparison(df, raw_df):
    st.markdown("### 📉 Zaman Serisi Karsilastirma: Normal vs Kacak")
    st.markdown("*Ayni profildeki normal ve kacak musterilerin tuketim paternlerini yan yana karsilastirin.*")

    col1, col2 = st.columns([1, 1])

    with col1:
        profile_sel = st.selectbox("Musteri Profili", ['residential', 'commercial', 'industrial'],
                                    format_func=lambda x: {'residential': '🏠 Konut', 'commercial': '🏢 Ticari', 'industrial': '🏭 Sanayi'}[x])

    with col2:
        days_sel = st.selectbox("Gosterilecek Sure", [3, 7, 14, 30], index=1, format_func=lambda x: f"{x} Gun")

    # Normal ve kacak musteri sec
    normal_pool = df[(df['label'] == 0) & (df['profile'] == profile_sel) & (df['customer_id'] < 200)]
    theft_pool = df[(df['label'] == 1) & (df['profile'] == profile_sel) & (df['customer_id'] < 200)]

    if len(normal_pool) == 0 or len(theft_pool) == 0:
        st.info("Bu profil icin yeterli veri yok (ilk 200 musteri icinde).")
        return

    normal_cust = normal_pool.iloc[0]
    theft_cust = theft_pool.iloc[0]

    n_points = days_sel * 96

    # Normal
    normal_raw = raw_df[raw_df['customer_id'] == normal_cust['customer_id']].head(n_points)
    theft_raw = raw_df[raw_df['customer_id'] == theft_cust['customer_id']].head(n_points)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=[
            f"✅ Normal Musteri #{int(normal_cust['customer_id'])} (Risk: {normal_cust['theft_probability']:.0%})",
            f"⚠️ Kacak Musteri #{int(theft_cust['customer_id'])} — {theft_cust['theft_type']} (Risk: {theft_cust['theft_probability']:.0%})"
        ]
    )

    # Normal tuketim
    fig.add_trace(go.Scatter(
        x=normal_raw['timestamp'], y=normal_raw['consumption_kw'],
        mode='lines', line=dict(color='#27AE60', width=1),
        fill='tozeroy', fillcolor='rgba(39,174,96,0.1)', name='Normal', showlegend=True
    ), row=1, col=1)

    # Kacak tuketim
    fig.add_trace(go.Scatter(
        x=theft_raw['timestamp'], y=theft_raw['consumption_kw'],
        mode='lines', line=dict(color='#E74C3C', width=1),
        fill='tozeroy', fillcolor='rgba(231,76,60,0.1)', name='Kacak', showlegend=True
    ), row=2, col=1)

    # Sifir noktalarini isaretle
    zero_points = theft_raw[theft_raw['consumption_kw'] < 0.01]
    if len(zero_points) > 0:
        fig.add_trace(go.Scatter(
            x=zero_points['timestamp'], y=zero_points['consumption_kw'],
            mode='markers', marker=dict(color='#F39C12', size=4, symbol='x'),
            name='Sifir Tuketim', showlegend=True
        ), row=2, col=1)

    fig.update_layout(height=550, hovermode='x unified', margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(title_text='kW', row=1, col=1)
    fig.update_yaxes(title_text='kW', row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Istatistik karsilastirma tablosu
    st.markdown("#### 📋 Istatistik Karsilastirma")
    comp_data = {
        'Metrik': ['Ort. Tuketim (kW)', 'Std Tuketim', 'Min Tuketim', 'Max Tuketim',
                   'Sifir Olcum %', 'Ani Degisim Orani', 'Gece/Gunduz Orani', 'Kacak Olasiligi'],
        f'Normal #{int(normal_cust["customer_id"])}': [
            f"{normal_cust['mean_consumption']:.3f}", f"{normal_cust['std_consumption']:.3f}",
            f"{normal_cust['min_consumption']:.3f}", f"{normal_cust['max_consumption']:.3f}",
            f"{normal_cust['zero_measurement_pct']:.1%}", f"{normal_cust['sudden_change_ratio']:.4f}",
            f"{normal_cust['night_day_ratio']:.3f}", f"{normal_cust['theft_probability']:.1%}"
        ],
        f'Kacak #{int(theft_cust["customer_id"])}': [
            f"{theft_cust['mean_consumption']:.3f}", f"{theft_cust['std_consumption']:.3f}",
            f"{theft_cust['min_consumption']:.3f}", f"{theft_cust['max_consumption']:.3f}",
            f"{theft_cust['zero_measurement_pct']:.1%}", f"{theft_cust['sudden_change_ratio']:.4f}",
            f"{theft_cust['night_day_ratio']:.3f}", f"{theft_cust['theft_probability']:.1%}"
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # Kacak turleri grid
    st.markdown("---")
    st.markdown("### 🔍 Tum Kacak Turleri — Ornek Tuketim Paternleri")

    theft_types = ['constant_reduction', 'night_zeroing', 'random_zeros', 'gradual_decrease', 'peak_clipping']
    type_labels = {
        'constant_reduction': 'Sabit Azaltma\n(Sayac Manipulasyonu)',
        'night_zeroing': 'Gece Sifirlamasi\n(Kablo Bypass)',
        'random_zeros': 'Rastgele Sifirlar\n(Sayac Durdurma)',
        'gradual_decrease': 'Kademeli Azalma\n(Yavas Hirsizlik)',
        'peak_clipping': 'Pik Kirpma\n(Akim Sinirlandirma)',
    }

    fig = make_subplots(rows=1, cols=5, subplot_titles=[type_labels[t].replace('\n', ' ') for t in theft_types])
    colors = ['#E74C3C', '#E67E22', '#F39C12', '#8E44AD', '#2980B9']

    for i, tt in enumerate(theft_types):
        tt_cust = df[(df['theft_type'] == tt) & (df['customer_id'] < 200)]
        if len(tt_cust) > 0:
            cid = tt_cust.iloc[0]['customer_id']
            cust_raw = raw_df[raw_df['customer_id'] == cid].head(96 * 3)  # 3 gun
            fig.add_trace(go.Scatter(
                y=cust_raw['consumption_kw'].values, mode='lines',
                line=dict(color=colors[i], width=1), fill='tozeroy',
                fillcolor=f'rgba({int(colors[i][1:3],16)},{int(colors[i][3:5],16)},{int(colors[i][5:7],16)},0.1)',
                name=tt, showlegend=False
            ), row=1, col=i+1)

    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    for i in range(1, 6):
        fig.update_yaxes(title_text='kW' if i == 1 else '', row=1, col=i)
        fig.update_xaxes(title_text='', showticklabels=False, row=1, col=i)
    st.plotly_chart(fig, use_container_width=True)


# ========== TAB 3: MODEL PERFORMANSI ==========
def render_model_performance(df, metrics):
    st.markdown("### 🧠 Model Performans Analizi")

    # Karsilastirma KPI
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RF ROC-AUC", f"{metrics['rf_auc']:.4f}")
    col2.metric("RF F1 Score", f"{metrics['rf_f1']:.4f}")
    col3.metric("IF ROC-AUC", f"{metrics['iso_auc']:.4f}")
    col4.metric("IF F1 Score", f"{metrics['iso_f1']:.4f}")

    st.markdown("---")

    # ROC ve PR curves
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics['rf_fpr'], y=metrics['rf_tpr'], mode='lines',
                                  name=f"Random Forest (AUC={metrics['rf_auc']:.3f})",
                                  line=dict(color='#2E86C1', width=2.5)))
        fig.add_trace(go.Scatter(x=metrics['iso_fpr'], y=metrics['iso_tpr'], mode='lines',
                                  name=f"Isolation Forest (AUC={metrics['iso_auc']:.3f})",
                                  line=dict(color='#E67E22', width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Random (AUC=0.500)',
                                  line=dict(color='gray', width=1, dash='dash')))
        fig.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            height=400, margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.4, y=0.1, bgcolor='rgba(255,255,255,0.8)')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Precision-Recall Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics['rf_rec'], y=metrics['rf_prec'], mode='lines',
                                  name='Random Forest', line=dict(color='#2E86C1', width=2.5)))
        fig.add_trace(go.Scatter(x=metrics['iso_rec'], y=metrics['iso_prec'], mode='lines',
                                  name='Isolation Forest', line=dict(color='#E67E22', width=2.5)))
        baseline = (metrics['y_test'] == 1).sum() / len(metrics['y_test'])
        fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode='lines',
                                  name=f'Baseline ({baseline:.2f})',
                                  line=dict(color='gray', width=1, dash='dash')))
        fig.update_layout(
            xaxis_title='Recall', yaxis_title='Precision',
            height=400, margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(x=0.02, y=0.1, bgcolor='rgba(255,255,255,0.8)')
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Confusion Matrices
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Random Forest — Confusion Matrix")
        cm = metrics['rf_cm']
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=['Normal', 'Kacak'], y=['Normal', 'Kacak'],
            colorscale='Blues', showscale=False,
            text=cm, texttemplate='%{text}', textfont={'size': 22}
        ))
        fig.update_layout(
            xaxis_title='Tahmin', yaxis_title='Gercek',
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Isolation Forest — Confusion Matrix")
        cm = metrics['iso_cm']
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=['Normal', 'Kacak'], y=['Normal', 'Kacak'],
            colorscale='Oranges', showscale=False,
            text=cm, texttemplate='%{text}', textfont={'size': 22}
        ))
        fig.update_layout(
            xaxis_title='Tahmin', yaxis_title='Gercek',
            height=350, margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(autorange='reversed')
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.markdown("#### 🎯 Ozellik Onemliligi (Random Forest)")
    imp = metrics['importance']
    top_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:12]
    feat_names = [f[0] for f in top_feats][::-1]
    feat_vals = [f[1] for f in top_feats][::-1]

    fig = go.Figure(go.Bar(
        x=feat_vals, y=feat_names, orientation='h',
        marker=dict(color=feat_vals, colorscale='Blues', showscale=False),
        text=[f'{v:.3f}' for v in feat_vals], textposition='outside'
    ))
    fig.update_layout(
        height=400, margin=dict(l=20, r=80, t=20, b=20),
        xaxis_title='Onem Skoru'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Kacak turune gore tespit basarisi
    st.markdown("#### 📋 Kacak Turune Gore Tespit Basarisi")
    col1, col2 = st.columns(2)

    with col1:
        theft_df = df[df['label'] == 1]
        type_detection = theft_df.groupby('theft_type').agg(
            toplam=('label', 'count'),
            tespit=('predicted_theft', 'sum')
        ).reset_index()
        type_detection['oran'] = type_detection['tespit'] / type_detection['toplam'] * 100

        fig = go.Figure(go.Bar(
            x=type_detection['theft_type'], y=type_detection['oran'],
            marker_color=['#1B4F72', '#2E86C1', '#5DADE2', '#85C1E9', '#AED6F1'],
            text=[f'{v:.0f}%' for v in type_detection['oran']], textposition='outside'
        ))
        fig.update_layout(
            yaxis_title='Tespit Orani (%)', yaxis_range=[0, 110],
            height=350, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Tum model karsilastirma tablosu
        st.markdown("#### Model Karsilastirma Tablosu")
        model_data = {
            'Model': ['Random Forest', 'XGBoost', 'Isolation Forest', 'LSTM Autoencoder'],
            'Tip': ['Supervised', 'Supervised', 'Unsupervised', 'Deep Learning'],
            'ROC-AUC': ['0.9471', '0.9373', '0.8208', '0.7482*'],
            'F1 Score': ['0.8704', '0.8468', '0.2609', '0.5600*'],
            'Avantaj': ['En yuksek dogruluk', 'Ozellik onemliligi', 'Etiket gerektirmez', 'Zaman serisi analizi']
        }
        st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
        st.caption("*Musteri bazli degerlendirme")


# ========== TAB 4: MUSTERI DETAY ==========
def render_customer_detail(df, raw_df):
    st.markdown("### 🔎 Musteri Detay Inceleme")

    col1, col2 = st.columns([1, 3])

    with col1:
        view_mode = st.radio("Gosterim", ["Yuksek Riskli", "Tum Musteriler"], label_visibility="collapsed")

        if view_mode == "Yuksek Riskli":
            pool = df[df['theft_probability'] > 0.5].sort_values('theft_probability', ascending=False)
        else:
            pool = df.sort_values('customer_id')

        if len(pool) == 0:
            st.info("Bu filtrede musteri yok.")
            return

        selected_id = st.selectbox(
            "Musteri Sec", options=pool['customer_id'].tolist(),
            format_func=lambda x: f"#{x} ({df[df['customer_id']==x]['theft_probability'].values[0]:.0%})"
        )

        cust = df[df['customer_id'] == selected_id].iloc[0]
        st.markdown(f"**Profil:** {cust['profile']}")
        st.markdown(f"**Risk:** {cust['risk_level']}")
        st.markdown(f"**Kacak Olasiligi:** {cust['theft_probability']:.1%}")
        st.markdown(f"**Anomali Skoru:** {cust['anomaly_score']:.3f}")

        if cust['label'] == 1:
            st.markdown(f"**Kacak Turu:** {cust['theft_type']}")

        if cust['theft_probability'] > 0.7:
            st.markdown('<div class="alert-box">⚠️ <strong>YUKSEK RISK</strong></div>', unsafe_allow_html=True)
        elif cust['theft_probability'] < 0.3:
            st.markdown('<div class="safe-box">✅ Normal</div>', unsafe_allow_html=True)

    with col2:
        cust_raw = raw_df[raw_df['customer_id'] == selected_id]
        if len(cust_raw) > 0:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                               subplot_titles=['Tuketim Profili', 'Gunluk Toplam Tuketim'], row_heights=[0.6, 0.4])

            # Zaman serisi
            fig.add_trace(go.Scatter(
                x=cust_raw['timestamp'], y=cust_raw['consumption_kw'],
                mode='lines', line=dict(color='#2E86C1', width=1),
                fill='tozeroy', fillcolor='rgba(46,134,193,0.1)', name='Tuketim'
            ), row=1, col=1)

            # Sifir noktalar
            zeros = cust_raw[cust_raw['consumption_kw'] < 0.01]
            if len(zeros) > 0:
                fig.add_trace(go.Scatter(
                    x=zeros['timestamp'], y=zeros['consumption_kw'],
                    mode='markers', marker=dict(color='red', size=3), name='Sifir'
                ), row=1, col=1)

            # Gunluk toplam
            daily = cust_raw.set_index('timestamp').resample('D')['consumption_kw'].sum().reset_index()
            bar_colors = ['#E74C3C' if v < daily['consumption_kw'].mean() * 0.3 else '#2E86C1'
                         for v in daily['consumption_kw']]
            fig.add_trace(go.Bar(
                x=daily['timestamp'], y=daily['consumption_kw'],
                marker_color=bar_colors, name='Gunluk', showlegend=False
            ), row=2, col=1)

            fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20), hovermode='x unified')
            fig.update_yaxes(title_text='kW', row=1, col=1)
            fig.update_yaxes(title_text='kWh/gun', row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Ham tuketim verisi mevcut degil (ilk 200 musteri).")

    # Ozellik radar grafigi
    st.markdown("---")
    st.markdown("#### 🕸️ Musteri Ozellik Profili")

    features_to_show = ['mean_consumption', 'std_consumption', 'zero_measurement_pct',
                        'sudden_change_ratio', 'night_day_ratio', 'cv_daily']
    labels = ['Ort. Tuketim', 'Std Sapma', 'Sifir %', 'Ani Degisim', 'Gece/Gunduz', 'Gunluk CV']

    # Normalize (0-1 arasi)
    cust_vals = []
    pop_vals = []
    for f in features_to_show:
        f_min = df[f].min()
        f_max = df[f].max()
        cust_vals.append((cust[f] - f_min) / (f_max - f_min + 1e-8))
        pop_vals.append((df[f].mean() - f_min) / (f_max - f_min + 1e-8))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=pop_vals + [pop_vals[0]], theta=labels + [labels[0]],
                                   fill='toself', fillcolor='rgba(46,134,193,0.1)',
                                   line=dict(color='#2E86C1'), name='Populasyon Ort.'))
    fig.add_trace(go.Scatterpolar(r=cust_vals + [cust_vals[0]], theta=labels + [labels[0]],
                                   fill='toself', fillcolor='rgba(231,76,60,0.15)',
                                   line=dict(color='#E74C3C'), name=f'Musteri #{int(selected_id)}'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                     height=400, margin=dict(l=60, r=60, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ========== TAB 5: CANLI SIMULASYON ==========
def render_live_simulation(df, raw_df):
    st.markdown('<p style="font-size:1.3rem; font-weight:600;">🔴 <span class="live-indicator"></span> Canli Simulasyon Modu</p>', unsafe_allow_html=True)
    st.markdown("*Gercek zamanli akan veri simulasyonu — MASS sayaclarindan gelen veriyi canli izleyin.*")

    col1, col2, col3 = st.columns(3)
    with col1:
        sim_speed = st.selectbox("Hiz", [0.05, 0.1, 0.2, 0.5], index=1, format_func=lambda x: f"{x}s aralik")
    with col2:
        n_customers = st.selectbox("Musteri Sayisi", [3, 5, 8], index=1)
    with col3:
        sim_points = st.selectbox("Veri Noktasi", [50, 100, 200], index=1)

    # Musterileri sec (karisik normal + kacak)
    normal_sample = df[(df['label'] == 0) & (df['customer_id'] < 200)].head(n_customers - 1)
    theft_sample = df[(df['label'] == 1) & (df['customer_id'] < 200)].head(1)
    sim_customers = pd.concat([normal_sample, theft_sample])

    if st.button("▶️ Simulasyonu Baslat", type="primary", use_container_width=True):
        # Canli metrik paneli
        metric_cols = st.columns(4)
        metric_cols[0].markdown("**Akan Olcum**")
        metric_cols[1].markdown("**Anomali Tespit**")
        metric_cols[2].markdown("**Ort. Tuketim**")
        metric_cols[3].markdown("**Alarm Sayisi**")

        m_count = metric_cols[0].empty()
        m_anomaly = metric_cols[1].empty()
        m_avg = metric_cols[2].empty()
        m_alarm = metric_cols[3].empty()

        chart_placeholder = st.empty()
        alert_placeholder = st.empty()
        progress = st.progress(0)

        alarm_count = 0
        anomaly_count = 0
        total_consumption = 0

        # Her musteri icin veri hazirla
        customer_data = {}
        for _, c in sim_customers.iterrows():
            cid = c['customer_id']
            craw = raw_df[raw_df['customer_id'] == cid].head(sim_points)
            customer_data[cid] = {
                'values': craw['consumption_kw'].values,
                'label': c['label'],
                'theft_type': c.get('theft_type', 'none'),
                'profile': c['profile'],
                'buffer_x': [],
                'buffer_y': []
            }

        for step in range(sim_points):
            fig = make_subplots(rows=len(sim_customers), cols=1, shared_xaxes=True,
                               vertical_spacing=0.04)

            for i, (cid, cdata) in enumerate(customer_data.items()):
                if step < len(cdata['values']):
                    val = cdata['values'][step]
                    cdata['buffer_x'].append(step)
                    cdata['buffer_y'].append(val)
                    total_consumption += val

                    color = '#E74C3C' if cdata['label'] == 1 else '#27AE60'
                    name = f"#{int(cid)} {'⚠️' if cdata['label']==1 else '✅'}"

                    fig.add_trace(go.Scatter(
                        x=cdata['buffer_x'], y=cdata['buffer_y'],
                        mode='lines', line=dict(color=color, width=1.5),
                        fill='tozeroy', fillcolor=f'{color}15',
                        name=name, showlegend=True
                    ), row=i+1, col=1)

                    # Anomali tespiti (sifir veya ani dusus)
                    if val < 0.01 and cdata['label'] == 1:
                        anomaly_count += 1
                        fig.add_trace(go.Scatter(
                            x=[step], y=[val], mode='markers',
                            marker=dict(color='red', size=8, symbol='x'),
                            showlegend=False
                        ), row=i+1, col=1)

                    if val < 0.01 and step > 5:
                        alarm_count += 1

            fig.update_layout(
                height=80 * len(sim_customers) + 100,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=True,
                legend=dict(orientation='h', y=1.02)
            )

            for i in range(len(sim_customers)):
                fig.update_yaxes(title_text='kW', row=i+1, col=1)

            chart_placeholder.plotly_chart(fig, use_container_width=True)

            # Metrikleri guncelle
            m_count.metric("Akan Olcum", f"{step + 1}/{sim_points}")
            m_anomaly.metric("Anomali Tespit", f"{anomaly_count}")
            avg = total_consumption / ((step + 1) * len(sim_customers))
            m_avg.metric("Ort. Tuketim", f"{avg:.2f} kW")
            m_alarm.metric("Alarm", f"{alarm_count}", delta="⚠️" if alarm_count > 0 else "")

            # Alarm bildirimi
            if alarm_count > 0 and step % 20 == 0 and step > 0:
                alert_placeholder.warning(f"🚨 {alarm_count} sifir tuketim alarmi tespit edildi!")

            progress.progress((step + 1) / sim_points)
            time.sleep(sim_speed)

        progress.empty()
        st.success(f"✅ Simulasyon tamamlandi! {sim_points} olcum islendi, {anomaly_count} anomali tespit edildi.")

    else:
        # Preview
        st.markdown("---")
        st.info("▶️ yukaridaki butona basarak canli simulasyonu baslatin. Simulasyon sırasında akan veriyi ve anomali tespitini gercek zamanli izleyebilirsiniz.")

        # Onizleme grafigi
        preview_cust = sim_customers.iloc[0]
        preview_raw = raw_df[raw_df['customer_id'] == preview_cust['customer_id']].head(96 * 3)
        if len(preview_raw) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=preview_raw['timestamp'], y=preview_raw['consumption_kw'],
                mode='lines', line=dict(color='#2E86C1', width=1),
                fill='tozeroy', fillcolor='rgba(46,134,193,0.1)', name='Onizleme'
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20),
                            title=f'Onizleme — Musteri #{int(preview_cust["customer_id"])}')
            st.plotly_chart(fig, use_container_width=True)


# ========== ANA UYGULAMA ==========
def main():
    st.markdown('<p class="main-header">⚡ MASS-AI Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Milli Akilli Sayac Sistemleri — Yapay Zeka Tabanli Anomali Tespit ve Kacak Elektrik Siniflandirma v2.0</p>', unsafe_allow_html=True)

    features_df, raw_df = load_data()
    features_df, metrics = run_models(features_df)
    filtered_df, threshold = render_sidebar(features_df)

    # Tab'lar
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Genel Bakis",
        "📉 Zaman Serisi Karsilastirma",
        "🧠 Model Performansi",
        "🔎 Musteri Detay",
        "🔴 Canli Simulasyon"
    ])

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

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#999; font-size:0.85rem;'>"
        "MASS-AI v2.0 | Omer Burak Kocak | Marmara Universitesi EEE | Mart 2026"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
