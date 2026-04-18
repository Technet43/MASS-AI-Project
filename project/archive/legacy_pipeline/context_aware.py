"""
MASS-AI: Context-Aware Anomali Tespiti
=======================================
Tatil/bayram false positive problemini cozen akilli ozellikler.

Yeni Ozellikler:
1. Tatil Takvimi — Resmi tatil ve bayramlarda tuketim dususu beklenir
2. Komsuluk Karsilastirmasi — Ayni trafodaki diger musterilerle kiyaslama
3. Gecis Analizi — Ani dusus/yukselis paternleri (tatil vs kacak farki)
4. Mevsimsel Baseline — Gecmis doneme gore sapma analizi
5. Profil Tutarliligi — Musteri profiline uygun mu?

Yazar: Omer Burak Kocak
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)


# ========== TURKIYE TATIL TAKVIMI ==========
def get_turkey_holidays_2026():
    """
    2026 yili Turkiye resmi tatilleri + dini bayramlar.
    Ramazan ve Kurban Bayrami tarihleri tahminidir (hicri takvim).
    """
    holidays = {
        # Resmi tatiller
        '2026-01-01': ('Yilbasi', 1),
        '2026-04-23': ('23 Nisan', 1),
        '2026-05-01': ('Isci Bayrami', 1),
        '2026-05-19': ('19 Mayis', 1),
        '2026-07-15': ('Demokrasi Bayrami', 1),
        '2026-08-30': ('Zafer Bayrami', 1),

        # Ramazan Bayrami (tahmini: 20-22 Mart 2026)
        '2026-03-20': ('Ramazan Bayrami 1. Gun', 3),
        '2026-03-21': ('Ramazan Bayrami 2. Gun', 3),
        '2026-03-22': ('Ramazan Bayrami 3. Gun', 3),

        # Kurban Bayrami (tahmini: 27-30 Mayis 2026)
        '2026-05-27': ('Kurban Bayrami 1. Gun', 4),
        '2026-05-28': ('Kurban Bayrami 2. Gun', 4),
        '2026-05-29': ('Kurban Bayrami 3. Gun', 4),
        '2026-05-30': ('Kurban Bayrami 4. Gun', 4),
    }

    # Yaz tatili donemi (okullar kapali, tatil sezonu)
    summer_period = [(datetime(2026, 6, 15) + timedelta(days=i)).strftime('%Y-%m-%d')
                     for i in range(75)]  # 15 Haziran - 28 Agustos

    return holidays, summer_period


def generate_vacation_patterns(all_data, timestamps, vacation_ratio=0.15):
    """
    Tatile giden musteri paternleri olustur.
    BUNLAR NORMAL MUSTERILER — kacak degil!

    Tatil Turleri:
    1. Kisa tatil (3-5 gun): Bayram tatili, kisa kacamak
    2. Uzun tatil (7-21 gun): Yaz tatili
    3. Bayram etkisi: Tum mahalle tuketimi duser
    4. Mevsimsel goc: Yazin koy/sahile tasinanlar
    """
    n_vacationers = int(len(all_data) * vacation_ratio)

    # Sadece normal musterilerden sec (kacaklar zaten isaretli)
    normal_indices = [i for i, d in enumerate(all_data) if d['label'] == 0]
    vac_indices = np.random.choice(normal_indices, min(n_vacationers, len(normal_indices)), replace=False)

    holidays, summer = get_turkey_holidays_2026()
    measurements_per_day = 96

    print(f"\n[TATIL] {len(vac_indices)} musteriye tatil senaryosu ekleniyor...")

    vacation_types = ['short_vacation', 'long_vacation', 'holiday_effect', 'seasonal_migration']
    type_counts = {t: 0 for t in vacation_types}

    for idx in vac_indices:
        vac_type = np.random.choice(vacation_types, p=[0.35, 0.25, 0.25, 0.15])
        type_counts[vac_type] += 1
        consumption = all_data[idx]['consumption'].copy()
        n_days = len(consumption) // measurements_per_day

        if vac_type == 'short_vacation':
            # 3-5 gun tatil — tuketim buzdolabi seviyesine duser
            duration = np.random.randint(3, 6)
            start_day = np.random.randint(10, n_days - duration - 10)

            for d in range(start_day, start_day + duration):
                day_start = d * measurements_per_day
                # Buzdolabi + standby: sabit dusuk tuketim
                base_standby = np.random.uniform(0.05, 0.15)
                consumption[day_start:day_start + measurements_per_day] = \
                    base_standby + np.random.normal(0, 0.02, measurements_per_day)
                consumption[day_start:day_start + measurements_per_day] = \
                    np.maximum(consumption[day_start:day_start + measurements_per_day], 0.02)

        elif vac_type == 'long_vacation':
            # 7-21 gun — yaz tatili
            duration = np.random.randint(7, 22)
            start_day = np.random.randint(60, min(n_days - duration - 5, 150))  # Yaz aylarinda

            for d in range(start_day, start_day + duration):
                day_start = d * measurements_per_day
                base_standby = np.random.uniform(0.03, 0.12)
                consumption[day_start:day_start + measurements_per_day] = \
                    base_standby + np.random.normal(0, 0.015, measurements_per_day)
                consumption[day_start:day_start + measurements_per_day] = \
                    np.maximum(consumption[day_start:day_start + measurements_per_day], 0.01)

            # Donus gunu: tuketim aniden normale doner (tatil imzasi!)
            return_day = start_day + duration
            if return_day < n_days:
                day_start = return_day * measurements_per_day
                # Geri donus gunu normalden biraz yuksek (camasir, temizlik vs)
                consumption[day_start:day_start + measurements_per_day] *= 1.3

        elif vac_type == 'holiday_effect':
            # Bayram etkisi — resmi tatil gunlerinde dusuk tuketim
            for date_str, (name, duration) in holidays.items():
                holiday_date = datetime.strptime(date_str, '%Y-%m-%d')
                start_date = timestamps[0] if isinstance(timestamps[0], datetime) else datetime.strptime(str(timestamps[0]), '%Y-%m-%d %H:%M:%S')
                day_offset = (holiday_date - start_date).days

                if 0 <= day_offset < n_days:
                    # Arife gunu de dahil
                    for d in range(max(0, day_offset - 1), min(n_days, day_offset + duration)):
                        day_start = d * measurements_per_day
                        # %40-60 dusus
                        reduction = np.random.uniform(0.4, 0.6)
                        consumption[day_start:day_start + measurements_per_day] *= reduction

        elif vac_type == 'seasonal_migration':
            # Yazin koye/sahile tasinanlar — 45-90 gun cok dusuk tuketim
            duration = np.random.randint(45, 91)
            start_day = np.random.randint(70, min(n_days - duration, 120))

            for d in range(start_day, start_day + duration):
                day_start = d * measurements_per_day
                # Neredeyse sifir ama tam sifir degil (sayac calisiyor)
                base = np.random.uniform(0.02, 0.08)
                consumption[day_start:day_start + measurements_per_day] = \
                    base + np.random.normal(0, 0.01, measurements_per_day)
                consumption[day_start:day_start + measurements_per_day] = \
                    np.maximum(consumption[day_start:day_start + measurements_per_day], 0.005)

        all_data[idx]['consumption'] = np.maximum(consumption, 0)
        all_data[idx]['vacation_type'] = vac_type
        # ONEMLI: label hala 0 (normal) — tatil kacak degil!

    for t, c in type_counts.items():
        print(f"    {t}: {c} musteri")

    return all_data


def extract_context_features(all_data, timestamps):
    """
    Context-aware ozellikler cikar.
    Bu ozellikler tatil vs kacak ayrimi yapmayi saglar.
    """
    print("\n[CONTEXT] Baglam-duyarli ozellikler cikariliyor...")

    holidays, summer = get_turkey_holidays_2026()
    measurements_per_day = 96
    features_list = []

    # Trafo bolgeleri simule et (her 50 musteri = 1 trafo)
    n_customers = len(all_data)
    transformer_groups = {i: i // 50 for i in range(n_customers)}
    n_transformers = max(transformer_groups.values()) + 1

    # Trafo bazli ortalama tuketim hesapla
    transformer_avg = {}
    for t_id in range(n_transformers):
        t_members = [i for i, g in transformer_groups.items() if g == t_id]
        t_consumptions = np.mean([all_data[i]['consumption'] for i in t_members], axis=0)
        transformer_avg[t_id] = t_consumptions

    for data in all_data:
        c = data['consumption']
        cid = data['customer_id']
        n_days = len(c) // measurements_per_day
        daily = [c[d * measurements_per_day:(d + 1) * measurements_per_day] for d in range(n_days)]
        daily_totals = [np.sum(d) for d in daily]

        # ===== 1. GECIS ANALIZI (Transition Analysis) =====
        # Tatil: ani dusus -> sabit dusuk -> ani yukselis (V-shape)
        # Kacak: ani dusus -> kalici dusuk VEYA kademeli dusus

        # Dusus kenar sayisi (tuketim %50'den fazla dusen gunler)
        daily_mean = np.mean(daily_totals)
        drop_edges = 0
        rise_edges = 0
        for i in range(1, n_days):
            if daily_totals[i] < daily_totals[i-1] * 0.5 and daily_totals[i-1] > daily_mean * 0.3:
                drop_edges += 1
            if daily_totals[i] > daily_totals[i-1] * 2.0 and daily_totals[i-1] < daily_mean * 0.5:
                rise_edges += 1

        # V-shape skoru: dusus ve yukselis ciftleri (tatil imzasi)
        v_shape_score = min(drop_edges, rise_edges) / max(n_days / 30, 1)

        # Dusuk tuketim blok suresi (ardisik dusuk gunler)
        low_threshold = daily_mean * 0.3
        low_blocks = []
        current_block = 0
        for dt in daily_totals:
            if dt < low_threshold:
                current_block += 1
            else:
                if current_block > 0:
                    low_blocks.append(current_block)
                current_block = 0
        if current_block > 0:
            low_blocks.append(current_block)

        max_low_block = max(low_blocks) if low_blocks else 0
        n_low_blocks = len(low_blocks)
        avg_low_block = np.mean(low_blocks) if low_blocks else 0

        # ===== 2. TATIL KORELASYONU =====
        # Resmi tatil gunlerinde tuketim dusuk mu?
        start_date = timestamps[0] if isinstance(timestamps[0], datetime) else datetime.strptime(str(timestamps[0]), '%Y-%m-%d %H:%M:%S')

        holiday_days = set()
        for date_str in holidays:
            h_date = datetime.strptime(date_str, '%Y-%m-%d')
            day_offset = (h_date - start_date).days
            if 0 <= day_offset < n_days:
                holiday_days.add(day_offset)

        if holiday_days:
            holiday_consumption = np.mean([daily_totals[d] for d in holiday_days if d < len(daily_totals)])
            non_holiday = [daily_totals[d] for d in range(n_days) if d not in holiday_days]
            non_holiday_consumption = np.mean(non_holiday) if non_holiday else daily_mean
            holiday_ratio = holiday_consumption / (non_holiday_consumption + 1e-6)
        else:
            holiday_ratio = 1.0

        # ===== 3. KOMSULUK KARSILASTIRMASI =====
        t_id = transformer_groups.get(cid, 0)
        t_avg = transformer_avg.get(t_id, c)

        # Musteri vs trafo ortalamasinin korelasyonu
        if len(c) == len(t_avg):
            correlation = np.corrcoef(c[:min(len(c), len(t_avg))], t_avg[:min(len(c), len(t_avg))])[0, 1]
        else:
            correlation = 0

        # Sapma skoru: musteri ne kadar komsularindan farkli?
        if len(c) == len(t_avg) and np.std(t_avg) > 0:
            deviation_score = np.mean(np.abs(c - t_avg)) / (np.std(t_avg) + 1e-6)
        else:
            deviation_score = 0

        # ===== 4. STANDBY TUKETIM ANALIZI =====
        # Tatilde buzdolabi calisiyor = dusuk ama sifir degil
        # Kacakta (night_zeroing, random_zeros) = tam sifir

        min_daily = [np.min(d) for d in daily]
        near_zero_days = sum(1 for m in min_daily if m < 0.005) / n_days  # Tam sifir gun orani
        standby_days = sum(1 for m in min_daily if 0.005 <= m <= 0.2) / n_days  # Standby gun orani

        # Ortalama minimum tuketim (dusuk donemde)
        low_period_mins = [np.min(d) for d in daily if np.sum(d) < daily_mean * 0.3]
        avg_min_in_low = np.mean(low_period_mins) if low_period_mins else 0

        # ===== 5. DONUS ETKISI =====
        # Tatilden donus: tuketim normalin uzerine cikar (camasir, temizlik)
        # Kacaktan sonra: tuketim dusuk kalir
        post_low_surge = 0
        for i in range(1, n_days):
            if daily_totals[i-1] < low_threshold and daily_totals[i] > daily_mean * 1.2:
                post_low_surge += 1

        # ===== 6. HAFTA ICI/SONU TUTARLILIGI =====
        weekday_totals = [daily_totals[i] for i in range(n_days) if i % 7 < 5]
        weekend_totals = [daily_totals[i] for i in range(n_days) if i % 7 >= 5]
        pattern_consistency = np.std(weekday_totals) / (np.mean(weekday_totals) + 1e-6) if weekday_totals else 0

        features = {
            'customer_id': cid,
            # Gecis analizi
            'drop_edge_count': drop_edges,
            'rise_edge_count': rise_edges,
            'v_shape_score': v_shape_score,
            'max_low_block_days': max_low_block,
            'n_low_blocks': n_low_blocks,
            'avg_low_block_days': avg_low_block,
            # Tatil korelasyonu
            'holiday_consumption_ratio': holiday_ratio,
            # Komsuluk
            'neighbor_correlation': correlation if not np.isnan(correlation) else 0,
            'neighbor_deviation_score': deviation_score,
            # Standby analizi
            'near_zero_day_ratio': near_zero_days,
            'standby_day_ratio': standby_days,
            'avg_min_in_low_period': avg_min_in_low,
            # Donus etkisi
            'post_low_surge_count': post_low_surge,
            # Tutarlilik
            'weekday_pattern_cv': pattern_consistency,
        }
        features_list.append(features)

    ctx_df = pd.DataFrame(features_list)
    print(f"    {len(ctx_df.columns) - 1} yeni context-aware ozellik cikarildi")
    return ctx_df


def main():
    """
    Context-aware ozellikleri mevcut veri setine ekle ve
    tatil vs kacak ayrimi performansini test et.
    """
    print("=" * 65)
    print("MASS-AI: Context-Aware Anomali Tespiti")
    print("  Tatil/Bayram False Positive Azaltma Modulu")
    print("=" * 65)

    # Mevcut veriyi yukle
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    features_df = pd.read_csv(os.path.join(base, 'features.csv'))
    raw_df = pd.read_csv(os.path.join(base, 'raw_consumption_sample.csv'))
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

    print(f"\nMevcut veri: {len(features_df)} musteri")

    # Veriyi yeniden uret (tatil paternleriyle)
    print("\n[1/4] Tatil paternleri ekleniyor...")

    # Orijinal veri ureticiyi cagir
    from generate_synthetic_data import generate_normal_consumption, inject_theft_patterns, create_feature_matrix

    all_data, timestamps = generate_normal_consumption(n_customers=2000, n_days=180)
    all_data = inject_theft_patterns(all_data, theft_ratio=0.12)

    # Tatil senaryolari ekle (SADECE normal musterilere)
    all_data = generate_vacation_patterns(all_data, timestamps, vacation_ratio=0.15)

    # Temel ozellikleri yeniden cikar
    print("\n[2/4] Temel ozellikler yeniden cikariliyor...")
    base_features = create_feature_matrix(all_data, timestamps)

    # Context-aware ozellikleri cikar
    print("\n[3/4] Context-aware ozellikler cikariliyor...")
    ctx_features = extract_context_features(all_data, timestamps)

    # Birlestir
    enhanced_df = base_features.merge(ctx_features, on='customer_id', how='left')

    # Tatil bilgisini ekle
    enhanced_df['has_vacation'] = enhanced_df['customer_id'].apply(
        lambda x: 1 if any(d.get('vacation_type') for d in all_data if d['customer_id'] == x) else 0
    )
    # Daha hizli yontem
    vac_lookup = {d['customer_id']: d.get('vacation_type', 'none') for d in all_data}
    enhanced_df['vacation_type'] = enhanced_df['customer_id'].map(vac_lookup)
    enhanced_df['has_vacation'] = (enhanced_df['vacation_type'] != 'none').astype(int)

    # Kaydet
    enhanced_path = os.path.join(base, 'features_enhanced.csv')
    enhanced_df.to_csv(enhanced_path, index=False)
    print(f"\n    Kaydedildi: {enhanced_path}")
    print(f"    Toplam ozellik: {len(enhanced_df.columns)}")

    # ===== MODEL KARSILASTIRMASI =====
    print("\n[4/4] Model karsilastirmasi: Orijinal vs Context-Aware...")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix

    meta_cols = ['customer_id', 'profile', 'label', 'theft_type', 'vacation_type', 'has_vacation']
    orig_feature_cols = [c for c in base_features.columns if c not in meta_cols]
    enhanced_feature_cols = [c for c in enhanced_df.columns if c not in meta_cols]

    y = enhanced_df['label'].values

    # ----- Model A: Orijinal ozellikler -----
    X_orig = enhanced_df[orig_feature_cols].values
    scaler_a = StandardScaler()
    X_orig_s = scaler_a.fit_transform(X_orig)
    X_tr_a, X_te_a, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_orig_s, y, np.arange(len(y)), test_size=0.25, random_state=42, stratify=y
    )

    rf_a = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    rf_a.fit(X_tr_a, y_tr)
    pred_a = rf_a.predict(X_te_a)
    prob_a = rf_a.predict_proba(X_te_a)[:, 1]

    # ----- Model B: Context-aware ozellikler -----
    X_enh = enhanced_df[enhanced_feature_cols].fillna(0).values
    scaler_b = StandardScaler()
    X_enh_s = scaler_b.fit_transform(X_enh)
    X_tr_b = X_enh_s[idx_tr]
    X_te_b = X_enh_s[idx_te]

    rf_b = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
    rf_b.fit(X_tr_b, y_tr)
    pred_b = rf_b.predict(X_te_b)
    prob_b = rf_b.predict_proba(X_te_b)[:, 1]

    # ----- Sonuclar -----
    test_df = enhanced_df.iloc[idx_te].reset_index(drop=True)
    vacation_mask = test_df['has_vacation'] == 1
    normal_no_vac = (test_df['label'] == 0) & (~vacation_mask)
    normal_vac = (test_df['label'] == 0) & (vacation_mask)

    print("\n" + "=" * 65)
    print("SONUCLAR")
    print("=" * 65)

    print(f"\n{'Metrik':<35} {'Orijinal':>12} {'Context-Aware':>14}")
    print("-" * 63)

    auc_a = roc_auc_score(y_te, prob_a)
    auc_b = roc_auc_score(y_te, prob_b)
    f1_a = f1_score(y_te, pred_a)
    f1_b = f1_score(y_te, pred_b)

    print(f"{'Genel ROC-AUC':<35} {auc_a:>12.4f} {auc_b:>14.4f}")
    print(f"{'Genel F1 (Kacak)':<35} {f1_a:>12.4f} {f1_b:>14.4f}")

    # False positive analizi
    fp_a = ((pred_a == 1) & (y_te == 0)).sum()
    fp_b = ((pred_b == 1) & (y_te == 0)).sum()
    print(f"{'Toplam False Positive':<35} {fp_a:>12} {fp_b:>14}")

    # Tatildeki musterilerde false positive
    if vacation_mask.sum() > 0:
        vac_indices = np.where(vacation_mask.values)[0]
        # Test setindeki tatil musteri indexleri
        fp_vac_a = ((pred_a[vac_indices] == 1) & (y_te[vac_indices] == 0)).sum()
        fp_vac_b = ((pred_b[vac_indices] == 1) & (y_te[vac_indices] == 0)).sum()
        total_vac_normal = (y_te[vac_indices] == 0).sum()

        print(f"{'Tatildeki FP':<35} {fp_vac_a:>12} {fp_vac_b:>14}")
        print(f"{'Tatildeki Normal Musteri':<35} {total_vac_normal:>12} {total_vac_normal:>14}")
        if total_vac_normal > 0:
            fpr_vac_a = fp_vac_a / total_vac_normal * 100
            fpr_vac_b = fp_vac_b / total_vac_normal * 100
            print(f"{'Tatildeki FP Orani':<35} {fpr_vac_a:>11.1f}% {fpr_vac_b:>13.1f}%")
            improvement = fpr_vac_a - fpr_vac_b
            print(f"\n    >>> Tatildeki false positive azalma: {improvement:+.1f} puan <<<")

    print("\n" + "-" * 63)
    print("\nContext-Aware Model — Siniflandirma Raporu:")
    print(classification_report(y_te, pred_b, target_names=['Normal', 'Kacak']))

    # En onemli context ozellikler
    new_ctx_cols = ['drop_edge_count', 'rise_edge_count', 'v_shape_score',
                    'max_low_block_days', 'n_low_blocks', 'avg_low_block_days',
                    'holiday_consumption_ratio', 'neighbor_correlation',
                    'neighbor_deviation_score', 'near_zero_day_ratio',
                    'standby_day_ratio', 'avg_min_in_low_period',
                    'post_low_surge_count', 'weekday_pattern_cv']

    importances = dict(zip(enhanced_feature_cols, rf_b.feature_importances_))
    ctx_importances = {k: importances.get(k, 0) for k in new_ctx_cols if k in importances}
    sorted_ctx = sorted(ctx_importances.items(), key=lambda x: x[1], reverse=True)

    print("Context-Aware Ozellik Onemliligi:")
    for feat, imp in sorted_ctx:
        bar = "█" * int(imp * 200)
        print(f"    {feat:<30} {imp:.4f} {bar}")

    # Tatil vs Kacak ayirimi ornekleri
    print("\n" + "=" * 65)
    print("TATIL vs KACAK — Ornek Karsilastirma")
    print("=" * 65)

    # Bir tatil musterisi
    vac_example = test_df[(test_df['has_vacation'] == 1) & (test_df['label'] == 0)].head(1)
    if len(vac_example) > 0:
        v = vac_example.iloc[0]
        print(f"\n  TATIL Musterisi #{int(v['customer_id'])} ({v['vacation_type']}):")
        print(f"    v_shape_score:        {v.get('v_shape_score', 0):.3f}  (yuksek = tatil)")
        print(f"    post_low_surge:       {v.get('post_low_surge_count', 0):.0f}     (donus etkisi)")
        print(f"    standby_day_ratio:    {v.get('standby_day_ratio', 0):.3f}  (buzdolabi calisiyor)")
        print(f"    near_zero_day_ratio:  {v.get('near_zero_day_ratio', 0):.3f}  (tam sifir yok)")
        print(f"    neighbor_correlation: {v.get('neighbor_correlation', 0):.3f}  (komsularla uyumlu)")

    # Bir kacak musterisi
    theft_example = test_df[test_df['label'] == 1].head(1)
    if len(theft_example) > 0:
        t = theft_example.iloc[0]
        print(f"\n  KACAK Musterisi #{int(t['customer_id'])} ({t['theft_type']}):")
        print(f"    v_shape_score:        {t.get('v_shape_score', 0):.3f}  (dusuk = kacak)")
        print(f"    post_low_surge:       {t.get('post_low_surge_count', 0):.0f}     (donus yok)")
        print(f"    standby_day_ratio:    {t.get('standby_day_ratio', 0):.3f}  (tam sifir var)")
        print(f"    near_zero_day_ratio:  {t.get('near_zero_day_ratio', 0):.3f}  (sifir gunler)")
        print(f"    neighbor_correlation: {t.get('neighbor_correlation', 0):.3f}  (komsulardan farkli)")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
