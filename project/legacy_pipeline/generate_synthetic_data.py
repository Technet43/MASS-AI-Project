"""
MASS-AI: Sentetik Akilli Sayac Verisi Uretici
==============================================
Gercek SGCC ve London Smart Meter veri setlerinin yapisini taklit eden
sentetik veri uretir. Kacak elektrik senaryolari enjekte eder.

Yazar: Omer Burak Kocak
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_normal_consumption(n_customers=1000, n_days=180):
    """
    Normal tuketim paternleri ureten fonksiyon.
    Her musteri icin 15 dakikalik aralikla tuketim verisi uretir.
    
    Parametreler:
    - n_customers: Musteri sayisi
    - n_days: Gun sayisi (6 ay default)
    """
    print(f"[1/4] {n_customers} musteri icin {n_days} gunluk normal tuketim verisi uretiliyor...")
    
    # 15 dakikalik araliklar (gunde 96 olcum)
    measurements_per_day = 96
    total_measurements = n_days * measurements_per_day
    
    # Zaman serisi olustur
    start_date = datetime(2026, 3, 1)  # MASS baslangic tarihi
    timestamps = [start_date + timedelta(minutes=15*i) for i in range(total_measurements)]
    
    all_data = []
    
    for customer_id in range(n_customers):
        # Musteri profili
        profile = np.random.choice(['residential', 'commercial', 'industrial'], p=[0.7, 0.2, 0.1])
        
        if profile == 'residential':
            base_load = np.random.uniform(0.3, 1.5)  # kW
            peak_multiplier = np.random.uniform(2.0, 4.0)
        elif profile == 'commercial':
            base_load = np.random.uniform(2.0, 8.0)
            peak_multiplier = np.random.uniform(1.5, 3.0)
        else:  # industrial
            base_load = np.random.uniform(10.0, 50.0)
            peak_multiplier = np.random.uniform(1.2, 2.0)
        
        consumption = np.zeros(total_measurements)
        
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60.0
            day_of_week = ts.weekday()
            month = ts.month
            
            # Gunluk patern (saat bazli)
            if profile == 'residential':
                # Sabah 7-9 ve aksam 18-23 pik
                if 7 <= hour <= 9:
                    daily_factor = 0.6 + 0.4 * np.sin((hour - 7) * np.pi / 2)
                elif 18 <= hour <= 23:
                    daily_factor = 0.7 + 0.3 * np.sin((hour - 18) * np.pi / 5)
                elif 0 <= hour <= 6:
                    daily_factor = 0.15 + 0.1 * np.random.random()
                else:
                    daily_factor = 0.3 + 0.2 * np.random.random()
            elif profile == 'commercial':
                # Is saatleri 9-18 pik
                if 9 <= hour <= 18:
                    daily_factor = 0.8 + 0.2 * np.random.random()
                else:
                    daily_factor = 0.1 + 0.1 * np.random.random()
            else:
                # 3 vardiya, surekli uretim
                daily_factor = 0.7 + 0.3 * np.random.random()
            
            # Hafta sonu etkisi
            if day_of_week >= 5:
                if profile == 'residential':
                    daily_factor *= 1.2
                elif profile == 'commercial':
                    daily_factor *= 0.3
            
            # Mevsimsel etki (yaz = klima, kis = isitma)
            seasonal = 1.0 + 0.3 * np.sin((month - 1) * np.pi / 6)
            
            # Gurultu
            noise = np.random.normal(0, 0.05 * base_load)
            
            consumption[i] = max(0, base_load * daily_factor * peak_multiplier * seasonal + noise)
        
        customer_data = {
            'customer_id': customer_id,
            'profile': profile,
            'consumption': consumption,
            'label': 0  # Normal
        }
        all_data.append(customer_data)
    
    print(f"    Tamamlandi: {n_customers} musteri x {total_measurements} olcum")
    return all_data, timestamps


def inject_theft_patterns(all_data, theft_ratio=0.12):
    """
    Kacak elektrik senaryolari enjekte eder.
    
    Kacak Turleri:
    1. Sabit azaltma: Sayac manipulasyonu (tum tuketim %30-70 dusuk gorunur)
    2. Gece sifirlamasi: Gece tuketimi sifira cekilir (kablo bypass)
    3. Rastgele sifirlar: Belirli gunlerde tuketim sifir gosterilir
    4. Kademeli azalma: Tuketim yavasyavas dusurulur (teshis zorlasmasi icin)
    5. Pik kirpma: Sadece yuksek tuketim anlari kesilir
    """
    n_thieves = int(len(all_data) * theft_ratio)
    thief_indices = np.random.choice(len(all_data), n_thieves, replace=False)
    
    theft_types = ['constant_reduction', 'night_zeroing', 'random_zeros', 'gradual_decrease', 'peak_clipping']
    
    print(f"[2/4] {n_thieves} musteriye kacak elektrik senaryosu enjekte ediliyor...")
    
    type_counts = {t: 0 for t in theft_types}
    
    for idx in thief_indices:
        theft_type = np.random.choice(theft_types)
        type_counts[theft_type] += 1
        consumption = all_data[idx]['consumption'].copy()
        
        if theft_type == 'constant_reduction':
            # Sayac manipulasyonu: tum tuketim %30-70 dusuk
            reduction = np.random.uniform(0.3, 0.7)
            consumption *= reduction
            
        elif theft_type == 'night_zeroing':
            # Gece 00:00-06:00 arasi bypass
            measurements_per_day = 96
            for day in range(len(consumption) // measurements_per_day):
                start = day * measurements_per_day
                # 00:00-06:00 = ilk 24 olcum (6 saat * 4)
                consumption[start:start+24] = np.random.uniform(0, 0.02, 24)
                
        elif theft_type == 'random_zeros':
            # Rastgele gunlerde sifir (sayac durdurma)
            measurements_per_day = 96
            n_days = len(consumption) // measurements_per_day
            zero_days = np.random.choice(n_days, int(n_days * 0.3), replace=False)
            for day in zero_days:
                start = day * measurements_per_day
                consumption[start:start+measurements_per_day] = 0
                
        elif theft_type == 'gradual_decrease':
            # Aylik %5-10 azalma
            n = len(consumption)
            decay = np.linspace(1.0, np.random.uniform(0.3, 0.6), n)
            consumption *= decay
            
        elif theft_type == 'peak_clipping':
            # Pik tuketim anlarini %40'ta sinirla
            threshold = np.percentile(consumption, 60)
            consumption = np.minimum(consumption, threshold)
        
        all_data[idx]['consumption'] = consumption
        all_data[idx]['label'] = 1  # Kacak
        all_data[idx]['theft_type'] = theft_type
    
    for t, c in type_counts.items():
        print(f"    {t}: {c} musteri")
    
    return all_data


def create_feature_matrix(all_data, timestamps):
    """
    Ham tuketim verisinden ozellik cikarimi (feature engineering).
    
    Ozellikler:
    - Istatistiksel: ortalama, std, min, max, medyan, carpiklik, basiklik
    - Zamansal: gece/gunduz orani, hafta ici/sonu farki, pik saati
    - Anomali gostergeleri: sifir yuzde, ani degisim sayisi, trend egimi
    """
    print("[3/4] Ozellik cikarimi yapiliyor...")
    
    features_list = []
    measurements_per_day = 96
    
    for data in all_data:
        c = data['consumption']
        n_days = len(c) // measurements_per_day
        
        # Gunluk tuketimler
        daily = [c[d*measurements_per_day:(d+1)*measurements_per_day] for d in range(n_days)]
        daily_totals = [np.sum(d) for d in daily]
        
        # Gece (00-06) vs gunduz (06-24) tuketim
        night_consumption = []
        day_consumption = []
        for d in daily:
            night_consumption.append(np.sum(d[:24]))   # 00:00-06:00
            day_consumption.append(np.sum(d[24:]))     # 06:00-24:00
        
        night_ratio = np.mean(night_consumption) / (np.mean(day_consumption) + 1e-6)
        
        # Hafta ici vs sonu
        weekday_totals = [daily_totals[i] for i in range(n_days) if i % 7 < 5]
        weekend_totals = [daily_totals[i] for i in range(n_days) if i % 7 >= 5]
        weekend_ratio = np.mean(weekend_totals) / (np.mean(weekday_totals) + 1e-6) if weekday_totals else 0
        
        # Sifir olcum yuzdesi
        zero_pct = np.sum(c == 0) / len(c)
        
        # Sifir gun yuzdesi (tum gun sifir olan)
        zero_days = sum(1 for dt in daily_totals if dt < 0.01) / n_days
        
        # Ani degisim (consecutive difference > threshold)
        diffs = np.abs(np.diff(c))
        mean_diff = np.mean(diffs)
        sudden_changes = np.sum(diffs > 3 * mean_diff) / len(diffs)
        
        # Trend (lineer regresyon egimi)
        x = np.arange(len(daily_totals))
        if len(daily_totals) > 1:
            slope = np.polyfit(x, daily_totals, 1)[0]
        else:
            slope = 0
        
        # Pik saat analizi
        hourly_avg = np.zeros(24)
        for d in daily:
            for h in range(24):
                hourly_avg[h] += np.sum(d[h*4:(h+1)*4])
        hourly_avg /= n_days
        peak_hour = np.argmax(hourly_avg)
        
        # Tuketim varyasyon katsayisi
        cv = np.std(daily_totals) / (np.mean(daily_totals) + 1e-6)
        
        features = {
            'customer_id': data['customer_id'],
            'profile': data['profile'],
            'label': data['label'],
            'theft_type': data.get('theft_type', 'none'),
            # Istatistiksel
            'mean_consumption': np.mean(c),
            'std_consumption': np.std(c),
            'min_consumption': np.min(c),
            'max_consumption': np.max(c),
            'median_consumption': np.median(c),
            'skewness': float(pd.Series(c).skew()),
            'kurtosis': float(pd.Series(c).kurtosis()),
            # Gunluk istatistikler
            'mean_daily_total': np.mean(daily_totals),
            'std_daily_total': np.std(daily_totals),
            'cv_daily': cv,
            # Zamansal
            'night_day_ratio': night_ratio,
            'weekend_weekday_ratio': weekend_ratio,
            'peak_hour': peak_hour,
            # Anomali gostergeleri
            'zero_measurement_pct': zero_pct,
            'zero_day_pct': zero_days,
            'sudden_change_ratio': sudden_changes,
            'trend_slope': slope,
            # Dagilim
            'q25': np.percentile(c, 25),
            'q75': np.percentile(c, 75),
            'iqr': np.percentile(c, 75) - np.percentile(c, 25),
        }
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    print(f"    {len(df)} musteri x {len(df.columns)} ozellik")
    return df


def main():
    print("=" * 60)
    print("MASS-AI: Sentetik Veri Uretimi")
    print("=" * 60)
    
    # Veri uret
    all_data, timestamps = generate_normal_consumption(n_customers=2000, n_days=180)
    
    # Kacak enjekte et (%12 oran - Turkiye gercegine yakin)
    all_data = inject_theft_patterns(all_data, theft_ratio=0.12)
    
    # Ozellik cikarimi
    df = create_feature_matrix(all_data, timestamps)
    
    # Kaydet
    print("[4/4] Veriler kaydediliyor...")
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(os.path.join(output_dir, 'features.csv'), index=False)
    
    # Ham tuketim verisi (ilk 200 musteri - tam veri cok buyuk)
    raw_records = []
    for data in all_data[:200]:
        for i, ts in enumerate(timestamps):
            raw_records.append({
                'customer_id': data['customer_id'],
                'timestamp': ts,
                'consumption_kw': data['consumption'][i],
                'label': data['label']
            })
    raw_df = pd.DataFrame(raw_records)
    raw_df.to_csv(os.path.join(output_dir, 'raw_consumption_sample.csv'), index=False)
    
    # Ozet
    print("\n" + "=" * 60)
    print("OZET")
    print("=" * 60)
    n_normal = (df['label'] == 0).sum()
    n_theft = (df['label'] == 1).sum()
    print(f"Toplam musteri: {len(df)}")
    print(f"Normal: {n_normal} ({n_normal/len(df)*100:.1f}%)")
    print(f"Kacak:  {n_theft} ({n_theft/len(df)*100:.1f}%)")
    print(f"\nKacak turleri:")
    for t in df[df['label']==1]['theft_type'].value_counts().items():
        print(f"  {t[0]}: {t[1]}")
    print(f"\nOzellik sayisi: {len(df.columns) - 4} (meta sutunlar haric)")
    print(f"\nDosyalar:")
    print(f"  data/processed/features.csv")
    print(f"  data/processed/raw_consumption_sample.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
