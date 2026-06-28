# MASS-AI Project - Current Status & Critical Gaps Report

**Tarih:** 2026  
**Amaç:** İTÜ Çekirdek ve Yıldız Teknik Kuluçka başvurularına hazırlık + Teknik kaliteyi 9+ seviyeye taşıma

---

## 1. Projenin Genel Durumu (Özet)

### Güçlü Yönler
- Gerçek ve büyük bir problem (Türkiye'de elektrik kaçak kaybı).
- Teknik olarak iddialı bir çekirdek (6 model + stacking + gelişmiş sentetik veri).
- Çalışan bir demo mevcut.
- Feature set, akademik literatüre göre oldukça zengin.
- Son dönemde ciddi yapısal ve dokümantasyon iyileştirmesi yapıldı.

### Zayıf Yönler (En Kritik Olanlar)
- İş modeli ve para kazanma modeli net değil.
- Müşteri doğrulaması (traction) neredeyse sıfır.
- Ekip anlatımı çok zayıf.
- Teknik borç (özellikle dashboard tarafında duplication) hâlâ yüksek.
- Proje hâlâ "güzel teknik proje" havasından tam çıkmadı.

---

## 2. Yapılan İyileştirmeler (Son Dönem)

### Yapısal ve Organizasyon
- `project/` klasörü kaldırıldı.
- Eski kodlar `legacy/` klasörüne taşındı.
- Root dizin büyük ölçüde temizlendi (`apps/`, `scripts/`, `web/` yapısı oluşturuldu).
- Launcher path'leri düzeltildi.

### Dokümantasyon ve Sunum Malzemeleri
- `docs/Feature_Catalog.md` oluşturuldu (global standartlarla karşılaştırma dahil).
- `docs/Pitch_Deck_Outline.md` oluşturuldu.
- `docs/One_Pager.md` oluşturuldu.
- `ARCHITECTURE.md` ve ana `README.md` güncellendi.

### Kod Kalitesi
- `dashboard_adapters.py` oluşturuldu ve kısmen kullanıma alındı.
- Birçok eski duplicate fonksiyon devre dışı bırakıldı veya deprecated olarak işaretlendi.
- Ana veri yükleme mantığı temizlendi.

### Test
- Test sayısı önemli ölçüde artırıldı (engine + adapter testleri).
- Bazı yeni testler eklendi.

---

## 3. Hâlâ Kritik Olan Eksiklikler (Sıralı)

### A. İş ve Pazar Tarafı (En Yüksek Risk)
- **İş modeli** çok yüzeysel ve inandırıcı değil.
- Gerçek bir dağıtım şirketiyle **hiç görüşme / pilot konuşması** yok.
- Rekabet analizi ve farklılaşma stratejisi zayıf.
- "Neden senin çözümün ölçeklenir?" sorusuna güçlü cevap yok.

### B. Ekip ve Güvenilirlik
- Ekip slaytı / anlatımı neredeyse yok.
- Enerji sektörü bağlantısı veya danışmanlık çok zayıf.

### C. Teknik Borç (Hâlâ Yüksek)
- `new_web/dashboard/app.py` hâlâ çok kalın ve içinde duplicate mantık var.
- Kod tabanı "araştırma projesi" havasından tam çıkmadı.
- Production-ready olma seviyesi düşük.
- Dockerfile, Compose ve Makefile ile tekrar üretilebilir lokal deployment eklendi; buna rağmen auth, monitoring, structured logging ve staging ortamı hâlâ eksik.

### D. Traction ve Doğrulama
- Her şey sentetik veri üzerine kurulu.
- Gerçek veri ile en azından küçük ölçekli bir temas yok.
- SGCC-style proxy benchmark var, ancak gerçek public SGCC dosyasıyla veya Türk DSO verisiyle rapor hâlâ gerekli.

### E. Sunum ve İletişim
- Profesyonel bir Pitch Deck henüz yok (sadece outline var).
- Proje hâlâ daha çok "teknik başarı" olarak anlatılıyor, "startup fırsatı" olarak değil.

---

## 4. Öncelikli Yapılması Gerekenler (Kısa Vadeli)

| Öncelik | İş | Tür | Neden Önemli? |
|--------|----|-----|---------------|
| 1 | İş modelini netleştir + basit bir revenue model oluştur | İş | Jürinin en çok sorduğu soru |
| 2 | Ekip anlatımını oluştur (en azından 1 slayt + kısa biyografi) | Sunum | Kuluçka'da ekip çok kritik |
| 3 | Pitch Deck'in içeriğini güçlendir (özellikle Problem, Solution, Business Model, Traction) | Sunum | Başvurunun kalbi |
| 4 | Dashboard'daki en ağır duplicate kodları temizle | Teknik | Profesyonellik algısı |
| 5 | En az 2-3 dağıtım şirketiyle görüşme başlat (hatta niyet mektubu hedefle) | Traction | En güçlü sinyal |
| 6 | Public SGCC dosyasıyla gerçek benchmark üret | Teknik/Validasyon | Proxy eleştirisini kapatır |
| 7 | PR-AUC, precision@K, recall@K ve threshold analizi ekle | Teknik/Ürün | Saha operasyonuna yakın metrik sağlar |

---

## 5. Tavsiye: Şu Anda Ne Yapmalısın?

**Kısa vadede (önümüzdeki 2-3 hafta) öncelik sırası önerim:**

1. **Pitch Deck**'i profesyonel seviyeye getir (en acil).
2. **İş modeli + traction** anlatısını güçlendir.
3. **Ekip** kısmını netleştir.
4. Teknik borçları azaltmaya devam et (ama Pitch Deck'ten sonra).

Teknik kaliteyi 9+ yapmak uzun vadeli bir hedef. Ancak kuluçka başvurusu için öncelik **"güven veren ve mantıklı duran bir hikaye"** anlatabilmek.

---

**Bu belgeyi düzenli olarak güncelleyeceğiz.** Her önemli adım sonrası buraya yeni durum + kalan eksikler eklenecek.

Son Güncelleme: 2026

---

## 5. Gerçek Veri İlerlemesi (27 Mayıs 2026)

Kullanıcı talebi üzerine **SGCC** (dünyada en çok kullanılan electricity theft benchmark veri seti) için ilk test yapıldı.

- `shared/core/real_data.py` içinde production-grade `extract_sgcc_style_features()` ve `generate_realistic_sgcc_proxy()` yazıldı.
- `scripts/benchmark_real_vs_synthetic.py` ile çalıştırılabilir hale getirildi.

**İlk somut sonuçlar (kontrollü proxy ile):**

- Sentetik (Turkey Urban) AUC: **1.000**
- SGCC-style proxy AUC: **0.9095** (F1: 0.714)
- Gap: **0.0905**

Bu, "sadece sentetik veriyle validasyon" riskinin ilk nicel kanıtı. Gerçek SGCC dosyası verildiğinde aynı script ile direkt çalıştırılabilir.

İnkübatör sunumları için güçlü bir "dürüstlük + çözüm" maddesi haline geldi.
