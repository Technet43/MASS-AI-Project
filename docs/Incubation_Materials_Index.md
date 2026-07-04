# MASS-AI — İnkübatör Malzemeleri Dizini (İndex)

**Son Güncelleme:** 27 Mayıs 2026  
**Amaç:** İTÜ Çekirdek ve Yıldız Teknik başvuruları için üretilen belgelerin tek yerden takip edilebilmesi.

> Not: Bu dizin hem dış paylaşıma uygun malzemeleri hem de ekip içi çalışma notlarını içerir. Dışa gönderilecek paket için yalnızca public olarak işaretlenen dokümanlar kullanılmalıdır.

---

## 1. Sunum ve Pitch Malzemeleri (En Kritik)

| Belge | Dosya | Açıklama | Durum |
|-------|-------|----------|-------|
| Tam Pitch Deck İçeriği | `docs/Pitch_Deck_Full_Content.md` | 11 slayt için konuşmaya hazır, dürüst ve güçlü metinler. Jüri soruları + cevaplar dahil. | Hazır (kopyala-yapıştır için optimize) |
| One Pager | `docs/One_Pager.md` | Kısa proje özeti | Mevcut (güncellenebilir) |

**Kullanım:** `Pitch_Deck_Full_Content.md` içinden slaytlara metinleri direkt aktar. Gerçek veri benchmark sonuçlarını Slayt 4'te kullan.

---

## 2. İş Modeli ve Gelir

| Belge | Dosya | Açıklama | Durum |
|-------|-------|----------|-------|
| İş Modeli + Revenue Derin | `docs/Business_Model_and_Revenue_v1.md` | 3 senaryo (Muhafazakâr / Gerçekçi / İddialı), birim ekonomisi, hibrit SaaS + başarı primi modeli | Hazır |

**Kullanım:** Jürinin en çok sorduğu "Nasıl para kazanacaksınız?" sorusuna somut cevap.

---

## 3. Traction ve Müşteri Doğrulama

| Belge | Dosya | Açıklama | Durum |
|-------|-------|----------|-------|
| Traction & Pilot Planı | `docs/Traction_and_Pilot_Plan_v1.md` | Hedef dağıtım şirketleri listesi (Tier 1-3), outreach mail şablonu, 6 aylık pilot checklist, 90 günlük aksiyon planı | Hazır |

**Kullanım:** "Müşteri doğrulaması nerede?" sorusuna en güçlü cevabımız.

---

## 4. Gerçek Veri Validasyonu (En Büyük Riskin Kapatılması)

| Belge / Araç | Konum | Açıklama |
|--------------|-------|----------|
| Real Data Integration Katmanı | `shared/core/real_data.py` | SGCC mapper, realistic proxy üreticisi, benchmark + otomatik rapor üreticisi |
| Real Data Validation Summary | `docs/Real_Data_Validation_Summary.md` | İnkübatör sunumları için hazır, net özet belge (rakamlar + stratejik mesaj + kullanım) |
| Incubation Readiness Checklist | `docs/Incubation_Readiness_Checklist.md` | Projenin mevcut durumunu, tamamlananları ve öncelikli eksiklikleri net listeleyen pratik takip aracı |
| Benchmark Script | `scripts/benchmark_real_vs_synthetic.py` | Tek komutla sentetik vs gerçekçi SGCC proxy karşılaştırması |
| Otomatik Rapor | `reports/real_data_validation_report.md` | İnkübatör sunumuna hazır güzel Markdown raporu |
| Testler | `shared/tests/test_real_data.py` | 8 test (proxy, extraction, benchmark, edge case'ler) |

**Mayıs 2026 Sonuçları (Özet):**
- Sentetik AUC: ~0.999
- SGCC-style Proxy AUC: ~0.912
- Gap: ~0.087

**Komutlar:**
```bash
python scripts/benchmark_real_vs_synthetic.py
# veya gerçek dosya ile
python scripts/benchmark_real_vs_synthetic.py --real /path/to/sgcc.csv
```

---

## 5. Public-Facing Technical Docs

| Belge | Dosya | Not |
|-------|-------|-----|
| Feature Catalog (Global Standartlar + Real Data) | `docs/Feature_Catalog.md` | ~40 özellik + SGCC karşılaştırması + ilk gerçek veri testi sonuçları |
| Architecture | `ARCHITECTURE.md` | Real Data Integration Layer bölümü eklendi |
| Real Data Requirements | `docs/REAL_DATA_REQUIREMENTS.md` | Sentetik, proxy ve gerçek veri seviyelerini ayıran teknik tanım |
| Real Data Validation Summary | `docs/Real_Data_Validation_Summary.md` | Kısa, paylaşılabilir validasyon özeti |

---

## 6. Internal Working Notes

| Belge | Dosya | Not |
|-------|-------|-----|
| Project Status & Gaps | `docs/Project_Status_and_Gaps.md` | İç değerlendirme ve açık risk listesi |
| Incubation Readiness Checklist | `docs/Incubation_Readiness_Checklist.md` | Başvuru öncesi takip listesi |
| While Eating Progress Log | `docs/WHILE_EATING_PROGRESS.md` | 27 Mayıs uzun çalışma oturumunun detaylı kaydı |
| May 2026 Deep Work Session Report | `docs/May_2026_Deep_Work_Session_Report.md` | Derin çalışma özeti ve ara notlar |
| PROJE_ICIN_YAPILACAKLAR_RAPORU | `docs/PROJE_ICIN_YAPILACAKLAR_RAPORU.md` | Çalışma / yapılacaklar raporu |

---

## 7. Önerilen Kullanım Sırası (Başvuru Hazırlığı)

1. `Pitch_Deck_Full_Content.md` → Slaytları hazırla (en acil).
2. `Business_Model_and_Revenue_v1.md` → İş modeli slaydını güçlendir.
3. `Traction_and_Pilot_Plan_v1.md` → 90 günlük planı gerçekçi hale getir (kişisel hedefler ekle).
4. `real_data_validation_report.md` → Gerçek veri slaydına ekle (dürüstlük + proaktiflik mesajı çok güçlü).
5. `Incubation_Materials_Index.md` (bu dosya) → Jüriye "her şey organize" hissi verir.

---

**Son Not:**  
Bu malzemeler "güzel teknik proje" algısından çıkıp "ciddi, kendini bilen, traction planı olan bir girişim" algısına taşımak için üretildi.

Kullanıcı (Burak) yemek yerken bu belgelerin büyük kısmı üretildi ve GitHub'a pushlandı.

Sonraki adımlar için: Dashboard temizliğine devam veya saha outreach maillerinin kişiselleştirilmesi yapılabilir.
