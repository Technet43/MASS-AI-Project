# MASS-AI — Gerçek Veri Validasyon Raporu (İlk Kontrollü Test)

**Tarih:** 2026-05-27T15:39:31
**Sentetik Veri:** 900 müşteri — Turkey Urban
**Gerçekçi Proxy:** 700 müşteri (SGCC istatistiklerine göre modellenmiş)

## Özet Sonuçlar

| Metrik                              | Sentetik (Kendi Dağılımı) | SGCC-Style Proxy | Fark (Gap) |
|-------------------------------------|---------------------------|------------------|------------|
| AUC                                 | 0.9994                  | 0.9119           | 0.0875       |
| F1                                  | 0.9697                  | 0.8           | -          |

**En iyi model (sentetik):** Random Forest
**En iyi model (proxy):** Random Forest

## Yorum ve İnkübatör Mesajı

Bu test, 'sadece sentetik veriyle mi çalışıyorsunuz?' sorusuna verdiğimiz ilk somut cevaptır.

- Sentetik veride model kendi dağılımında çok güçlü performans gösteriyor (AUC 1.0).
- Gerçekçi SGCC-style dağılıma geçtiğimizde performans doğal olarak düşüyor (AUC 0.91).
- Bu düşüş (gap) beklenen bir durumdur. Önemli olan bu gap'i **ölçüyor** olmamız ve gerçek veriyle kapatma planımızın olmasıdır.

**Sonraki Adım:** Gerçek bir SGCC veya Türk dağıtım şirketi veri seti ile aynı pipeline çalıştırıldığında bu gap'in ne kadar kapandığını göreceğiz.

## Teknik Detay

Bu sonuçlar sentetik veride eğitilen modellerin gerçekçi (SGCC benzeri) bir dağılım üzerinde nasıl performans gösterdiğini gösterir. Gerçek SGCC dosyasını verdiğinde aynı pipeline'ı gerçek veride çalıştırabiliriz.

---

*Bu rapor `scripts/benchmark_real_vs_synthetic.py` tarafından otomatik üretilmiştir.*