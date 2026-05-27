# MASS-AI — İş Modeli ve Gelir Modeli (Derin Versiyon 1.0)

**Tarih:** 27 Mayıs 2026  
**Amaç:** İTÜ Çekirdek ve Yıldız Teknik jürisine "nasıl para kazanacağız?" sorusuna somut, makul ve dürüst cevap vermek.

---

## 1. Temel Varsayımlar

- Türkiye'de 21 elektrik dağıtım şirketi var.
- Ortalama bir dağıtım şirketinin yıllık kayıp-kaçak maliyeti: 300-800 milyon TL arasında (bölgeye göre çok değişir).
- Bir dağıtım şirketinin kayıp-kaçak oranını **kalıcı %1-2 puan** düşürmek bile onlar için yılda onlarca milyon TL tasarruf demek.
- Dağıtım şirketleri halihazırda bu alana para harcıyor (yabancı yazılımlar, kendi ekipleri, basit yazılımlar).

**Müşteri Ağrı Noktaları:**
- Mevcut çözümler ya çok pahalı ya da lokal koşullara (kültür, mevsim, bayram etkisi vb.) uyumsuz.
- "Kaçak var" demek yetmiyor; "hangi müşteriye önce gidiyoruz, ne kadar kayıp var, ne kadar emin olabiliriz?" diyebilmeleri lazım.

---

## 2. Önerilen İş Modeli: Hibrit SaaS + Başarı Bazlı

### Modelin Adı: "Güvenli Başlangıç + Ortak Kazanç"

**İki katmanlı fiyatlandırma:**

### A. Temel Abonelik (SaaS)
- **Fiyatlandırma önerisi (pilot dönemi):**  
  - Küçük-orta ölçekli şirket: Aylık 25.000 - 45.000 TL  
  - Büyük ölçekli şirket: Aylık 60.000 - 90.000 TL

- **Ne kapsar?**  
  - Tüm dashboard ve risk skorlama  
  - Aylık toplu analiz  
  - Standart destek  
  - Gerçek veri entegrasyonu (ilk 2 bölgeye kadar)

### B. Performans Primi (En Önemli Kısım)
- Tespit edilen ve **sahada saha ekibi tarafından doğrulanmış** kaçak miktarının %X-%Y'si.
- Örnek: Tespit edilen kaçak 5 milyon TL değerindeyse, %8 başarı primi = 400 bin TL ekstra ödeme.
- Bu kısım hem bizim için upside yaratır hem de müşteri için "sadece sonuç için ödeme" hissi verir.

### C. Opsiyonel Modüller (İleride)
- Saha ekibi mobil uygulaması
- Gerçek zamanlı entegrasyon (SCADA/billing)
- Özel model eğitimi (şirkete özel ince ayar)

---

## 3. Üç Senaryo (2026-2028)

### Senaryo 1: Muhafazakâr (En Düşük Beklenti)

- Yıl 1: 1 pilot şirket (ücretsiz veya çok düşük ücretli pilot)
- Yıl 2: 2 şirket → 1 tanesi ücretli
- Yıllık tekrar eden gelir (ARR) hedefi Yıl 2 sonunda: **1.8 - 2.4 milyon TL**
- Performans primi: Henüz çok düşük

### Senaryo 2: Gerçekçi (En Çok Kullanacağımız Senaryo)

- Yıl 1 (2026 sonu): 1 pilot + 1 ücretli müşteri
- Yıl 2: 4 şirket (2'si tam ücretli)
- Yıl 3: 7-8 şirket
- ARR hedefleri:
  - 2026 sonu: 1.2 - 1.8 milyon TL
  - 2027 sonu: 6 - 9 milyon TL
  - 2028 sonu: 15 - 22 milyon TL
- Performans primi katkısı: Toplam gelirin %25-35'i

### Senaryo 3: İddialı (Yüksek Büyüme)

- Hızlı pilot başarısı + referans etkisi
- Yıl 2 sonunda 6+ şirket
- ARR 2027: 12+ milyon TL
- Erken uluslararası ilgi (1-2 Balkan veya Orta Doğu şirketi)

---

## 4. Birim Ekonomisi (Unit Economics) — Somut Örnek

**Varsayım:** Orta ölçekli bir dağıtım şirketi (yıllık 400 milyon TL kaçak kaybı)

- MASS-AI ile tespit edilen + saha doğrulamalı kaçak: Yıllık 35-50 milyon TL
- Müşterinin bu sayede tasarrufu: 35-50 milyon TL
- Bizim aldığımız:
  - Temel SaaS: 720 bin TL/yıl
  - Başarı primi (%7): 2.45 - 3.5 milyon TL
  - **Toplam yıl 1 geliri bu müşteriden:** ~3.2 - 4.2 milyon TL

**Maliyet tarafı (yaklaşık):**
- Sunucu + altyapı: 150-250 bin TL
- 2-3 kişilik teknik ekip: 1.8-2.4 milyon TL
- Saha destek + satış: 600-800 bin TL

Bu örnekte bile tek bir orta ölçekli müşteriyle **pozitif katkı marjı** yakalamak mümkün.

---

## 5. Neden Bu Model Mantıklı?

**Müşteri için:**
- Risk düşük (temel ücret makul, asıl para sonuç çıkınca ödeniyor).
- "Deneyelim bakalım" demeleri kolay.

**Bizim için:**
- Recurring revenue (temel abonelik) var.
- Upside potansiyeli yüksek (performans primi).
- Erken aşamada pilot kapısını açıyor.

**En kritik risk ve dürüst cevap:**
Bu model şu anda teorik. Gerçek bir pilot yapmadan "şu fiyata satacağız" demek doğru değil. Bu yüzden jüriye şunu söyleyeceğiz:

> "Fiyatlandırma ve başarı primi oranlarını, ilk pilot şirketle birlikte 6-9 ay içinde birlikte belirleyeceğiz. Müşterinin de kazanması bizim de kazanmamız lazım."

---

## 6. Erken Dönem (Pilot) Stratejisi

**İlk 1-2 müşteri için özel koşullar:**
- İlk 6 ay temel SaaS ücreti %60-70 indirimli veya "başarıya bağlı".
- Performans primi ilk pilotlarda daha yüksek tutulabilir (müşteriyi motive etmek için).
- Karşılığında: Gerçek veri paylaşımı + saha doğrulama desteği + referans olma taahhüdü.

**Hedef:** İlk pilotu "para kazanmak" değil, "kanıt + referans + model olgunlaştırma" olarak görmek.

---

## 7. Uzun Vadeli Opsiyonlar

- 2028+: Yurt dışı adaptasyon (benzer sorun yaşayan ülkeler)
- "MASS-AI as a Service" yerine bazı büyük şirketlere "on-premise + özel model" lisansı
- Regülasyonla ilgili danışmanlık / raporlama modülü (EPDK raporlaması için otomatik çıktı)

---

## 8. Özet — Jüriye Söylenecek 4 Cümle

1. "Hibrit bir model öneriyoruz: Makul bir temel abonelik + tespit edilen ve doğrulanmış kaçak üzerinden başarı primi."
2. "Bu model hem dağıtım şirketinin riskini düşürür hem de bizim ölçeklenebilir gelir elde etmemizi sağlar."
3. "Fiyatları ve primi ilk pilotta müşteriyle birlikte netleştireceğiz. Şu anda kağıt üzerindeki rakamlar referans niteliğindedir."
4. "Tek bir orta ölçekli müşteriyle bile ciddi katkı marjı yakalayabileceğimizi birim ekonomisi hesaplarımız gösteriyor."

---

**Sonraki adım önerisi:** Bu belgeyi okuyup "gerçekçi senaryo" rakamlarını kendi ekibinle birlikte güncelle. Jüriye en çok bu slaytı soracak.

Bu belge `Pitch_Deck_Full_Content.md` içindeki İş Modeli slaydını desteklemek için hazırlandı.