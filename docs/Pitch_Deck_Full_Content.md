# MASS-AI — Tam Pitch Deck İçeriği (İTÜ Çekirdek & Yıldız Teknik)

**Versiyon:** 1.0 — 27 Mayıs 2026  
**Not:** Bu belge, Pitch_Deck_Outline.md'in üzerine yazılmış, jüriye doğrudan anlatılabilecek güçlü ve dürüst metinlerden oluşur. Her slayt için konuşma metni + görsel önerileri içerir.

---

## Slayt 1: Kapak

**Başlık:**  
MASS-AI  
Türkiye'nin Elektrik Kaçaklarını Yapay Zekâ ile Tespit Ediyoruz

**Alt başlık:**  
Akıllı Sayaç Verilerinden Gerçek Zamanlı Kaçak Tespiti ve Operasyonel Önceliklendirme

**Sunucu:** Founder  
**İletişim:** Available on request  
**Tarih:** Mayıs 2026

**Görsel önerisi:** Güçlü bir görsel — gece elektrik direkleri + veri akışı + kırmızı risk haritası.

**Konuşma notu (30-40 sn):**  
"Merhaba, ben MASS-AI'nin kurucusuyum. MASS-AI ile Türkiye'deki en büyük teknik kayıplardan birini, elektrik kaçaklarını, yapay zekâ ile çözmeye çalışıyoruz. Bugün size hem teknik olarak iddialı hem de piyasaya çok yakın bir çözüm sunacağız."

---

## Slayt 2: Problem (En Güçlü Slayt)

**Başlık:** Problem — Milli Bir Kaynak İsrafı

**İçerik (madde madde):**

- Türkiye'de bazı bölgelerde elektrik kaçak oranı %25-28'leri buluyor (resmi veriler ve sektör raporları).
- Yıllık ekonomik kayıp: **10 milyar TL'nin üzerinde**.
- Bu kayıp sadece para değil; aynı zamanda **dürüst tüketicinin cebinden çıkan** parayla sübvanse ediliyor.
- Mevcut yöntemler: Saha ekipleriyle manuel kontrol, basit kural bazlı alarmlar. Bunlar ne ölçeklenebilir ne de yeterince erken tespit yapabiliyor.
- 2025-2028 arasında 50 milyondan fazla akıllı sayaç devreye girecek. Veri bolluğu var, ancak bu veriyi **anlamlı ve aksiyon alınabilir** hale getiren sistem neredeyse yok.

**Vurgu kutusu (büyük font):**  
"Kaçak elektrik sadece teknik bir kayıp değil, aynı zamanda sosyal adalet ve ulusal ekonomi sorunudur."

**Jüriye soru sorma ihtimali:**  
"Bu rakamlar nereden geliyor?" → Resmi EPDK raporları + dağıtım şirketlerinin kendi yayınladığı kayıp-kaçak oranları.

**Konuşma notu:**  
"Her yıl 10 milyar TL'nin üzerinde para sistemden kayboluyor. Bu parayı ya faturalara yansıtıyoruz ya da devlet sübvanse ediyor. Her iki durumda da kaybeden vatandaş."

---

## Slayt 3: Çözüm

**Başlık:** MASS-AI — Veri Zenginliğini Karara Dönüştüren Sistem

**İçerik:**

- Akıllı sayaç verilerinden elektrik kaçak tespiti yapan, 6 farklı makine öğrenmesi modelini (Isolation Forest, XGBoost, LSTM Autoencoder, Stacking Ensemble vb.) bir arada kullanan hibrit bir yapay zekâ sistemi.
- Türkiye'ye özel tasarlanmış **8 farklı kaçak tipi** modellemesi (gece sıfırlama, hafta sonu maskeleme, kademeli düşüş, tamper spike vb.).
- **Açıklanabilirlik** çok önemli: Sadece "kaçak var" demiyoruz. "Bu müşteri şu 3 nedenden dolayı yüksek riskli: peer'lerine göre %40 daha düşük tüketim, son 30 günde 7 gece sıfırlama, trafo ortalamasından 2.3 sigma sapma" diyebiliyoruz.
- Saha ekiplerine **önceliklendirilmiş vaka listesi** sunuyoruz (sadece "şüpheli" demek yerine kime önce gidilmesi gerektiğini söylüyoruz).

**Görsel:** Dashboard'dan 1-2 güçlü ekran görüntüsü (risk haritası + müşteri detay kartı).

---

## Slayt 4: Teknik Fark ve Yenilik (Real Data Vurgusu Burada Güçlü)

**Başlık:** Neden Biz? Teknik Üstünlük

**İçerik:**

- Standart akademik yaklaşımlar (SGCC benchmark) genellikle 10-20 basit özellik kullanır. Biz **40+ özellik** ile çalışıyoruz (peer bazlı analiz, trafo/feeder hiyerarşisi, mevsimsel ve kültürel etkiler dahil).
- Dünyada en çok kullanılan benchmark olan **SGCC** (State Grid Corporation of China) veri seti üzerinde ilk kontrollü testlerimizi yaptık (Mayıs 2026).
  - Sentetik veride kendi içinde AUC: 1.0
  - Gerçekçi SGCC-style dağılımda AUC: **0.91** (F1: 0.71)
  - Gap'i açıkça ölçüyoruz ve gerçek veriyle kapatma planımız var.
- Bu, "sadece sentetik veriyle mi çalışıyorsunuz?" sorusuna verdiğimiz en dürüst ve en güçlü cevap.

**Vurgu:**  
"Birçok akademik çalışma sentetik veride %95+ başarı iddia eder. Biz sentetik ile gerçek arasındaki farkı ölçüyor ve bunu avantaja çevirme planı yapıyoruz."

---

## Slayt 5: Ürün ve MVP Durumu

**Başlık:** Bugün Neyi Gösterebiliyoruz?

- Tam çalışan web tabanlı demo (yeni nesil dashboard)
- Gerçek zamanlı / toplu risk skorlama
- Açıklanabilir risk nedenleri (3 ana neden + özet metin)
- Ops Center: Vaka oluşturma, takip, not alma, önceliklendirme
- Sentetik veri motoru ile farklı Türkiye bölgeleri ve kültürel senaryolarda test imkanı

**Durum:** MVP hazır, pilot için olgun. Gerçek veri entegrasyonu için adaptör katmanı geliştirme aşamasında.

---

## Slayt 6: Pazar ve Müşteri

**Başlık:** Kim Para Ödeyecek?

- **Birincil müşteri:** Türkiye'deki 21 elektrik dağıtım şirketi (TEDAŞ, BAŞKENT EDAŞ, AYDEM, UEDAŞ, Toroslar EDAŞ vb.).
- Her şirketin kayıp-kaçak oranı %15-30 arasında değişiyor. Bu şirketler için her %1'lik iyileşme bile ciddi para demek.
- İkincil pazar: Kuzey Afrika, Orta Doğu, Balkanlar, Latin Amerika'da benzer sorun yaşayan ülkeler.
- TAM (toplam adreslenebilir pazar): Sadece Türkiye'de yıllık kaçak kaybının %5-10'unu hedeflesek bile yüz milyonlarca TL'lik bir fırsat.

**Jüri sorusu ihtimali:** "Dağıtım şirketleri bu tip çözümlere para verir mi?"  
Cevap: Birçok şirket halihazırda yabancı yazılımlara (bazıları çok pahalı ve lokal adaptasyonu zayıf) veya kendi geliştirdikleri yetersiz sistemlere para ödüyor.

---

## Slayt 7: İş Modeli (En Çok Sorulan Slayt — Dürüst ve Somut)

**Başlık:** Nasıl Para Kazanacağız?

**Önerilen Hibrit Model (Pilotla Birlikte Netleşecek):**

1. **Temel SaaS Ücreti**  
   - Aylık/ yıllık abonelik (müşteri başına veya trafo bölgesi başına).
   - Erken pilot müşteriler için indirimli veya "başarıya bağlı" başlangıç.

2. **Başarı Bazlı Prim**  
   - Tespit edilen ve sahada doğrulanmış kaçak miktarı üzerinden %X-%Y başarı payı.
   - Bu model dağıtım şirketi için riski çok düşürür.

3. **Opsiyonel Modüller**  
   - Saha ekibi için mobil uygulama + offline raporlama
   - Entegrasyon desteği (mevcut SCADA / billing sistemleriyle)

**Neden bu model mantıklı?**
- Müşteri sadece "vaat" için değil, "kanıtlanmış sonuç" için ödüyor.
- Bizim için recurring revenue + upside potansiyeli var.
- Erken aşamada pilotu kolaylaştırıyor.

**Dürüstlük vurgusu (çok önemli):**  
"Bu modeli şu anda kağıt üzerinde tasarladık. Gerçek bir dağıtım şirketiyle 6-9 aylık bir pilot yaparak hem teknik performansı hem de ticari modeli birlikte olgunlaştıracağız."

---

## Slayt 8: Yol Haritası + Traction Planı

**Başlık:** 18 Aylık Yol Haritası

**Kısa Vadeli (0-6 ay):**
- 1 dağıtım şirketiyle pilot proje (hedef: en az 1 MoU veya niyet mektubu)
- Gerçek veri entegrasyonu ve model adaptasyonu
- İlk saha doğrulamaları

**Orta Vadeli (6-12 ay):**
- 3-4 şirketle genişleme
- Gerçek zamanlı izleme modülünün olgunlaşması
- İlk gelir

**Uzun Vadeli (12-24 ay):**
- 8+ şirket
- Mobil saha uygulaması
- Potansiyel yurt dışı adaptasyon

**Mevcut Durum (27 Mayıs 2026):**
- Sentetik + gerçekçi proxy ile ilk benchmark tamamlandı.
- Gerçek SGCC formatı için feature mapper üretime hazır.
- Pilot için hedef şirket listesi ve outreach planı hazır (ayrı belge).

---

## Slayt 9: Ekip

**Başlık:** Ekip

(Bu slayt şu anda en zayıf nokta. Kullanıcı burayı doldurmalı.)

**Örnek yapı:**
- Founder — AI & Product
- [Varsa ortak] — ...
- Danışman / Mentor (varsa): Enerji sektörü deneyimli isimler

**Jüri çok önemser.** En azından "bu işi neden ben yapabilirim?" sorusuna güçlü cevap lazım.

---

## Slayt 10: Kuluçkadan İhtiyaçlarımız

**Başlık:** Sizden Ne İstiyoruz?

- Enerji sektörü + regülasyon deneyimli mentorluk
- Dağıtım şirketleriyle bağlantı ve pilot kolaylaştırma
- Ofis + altyapı desteği
- Potansiyel hibe / yatırım bağlantıları

**Vurgu:**  
"Para kadar önemli olan şey, bu ekosisteme hızlı girebilmemiz için kapıların açılması."

---

## Slayt 11: Kapanış + Risklere Dürüst Yaklaşım

**Başlık:** Neden Şimdi? Neden Biz?

- Problem çok büyük ve acil.
- Teknolojimiz iddialı ve sürekli iyileştiriyoruz (gerçek veri gap'ini açıkça ölçüyoruz).
- İş modelini pilotla birlikte müşterinin yanında şekillendireceğiz.
- En kritik riskleri (sentetik veri, iş modeli, ekip) farkındayız ve üzerlerinde aktif çalışıyoruz.

**Son cümle:**  
"MASS-AI, sadece güzel bir teknik proje değil. Hem Türkiye'nin parasını korumak hem de dürüst vatandaşın hakkını savunmak için somut bir araç. Bu yolculuğu sizinle birlikte büyütmek istiyoruz."

---

## Ek: Jürinin En Çok Soracağı 5 Soru ve Önerilen Cevaplar

1. **"Sadece sentetik veriyle mi çalışıyorsunuz?"**  
   → Hayır. Bugün size SGCC benchmark üzerinde ilk kontrollü sonuçlarımızı gösteriyoruz. Gap'i ölçüyoruz ve gerçek veriyle kapatma planımız var. (0.91 AUC rakamı ver)

2. **"Dağıtım şirketleri neden size para versin?"**  
   → Zaten para veriyorlar. Ya yabancı çok pahalı çözümlere ya da kendi yetersiz sistemlerine. Biz daha ucuz, daha lokal ve başarı bazlı model sunuyoruz.

3. **"Rekabetiniz kim?"**  
   → Büyük uluslararası firmalar + birkaç yerli girişim. Bizim farkımız: Türk tüketim kültürüne özel model + peer analizi + açıklanabilirlik.

4. **"6 ay sonra neyi başarmış olmanızı beklersiniz?"**  
   → En az 1 dağıtım şirketiyle imzalanmış pilot + gerçek veride çalışan sistem.

5. **"Siz bu işi neden yapabilirsiniz?" (Ekip sorusu)**  
   → [Kişisel cevap — hazırlanmalı]

---

**Bu belge hazır olduğunda:**
- Her slayt için 1 sayfa PowerPoint'e kopyala-yapıştır yapılabilir.
- Görsellerle (dashboard screenshot, Türkiye elektrik şebekesi haritası, risk eğrisi) desteklenmeli.

Sonraki adım: Business Model belgesini derinleştirmek.
