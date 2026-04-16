# MASS-AI Proje İçin Yapılacaklar Raporu

Hazırlanma tarihi: 2026-04-16

İncelenen kaynaklar:
- `C:\Users\kocak\Downloads\UI Eleştirisi ve Geliştirme (1).txt`
- `C:\Users\kocak\Downloads\UI Eleştirisi ve Geliştirme (1).pdf`
- `C:\Users\kocak\Downloads\Mass Aı yapılıcaklar.docx`
- `C:\Users\kocak\Downloads\Akıllı Sayaç Verisi Analizi.pdf`
- repo branch: `codex/update-dashboard-and-realtime-ingest`
- kod tabanı: `README.md`, `project/dashboard/app.py`, `project/mass_ai_engine.py`, `project/realtime_ingest/*`, `project/ops_store.py`
- doğrulama: `python -m unittest discover project/tests -v` sonucu 13 test geçti

---

## 1. Bu Rapor Nasıl Kullanılmalı?

Bu rapor üç amaç için yazıldı:

1. Senin projeyi 1 haftalık yoğun çalışma içinde toparlaman için bir yürütme planı vermek.
2. `Codex` ve `Claude Code` gibi ajanlara doğrudan görev dağıtımı yapılabilecek kadar net teknik çerçeve sunmak.
3. Potansiyel bir firmaya bu projeyi “ham öğrenci projesi” gibi değil, “pilot seviyeye taşınabilecek yapay zeka destekli kaçak elektrik tespit platformu” olarak anlatabilmek.

Bu raporu bir “tek seferlik not” gibi değil, aktif çalışma dokümanı gibi kullanmak gerekir. Kod yazdırırken, UI düzenlerken, veri işleme mimarisini değiştirirken ve satış/pitch metni hazırlarken aynı belge referans alınmalıdır.

---

## 2. Yönetici Özeti

MASS-AI şu anda güçlü bir çekirdeğe sahip:

- akıllı sayaç/kaçak elektrik problemi doğru seçilmiş,
- ürün fikri ticari olarak anlamlı,
- tek model değil çok modelli bir yapı düşünülmüş,
- dashboard, desktop app, ops center, realtime ingest gibi “ürünleşme” sinyalleri mevcut,
- sentetik veri motoru demo üretmek için güçlü,
- case management tarafı beklenenden daha olgun,
- temel testler geçiyor.

Ama aynı anda kritik eksikler de var:

- proje anlatısı ile gerçek implementasyon bire bir örtüşmüyor,
- gerçek saha verisine geçiş için veri şeması uyumu henüz tamamlanmamış,
- realtime pipeline ile ana model pipeline arasında ciddi mimari kopukluk var,
- UI “decision dashboard” değil hâlâ yer yer “ML raporu / veri dökümü” gibi davranıyor,
- README ve ürün iddiaları bazı yerlerde implementasyonun önünde gidiyor,
- kurulum ve paket ayrımı kullanıcıyı gereksiz ağır bağımlılıklara sürüklüyor,
- güvenlik ve kurumsal entegrasyon tarafı demo seviyesinde.

En önemli sonuç şu:

Bu proje bugün doğrudan “kurumsal üretim sistemi” diye satılmamalı. Ama doğru paketlenirse çok rahat biçimde şu şekilde konumlandırılabilir:

> “Kaçak elektrik ve anomali tespiti için utility pilot / proof-of-value platformu”

Yani bir hafta içinde hedef “her şeyi bitirmek” değil, “ikna edici, temiz, güven veren, pilota hazır görünen bir ürün demosu + teknik yol haritası” üretmek olmalıdır.

---

## 3. Doğrulanmış Mevcut Durum

### 3.1 Güçlü Taraflar

- Problem alanı doğru: kaçak elektrik tespiti enerji şirketleri için gerçek parasal karşılığı olan bir problem.
- Ürün düşüncesi doğru: sadece model değil, analist iş akışı da düşünülmüş.
- Kod tabanı modüler: `mass_ai_engine`, `dashboard`, `realtime_ingest`, `ops_store`, `desktop` ayrışmış.
- Ops Center var: SQLite tabanlı vaka, not, geçmiş, follow-up, inspection kayıtları mevcut.
- Realtime ingest tarafı düşünülmüş: Postgres, MQTT, worker, alert üretimi ve watchdog ile dosya bazlı tetikleme var.
- Sentetik veri motoru güçlü: profiller, bölgeler, theft pattern’lar ve çok sayıda türetilmiş feature üretiyor.
- Test yüzeyi var: smoke ve ops testleri geçiyor.

### 3.2 Zayıf Taraflar

- “Production-ready” anlatısı erken.
- Gerçek veri entegrasyonu roadmap’te, ama ürünün merkezinde olması gereken konu hâlâ tamamlanmamış.
- Ana engine, dashboard ve realtime ingest farklı veri mantıklarıyla yaşıyor.
- Model katmanında “tek gerçek kaynak” yok.
- UI üretim seviyesine yaklaşsa da karar destek akışını henüz tam taşımıyor.

### 3.3 Kritik Tutarsızlıklar

#### A. 6 model anlatısı ile gerçek uygulama akışı tam örtüşmüyor

README 6 modelden bahsediyor:
- Isolation Forest
- XGBoost
- Random Forest
- Gradient Boosting
- LSTM Autoencoder
- Stacking Ensemble

Ama doğrulanmış kod akışında:

- `project/mass_ai_engine.py` içindeki ana `train_models()` yolu 5 modeli eğitiyor:
  - Isolation Forest
  - XGBoost
  - Random Forest
  - Gradient Boosting
  - Stacking Ensemble
- LSTM ana production path içinde değil, daha çok legacy/research path tarafında duruyor.
- `project/realtime_ingest/inference.py` tarafında ise sadece 3 feature + Isolation Forest benzeri bir anomali skorlama akışı var.
- dashboard tarafındaki güncel karşılaştırma ekranı branch özel değerlendirme mantığında 6 model gösterebiliyor; ama bu set README’de anlatılan 6 modelle bire bir aynı değil.

Bu şu anlama gelir:

Ürün anlatısı, model katmanı ve gerçek zamanlı skorlayıcı aynı hikâyeyi anlatmıyor.

#### B. Sentetik veri formatı ile gerçek ingest formatı uyumsuz

Kullanıcı dokümanında da doğru tespit edilmiş:

- sentetik üretim ana olarak feature-table / müşteri-bazlı skorlamaya dönük,
- realtime ingest ise long-format telemetry bekliyor:
  - `meter_id`
  - `voltage`
  - `current`
  - `active_power`
  - opsiyonel `timestamp`

Bu kopukluk giderilmeden gerçek utility verisine geçiş hikâyesi zayıf kalır.

#### C. SHAP anlatısı ile ana kod yolu ayrışıyor

- README ve legacy pipeline içinde SHAP anlatısı var.
- Ancak aktif ana path içinde SHAP tabanlı açıklanabilirlik merkezde değil.
- Mevcut açıklanabilirlik daha çok engineered drivers ve rule-based reason üretimi tarafında.

Yani “explainability var” denebilir, ama “ürün seviyesinde tek ve tutarlı explainability sistemi oturdu” denemez.

#### D. Kurulum deneyimi dağınık

- `requirements-full.txt` TensorFlow dahil ağır bağımlılık kuruyor.
- dashboard açmak isteyen kullanıcı gereksiz ağır paketleri yüklemeye kayabiliyor.
- bu durum bizzat kullanım deneyiminde sürtünme yaratıyor.

---

## 4. Esas Problem Tanımı

Bu proje teknik olarak ilginç bir sistem olmaya başlamış. Fakat satış, firma görüşmesi ve pilot ikna süreci açısından asıl problem teknik değil:

> Proje şu anda “özellikler toplamı” gibi duruyor; henüz tam anlamıyla “tek ürün” gibi davranmıyor.

Bu yüzden yapılması gereken işlerin özü:

1. Hikâyeyi birleştirmek
2. Veri akışını birleştirmek
3. Model akışını birleştirmek
4. UI’ı karar destek ekranına çevirmek
5. Satılabilir pilot paketini üretmek

---

## 5. Ürün, Teknik ve Ticari Açıdan Ana Çalışma Alanları

## 5.1 Ürün Konumlandırma

Şu an proje yanlış satılırsa güven kaybettirir.

Şu söylem riskli:
- “tam üretim sistemi”
- “kurumsal kullanıma hazır”
- “sahada deploy edilir”

Şu söylem doğru ve güçlü:
- “utility pilot platformu”
- “kaçak elektrik ve anomali tespiti için decision-support çözümü”
- “gerçek veri entegrasyonuna hazır mimari iskelet”
- “analist dashboard + vaka yönetimi + çok modelli skorlayıcı prototipi”

Önerilen ürün tanımı:

> MASS-AI, akıllı sayaç verilerinden kaçak elektrik ve anomali sinyallerini tespit eden; risk skoru, vaka önceliklendirme, analist ekranı ve operasyonel inceleme iş akışı sunan pilot seviyede yapay zeka platformudur.

Bu tanım hem iddialı hem savunulabilir.

---

## 5.2 UI / UX

UI Eleştirisi dokümanından çıkan ana tez çok doğru:

> Sistem şu anda veri dashboard’u; olması gereken karar dashboard’u.

### Mevcut UI sorunu

- overview çok şey gösteriyor ama yön göstermiyor,
- KPI’lar aksiyon üretmiyor,
- map insight üretmiyor,
- tablolar worklist gibi davranmıyor,
- siyah tablolar genel görsel dili bozuyor,
- kullanıcıya hata/debug dump gösteriliyor,
- boş canlı alanlar “ölü sistem” hissi veriyor.

### UI tarafında hedef durum

Overview şu üç soruya cevap vermeli:

1. Neresi riskli?
2. Risk ne kadar ciddi?
3. Analist şimdi ne yapmalı?

### UI için yapılacaklar

P0:
- overview’ü karar odaklı yeniden hiyerarşiklemek
- KPI kartlarını aksiyonlu hale getirmek
- siyah tablo ve siyah comparison table’ları açık temalı worklist/card table’a çevirmek
- traceback / raw error dump’ları kullanıcıdan gizlemek
- live alerts alanını boş ekran olmaktan çıkarmak

P1:
- map için cluster / heatmap toggle
- hot zone side panel
- reason / trend / last seen kolonları
- suspicious customers mini worklist
- customer detail sayfasında karşılaştırmalı davranış anlatısı

P2:
- rol bazlı dashboard varyasyonları
- daha ileri animasyon/micro-feedback
- mobil/exec brief görünümü

---

## 5.3 Veri ve Feature Engineering

Bu proje gerçek dünyaya açılacaksa en kritik katman model değil, veri katmanıdır.

### Ana sorunlar

- sentetik veri güçlü ama ana çıktı formatı utility ingest ile tam uyumlu değil,
- feature set iyi başlangıç seviyesi ama production için muhtemelen yetersiz,
- gerçek veri etiket kalitesi henüz yok,
- validasyon hâlâ sentetik başarıya fazla dayanıyor.

### Genişletilmesi gereken feature alanları

Dokümanlardaki yön doğru:

- self-history deviation
- peer-group deviation
- seasonality / holiday / tariff context
- persistence / sustained anomaly duration
- meter health / outage / device anomaly signals
- transformer / feeder context

Özellikle eklenmesi gerekenler:

- geçmişe göre sapma feature’ları
- benzer müşteri grubuna göre sapma
- mevsimsellik ve tatil feature’ları
- süreklilik ve anomaly burst feature’ları
- sayaç sağlık / outage / tamper sıklığı feature’ları
- operasyonel feedback ile etiket düzeltme alanı

### En kritik veri görevi

Sentetik üretim iki modda yaşamalı:

1. feature-table output
2. long-format telemetry output

Böylece:
- batch model eğitimi,
- realtime ingest testi,
- demo senaryosu,
- müşteriye örnek veri gösterimi

tek veri üretim mantığından beslenebilir.

---

## 5.4 Model Katmanı

Model kalbi iyi düşünülmüş ama henüz ürünle tam senkron değil.

### Gerçek durum

- ana engine çok modelli
- realtime ingest sadeleşmiş ve farklı
- legacy tarafında daha zengin araştırma kodu var
- README’de anlatılan model hikâyesi ile aktif yol arasında fark var

### Yapılması gereken

#### Kısa vadede

- “ürünün resmi model seti” belirlenmeli
- README, dashboard ve engine aynı model hikâyesini anlatmalı
- LSTM gerçekten üründe yer alacaksa ana path’e bağlanmalı
- yer almayacaksa “research module” olarak yeniden konumlandırılmalı

#### Orta vadede

- model registry gerçek anlamda kullanılmalı
- versioning, schema compatibility, threshold governance tanımlanmalı
- experiment tracking eklenmeli
- real dataset benchmark seti oluşturulmalı

### En kritik model kararı

1 haftalık ticari hazırlık için hedef:

> Çok model göstermek değil, güvenilir model hikâyesi göstermek.

Yani bir firmaya şu daha güçlü gelir:

- “Bizim ana scoring modelimiz şu”
- “ikinci seviye doğrulama modelimiz şu”
- “realtime tarafında şu hafif model çalışıyor”

yerine

- “aslında 6 model var ama hepsi her yerde aynı değil”

demekten kaçınmak gerekir.

---

## 5.5 Realtime Pipeline

Realtime katman projeyi sıradan ML demosundan ayırabilecek alanlardan biri.

Ama şu anda önemli bir sınırlama var:

- realtime worker yalnızca 3 feature ile çalışıyor,
- ana engine’deki zengin feature mantığı ile birleşmiyor,
- realtime skor ile batch skor aynı açıklanabilirlik düzeyine sahip değil.

### Bu neden önemli?

Firma tarafı genelde şunu sorar:

> “Gerçekten akan veriyle çalışıyor mu?”

Senin cevabın idealde şu olmalı:

- evet, telemetry ingestion var,
- sliding window feature extraction var,
- alert üretimi var,
- ama production eşleniği için ikinci fazda model konsolidasyonu gerekiyor.

Bu dürüst ve güçlü bir cevap olur.

### Realtime için hedef mimari

- telemetry input standardize edilmeli
- batch ve realtime ortak feature sözlüğü kullanmalı
- lightweight realtime scorer + periodic offline retraining modeli kurulmalı
- alert severity mantığı batch risk band mantığı ile tutarlı hale gelmeli

---

## 5.6 Ops Center ve Operasyon Akışı

Ops Center bu projenin en değerli taraflarından biri olabilir; çünkü çoğu öğrenci projesi burada biter:

- model tahmin eder
- sonra hiçbir operasyon akışı yoktur

Burada ise:
- cases
- notes
- history
- inspections
- priority
- follow-up

gibi alanlar mevcut.

Bu çok iyi.

### Ama kritik eksik

Varsayılan kullanıcı/şifrelerin düz metin ve demo seviyesinde olması kurumsal güven için sorun:

- `admin/admin`
- `analyst/analyst`
- `field/field`

Bu yapı demo için kabul edilebilir ama satış görüşmesinde mutlaka “demo auth layer” diye çerçevelenmelidir.

### Ops tarafında yapılacaklar

P0:
- demo kullanıcılarını güvenli placeholder anlatısıyla sunmak
- audit trail anlatısını demo sunumunda öne çıkarmak
- recommended action metinlerini daha utility diliyle güçlendirmek

P1:
- gerçek RBAC
- hashed password
- company/region/team bazlı ayrım
- SLA ve escalation rules

---

## 5.7 Kurulum, Paketleme ve Developer Experience

Bu alan küçük görünür ama satışta çok önemlidir. Çünkü demo açılmazsa ürün iyi olsa da güven kaybettirir.

### Sorunlar

- dashboard için gereksiz ağır kurulum riski var
- full requirements TensorFlow gibi ağır bağımlılık çekiyor
- hangi kullanım için hangi requirements dosyasının gerektiği net değil

### Yapılması gereken

P0:
- “dashboard only” kurulum yolu netleştirilmeli
- “full research stack” ayrı tutulmalı
- tek komutla demo açılmalı
- README’de kullanım yolları rol bazlı ayrılmalı:
  - sadece dashboard
  - desktop app
  - research pipeline
  - realtime pipeline

P1:
- installer veya bootstrap script
- environment self-check
- dependency health screen

---

## 5.8 Ticari Paketleme ve Satış Hazırlığı

Bir firmaya doğrudan kod satılmaz. Hikâye, çıktı ve güven satılır.

### Şu anda satılabilir olan şey

Kodun tamamı değil; şu paket:

> “kaçak elektrik analitiği için pilot gösterim paketi”

### Paketin içinde olması gerekenler

1. Temiz dashboard demo
2. 1 veya 2 güçlü demo veri senaryosu
3. Yönetici özeti
4. Teknik mimari sayfası
5. Model ve karar mantığı sayfası
6. Pilot uygulama yol haritası
7. Açıkça yazılmış kapsam dışı maddeler

### Firmaya söylenecek doğru cümle

“Bu ürün bugün tam saha üretim sistemi olarak değil, hızlı pilot ve karar destek platformu olarak en güçlü noktasında.”

### Firmaya söylenmemesi gereken cümle

- “tam hazır”
- “production deploy edelim”
- “her veri setine direkt uyar”
- “sahadaki kaçak oranını hemen düşürür”

Bunlar riskli.

### Daha doğru vaat

- pilot kurulum
- veri keşfi
- alarm önceliklendirme
- analist iş akışı
- modelleme çerçevesi
- gerçek veri uyarlama fazı

---

## 6. Önceliklendirilmiş Yapılacaklar Listesi

## P0 — Bu Hafta Bitmesi Gerekenler

| Alan | Yapılacak | Neden kritik |
|---|---|---|
| Ürün anlatısı | README, rapor ve demo söylemini “pilot platform” çizgisine çek | Güven kaybını önler |
| Model anlatısı | Resmi model setini netleştir, tutarsızlığı kaldır | Teknik güven sağlar |
| Veri şeması | Sentetik veriye long-format telemetry export ekle | Realtime ve gerçek veri hikâyesini güçlendirir |
| UI | Overview’ü decision-first yap, siyah tablo dilini kaldır | Demo kalitesini ciddi artırır |
| Error UX | Traceback ve debug dump’ı kullanıcıya göstermeyi bitir | Kurumsal görünüm için şart |
| Worklist | Table’a reason, trend, last seen, priority ekle | Dashboard’u aksiyona bağlar |
| Kurulum | Hafif dashboard kurulum yolu netleştir | Demo açılış riskini düşürür |
| Satış paketi | Pitch anlatısı + pilot scope + teslimatlar hazırla | Firma görüşmesi için gerekli |

## P1 — Sonraki 2-4 Hafta

| Alan | Yapılacak | Neden önemli |
|---|---|---|
| Gerçek veri | SGCC / London / utility-like dataset adaptasyonu | Sentetik başarıdan çıkış |
| Feature engineering | self-history, peer, seasonality, persistence, meter health | Gerçek saha doğruluğu |
| Registry | model versioning, threshold registry, metadata | Operasyonel güven |
| Explainability | SHAP veya tutarlı açıklama katmanı | Analist güveni |
| Realtime konsolidasyon | batch ve realtime feature sözlüğünü birleştir | Ürün bütünlüğü |
| API | REST veya service katmanı | Kurumsal entegrasyon |
| Auth | hashed auth + gerçek roller | Demo’dan pilota geçiş |

## P2 — Orta Vade

| Alan | Yapılacak | Neden stratejik |
|---|---|---|
| MLOps | experiment tracking, drift monitoring, model promotion | Sürdürülebilirlik |
| SaaS / on-prem opsiyonları | deployment strategy | Satış opsiyonu |
| Hardware / edge | ESP32 / CT / field prototype | Gösterim gücü |
| Kurumsal raporlama | executive brief, PDF packs, audit export | Yönetici kabulü |
| Akademik çıktı | IEEE paper, benchmark study | İtibar ve görünürlük |

---

## 7. 1 Haftalık Uygulanabilir Sprint Planı

## Gün 1

- resmi ürün konumunu netleştir
- model hikâyesini sabitle
- veri şeması kararını ver
- bu raporu referans belge olarak kilitle

## Gün 2

- sentetik veriye long telemetry export ekle
- dashboard ingest ile sentetik output’u bağla
- demo verisi üret

## Gün 3

- overview redesign
- KPI ve alert hierarchy
- siyah tablo temizliği
- error UI temizliği

## Gün 4

- customer list ve model performance ekranlarını worklist / comparison mantığına çek
- reason / trend / last seen alanlarını netleştir

## Gün 5

- pitch demo hazırlığı
- yönetici özeti
- ürün sunum akışı
- 3 dakikalık demo senaryosu

## Gün 6

- kurulum sadeleştirme
- demo runbook
- screenshot / kısa video / akış kontrolü

## Gün 7

- firma için gönderilecek versiyon
- pilot teklif dili
- “şu an ne var / faz 2’de ne yapılır” ayrımı

---

## 8. Ajanlara Verilebilecek Görev Paketleri

Bu bölüm doğrudan ajanlara verilebilir.

## Paket 1 — Codex: Veri Şeması Birleştirme

Amaç:
Sentetik veri üretimi ile realtime ingest arasında format uyumu kurmak.

Beklenen çıktı:
- sentetik veriyi long-format telemetry olarak export eden modül
- örnek CSV şablonları
- dashboard ve ingest ile çalışan demo veri seti
- kısa teknik not

Başarı kriteri:
- `generate_synthetic(...)` sonrası hem feature table hem telemetry table elde edilebilmesi
- `project/realtime_ingest/data_loader.py` şablonuyla uyum

Kısıt:
- mevcut feature-engine akışını bozma
- demo ve gerçek veri için ayrı adapter mantığı kur

## Paket 2 — Codex: Decision-First Dashboard Revizyonu

Amaç:
Dashboard’u veri ekranından analist karar ekranına çevirmek.

Beklenen çıktı:
- overview information hierarchy revizyonu
- KPI redesign
- hot zone / reason / priority / trend / last seen alanları
- siyah tabloların açık temalı worklist yapısına dönmesi
- kullanıcıya raw traceback gösterilmemesi

Başarı kriteri:
- ilk bakışta “neresi riskli, neden riskli, şimdi ne yapılmalı” görülmeli

## Paket 3 — Claude Code: Ürün ve Mimari Tutarlılık Denetimi

Amaç:
README, dashboard, engine, realtime ve ops center arasındaki anlatı ve teknik farkları çıkarıp tek ürün anlatısına dönüştürmek.

Beklenen çıktı:
- mevcut tutarsızlık listesi
- resmi ürün kapsamı önerisi
- “demo / pilot / production” ayrım dokümanı
- README revizyon önerisi

Başarı kriteri:
- 6 model, realtime, explainability, production-ready gibi iddialar net ve savunulabilir hale gelmeli

## Paket 4 — Claude Code veya Codex: Ticari Sunum Dosyası

Amaç:
Firmaya gösterilecek kısa ama ciddi pilot sunum paketi hazırlamak.

Beklenen çıktı:
- 1 sayfa executive summary
- 1 sayfa architecture
- 1 sayfa pilot scope
- 1 sayfa business value + ROI mantığı
- 1 sayfa riskler ve faz 2 planı

Başarı kriteri:
- “öğrenci projesi” algısından çıkıp “pilot platform” algısı oluşmalı

---

## 9. Firma Görüşmesinde Kullanılacak Ana Mesaj

### Söylenecek ana şey

MASS-AI; akıllı sayaç verilerinden kaçak elektrik ve tüketim anomalilerini tespit etmek için geliştirilmiş, çok modelli skorlayıcı, analist dashboard’u ve operasyonel vaka yönetimi içeren pilot seviyede bir yapay zeka platformudur.

### Vurgulanacak farklar

- sadece model değil, analist iş akışı var
- sadece batch değil, realtime ingest iskeleti de var
- sadece skor yok, ops center var
- sadece akademik değil, ürünleşme yönü var

### Dürüstçe söylenecek eksikler

- gerçek utility veri adaptasyonu ikinci faz işidir
- deployment ve security hardening pilot sonrası fazdır
- bazı araştırma modülleri ile aktif ürün yolu arasındaki konsolidasyon sürmektedir

Bu dürüstlük güven yaratır.

---

## 10. Ne Satılmalı?

Bu hafta sonunda satılması gereken şey:

> “tam bitmiş platform” değil, “pilot çalışma paketi”

Önerilen paket:

- dashboard demo
- örnek veri akışı
- vaka yönetimi ekranı
- mimari anlatım
- gerçek veri onboarding planı
- faz 2 geliştirme planı

Bu, hem daha gerçekçi hem daha ikna edici.

---

## 11. En Kritik Karar

Bu projenin kısa vadeli başarısını belirleyecek tek karar şu:

> Sen bunu “her şeyi yapan ama dağınık sistem” olarak mı göstereceksin, yoksa “tek probleme odaklı pilot ürün” olarak mı göstereceksin?

Doğru cevap ikincisi.

Çünkü şu an en büyük fırsat:
- problemi doğru seçmiş olman,
- ürün hissi veren modüller kurmuş olman,
- bunu kısa sürede ciddi görünen pilot haline getirebilecek noktada olman.

---

## 12. Nihai Tavsiye

Önümüzdeki 1 haftada odak şu olmalı:

1. ürün anlatısını tekleştir
2. sentetik veri ile realtime veri hikâyesini birleştir
3. UI’ı decision-first hale getir
4. model hikâyesindeki tutarsızlığı temizle
5. firmaya gösterilecek pilot paketi hazırla

Bu yapılırsa proje:

- sadece “güzel GitHub reposu” olmaktan çıkar,
- satılabilir pilot demoya dönüşür,
- gerçek veri adaptasyonu için ciddi görünen bir temel haline gelir,
- hatta ileride IEEE paper veya tez çalışması için de güçlü bir gövde sunar.

En net cümleyle:

> MASS-AI’nin şu an ihtiyacı daha fazla özellik değil, daha fazla bütünlük.

