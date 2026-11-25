<div align="center">

# T.C. SAMSUN ÜNİVERSİTESİ
## MÜHENDİSLİK FAKÜLTESİ
### YAZILIM MÜHENDİSLİĞİ BÖLÜMÜ

<br>

# YAPAY ZEKA DERSİ
# FİNAL PROJE RAPORU

<br>

## PREDICTIVE-EDGE
### Sinyal İşleme ve Yapay Sinir Ağı Tabanlı  
### Endüstriyel Motor Kestirimci Bakım Sistemi

<br><br>

**Hazırlayan**

**Abdulkadir ELALDI**
**Öğrenci No: 231118077**

<br>

**Ders:** Yapay Zeka
**Öğretim Üyesi:** Doç. Dr. Muammer TÜRKOĞLU

<br>

**Proje Deposu (GitHub):** https://github.com/abdulkadirelaldi/Predictive-edge-v2

<br>

**Haziran 2026**
**SAMSUN**

</div>

<br>

---

<div align="center">

## ÖZET

</div>

Bu projede, endüstriyel elektrik motorlarının **titreşim, sıcaklık ve akım** verilerini gerçek zamanlı olarak işleyerek olası rulman arızalarını önceden tahmin eden ve motorun **kalan kullanım ömrünü (RUL)** kestiren uçtan uca bir **kestirimci bakım (predictive maintenance)** sistemi geliştirilmiştir. Sistemin adı **Predictive-Edge**'dir.

Çalışmada, rulman dinamiği literatürüne (ISO 15243) dayanan fiziksel kurallarla **sentetik titreşim sinyalleri** üretilmiş; bu sinyaller **Hızlı Fourier Dönüşümü (FFT)** ile frekans domenine taşınarak **8 ayırt edici özellik** çıkarılmıştır. Çıkarılan özellikler, **scikit-learn** kütüphanesi ile eğitilen **Çok Katmanlı Algılayıcı (MLP)** tabanlı bir yapay sinir ağına verilerek motorun sağlıklı/arızalı sınıflandırması yapılmıştır.

Model, hold-out test setinde **%98,00 doğruluk**, **1,0000 duyarlılık (recall)** ve **1,0000 ROC-AUC** değerlerine ulaşmıştır. **5-katlı çapraz doğrulama (5-fold cross validation)** ile F1 skoru **0,9910 ± 0,0020** olarak doğrulanmış; bu da modelin şansa bağlı olmadığını ve tutarlı çalıştığını göstermiştir. Geliştirilen model yalnızca **289 parametre (1,13 KB)** içermekte olup, **STM32H7** gibi mikrodenetleyicilerde çalışabilecek **TinyML** ölçeğindedir.

Son olarak sonuçlar, **Flask + Chart.js** tabanlı gerçek zamanlı bir **web dashboard** üzerinde 200 motorluk bir filo için görselleştirilmiştir.

**Anahtar Kelimeler:** Kestirimci Bakım, Yapay Sinir Ağı, MLP, Sinyal İşleme, FFT, TinyML, Rulman Arıza Teşhisi, Makine Öğrenmesi

---

## İÇİNDEKİLER

1. [Giriş](#1-giriş)
2. [Problem Tanımı ve Motivasyon](#2-problem-tanımı-ve-motivasyon)
3. [Sistem Mimarisi](#3-sistem-mimarisi)
4. [Veri Üretimi (Fizik Tabanlı Sentetik Veri)](#4-veri-üretimi-fizik-tabanlı-sentetik-veri)
5. [Özellik Çıkarımı ve Sinyal İşleme](#5-özellik-çıkarımı-ve-sinyal-i̇şleme)
6. [Makine Öğrenmesi Modeli](#6-makine-öğrenmesi-modeli)
7. [Model Karşılaştırması ve Hiperparametre Optimizasyonu](#7-model-karşılaştırması-ve-hiperparametre-optimizasyonu)
8. [Deneysel Sonuçlar ve Değerlendirme](#8-deneysel-sonuçlar-ve-değerlendirme)
9. [Gömülü Sistem (TinyML / STM32H7) Analizi](#9-gömülü-sistem-tinyml--stm32h7-analizi)
10. [Web Dashboard ve Gerçek Zamanlı Çıkarım](#10-web-dashboard-ve-gerçek-zamanlı-çıkarım)
11. [Karşılaşılan Sorunlar ve Çözümleri](#11-karşılaşılan-sorunlar-ve-çözümleri)
12. [Sınırlılıklar ve Gelecek Çalışmalar](#12-sınırlılıklar-ve-gelecek-çalışmalar)
13. [Sonuç](#13-sonuç)
14. [Kaynaklar](#14-kaynaklar)
15. [Ekler](#15-ekler)

---

## 1. GİRİŞ

Endüstriyel tesislerde elektrik motorları, üretim hatlarının en kritik bileşenlerindendir. Bir motorun beklenmedik şekilde arızalanması yalnızca onarım maliyeti değil, aynı zamanda **üretim duruşu**, **iş gücü kaybı** ve **teslimat gecikmeleri** gibi dolaylı maliyetler de doğurur. Bu nedenle modern Endüstri 4.0 yaklaşımında, arızayı **olmadan önce tahmin etmek** büyük önem taşımaktadır.

Bu proje, **Yapay Zeka** dersi kapsamında, makine öğrenmesi ve sinyal işleme tekniklerini birleştirerek bir motorun arıza olasılığını ve kalan ömrünü tahmin eden çalışan bir prototip geliştirmeyi amaçlamaktadır. Proje; **veri üretimi → sinyal işleme → özellik çıkarımı → model eğitimi → değerlendirme → gerçek zamanlı arayüz** olmak üzere tüm aşamaları uçtan uca kapsamaktadır.

> **Proje kaynak kodu ve tüm dosyalar GitHub üzerinde açık olarak yayımlanmıştır:**
> **https://github.com/abdulkadirelaldi/Predictive-edge-v2**

### 1.1 Projenin Amaçları

- Rulman arızalarının fiziksel temelini (arıza frekansları) modelleyerek **gerçekçi sentetik veri** üretmek,
- FFT tabanlı sinyal işleme ile **ayırt edici özellikler** çıkarmak,
- Bir **MLP yapay sinir ağı** eğitip arıza sınıflandırması yapmak,
- Modeli **bilimsel olarak doğru bir metodoloji** (çapraz doğrulama, veri sızıntısı önleme) ile değerlendirmek,
- Modelin **gömülü sistemde (TinyML)** çalışabilirliğini analiz etmek,
- Sonuçları **gerçek zamanlı bir web panelinde** görselleştirmek.

### 1.2 Kullanılan Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Programlama Dili | Python 3.14 |
| Makine Öğrenmesi | scikit-learn (MLPClassifier) |
| Sinyal İşleme | NumPy, SciPy (`scipy.fft`, `scipy.signal`) |
| Veri Görselleştirme | Matplotlib, Seaborn |
| Web Backend | Flask 3.1 |
| Web Frontend | HTML5, CSS3, JavaScript, Chart.js |
| Model Kalıcılığı | joblib (`model.pkl`, `scaler.pkl`) |
| Geliştirme Ortamı | Jupyter Notebook |

---

## 2. PROBLEM TANIMI VE MOTİVASYON

### 2.1 Endüstriyel Motorlarda Arıza Mekanizması

Bir elektrik motoru temel olarak şu bileşenlerden oluşur: **rotor** (dönen parça), **stator** (sabit parça), **rulmanlar** (rotoru taşıyan bilyeli yataklar) ve **mil** (gücü ileten çubuk). Sahadaki istatistiklere göre en sık arızalanan bileşen **rulmanlardır**. Rulman arızasının başlıca nedenleri metal yorgunluğu, yetersiz yağlama, aşırı yük ve titreşim birikimidir.

Önemli olan nokta şudur: **rulman arızası aniden gerçekleşmez.** Arıza, haftalar hatta aylar boyunca titreşim ve sıcaklık değişimleri şeklinde belirtiler verir. Bu belirtiler sensörlerle ölçülebildiği için, makine öğrenmesi ile önceden tahmin edilebilirler.

### 2.2 Bakım Stratejilerinin Karşılaştırması

| Strateji | Yaklaşım | Dezavantaj |
|----------|----------|------------|
| **Düzeltici Bakım** | Arıza çıkınca tamir et | Motor durunca üretim durur, acil tamir pahalı |
| **Önleyici Bakım** | Periyodik bakım (örn. 3 ayda bir) | Gereksiz bakım maliyeti, çoğu motor bakım gerektirmez |
| **Kestirimci Bakım** | Sensöre bak, arızayı tahmin et | Veri ve model gerektirir → **bu projenin konusu** |

**Predictive-Edge**, üçüncü stratejiyi (kestirimci bakım) uygulayarak yalnızca gerçekten gerektiğinde bakım yapılmasını sağlar; böylece hem maliyeti düşürür hem de üretim kesintilerini engeller.

---

## 3. SİSTEM MİMARİSİ

Sistem, fiziksel sinyalden mühendise ulaşan bildirime kadar bütünleşik bir veri akış hattı (pipeline) üzerine kuruludur:

```
   Motor Sensörleri (Titreşim + Sıcaklık + Akım)
              │
              ▼
   Fizik Tabanlı Sinyal Üretimi (50/100/150 Hz harmonikleri + gürültü)
              │
              ▼
   Sinyal İşleme — FFT (Frekans Domeni Analizi)
              │
              ▼
   Özellik Çıkarımı (8 Özellik)
              │
              ▼
   StandardScaler → MLP Yapay Sinir Ağı
              │
        ┌─────┴─────┐
        ▼           ▼
   Arıza Olasılığı   Kalan Ömür (RUL)
        │           │
        └─────┬─────┘
              ▼
   Bakım Önceliği Kararı (LOW / MEDIUM / HIGH / CRITICAL)
              │
              ▼
   Flask REST API → Chart.js Web Dashboard → Mühendise Bildirim
```

### 3.1 Proje Dosya Yapısı

```
predictive-edge/
├── predictive_edge.ipynb     # Ana çalışma defteri (model eğitimi, analiz)
├── app.py                    # Flask backend (gerçek zamanlı çıkarım)
├── model.pkl                 # Eğitilmiş MLP modeli (21 KB)
├── scaler.pkl                # Eğitilmiş StandardScaler (807 B)
├── dashboard_output.json     # 200 motorluk ön-hesaplı filo verisi
├── templates/
│   └── index.html            # Dashboard arayüzü
└── *.png                     # Analiz görselleri (10 adet)
```

---

## 4. VERİ ÜRETİMİ (FİZİK TABANLI SENTETİK VERİ)

### 4.1 Neden Sentetik Veri?

Gerçek endüstriyel arıza verisi elde etmek pratikte çok zordur: bir motorun arızalanmasını beklemek aylar sürer, arıza etiketlemesi uzman gerektirir ve endüstriyel veriler genellikle gizlidir. Bu nedenle, **rulman fiziğine dayalı** sentetik veri üretilmiştir. Bu yaklaşım literatürde yaygın ve kabul gören bir yöntemdir.

### 4.2 Rulman Arıza Frekanslarının Hesaplanması (ISO 15243)

Motor parametreleri: **3000 RPM** → temel dönüş frekansı `f_rot = 3000/60 = 50 Hz`, **6 bilyeli rulman**, bilye/piston çapı oranı `d/D = 1/3`, temas açısı `α = 0°`.

**Dış Bilezik Arıza Frekansı (BPFO — Ball Pass Frequency Outer race):**

```
BPFO = (n/2) × f_rot × (1 − (d/D) × cos α)
     = (6/2) × 50 × (1 − 1/3 × 1)
     = 3 × 50 × 0,667
     = 100 Hz
```

Ayrıca **3 × f_rot = 150 Hz** mekanik gevşeklik (mechanical looseness) harmoniğine karşılık gelir. Bu iki frekans (100 Hz ve 150 Hz), tüm proje boyunca arıza göstergesi olarak kullanılan sabitlerdir.

### 4.3 Sinyal Modeli

Titreşim sinyali, fiziksel olarak anlamlı üç bileşenin ve gürültünün toplamı olarak modellenir:

```python
vibration = (
    1.0    * sin(2π · 50  · t) +   # Temel motor frekansı
    h2_amp * sin(2π · 100 · t) +   # BPFO — rulman dış bilezik arızası
    h3_amp * sin(2π · 150 · t) +   # 3× mil harmoniği — mekanik gevşeklik
    normal(0, noise_std, 1000)      # Gerçekçi ölçüm gürültüsü
)
temperature = clip(normal(45 + 23 · fault_level, 5),  18, 110)  # °C
current     = clip(normal(5.0 + 3.5 · fault_level, 0.5), 2.5, 14) # A
```

Burada `fault_level ∈ [0, 1]` arıza şiddetini temsil eder. Arıza şiddeti arttıkça harmonik genlikleri (`h2_amp`, `h3_amp`), gürültü, sıcaklık ve akım birlikte yükselir.

### 4.4 Örtüşen Dağılımlar — Gerçekçilik İçin Bilinçli Tasarım

Projenin ilk sürümünde sağlıklı ve arızalı motor dağılımları hiç örtüşmüyordu; bu durum model için problemi yapay olarak kolaylaştırıyor ve **%100 doğruluk** gibi gerçekçi olmayan bir sonuç veriyordu. Bu sorunu gidermek için sağlıklı motorların bazen düşük harmonik gösterebileceği, arızalı motorların bazen yüksek harmonik gösterebileceği şekilde dağılımlar **bilinçli olarak örtüştürülmüştür**. Bu sayede sınıflar arasında gerçek dünyadaki gibi bir belirsizlik bölgesi oluşmuş ve sonuç %100 yerine **%98** seviyesine inerek gerçekçi hale gelmiştir.

Toplam **1000 örnek** üretilmiştir: **500 sağlıklı + 500 arızalı** (dengeli veri kümesi).

---

## 5. ÖZELLİK ÇIKARIMI VE SİNYAL İŞLEME

### 5.1 FFT (Hızlı Fourier Dönüşümü)

Rulman arızaları zaman domeninde gözle ayırt edilmesi zor, ancak **frekans domeninde** belirgin enerji artışları olarak görünür. Bu nedenle titreşim sinyali FFT ile frekans domenine taşınır:

```python
freqs = scipy.fft.rfftfreq(N, d=1.0/SAMPLE_RATE)   # 0–500 Hz frekans ekseni
mag   = np.abs(scipy.fft.rfft(signal)) / N          # Her frekanstaki büyüklük
```

Örnekleme hızı `SAMPLE_RATE = 1000 Hz`'dir. **Nyquist teoremi** gereği bu hız ile 500 Hz'e kadar tüm frekanslar doğru ölçülebilir; ilgilenilen en yüksek frekans (150 Hz) bu sınırın çok altındadır.

### 5.2 Çıkarılan 8 Özellik

| # | Özellik | Hesaplama | Domain | Arıza Göstergesi |
|---|---------|-----------|--------|------------------|
| 1 | `vibration_rms` | √(ortalama(x²)) | Zaman | Yüksekse kötü |
| 2 | `peak_to_peak` | max − min | Zaman | Yüksekse kötü |
| 3 | `mean_abs` | ortalama(\|x\|) | Zaman | Yüksekse kötü |
| 4 | `harmonic_ratio` | (E₁₀₀ + E₁₅₀) / E₅₀ | Frekans | Yüksekse kötü |
| 5 | `spectral_entropy` | −Σ pᵢ·log(pᵢ) | Frekans | Yüksekse kötü |
| 6 | `n_spectral_peaks` | belirgin tepe sayısı | Frekans | Yüksekse kötü |
| 7 | `temperature_c` | sensör ölçümü | — | Yüksekse kötü |
| 8 | `current_a` | sensör ölçümü | — | Yüksekse kötü |

**En güçlü ayırt edici özellik `harmonic_ratio`'dur:** Arıza frekanslarındaki (100/150 Hz) enerjinin, temel motor frekansındaki (50 Hz) enerjiye oranıdır. Sağlıklı motorda bu oran küçük (~0,02), arızalı motorda büyüktür (~0,45).

**Spektral entropi**, Shannon entropisinin FFT büyüklüğüne uygulanmış halidir. Sağlıklı motorda enerji 50 Hz'de yoğunlaştığından entropi düşük; arızalı motorda enerji birçok frekansa yayıldığından (kaotik) entropi yüksektir.

---

## 6. MAKİNE ÖĞRENMESİ MODELİ

### 6.1 Çok Katmanlı Algılayıcı (MLP) Mimarisi

```
Giriş Katmanı  →  Gizli Katman 1  →  Gizli Katman 2  →  Çıkış Katmanı
   8 nöron      →     16 nöron      →      8 nöron      →     1 nöron
  (özellikler)      (ReLU)             (ReLU)            (Sigmoid → olasılık)
```

**Huni (funnel) mimarisi** tercih edilmiştir: her katman daha az nöronla daha soyut özellikler öğrenir. Toplam parametre sayısı:

```
Katman 1:  8×16 + 16 = 144
Katman 2: 16×8  +  8 = 136
Çıkış:     8×1  +  1 =   9
─────────────────────────────
TOPLAM             = 289 parametre
```

### 6.2 Aktivasyon Fonksiyonları

- **Gizli katmanlar — ReLU:** `ReLU(x) = max(0, x)`. Hesabı hızlıdır ve gradyan kaybı (vanishing gradient) problemini hafifletir.
- **Çıkış katmanı — Sigmoid:** `σ(x) = 1/(1+e⁻ˣ)`. Çıktıyı [0, 1] aralığına sıkıştırır ve doğrudan **arıza olasılığı** olarak yorumlanır.

### 6.3 Eğitim Yapılandırması ve Pipeline

Veri sızıntısını (data leakage) önlemek için **ölçekleme (scaler) ve model bir Pipeline içinde birleştirilmiştir.** Bu sayede çapraz doğrulamanın her katında scaler yalnızca o katın eğitim verisine `fit` edilir.

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.005,
        max_iter=300,
        random_state=42
    ))
])
```

- **Optimizasyon:** Adam (momentum + adaptif öğrenme hızı)
- **Kayıp fonksiyonu:** Binary cross-entropy → `−[y·log(ŷ) + (1−y)·log(1−ŷ)]`
- **Yakınsama:** Eğitim **137 iterasyonda** tamamlanmıştır.

### 6.4 Veri Sızıntısının Önlenmesi (Kritik Metodoloji)

StandardScaler **yalnızca eğitim setine** `fit` edilir; test setine sadece `transform` uygulanır. Aksi halde test setinin istatistikleri modele sızar ve performans yapay olarak şişer. Bu kural, projenin metodolojik doğruluğunun temel taşıdır.

```
z = (x − μ_train) / σ_train      # μ ve σ yalnızca eğitim verisinden
```

---

## 7. MODEL KARŞILAŞTIRMASI VE HİPERPARAMETRE OPTİMİZASYONU

### 7.1 Baseline Karşılaştırması

Beş farklı sınıflandırıcı aynı veri üzerinde karşılaştırılmıştır:

| Model | F1 Skoru | Elenme Nedeni |
|-------|----------|---------------|
| Lojistik Regresyon | ~0,990 | Yalnızca doğrusal karar sınırı |
| Random Forest | ~0,990 | 100+ ağaç → yüzlerce KB bellek |
| SVM (RBF) | ~0,990 | Destek vektörü başına O(n) kernel hesabı |
| Gradient Boosting | ~0,990 | Ardışık ağaçlar → yavaş çıkarım |
| **MLP (Seçilen)** | **~0,990** | **289 parametre → 1,13 KB, gömülü sistem uyumlu** |

**Önemli bulgu:** Tüm modellerin performansı birbirine çok yakındır. Bu nedenle seçim kriteri performans değil, **gömülü sistemde dağıtılabilirlik** olmuştur.

### 7.2 GridSearchCV — Hiperparametre Araması

```
6 mimari × 3 öğrenme hızı × 2 aktivasyon = 36 kombinasyon
Her kombinasyon 5-katlı CV ile test edildi = 180 model eğitimi
```

En iyi 5 sonuç:

| Sıra | Mimari | Aktivasyon | LR | F1 |
|------|--------|-----------|-----|-----|
| 1 | (32, 16) | relu | 0,005 | 0,9919 |
| 2 | **(16, 8)** | **relu** | **0,005** | **0,9910** |
| 3 | (16, 8) | tanh | 0,001 | 0,9908 |
| 4 | (32, 16) | tanh | 0,001 | 0,9906 |
| 5 | (8, 4) | relu | 0,005 | 0,9895 |

**Mühendislik kararı:** GridSearch en yüksek F1'i (32,16) mimarisinde bulmuş olsa da, aradaki fark yalnızca **0,0009**'dur ve bu standart sapmanın (≈0,004) altında kaldığından **istatistiksel olarak anlamsızdır.** Buna karşılık (32,16) mimarisi (16,8)'e göre **~3 kat daha fazla parametre** içerir. Bu nedenle, gömülü sistem kısıtı altında daha küçük olan **(16,8) mimarisi** tercih edilmiştir. Bu, performans ve donanım maliyeti arasında doğru bir ödünleşim (trade-off) örneğidir.

---

## 8. DENEYSEL SONUÇLAR VE DEĞERLENDİRME

### 8.1 Hold-out Test Sonuçları (200 örnek, hiç görülmemiş)

| Metrik | Değer | Yorum |
|--------|-------|-------|
| **Accuracy (Doğruluk)** | %98,00 | 200 örnekten 196'sı doğru |
| **Precision (Kesinlik)** | 0,9709 | Düşük yanlış alarm oranı |
| **Recall (Duyarlılık)** | **1,0000** | **Hiçbir arızalı motor kaçırılmadı** |
| **F1 Score** | 0,9804 | Precision ve recall'un harmonik ortalaması |
| **ROC-AUC** | 1,0000 | Her eşik değerinde mükemmel ayrım |

**Karışıklık Matrisi (Confusion Matrix):**

```
                  Tahmin: Sağlıklı    Tahmin: Arızalı
Gerçek: Sağlıklı        96                  3   (FP — gereksiz bakım)
Gerçek: Arızalı          0                 101  (TP — doğru yakalama)
```

### 8.2 Metriklerin Yorumu — Kestirimci Bakımda Neden Recall Önemlidir?

Kestirimci bakımda iki tür hata vardır:
- **Yanlış Pozitif (FP):** Sağlıklı motora "arızalı" demek → sadece **gereksiz bakım maliyeti**.
- **Yanlış Negatif (FN):** Arızalı motora "sağlıklı" demek → **fabrika duruşu, en tehlikeli hata.**

Bu nedenle bu uygulamada **Recall (duyarlılık)** en kritik metriktir. Modelimiz **Recall = 1,0000** ile tüm arızalı motorları yakalamış, yalnızca 3 sağlıklı motoru yanlışlıkla arızalı işaretlemiştir (kabul edilebilir bir takas).

### 8.3 5-Katlı Çapraz Doğrulama Sonuçları

Tek bir hold-out bölmesinin şansa bağlı olabileceği endişesini gidermek için 5-katlı çapraz doğrulama yapılmıştır:

| Metrik | Ortalama | Std. Sapma |
|--------|----------|------------|
| Accuracy | 0,9810 | 0,0080 |
| Precision | 0,9615 | 0,0120 |
| Recall | 1,0000 | 0,0000 |
| F1 | 0,9910 | 0,0020 |
| ROC-AUC | 0,9994 | 0,0006 |

**Çok düşük standart sapmalar**, modelin farklı veri bölmelerinde tutarlı çalıştığını ve sonucun şansa bağlı olmadığını kanıtlamaktadır.

### 8.4 Kalan Kullanım Ömrü (RUL) Hesabı

Arıza olasılığından kalan ömür, lineer degradasyon modeliyle kestirilir:

```
RUL = T_max × (1 − P_arıza) + ε      (T_max = 8760 saat = 1 yıl)
```

Örnek: `P_arıza = 0,30` → `RUL = 8760 × 0,70 ≈ 6132 saat ≈ 255 gün`. Hesaplanan ömre göre bakım önceliği belirlenir:

| Arıza Olasılığı | Öncelik | Önerilen Eylem |
|-----------------|---------|----------------|
| < %35 | LOW | Rutin bakım planla |
| %35–65 | MEDIUM | 72 saat içinde kontrol et |
| %65–85 | HIGH | 24 saat içinde müdahale et |
| > %85 | CRITICAL | Motoru hemen durdur |

---

## 9. GÖMÜLÜ SİSTEM (TinyML / STM32H7) ANALİZİ

Projenin en özgün yönlerinden biri, eğitilen modelin **mikrodenetleyicide çalışabilirliğinin** somut hesaplarla gösterilmesidir (TinyML).

**Hedef donanım — STM32H7:** ARM Cortex-M7 @ 480 MHz, 1 MB SRAM, 2 MB Flash.

| Ölçüt | Değer |
|-------|-------|
| Model boyutu (Float32) | 289 × 4 byte = **1,13 KB** |
| Int8 niceleme sonrası | **0,28 KB** (4× küçülme) |
| Flash kullanımı | **%0,055** (2048 KB'ın) |
| MLP ileri besleme süresi | ~0,6 μs |
| FFT (1000 nokta) süresi | ~20 μs |
| Toplam çıkarım süresi | **~21 μs** |
| Maksimum çıkarım hızı | **~47.000 çıkarım/saniye** |

**STM32Cube.AI** araç zinciri ile `model.pkl` → ONNX → otomatik C kodu üretimi mümkündür. Bu, modelin internet bağlantısı gerektirmeden, düşük gecikmeyle ve güvenli şekilde **edge** üzerinde çalışabileceği anlamına gelir.

---

## 10. WEB DASHBOARD VE GERÇEK ZAMANLI ÇIKARIM

Sistem, **Flask** backend ve **Chart.js** frontend ile gerçek zamanlı bir izleme paneli sunar. Panel 200 motorluk bir filoyu yönetir.

### 10.1 REST API Uç Noktaları

| Endpoint | İşlev |
|----------|-------|
| `GET /` | HTML dashboard sayfası |
| `GET /api/simulate` | Gerçek zamanlı model çıkarımı (canlı sinyal → FFT → MLP) |
| `GET /api/motors` | 200 motorun listesi |
| `GET /api/stats` | Filo istatistikleri |

### 10.2 Gerçek Zamanlı Çıkarım Akışı

`/api/simulate` çağrıldığında **gerçek bir çıkarım** yapılır (önceden hesaplanmış değer döndürülmez):

```
Her 3 saniyede:
  1. JavaScript → fetch('/api/simulate')
  2. Flask → generate_live_signal()  (fiziksel sinyal üret)
          → extract_features()       (FFT'yi gerçekten hesapla)
          → run_inference()          (StandardScaler + MLP.predict_proba)
          → RUL ve bakım önceliği hesapla
  3. JSON yanıt → gauge, zaman serisi ve KPI kartlarını güncelle
  4. priority='CRITICAL' ise → alarm tetikle
```

Ölçülen tipik çıkarım süresi (Python ek yükü dahil) **2–8 ms**'dir.

### 10.3 Arayüz Bileşenleri

- **Gauge Chart:** Yarım daire arıza göstergesi (yeşil/sarı/turuncu/kırmızı eşikler).
- **Zaman Serisi:** Sıcaklık, akım ve arıza olasılığının son 40 okuması, canlı akış.
- **Motor Grid:** 200 motor kartı; filtreleme ve detay modalı.
- **KPI Kartları:** Toplam / sağlıklı / uyarı / yüksek risk / kritik motor sayıları.
- **Model Metrikleri Paneli:** CV skorları, mimari, parametre sayısı.

---

## 11. KARŞILAŞILAN SORUNLAR VE ÇÖZÜMLERİ

Bu bölüm, projenin geliştirilmesi sırasında yaşanan gerçek sorunları ve çözümlerini şeffaf biçimde belgeler — iyi mühendislik pratiğinin bir göstergesi olarak.

### Sorun 1 — Model %50 Doğrulukta Kaldı (Hiçbir Şey Öğrenmedi)
**Belirti:** Tüm örnekler "arızalı" tahmin edildi.
**Neden:** `early_stopping=True` parametresi nedeniyle eğitim, model yakınsamadan 22. iterasyonda durdu.
**Çözüm:** `early_stopping=False` ve `learning_rate_init=0.005` ile 137 iterasyonda gerçek yakınsama sağlandı.
**Ders:** Erken durdurma her zaman faydalı değildir; kayıp eğrisi izlenmeli, gerçekten platoya mı ulaşıldığı doğrulanmalıdır.

### Sorun 2 — %100 Doğruluk (Gerçekçi Değil)
**Belirti:** Model tüm test örneklerini hatasız sınıflandırdı.
**Neden:** Sağlıklı ve arızalı dağılımları (sıcaklık, harmonik) hiç örtüşmüyordu.
**Çözüm:** Dağılımlar gerçekçi belirsizliği yansıtacak şekilde örtüştürüldü → %98 doğruluk.
**Ders:** Mükemmel sonuç istatistiksel olarak şüphelidir; gerçek veri asla %100 ayrılabilir değildir.

### Sorun 3 — Seaborn Violin Plot Hatası
**Neden:** Yeni Seaborn sürümleri palette anahtarı olarak string bekliyor (integer değil).
**Çözüm:** Etiketler `{0:'Saglikli', 1:'Arizali'}` olarak string'e eşlendi.
**Ders:** Kütüphane sürüm farkları küçük API değişimlerine yol açar; sürüm sabitleme önemlidir.

### Sorun 4 — Flask Port Çakışması
**Neden:** Eski Python süreci arka planda çalışmaya devam etti (port 5050 meşgul).
**Çözüm:** `pkill -f "python3 app.py"` ile eski süreç sonlandırıldı.
**Ders:** Sunucu süreçleri düzgün sonlandırılmalı; üretimde systemd/supervisor kullanılır.

---

## 12. SINIRLILIKLAR VE GELECEK ÇALIŞMALAR

Bilimsel dürüstlük gereği projenin sınırlılıkları açıkça belirtilmiştir:

**Sınırlılıklar:**
1. **Sentetik veri kullanılmıştır** — gerçek sensör verisiyle doğrulama yapılmamıştır.
2. **RUL modeli basittir** — lineer degradasyon kullanılmıştır; Weibull veya LSTM daha gerçekçi olurdu.
3. **Test seti küçüktür** (200 örnek) — gerçek dağıtım için binlerce örnek gerekir.
4. **Gerçek streaming yoktur** — Kafka/MQTT gibi gerçek akış altyapısı yerine simülasyon kullanılmıştır.

**Gelecek Çalışmalar:**
- Gerçek **CWRU (Case Western Reserve University)** rulman veri seti ile test,
- **Weibull dağılımı** tabanlı istatistiksel RUL modeli,
- **LSTM** ile zaman serisi anomali tespiti,
- STM32H7 üzerinde **gerçek firmware dağıtımı**,
- **OPC-UA** ile endüstriyel SCADA entegrasyonu.

---

## 13. SONUÇ

Bu projede, endüstriyel kestirimci bakımın temel prensiplerini uçtan uca uygulayan **Predictive-Edge** prototip sistemi geliştirilmiştir. Fizik tabanlı veri üretiminden FFT sinyal işlemeye, makine öğrenmesi modellemesinden gerçek zamanlı web panele kadar tüm bileşenler bütünleşik biçimde çalışmaktadır.

Geliştirilen MLP modeli **%98 doğruluk**, **1,0000 recall** ve **0,9910 ± 0,0020 (CV) F1** skorlarıyla yüksek ve **tutarlı** bir performans göstermiştir. Yalnızca **289 parametre (1,13 KB)** ile model, **STM32H7** mikrodenetleyicide çalışabilecek TinyML ölçeğindedir.

**Projenin akademik katkıları:**
1. Sinyal işleme ve makine öğrenmesinin entegrasyonu,
2. TinyML kısıtları altında metodolojik model seçimi,
3. Gerçekçi (örtüşen dağılımlı) sentetik veri üretimi,
4. Pipeline tabanlı veri sızıntısı önleme.

Bu çalışma, yapay zekânın yalnızca bir sınıflandırma probleminden ibaret olmadığını; **veri kalitesi, doğru değerlendirme metodolojisi, donanım kısıtları ve gerçek dünya dağıtımı** gibi mühendislik boyutlarının da en az model kadar önemli olduğunu göstermektedir.

---

## 14. KAYNAKLAR

1. ISO 15243:2017 — *Rolling bearings — Damage and failures — Terms, characteristics and causes.*
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.
3. Harris, C. R. et al. (2020). *Array programming with NumPy.* Nature, 585, 357–362.
4. Virtanen, P. et al. (2020). *SciPy 1.0: fundamental algorithms for scientific computing in Python.* Nature Methods, 17, 261–272.
5. Cooley, J. W. & Tukey, J. W. (1965). *An algorithm for the machine calculation of complex Fourier series.* Mathematics of Computation, 19(90), 297–301.
6. Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
7. STMicroelectronics. *STM32Cube.AI — Artificial Intelligence (AI) Documentation.*
8. Case Western Reserve University Bearing Data Center — *Bearing Fault Dataset.*

---

## 15. EKLER

Proje dizininde yer alan üretilmiş analiz görselleri:

| Dosya | İçerik |
|-------|--------|
| `bearing_physics.png` | Rulman arıza frekansı analizi |
| `signal_analysis.png` | Zaman domeni + FFT frekans spektrumu |
| `feature_distribution.png` | Özelliklerin sağlıklı/arızalı violin grafiği |
| `correlation_matrix.png` | Özellikler arası korelasyon haritası |
| `model_performance.png` | Karışıklık matrisi + ROC eğrisi |
| `learning_curve.png` | Öğrenme eğrisi (overfitting analizi) |
| `cv_metrics.png` | 5-katlı çapraz doğrulama metrikleri |
| `baseline_comparison.png` | Model karşılaştırma grafiği |
| `gridsearch_results.png` | Hiperparametre ısı haritası |
| `stm32_analysis.png` | Gömülü sistem (STM32H7) analizi |

**Sistemi çalıştırmak için:**
```bash
git clone https://github.com/abdulkadirelaldi/Predictive-edge-v2
cd Predictive-edge-v2
python3 app.py     →     http://localhost:5050
```

<br>

<div align="center">

---

*Bu rapor, Samsun Üniversitesi Yazılım Mühendisliği Bölümü Yapay Zeka dersi final ödevi kapsamında hazırlanmıştır.*

**Abdulkadir ELALDI — 231118077 — Haziran 2026**

</div>
