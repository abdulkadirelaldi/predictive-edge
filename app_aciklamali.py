"""
=============================================================
Predictive-Edge Dashboard v2.0 — Flask Backend
=============================================================

Bu dosya sistemin beynidir. Üç temel görevi var:
  1. Model dosyalarını yükleyip çıkarım yapmak (scaler + MLP)
  2. API endpoint'leri sunmak (/api/simulate, /api/motors, ...)
  3. HTML dashboard sayfasını render etmek (/)

Çalıştırma: python3 app.py → http://localhost:5050
=============================================================
"""

# ── KÜTÜPHANE İMPORTLARI ─────────────────────────────────────────────────────

import json      # JSON dosyası okuma ve HTTP yanıtı oluşturma için
import random    # Rastgele arıza seviyesi ve motor ID üretimi için
import time      # Çıkarım süresini ölçmek için (perf_counter)

import numpy as np                          # Sayısal hesaplama: sinüs, sqrt, clip, vb.
import scipy.fft as spfft                   # FFT hesabı — zaman domenini frekans domenine çevirir
from scipy.signal import find_peaks         # FFT büyüklük grafiğindeki tepe noktaları
from scipy.stats import entropy as scipy_entropy  # Shannon entropi — spektral düzensizlik ölçer
import joblib                               # .pkl model dosyalarını yükler/kaydeder

from flask import Flask, render_template, jsonify
# Flask      → web sunucusu çerçevesi
# render_template → HTML dosyasını Jinja2 şablonuyla render eder
# jsonify    → Python dict'i JSON HTTP yanıtına çevirir

# ── FLASK UYGULAMASI OLUŞTURMA ────────────────────────────────────────────────

app = Flask(__name__)
# __name__ → Python'un bulunduğu modülün adı.
# Flask bu değeri kullanarak templates/ ve static/ klasörlerini bulur.

# ── MODEL VE SCALER YÜKLEME ───────────────────────────────────────────────────

model  = joblib.load('model.pkl')
# model.pkl: Eğitilmiş MLPClassifier nesnesi (21 KB)
# Ağırlıklar (289 parametre) bu dosyada saklanıyor.
# Yeniden eğitmeden predict_proba() çağrısı yapılabilir.

scaler = joblib.load('scaler.pkl')
# scaler.pkl: Eğitilmiş StandardScaler nesnesi (807 byte)
# Her özelliği z = (x - ortalama) / std_sapma şeklinde ölçekler.
# SADECE transform() kullanılır — fit() değil! (data leakage önlemi)

# ── SİNYAL PARAMETRELERİ ──────────────────────────────────────────────────────

SAMPLE_RATE = 1000
# Saniyede 1000 örnekleme noktası.
# Nyquist teoremi: 500 Hz'ye kadar olan frekansları doğru ölçebiliriz.
# Pratikte: motor 50 Hz, BPFO 100 Hz, SHAFT_3X 150 Hz → hepsi 500 Hz altında, güvenli.

t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
# 0'dan 1.0 saniyeye kadar 1000 eşit aralıklı zaman noktası üret.
# endpoint=False → son nokta (1.0) dahil edilmez, böylece sinüs periyodik kalır.
# Örnek değerler: [0.000, 0.001, 0.002, ..., 0.999]
# Bu dizi tüm sinüs hesaplarında kullanılacak.

# ── RULMAN ARIZA FREKANSLARI ──────────────────────────────────────────────────

BPFO     = 100.0
# Ball Pass Frequency Outer Race (Dış Bilezik Arıza Frekansı)
# Formül: BPFO = (n/2) × f_rot × (1 - d/D × cosα)
#   n = 6 bilye, f_rot = 50 Hz (3000 RPM), d/D = 1/3, α = 0°
#   → (6/2) × 50 × (1 - 1/3) = 3 × 50 × 0.667 = 100 Hz
# Dış bilezik arızası varsa → bu frekansta güçlü titreşim

SHAFT_3X = 150.0
# 3. Harmonik (3 × f_rot = 3 × 50 = 150 Hz)
# Mekanik gevşeklik (mechanical looseness) arızasının imzası.
# Rulman/yatak gevşediğinde 3. harmonik belirgin şekilde yükselir.

# ── ÖN HESAPLANMIŞ FİLO VERİSİ ───────────────────────────────────────────────

with open('dashboard_output.json', encoding='utf-8') as f:
    MOTORS = json.load(f)
# 200 motorun sensor okumalarını ve tahminlerini içeren JSON.
# Notebook'ta oluşturuldu, her motor için fault_prob, RUL, bakım önceliği var.
# encoding='utf-8' → Türkçe karakterlerin bozulmaması için zorunlu.

# ── MODEL METRİKLERİ ──────────────────────────────────────────────────────────

MODEL_METRICS = {
    'accuracy'  : 98.00,     # Hold-out test seti üzerinde: 200'den 196 doğru
    'f1'        : 0.9804,    # Precision ve Recall'un harmonik ortalaması
    'precision' : 0.9615,    # "Arızalı" dediğimizin kaçı gerçekten arızalı?
    'recall'    : 1.0000,    # Gerçek arızaların kaçını yakaladık? (1.0 = hiç kaçırmadık)
    'roc_auc'   : 1.0000,    # Her eşik değerinde mükemmel ayrım (1.0 = ideal)
    'cv_f1'     : '0.9910 ± 0.0020',  # 5-Fold CV ortalaması ± standart sapma
    'cv_auc'    : '0.9994 ± 0.0006',  # CV ROC-AUC: tutarlı, şansa bağlı değil
    'architecture': '8 → 16 → 8 → 1', # Giriş → Gizli1 → Gizli2 → Çıkış
    'iterations': 137,        # Modelin yakınsaması için gereken iterasyon sayısı
    'samples'   : 1000,       # Toplam eğitim veri seti boyutu (500 sağlıklı + 500 arızalı)
    'features'  : 8,          # Giriş katmanındaki özellik sayısı
    'params'    : 289,        # Toplam eğitilebilir parametre (ağırlıklar + biaslar)
    'memory_kb' : 1.13,       # Float32 formatında model boyutu (289 × 4 byte = 1156 byte)
}
# Bu sözlük doğrudan HTML dashboard'una aktarılıyor ({{ metrics.accuracy }} gibi)

# ═══════════════════════════════════════════════════════════════════════════════
# GERÇEK ZAMANLI SİNYAL ÜRETİMİ
# ═══════════════════════════════════════════════════════════════════════════════

def generate_live_signal(fault_level: float):
    """
    fault_level ∈ [0, 1] değerine göre fiziksel motor sinyali üretir.

    0.0 = sağlıklı motor (düşük gürültü, zayıf harmonikler)
    1.0 = ağır arızalı motor (yüksek gürültü, güçlü BPFO+3X harmonikleri)

    Döndürür: (vibration[1000], temperature_float, current_float)
    """

    # ── Gürültü seviyesi ──────────────────────────────────────────────────────
    noise_std = 0.05 + fault_level * 0.45
    # fault_level=0   → noise_std=0.05 (çok temiz sinyal)
    # fault_level=0.5 → noise_std=0.275
    # fault_level=1.0 → noise_std=0.50 (çok gürültülü)
    # Arızalı motor: rulman hasarı mekanik titreşimi geniş frekanslara yayar = daha fazla gürültü

    # ── Harmonik amplitüdler ──────────────────────────────────────────────────
    h2_amp = fault_level * np.random.uniform(0.15, 0.75)
    # BPFO (100 Hz) harmonik amplitüdü
    # fault_level=0   → h2_amp=0 (harmonik yok)
    # fault_level=1.0 → h2_amp=0.15 ile 0.75 arası rastgele (güçlü harmonik)
    # np.random.uniform → her çağrıda biraz farklı → gerçekçilik

    h3_amp = fault_level * np.random.uniform(0.08, 0.45)
    # SHAFT_3X (150 Hz) harmonik amplitüdü
    # Mekanik gevşeklik harmonik — h2_amp'tan biraz daha zayıf

    # ── Titreşim sinyali (sinüs superposizyonu) ───────────────────────────────
    vibration = (
        1.0    * np.sin(2 * np.pi * 50       * t) +
        # Ana motor frekansı: 3000 RPM → 50 Hz. Her zaman var, amplitude=1.0 sabit.
        # sin(2π × 50 × t): saniyede 50 kez tam salınım

        h2_amp * np.sin(2 * np.pi * BPFO     * t) +
        # BPFO arıza harmonik: sadece arızalı motorda belirgin
        # sin(2π × 100 × t): saniyede 100 kez salınım

        h3_amp * np.sin(2 * np.pi * SHAFT_3X * t) +
        # Mekanik gevşeklik harmonik: 3×50=150 Hz
        # sin(2π × 150 × t): saniyede 150 kez salınım

        np.random.normal(0, noise_std, SAMPLE_RATE)
        # Gaussian gürültü: ortalama=0, std=noise_std, 1000 nokta
        # Gerçek sensörlerde elektronik ve çevresel gürültü kaçınılmaz
    )
    # Sonuç: 1000 uzunlukta NumPy array, fiziksel motor titreşimini simüle eder

    # ── Sıcaklık ──────────────────────────────────────────────────────────────
    temperature = float(np.clip(
        np.random.normal(45.0 + fault_level * 23.0, 5.0), 18, 110
    ))
    # np.random.normal(ortalama, std): Gaussian dağılım
    #   Sağlıklı (fault=0): ortalama=45°C, std=5°C
    #   Arızalı  (fault=1): ortalama=68°C, std=5°C
    #   Arıza → sürtünme artışı → sıcaklık yükselir (fiziksel gerçek)
    # np.clip(değer, 18, 110): 18°C altına veya 110°C üstüne çıkmasını önler
    # float() → numpy float'ı Python float'a çevirir (JSON serialize için gerekli)

    # ── Akım ──────────────────────────────────────────────────────────────────
    current = float(np.clip(
        np.random.normal(5.0 + fault_level * 3.5, 0.5), 2.5, 14.0
    ))
    # Sağlıklı (fault=0): ortalama=5.0A, std=0.5A
    # Arızalı  (fault=1): ortalama=8.5A, std=0.5A
    # Arıza → mekanik yük artışı → motor daha fazla akım çeker
    # np.clip(değer, 2.5, 14.0): gerçekçi amper aralığında kalır

    return vibration, temperature, current
    # Üç değer döndürülüyor:
    #   vibration   → 1000 noktalık NumPy array
    #   temperature → tek float (°C)
    #   current     → tek float (Amper)


# ═══════════════════════════════════════════════════════════════════════════════
# ÖZELLİK ÇIKARIMI (FEATURE EXTRACTION)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(signal, temperature, current):
    """
    Ham sensör verisinden 8 sayısal özellik çıkarır.
    Bu özellikler MLP modelinin giriş vektörünü oluşturur.

    Notebook Modül 2 ile birebir aynı — çıkarım sırasında da
    eğitim zamanındaki aynı özellikler üretilmeli (pipeline tutarlılığı).
    """

    N     = len(signal)           # 1000 (sinyal uzunluğu)
    freqs = spfft.rfftfreq(N, d=1.0 / SAMPLE_RATE)
    # rfftfreq: FFT çıktısındaki her bin'e karşılık gelen frekansları üretir
    # N=1000, SAMPLE_RATE=1000 → 0'dan 500 Hz'e kadar 501 frekans noktası
    # d=1.0/SAMPLE_RATE → örnekleme periyodu (saniye cinsinden)

    mag = np.abs(spfft.rfft(signal)) / N
    # rfft: sadece pozitif frekansları hesaplar (reel sinyal için yeterli, 2× hızlı)
    # np.abs: karmaşık sayının büyüklüğünü alır → gerçek amplitüd
    # / N: normalize eder → amplitüd gerçek sinüs amplitüdüne karşılık gelir

    # ── İç yardımcı fonksiyon: frekans etrafındaki enerji ─────────────────────
    def freq_energy(f_target, bw=2.0):
        """
        f_target Hz etrafında ±bw Hz penceredeki toplam enerji.
        Enerji = büyüklük²'nin toplamı (Parseval teoremi)
        """
        mask = (freqs >= f_target - bw) & (freqs <= f_target + bw)
        # mask: f_target-2 ile f_target+2 Hz arasındaki True/False dizisi
        # ±2 Hz pencere: frekans çözünürlüğü 1 Hz olduğundan 4-5 bin içerir
        return float(np.sum(mag[mask] ** 2)) + 1e-10
        # mag[mask]²'yi topla → o frekans bandındaki güç
        # + 1e-10 → sıfıra bölme önlemi (sağlıklı motorda payda sıfır olabilir)

    # ── Özellik 4: Harmonik Oran ───────────────────────────────────────────────
    harmonic_ratio = (freq_energy(BPFO) + freq_energy(SHAFT_3X)) / freq_energy(50.0)
    # Arıza frekanslarındaki toplam enerji / temel motor frekansı enerjisi
    # Sağlıklı: 100 Hz ve 150 Hz'de az enerji → oran küçük (~0.02)
    # Arızalı:  100 Hz ve 150 Hz'de fazla enerji → oran büyük (~0.40)
    # Model için en ayırt edici özelliklerden biri

    # ── Özellik 5: Spektral Entropi ────────────────────────────────────────────
    mag_norm = mag / (mag.sum() + 1e-10)
    # FFT büyüklüğünü normalize et → toplam 1'e eşit olacak şekilde
    # Böylece bir olasılık dağılımı elde edilir
    # + 1e-10 → sıfır mag dizisi durumunda bölme hatasını önler

    spectral_entropy = float(scipy_entropy(mag_norm + 1e-12))
    # Shannon entropi: H = -Σ p × log(p)
    # Sağlıklı motor: enerji 50 Hz'de yoğunlaşmış → düşük entropi (~4.0)
    # Arızalı motor: enerji birçok frekansa dağılmış → yüksek entropi (~5.5)
    # + 1e-12 → log(0) tanımsızlığını önler

    # ── Özellik 6: Spektral Tepe Sayısı ───────────────────────────────────────
    peaks, _ = find_peaks(mag, height=mag.mean() * 3, distance=5)
    # mag grafiğindeki belirgin tepeleri bulur
    # height=mag.mean()*3 → ortalamanın 3 katından yüksek tepeler aranır
    # distance=5 → iki tepe arası en az 5 bin (5 Hz) mesafe olmalı
    # Sağlıklı: 1-3 tepe (50 Hz + belki harmonikler)
    # Arızalı: daha fazla tepe (BPFO, SHAFT_3X + yan bantlar)

    # ── 8 özelliği bir liste olarak döndür ────────────────────────────────────
    return [
        float(np.sqrt(np.mean(signal ** 2))),   # Özellik 1: vibration_rms
        # RMS = √(mean(x²)) → sinyalin ortalama güç seviyesi
        # Arızalı motor daha fazla titreşir → RMS yüksek

        float(signal.max() - signal.min()),       # Özellik 2: peak_to_peak
        # Sinyalin en yüksek ile en düşük noktası arasındaki fark
        # Ani darbeler (rulman çarpması) bu değeri artırır

        float(np.mean(np.abs(signal))),           # Özellik 3: mean_abs
        # Sinyalin mutlak değerinin ortalaması (MAV - Mean Absolute Value)
        # RMS'e benzer ama kare almadan → farklı hassasiyet

        harmonic_ratio,                           # Özellik 4: yukarıda hesaplandı

        spectral_entropy,                         # Özellik 5: yukarıda hesaplandı

        len(peaks),                               # Özellik 6: tepe sayısı (tam sayı)

        temperature,                              # Özellik 7: °C (float)
        # Fiziksel ölçüm → arıza korelasyonu var ama gürültülü

        current,                                  # Özellik 8: Amper (float)
        # Yük artışı → akım artışı → arıza göstergesi
    ]
    # Bu 8 sayı, StandardScaler'a ve ardından MLP'ye girecek vektör


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ÇIKARIMI (INFERENCE)
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(features):
    """
    8 ham özelliği alır, scaler + MLP pipeline'ını çalıştırır.
    Döndürür: (prob, pred)
      prob → float [0,1]: arızalı olma olasılığı
      pred → int {0,1}: 0=sağlıklı, 1=arızalı
    """

    features_scaled = scaler.transform([features])
    # scaler.transform: her özelliği z = (x - μ) / σ formülüyle normalleştirir
    # [features] → tek örnek için 2D array gerekli: shape (1, 8)
    # μ ve σ eğitim sırasında öğrenildi, scaler.pkl'da saklı
    # NEDEN: 1000+ sıcaklık vs 0.02 harmonik oran → model dengeli öğrenemez
    # Normalleştirme ile her özellik yaklaşık [-3, +3] aralığına çekilir

    prob = float(model.predict_proba(features_scaled)[0][1])
    # predict_proba → her sınıf için olasılık döndürür
    # [0] → tek örnek (batch'in ilk elemanı)
    # [1] → "arızalı" sınıfının olasılığı (index 0 = sağlıklı, 1 = arızalı)
    # float() → numpy float64'ü Python float'a çevirir (JSON serialize)
    # Örnek: 0.87 → %87 arızalı olasılığı

    pred = int(model.predict(features_scaled)[0])
    # predict → olasılığı 0.5 eşiğine göre kesin sınıfa çevirir
    # 0.5'in altı → 0 (sağlıklı), üstü → 1 (arızalı)
    # int() → numpy int'i Python int'e çevirir

    return prob, pred


# ═══════════════════════════════════════════════════════════════════════════════
# BAKIM KARARI VE RUL HESABI
# ═══════════════════════════════════════════════════════════════════════════════

def get_maintenance(fault_prob, rul_hours):
    """
    Arıza olasılığına göre bakım önceliği ve önerilen eylemi belirler.
    """

    # ── Eşik tabanlı karar ────────────────────────────────────────────────────
    if fault_prob < 0.35:
        priority, status = 'LOW',      'NORMAL'
        # Arıza olasılığı düşük → rutin bakım planla
    elif fault_prob < 0.65:
        priority, status = 'MEDIUM',   'UYARI'
        # Geçiş bölgesi → 72 saat içinde kontrol et
    elif fault_prob < 0.85:
        priority, status = 'HIGH',     'KRITIK ALARM'
        # Yüksek olasılık → 24 saat içinde müdahale
    else:
        priority, status = 'CRITICAL', 'ACIL MUDAHALE'
        # %85+ → hemen durdur, üretim kaybından kaçın

    return {
        'status'  : status,
        'priority': priority,
        'alarm'   : fault_prob >= 0.65,    # True ise dashboard'da alarm çalar
        'rul_hours': round(rul_hours, 1),  # Saat cinsinden kalan ömür
        'rul_days' : round(rul_hours / 24, 1),  # Gün cinsinden kalan ömür (1 gün = 24 saat)
        'recommended_action': {
            'LOW'     : 'Rutin bakim planla',
            'MEDIUM'  : '72 saat icinde kontrol et',
            'HIGH'    : '24 saat icinde mudahale et',
            'CRITICAL': 'Motoru hemen durdur',
        }[priority],
        # priority değerine göre sözlükten doğrudan string seç
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FİLO İSTATİSTİKLERİ
# ═══════════════════════════════════════════════════════════════════════════════

def _stats():
    """
    MOTORS listesinden filo özet istatistiklerini hesaplar.
    Bu değerler dashboard KPI kartlarında gösterilir.
    """
    p = [m['maintenance']['priority'] for m in MOTORS]
    # List comprehension: her motor için bakım önceliğini al
    # Örnek: ['LOW', 'LOW', 'HIGH', 'CRITICAL', ...]

    return {
        'total'   : len(MOTORS),       # Toplam motor sayısı (200)
        'low'     : p.count('LOW'),    # Sağlıklı motor sayısı
        'medium'  : p.count('MEDIUM'),  # Uyarı durumundaki
        'high'    : p.count('HIGH'),   # Yüksek riskli
        'critical': p.count('CRITICAL'), # Kritik durumda
        'alarm'   : sum(1 for m in MOTORS if m['maintenance']['alarm']),
        # alarm=True olan motorları say → header'daki alarm rozeti
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP ENDPOINT'LERİ (ROUTE'LAR)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """
    Ana sayfa: HTML dashboard'u render eder.
    Jinja2 şablon motoru Flask.render_template ile çalışır.
    """
    return render_template(
        'index.html',
        stats=_stats(),              # Filo istatistikleri → KPI kartları
        metrics=MODEL_METRICS,       # Model performans metrikleri → sol panel
        motors_json=json.dumps(MOTORS)  # 200 motor JSON → motor grid
        # json.dumps → Python listesini JSON string'ine çevirir
        # index.html'de {{ motors_json|safe }} ile JavaScript'e aktarılır
    )


@app.route('/api/simulate')
def simulate():
    """
    GERÇEK ZAMANLI ÇIKARIM ENDPOİNT'İ

    Her 2 saniyede bir JavaScript bu endpoint'i çağırır.
    Pipeline: Sinyal Üret → FFT → Özellik → Scaler → MLP → RUL → JSON

    Bu endpoint'in asıl önemi: gerçek model çalıştırması yapıyor.
    Pre-computed JSON değil, anlık fiziksel sinyal → anlık tahmin.
    """

    t0 = time.perf_counter()
    # Yüksek çözünürlüklü zamanlayıcı başlat
    # perf_counter: işletim sistemi saatine bağlı değil, CPU cycle'larına dayanır
    # millisaniye hassasiyetinde çıkarım süresi ölçümü için

    # ── Arıza seviyesi belirle ────────────────────────────────────────────────
    fault_level = random.choices(
        [random.uniform(0.0, 0.30),   # Sağlıklı aralık: 0-30%
         random.uniform(0.35, 0.65),  # Geçiş bölgesi: 35-65% (belirsiz durum)
         random.uniform(0.70, 1.00)], # Arızalı aralık: 70-100%
        weights=[45, 10, 45],         # Olasılık ağırlıkları (toplam 100 değil, oran)
        k=1                           # 1 eleman seç
    )[0]
    # weights=[45,10,45]: %45 sağlıklı, %10 geçiş, %45 arızalı
    # Gerçek fabrikada da motorların önemli kısmı ya sağlıklı ya arızalı
    # Geçiş bölgesi az → model için zor örnekler, ama gerçekçi dağılım

    # ── Tam pipeline çalıştır ─────────────────────────────────────────────────
    vibration, temperature, current = generate_live_signal(fault_level)
    # Fiziksel sinyal üret: 1000 nokta titreşim + sıcaklık + akım

    features = extract_features(vibration, temperature, current)
    # FFT uygula, harmonik oran ve spektral entropi hesapla
    # Sonuç: 8 elemanlı Python listesi

    prob, pred = run_inference(features)
    # StandardScaler normalize → MLP.predict_proba() çalıştır
    # prob: [0,1] arıza olasılığı, pred: {0,1} sınıf etiketi

    # ── RUL hesabı (lineer degradasyon modeli) ────────────────────────────────
    rul_nominal = 8760.0 * (1 - prob)
    # 8760 = bir yılın saati (365 × 24)
    # prob=0.0 (sağlıklı) → rul_nominal=8760 saat (1 yıl kaldı)
    # prob=0.5            → rul_nominal=4380 saat (6 ay kaldı)
    # prob=1.0 (arızalı)  → rul_nominal=0    saat (hemen bakım gerekli)

    rul_hours = max(0.0, rul_nominal + np.random.normal(0, rul_nominal * 0.05))
    # Gaussian gürültü ekle → ölçüm belirsizliğini simüle eder
    # Standart sapma = rul_nominal'ın %5'i (bağıl belirsizlik)
    # max(0.0, ...) → negatif RUL fiziksel anlam taşımaz, sıfırda kesilir

    maintenance = get_maintenance(prob, rul_hours)
    # Olasılığa göre öncelik ve önerilen eylem hesapla

    inference_ms = (time.perf_counter() - t0) * 1000
    # Geçen süre = (bitiş - başlangıç) saniye cinsinden → ×1000 = milisaniye
    # Tipik: 2-8 ms (Python + NumPy + FFT + model overhead)

    # ── JSON yanıt oluştur ────────────────────────────────────────────────────
    return jsonify({
        'motor_id'      : f'MOTOR-{random.randint(0, 199):04d}',
        # Rastgele motor ID: MOTOR-0042, MOTOR-0158, vb.
        # :04d → 4 basamaklı, sıfır dolgulu format

        'inference_ms'  : round(inference_ms, 2),
        # Çıkarım süresi (milisaniye, 2 ondalık)

        'readings'      : {
            'vibration_rms'   : round(features[0], 4),  # Titreşim RMS
            'peak_to_peak'    : round(features[1], 4),  # Tepe-tepe fark
            'temperature_c'   : round(temperature, 2),  # Sıcaklık °C
            'current_a'       : round(current, 3),      # Akım Amper
            'harmonic_ratio'  : round(features[3], 4),  # Arıza harmonik oranı
            'spectral_entropy': round(features[4], 4),  # FFT entropi
        },
        # round() → JSON çıktısını okunabilir kılmak için

        'fault_prob'    : round(prob * 100, 1),
        # 0-1 arası olasılığı 0-100 yüzdeye çevir (1 ondalık)
        # Örnek: 0.8734 → 87.3

        'class_name'    : 'Arizali' if pred == 1 else 'Saglikli',
        # pred=1 → arızalı, pred=0 → sağlıklı

        'priority'      : maintenance['priority'],   # LOW/MEDIUM/HIGH/CRITICAL
        'status'        : maintenance['status'],     # NORMAL/UYARI/KRITIK ALARM/...
        'rul_hours'     : maintenance['rul_hours'],  # Saat (float)
        'rul_days'      : maintenance['rul_days'],   # Gün (float)
    })


@app.route('/api/motors')
def api_motors():
    """200 motorun tüm verisini JSON olarak döndürür."""
    return jsonify(MOTORS)
    # MOTORS listesi sunucu başlarken yüklendi, her istekte JSON'a serialize edilir


@app.route('/api/stats')
def api_stats():
    """Filo istatistiklerini JSON olarak döndürür (gerçek zamanlı güncellemeler için)."""
    return jsonify(_stats())


# ═══════════════════════════════════════════════════════════════════════════════
# SUNUCU BAŞLATMA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Bu dosya doğrudan çalıştırıldığında (import edilmediğinde) bu blok çalışır
    # python3 app.py → __name__ == '__main__'
    # from app import something → __name__ == 'app' → bu blok çalışmaz

    print()
    print('  ╔══════════════════════════════════════════╗')
    print('  ║   Predictive-Edge Dashboard v2.0        ║')
    print('  ║   Gerçek Zamanlı MLP Çıkarım Aktif      ║')
    print(f'  ║   Model: {MODEL_METRICS["params"]} param · '
          f'{MODEL_METRICS["memory_kb"]} KB · 8→16→8→1  ║')
    print('  ║   http://localhost:5050                  ║')
    print('  ╚══════════════════════════════════════════╝')
    print()

    app.run(
        host='0.0.0.0',   # Tüm ağ arayüzlerinde dinle (sadece localhost değil)
        port=5050,         # HTTP port numarası
        debug=False        # Production modda debug kapalı (otomatik reload ve hata sayfaları yok)
    )
