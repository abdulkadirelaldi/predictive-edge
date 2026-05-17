"""
Predictive-Edge Dashboard v2.0 — Flask Backend
Gerçek zamanlı model çıkarımı: joblib → FFT pipeline → MLP.predict()
"""

import json, random, time
import numpy as np
import scipy.fft as spfft
from scipy.signal import find_peaks
from scipy.stats import entropy as scipy_entropy
import joblib
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# ── Model ve scaler yükle ────────────────────────────────────────────────────
model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ── Sinyal parametreleri (notebook ile aynı) ─────────────────────────────────
SAMPLE_RATE = 1000
t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)

# Rulman arıza frekansları (Modül 0 ile aynı)
BPFO     = 100.0   # Hz — dış bilezik arıza frekansı
SHAFT_3X = 150.0   # Hz — 3× mil harmonik (gevşeklik)

# ── Ön hesaplanmış filo verisi (dashboard motorları için) ────────────────────
with open('dashboard_output.json', encoding='utf-8') as f:
    MOTORS = json.load(f)

MODEL_METRICS = {
    'accuracy'  : 98.00,
    'f1'        : 0.9804,
    'precision' : 0.9615,
    'recall'    : 1.0000,
    'roc_auc'   : 1.0000,
    'cv_f1'     : '0.9910 ± 0.0020',
    'cv_auc'    : '0.9994 ± 0.0006',
    'architecture': '8 → 16 → 8 → 1',
    'iterations': 137,
    'samples'   : 1000,
    'features'  : 8,
    'params'    : 289,
    'memory_kb' : 1.13,
}


# ── Gerçek Zamanlı Sinyal Üretimi ────────────────────────────────────────────

def generate_live_signal(fault_level: float):
    """
    fault_level ∈ [0, 1] değerine göre fiziksel motor sinyali üretir.
    Notebook'taki generate_healthy / generate_faulty fonksiyonlarının
    sürekli versiyonu — aynı dağılım parametreleri.
    """
    noise_std = 0.05 + fault_level * 0.45
    h2_amp    = fault_level * np.random.uniform(0.15, 0.75)
    h3_amp    = fault_level * np.random.uniform(0.08, 0.45)

    vibration = (
        1.0    * np.sin(2 * np.pi * 50       * t) +
        h2_amp * np.sin(2 * np.pi * BPFO     * t) +
        h3_amp * np.sin(2 * np.pi * SHAFT_3X * t) +
        np.random.normal(0, noise_std, SAMPLE_RATE)
    )
    temperature = float(np.clip(
        np.random.normal(45.0 + fault_level * 23.0, 5.0), 18, 110
    ))
    current = float(np.clip(
        np.random.normal(5.0 + fault_level * 3.5, 0.5), 2.5, 14.0
    ))
    return vibration, temperature, current


def extract_features(signal, temperature, current):
    """
    Notebook Modül 2 ile birebir aynı özellik çıkarımı.
    Burada FFT gerçekten hesaplanıyor — pre-computed değil.
    """
    N     = len(signal)
    freqs = spfft.rfftfreq(N, d=1.0 / SAMPLE_RATE)
    mag   = np.abs(spfft.rfft(signal)) / N

    def freq_energy(f_target, bw=2.0):
        mask = (freqs >= f_target - bw) & (freqs <= f_target + bw)
        return float(np.sum(mag[mask] ** 2)) + 1e-10

    harmonic_ratio   = (freq_energy(BPFO) + freq_energy(SHAFT_3X)) / freq_energy(50.0)
    mag_norm         = mag / (mag.sum() + 1e-10)
    spectral_entropy = float(scipy_entropy(mag_norm + 1e-12))
    peaks, _         = find_peaks(mag, height=mag.mean() * 3, distance=5)

    return [
        float(np.sqrt(np.mean(signal ** 2))),   # vibration_rms
        float(signal.max() - signal.min()),       # peak_to_peak
        float(np.mean(np.abs(signal))),           # mean_abs
        harmonic_ratio,                           # harmonic_ratio
        spectral_entropy,                         # spectral_entropy
        len(peaks),                               # n_spectral_peaks
        temperature,
        current,
    ]


def run_inference(features):
    """Scaler + MLP.predict_proba() — tam pipeline çalıştırır."""
    features_scaled = scaler.transform([features])
    prob = float(model.predict_proba(features_scaled)[0][1])
    pred = int(model.predict(features_scaled)[0])
    return prob, pred


def get_maintenance(fault_prob, rul_hours):
    if fault_prob < 0.35:
        priority, status = 'LOW',      'NORMAL'
    elif fault_prob < 0.65:
        priority, status = 'MEDIUM',   'UYARI'
    elif fault_prob < 0.85:
        priority, status = 'HIGH',     'KRITIK ALARM'
    else:
        priority, status = 'CRITICAL', 'ACIL MUDAHALE'
    return {
        'status'  : status,
        'priority': priority,
        'alarm'   : fault_prob >= 0.65,
        'rul_hours': round(rul_hours, 1),
        'rul_days' : round(rul_hours / 24, 1),
        'recommended_action': {
            'LOW'     : 'Rutin bakim planla',
            'MEDIUM'  : '72 saat icinde kontrol et',
            'HIGH'    : '24 saat icinde mudahale et',
            'CRITICAL': 'Motoru hemen durdur',
        }[priority],
    }


# ── Stats ─────────────────────────────────────────────────────────────────────

def _stats():
    p = [m['maintenance']['priority'] for m in MOTORS]
    return {
        'total'   : len(MOTORS),
        'low'     : p.count('LOW'),
        'medium'  : p.count('MEDIUM'),
        'high'    : p.count('HIGH'),
        'critical': p.count('CRITICAL'),
        'alarm'   : sum(1 for m in MOTORS if m['maintenance']['alarm']),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           stats=_stats(),
                           metrics=MODEL_METRICS,
                           motors_json=json.dumps(MOTORS))


@app.route('/api/simulate')
def simulate():
    """
    GERÇEKÇİ ÇIKARIM:
      1. Fiziksel sinyal üret (FFT harmonikleri dahil)
      2. Özellik çıkar (notebook pipeline ile aynı)
      3. StandardScaler + MLP.predict_proba() çalıştır
      4. RUL hesapla (lineer degradasyon modeli)
    """
    t0 = time.perf_counter()

    # Motor sağlık durumunu rastgele seç — ağırlıklı dağılım
    fault_level = random.choices(
        [random.uniform(0.0, 0.30),   # sağlıklı
         random.uniform(0.35, 0.65),  # geçiş bölgesi
         random.uniform(0.70, 1.00)], # arızalı
        weights=[45, 10, 45],
        k=1
    )[0]

    # Sinyal üret → özellik çıkar → model çalıştır
    vibration, temperature, current = generate_live_signal(fault_level)
    features  = extract_features(vibration, temperature, current)
    prob, pred = run_inference(features)

    # Lineer degradasyon RUL
    rul_nominal = 8760.0 * (1 - prob)
    rul_hours   = max(0.0, rul_nominal + np.random.normal(0, rul_nominal * 0.05))

    maintenance = get_maintenance(prob, rul_hours)

    inference_ms = (time.perf_counter() - t0) * 1000

    return jsonify({
        'motor_id'      : f'MOTOR-{random.randint(0, 199):04d}',
        'inference_ms'  : round(inference_ms, 2),
        'readings'      : {
            'vibration_rms'   : round(features[0], 4),
            'peak_to_peak'    : round(features[1], 4),
            'temperature_c'   : round(temperature, 2),
            'current_a'       : round(current, 3),
            'harmonic_ratio'  : round(features[3], 4),
            'spectral_entropy': round(features[4], 4),
        },
        'fault_prob'    : round(prob * 100, 1),
        'class_name'    : 'Arizali' if pred == 1 else 'Saglikli',
        'priority'      : maintenance['priority'],
        'status'        : maintenance['status'],
        'rul_hours'     : maintenance['rul_hours'],
        'rul_days'      : maintenance['rul_days'],
    })


@app.route('/api/motors')
def api_motors():
    return jsonify(MOTORS)


@app.route('/api/stats')
def api_stats():
    return jsonify(_stats())


if __name__ == '__main__':
    print()
    print('  ╔══════════════════════════════════════════╗')
    print('  ║   Predictive-Edge Dashboard v2.0        ║')
    print('  ║   Gerçek Zamanlı MLP Çıkarım Aktif      ║')
    print(f'  ║   Model: {MODEL_METRICS["params"]} param · '
          f'{MODEL_METRICS["memory_kb"]} KB · 8→16→8→1  ║')
    print('  ║   http://localhost:5050                  ║')
    print('  ╚══════════════════════════════════════════╝')
    print()
    app.run(host='0.0.0.0', port=5050, debug=False)
