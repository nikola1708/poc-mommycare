#  Mommy Care — AI Pregnancy Monitoring App

> **Proof of Concept · Team Anomali · Hackathon 2025**

Aplikasi pemantauan tumbuh kembang janin berbasis AI yang menggabungkan **Machine Learning** (Random Forest, Logistic Regression, Gradient Boosting) dengan **kalkulasi klinis** (Naegele's Rule) untuk memprediksi risiko kehamilan dan merekomendasikan perencanaan finansial persalinan.

---

## Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
|  **Kalkulasi HPL** | Otomatis via Naegele's Rule dari input HPHT |
|  **Prediksi Risiko AI** | Random Forest / Logistic Regression / Gradient Boosting |
|  **Visualisasi Interaktif** | Gauge, Pie chart, Bar chart, Confusion Matrix |
|  **Perencanaan Finansial** | Target tabungan otomatis berdasarkan skor risiko |
|  **Rekomendasi Personalisasi** | Berubah dinamis sesuai kondisi input |
|  **Edukasi Trimester** | Konten klinis per trimester |
|  **Retrain Model** | Tombol retrain langsung dari UI |

---

##  Struktur Project

```
mommy_care/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md
│
├── .streamlit/
│   └── config.toml             # Streamlit theme & server config
│
├── data/
│   ├── __init__.py
│   ├── generate_data.py        # Synthetic dataset generator (2000 records)
│   └── maternal_health_data.csv  # ← auto-generated saat pertama run
│
├── models/
│   └── bundle.pkl              # ← auto-generated (scaler + 3 model tersimpan)
│
└── utils/
    ├── __init__.py
    ├── ml_engine.py            # Training, loading, prediksi ML
    └── helpers.py              # HPL, finansial, rekomendasi, edukasi
```

---

##  Cara Menjalankan

### 1. Clone / Download project

```bash
# Jika dari zip:
unzip mommy_care.zip
cd mommy_care

# Jika dari git:
git clone <repo-url>
cd mommy_care
```

### 2. Buat Virtual Environment (disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser: `http://localhost:8501`

> **Catatan:** Saat pertama kali dijalankan, aplikasi akan otomatis:
> 1. Generate dataset sintetis (`data/maternal_health_data.csv`)
> 2. Melatih 3 model ML (`models/bundle.pkl`)
> 
> Proses ini hanya terjadi **sekali** — selanjutnya model di-cache.

---

##  Detail Model ML

### Dataset

Dataset sintetis 2000 sampel yang dihasilkan dari pola klinis dataset **UCI Maternal Health Risk**, dengan distribusi:
- **Low Risk** (50%): Usia 18–35, tekanan darah normal, gula darah normal
- **Mid Risk** (30%): Usia 25–42, hipertensi ringan, gula darah batas
- **High Risk** (20%): Usia 35+, hipertensi berat, gula darah tinggi

### Fitur Input (10 features)

| Feature | Deskripsi | Satuan |
|---------|-----------|--------|
| `age` | Usia ibu | tahun |
| `systolic_bp` | Tekanan darah sistolik | mmHg |
| `diastolic_bp` | Tekanan darah diastolik | mmHg |
| `blood_glucose` | Kadar gula darah | mg/dL |
| `body_temp` | Suhu tubuh | °C |
| `heart_rate` | Detak jantung | bpm |
| `weight_gain_kg` | Kenaikan berat badan | kg |
| `gestational_age_weeks` | Usia kandungan | minggu |
| `previous_pregnancies` | Jumlah kehamilan sebelumnya | - |
| `previous_complications` | Riwayat komplikasi | 0/1 |

### Target Label

`low` / `mid` / `high` (multiclass classification)

### Model & Performa (Typical)

| Model | Accuracy | CV Mean |
|-------|----------|---------|
| Random Forest | ~99–100% | ~99% |
| Logistic Regression | ~97–99% | ~98% |
| Gradient Boosting | ~99–100% | ~99% |

> Performa tinggi karena dataset sintetis dengan pola yang jelas. Pada data riil, performa akan disesuaikan dengan variasi klinis sesungguhnya.

### Sub-risk Estimation

Selain prediksi utama, sistem menghitung estimasi risiko spesifik berbasis rule-enhancement:

- **Preeklamsia**: Diperkuat jika sistolik ≥ 140 atau usia ≥ 35
- **Diabetes Gestasional**: Diperkuat jika gula darah ≥ 140 mg/dL
- **Persalinan Caesar**: Diperkuat jika berat badan berlebih atau usia ≥ 35

---

##  Arsitektur Sistem

```
User Input (Sidebar)
       │
       ▼
  Streamlit Frontend (app.py)
       │
       ├──► utils/helpers.py
       │    ├── calculate_hpl()          → HPL, usia kandungan, trimester
       │    ├── calculate_financial_plan() → Target tabungan, cicilan/bulan
       │    ├── generate_recommendations() → Rekomendasi personalisasi
       │    └── TRIMESTER_CONTENT        → Konten edukasi
       │
       └──► utils/ml_engine.py
            ├── train_all_models()       → Training RF + LR + GB
            ├── load_bundle()            → Load model dari disk
            ├── predict_risk()           → Prediksi + sub-risks
            └── get_feature_importance() → Feature importance RF/GB
```

---

##  Kustomisasi

### Mengganti Dataset

Ganti file `data/maternal_health_data.csv` dengan dataset riil. Pastikan kolom sesuai:
```
age, systolic_bp, diastolic_bp, blood_glucose, body_temp, heart_rate,
weight_gain_kg, gestational_age_weeks, previous_pregnancies,
previous_complications, risk_level
```
Nilai `risk_level` harus: `low`, `mid`, atau `high`.

Lalu hapus `models/bundle.pkl` dan restart app untuk retrain otomatis.

### Menambah Model Baru

Di `utils/ml_engine.py`, tambahkan entry baru di dict `models`:
```python
"XGBoost": XGBClassifier(n_estimators=100, random_state=42),
```
Tambahkan juga opsinya di `app.py` pada `st.selectbox("Algoritma ML", [...])`.

### Mengubah Estimasi Biaya

Di `utils/helpers.py`, ubah konstanta:
```python
NORMAL_COST = 8_000_000   # Rp 8 juta
CAESAR_COST = 25_000_000  # Rp 25 juta
```


##  Disclaimer

Aplikasi ini adalah **Proof of Concept** untuk keperluan hackathon. **Bukan pengganti konsultasi medis profesional.** Semua prediksi bersifat estimasi berbasis data sintetis dan harus divalidasi oleh tenaga kesehatan berlisensi sebelum digunakan secara klinis.

---


