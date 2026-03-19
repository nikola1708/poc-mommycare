"""
Mommy Care - Clinical & Financial Helper Utilities
"""

from datetime import datetime, timedelta
import math


# ── Naegele's Rule ─────────────────────────────────────────────────────────────
def calculate_hpl(hpht: datetime) -> dict:
    """Calculate HPL using Naegele's Rule: HPHT - 3 months + 1 year + 7 days."""
    hpl = hpht + timedelta(days=280)   # 40 weeks = 280 days
    today = datetime.today()
    gestational_days = (today - hpht).days
    gestational_weeks = gestational_days // 7
    gestational_days_rem = gestational_days % 7
    days_to_hpl = (hpl - today).days

    if gestational_weeks <= 12:
        trimester = 1
        trimester_label = "Trimester 1"
        trimester_desc = "Pembentukan organ janin"
    elif gestational_weeks <= 26:
        trimester = 2
        trimester_label = "Trimester 2"
        trimester_desc = "Pertumbuhan & perkembangan aktif"
    else:
        trimester = 3
        trimester_label = "Trimester 3"
        trimester_desc = "Persiapan persalinan"

    progress_pct = min(max(gestational_days / 280 * 100, 0), 100)

    return {
        "hpl": hpl,
        "gestational_weeks": max(gestational_weeks, 0),
        "gestational_days_rem": gestational_days_rem,
        "days_to_hpl": max(days_to_hpl, 0),
        "trimester": trimester,
        "trimester_label": trimester_label,
        "trimester_desc": trimester_desc,
        "progress_pct": round(progress_pct, 1),
    }


# ── Financial Planning ─────────────────────────────────────────────────────────
def calculate_financial_plan(risk_level: str, cs_risk: float, gestational_weeks: int) -> dict:
    """Generate personalized financial recommendations."""
    NORMAL_COST = 8_000_000
    CAESAR_COST = 25_000_000

    if cs_risk >= 0.50:
        base_target = CAESAR_COST * 1.1
        plan_type = "Caesar"
        insurance_rec = True
    elif cs_risk >= 0.30 or risk_level in ("mid", "high"):
        base_target = (NORMAL_COST + CAESAR_COST) / 2 * 1.05
        plan_type = "Campuran (cadangan caesar)"
        insurance_rec = True
    else:
        base_target = NORMAL_COST * 1.2
        plan_type = "Normal"
        insurance_rec = False

    weeks_remaining = max(280 // 7 - gestational_weeks, 1)
    months_remaining = max(weeks_remaining / 4.33, 1)
    monthly_savings = math.ceil(base_target / months_remaining / 100_000) * 100_000

    return {
        "target": int(base_target),
        "plan_type": plan_type,
        "monthly_savings": int(monthly_savings),
        "normal_cost": NORMAL_COST,
        "caesar_cost": CAESAR_COST,
        "insurance_rec": insurance_rec,
        "months_remaining": round(months_remaining, 1),
    }


# ── Personalized Recommendations ───────────────────────────────────────────────
def generate_recommendations(input_data: dict, prediction: dict) -> list:
    recs = []
    sbp = input_data.get('systolic_bp', 110)
    gluc = input_data.get('blood_glucose', 90)
    wt = input_data.get('weight_gain_kg', 9)
    age = input_data.get('age', 28)
    gest_weeks = input_data.get('gestational_age_weeks', 20)
    risk = prediction['risk_level']
    cs_risk = prediction['cs_risk']
    preec_risk = prediction['preeclampsia_risk']
    gd_risk = prediction['gd_risk']

    # Blood Pressure
    if sbp >= 140:
        recs.append({
            "icon": "", "category": "Tekanan Darah",
            "title": "Tekanan Darah Tinggi — Segera Konsultasi Dokter",
            "detail": f"Tekanan darah Anda {sbp} mmHg berada di zona hipertensi berat. Risiko preeklamsia {preec_risk*100:.0f}%. Segera hubungi SpOG.",
            "urgency": "high"
        })
    elif sbp >= 130:
        recs.append({
            "icon": "", "category": "Tekanan Darah",
            "title": "Tekanan Darah Pra-Hipertensi",
            "detail": f"Sistolik {sbp} mmHg perlu dipantau. Kurangi garam, istirahat cukup, dan monitor 2x sehari.",
            "urgency": "mid"
        })
    else:
        recs.append({
            "icon": "", "category": "Tekanan Darah",
            "title": "Tekanan Darah Normal",
            "detail": "Pertahankan pola hidup aktif. Jalan kaki ringan 30 menit/hari sangat disarankan.",
            "urgency": "low"
        })

    # Blood Glucose
    if gluc >= 140:
        recs.append({
            "icon": "", "category": "Gula Darah",
            "title": "Gula Darah Tinggi — Risiko Diabetes Gestasional",
            "detail": f"Gula darah {gluc} mg/dL. Risiko GD {gd_risk*100:.0f}%. Lakukan GTT (tes toleransi glukosa) segera. Batasi karbohidrat sederhana.",
            "urgency": "high"
        })
    elif gluc >= 120:
        recs.append({
            "icon": "", "category": "Gula Darah",
            "title": "Gula Darah Perlu Diawasi",
            "detail": "Gula darah mendekati ambang batas. Hindari minuman manis, konsumsi serat tinggi.",
            "urgency": "mid"
        })
    else:
        recs.append({
            "icon": "", "category": "Nutrisi",
            "title": "Gula Darah Normal",
            "detail": "Pertahankan pola makan seimbang: protein cukup, sayur 5 porsi/hari, karbohidrat kompleks.",
            "urgency": "low"
        })

    # Weight gain
    if gest_weeks > 0:
        expected_wt = gest_weeks * 0.45 if gest_weeks <= 13 else 2 + (gest_weeks - 13) * 0.45
        if wt > expected_wt * 1.3:
            recs.append({
                "icon": "", "category": "Berat Badan",
                "title": "Kenaikan Berat Badan Berlebih",
                "detail": f"Kenaikan {wt} kg melebihi ekspektasi. Risiko caesar meningkat ke {cs_risk*100:.0f}%. Konsultasikan diet dengan ahli gizi.",
                "urgency": "mid"
            })
        else:
            recs.append({
                "icon": "", "category": "Berat Badan",
                "title": "Kenaikan Berat Badan Normal",
                "detail": f"Kenaikan {wt} kg sesuai target gestasi {gest_weeks} minggu. Pertahankan aktivitas fisik ringan.",
                "urgency": "low"
            })

    # Age
    if age >= 35:
        recs.append({
            "icon": "", "category": "Usia",
            "title": "Kehamilan Risiko Usia > 35 Tahun",
            "detail": "Usia di atas 35 meningkatkan risiko komplikasi. Lakukan pemeriksaan amniosentesis atau NIPT bila disarankan dokter.",
            "urgency": "mid"
        })

    # Trimester-based
    if gest_weeks <= 12:
        recs.append({
            "icon": "", "category": "Suplemen",
            "title": "Asam Folat & Suplemen Trimester 1",
            "detail": "Konsumsi asam folat 400–800 mcg/hari. Hindari obat tanpa resep. Jadwalkan USG pertama di minggu 8–12.",
            "urgency": "low"
        })
    elif gest_weeks <= 26:
        recs.append({
            "icon": "", "category": "Aktivitas",
            "title": "Aktivitas & Pemeriksaan Trimester 2",
            "detail": "Lakukan senam hamil ringan. Jadwalkan USG morfologi minggu 18–22. Tes gula darah (GTT) minggu 24–28.",
            "urgency": "low"
        })
    else:
        recs.append({
            "icon": "", "category": "Persiapan",
            "title": "Persiapan Persalinan Trimester 3",
            "detail": "Siapkan tas rumah sakit. Kenali tanda persalinan. Pantau gerakan janin minimal 10x/12 jam. Diskusikan rencana persalinan dengan dokter.",
            "urgency": "low"
        })

    return recs


# ── Trimester Education Content ─────────────────────────────────────────────
TRIMESTER_CONTENT = {
    1: {
        "title": "Trimester 1 (Minggu 1–12): Pembentukan Organ",
        "items": [
            ("", "Asam Folat", "400–800 mcg/hari — mencegah cacat tabung saraf"),
            ("", "Pantangan", "Alkohol, rokok, obat tanpa resep, paparan radiasi"),
            ("", "Pemeriksaan", "USG pertama minggu 8–12, tes darah lengkap"),
            ("", "Kenaikan BB", "Normal hanya 1–2 kg seluruh trimester 1"),
            ("", "Gejala Normal", "Mual, lelah, pusing — biasa terjadi"),
            ("", "Tes Wajib", "Golongan darah, Rh, hemoglobin, hepatitis B, toksoplasmosis"),
        ]
    },
    2: {
        "title": "Trimester 2 (Minggu 13–26): Pertumbuhan Aktif",
        "items": [
            ("", "Olahraga", "Senam hamil, yoga prenatal, jalan kaki — 30 menit/hari"),
            ("", "USG Morfologi", "Dilakukan minggu 18–22 untuk evaluasi anatomi janin"),
            ("", "Tes GTT", "Glukosa toleransi test minggu 24–28 untuk deteksi DM gestasional"),
            ("", "Zat Besi", "Suplemen Fe 30–60 mg/hari — cegah anemia"),
            ("", "Gerakan Janin", "Mulai terasa minggu 16–20, catat pergerakan harian"),
            ("", "Finansial", "Mulai tabungan persalinan & riset asuransi kesehatan"),
        ]
    },
    3: {
        "title": "Trimester 3 (Minggu 27–40): Persiapan Persalinan",
        "items": [
            ("", "Tas Rumah Sakit", "Siapkan dari minggu 36: dokumen, pakaian bayi, perlengkapan ibu"),
            ("", "Monitor Gerakan", "Hitung gerakan janin: minimal 10 gerakan dalam 12 jam"),
            ("", "Senam Kegel", "Perkuat otot dasar panggul untuk proses persalinan normal"),
            ("", "Tanda Darurat", "Kontraksi teratur, ketuban pecah, perdarahan → langsung ke RS"),
            ("", "Suplemen", "Lanjutkan vitamin prenatal, kalsium 1000 mg/hari"),
            ("", "Rencana Persalinan", "Diskusikan pilihan persalinan (normal/caesar) dengan SpOG"),
        ]
    }
}
