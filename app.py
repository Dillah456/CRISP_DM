import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Depresi Remaja",
    page_icon="🧠",
    layout="centered"
)

# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("Mental_Health_CRISP.pkl")

model = load_model()

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("🧠 Prediksi Risiko Depresi Remaja")
st.markdown(
    "Aplikasi ini memprediksi **risiko depresi** berdasarkan pola penggunaan "
    "media sosial menggunakan model **Decision Tree Classifier**."
)
st.divider()

# ─── Sidebar Info ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📌 Tentang Aplikasi")
    st.info(
        "Model dilatih menggunakan **Teen Mental Health Dataset** dari Kaggle "
        "dengan metode CRISP-DM.\n\n"
        "🎯 **Target**: `depression_label`\n"
        "- `0` = Tidak Depresi\n"
        "- `1` = Berisiko Depresi"
    )
    st.markdown("---")
    st.caption("👥 Dibuat oleh Tim (2 Orang)")
    st.caption("📊 Algoritma: Decision Tree | max_depth=4")

# ─── Input Form ────────────────────────────────────────────────────────────────
st.subheader("📝 Masukkan Data Remaja")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**👤 Data Demografis**")

    age = st.slider(
        "Usia (tahun)",
        min_value=10, max_value=25, value=17, step=1,
        help="Usia remaja dalam tahun"
    )

    gender = st.selectbox(
        "Jenis Kelamin",
        options=[("Laki-laki", 1), ("Perempuan", 0)],
        format_func=lambda x: x[0]
    )
    gender_val = gender[1]

    sleep_hours = st.slider(
        "Jam Tidur per Malam (jam)",
        min_value=3.0, max_value=12.0, value=7.0, step=0.5,
        help="Rata-rata jam tidur per malam"
    )

    academic_performance = st.slider(
        "Performa Akademik (1–10)",
        min_value=1, max_value=10, value=6, step=1,
        help="Nilai rata-rata akademik (1=sangat buruk, 10=sangat baik)"
    )

    physical_activity = st.slider(
        "Aktivitas Fisik per Minggu (jam)",
        min_value=0.0, max_value=20.0, value=3.0, step=0.5,
        help="Total jam olahraga/aktivitas fisik per minggu"
    )

with col2:
    st.markdown("**📱 Penggunaan Media Sosial**")

    daily_social_media_hours = st.slider(
        "Durasi Media Sosial per Hari (jam)",
        min_value=0.0, max_value=15.0, value=4.0, step=0.5,
        help="Rata-rata jam penggunaan media sosial per hari"
    )

    platform_usage = st.selectbox(
        "Platform Utama yang Digunakan",
        options=[
            ("Instagram", 0),
            ("TikTok", 1),
            ("Twitter/X", 2),
            ("YouTube", 3),
            ("Facebook", 4),
        ],
        format_func=lambda x: x[0]
    )
    platform_val = platform_usage[1]

    screen_time_before_sleep = st.slider(
        "Screen Time Sebelum Tidur (jam)",
        min_value=0.0, max_value=5.0, value=1.5, step=0.5,
        help="Durasi penggunaan layar dalam 1 jam sebelum tidur"
    )

    social_interaction_level = st.selectbox(
        "Tingkat Interaksi Sosial",
        options=[("Rendah", 0), ("Sedang", 1), ("Tinggi", 2)],
        format_func=lambda x: x[0]
    )
    social_val = social_interaction_level[1]

st.divider()
st.subheader("😰 Kondisi Psikologis")

col3, col4, col5 = st.columns(3)

with col3:
    stress_level = st.slider(
        "Tingkat Stres (1–10)",
        min_value=1, max_value=10, value=5, step=1,
        help="1 = sangat rendah, 10 = sangat tinggi"
    )

with col4:
    anxiety_level = st.slider(
        "Tingkat Kecemasan (1–10)",
        min_value=1, max_value=10, value=5, step=1,
        help="1 = sangat rendah, 10 = sangat tinggi"
    )

with col5:
    addiction_level = st.slider(
        "Tingkat Kecanduan Media Sosial (1–10)",
        min_value=1, max_value=10, value=4, step=1,
        help="1 = tidak kecanduan, 10 = sangat kecanduan"
    )

# ─── Prediction ────────────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Prediksi Sekarang", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        "age"                     : age,
        "gender"                  : gender_val,
        "daily_social_media_hours": daily_social_media_hours,
        "platform_usage"          : platform_val,
        "sleep_hours"             : sleep_hours,
        "screen_time_before_sleep": screen_time_before_sleep,
        "academic_performance"    : academic_performance,
        "physical_activity"       : physical_activity,
        "social_interaction_level": social_val,
        "stress_level"            : stress_level,
        "anxiety_level"           : anxiety_level,
        "addiction_level"         : addiction_level,
    }])

    prediction    = model.predict(input_data)[0]
    proba         = model.predict_proba(input_data)[0]
    proba_depresi = proba[1] * 100
    proba_aman    = proba[0] * 100

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(
            f"⚠️ **Berisiko Depresi**\n\n"
            f"Model memprediksi remaja ini **berpotensi mengalami depresi** "
            f"dengan probabilitas **{proba_depresi:.1f}%**."
        )
    else:
        st.success(
            f"✅ **Tidak Berisiko Depresi**\n\n"
            f"Model memprediksi remaja ini **tidak berisiko mengalami depresi** "
            f"dengan probabilitas aman **{proba_aman:.1f}%**."
        )

    # Probability bar
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("🟢 Probabilitas Tidak Depresi", f"{proba_aman:.1f}%")
    with col_b:
        st.metric("🔴 Probabilitas Depresi", f"{proba_depresi:.1f}%")

    st.progress(int(proba_depresi), text=f"Risiko Depresi: {proba_depresi:.1f}%")

    # Input summary
    with st.expander("📋 Ringkasan Input yang Dimasukkan"):
        summary = {
            "Usia"                           : f"{age} tahun",
            "Jenis Kelamin"                  : "Laki-laki" if gender_val == 1 else "Perempuan",
            "Durasi Media Sosial / Hari"     : f"{daily_social_media_hours} jam",
            "Platform Utama"                 : platform_usage[0],
            "Jam Tidur"                      : f"{sleep_hours} jam",
            "Screen Time Sebelum Tidur"      : f"{screen_time_before_sleep} jam",
            "Performa Akademik"              : f"{academic_performance}/10",
            "Aktivitas Fisik / Minggu"       : f"{physical_activity} jam",
            "Interaksi Sosial"               : social_interaction_level[0],
            "Tingkat Stres"                  : f"{stress_level}/10",
            "Tingkat Kecemasan"              : f"{anxiety_level}/10",
            "Tingkat Kecanduan Medsos"       : f"{addiction_level}/10",
        }
        st.table(pd.DataFrame(summary.items(), columns=["Fitur", "Nilai"]))

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ *Aplikasi ini bersifat edukatif dan tidak menggantikan diagnosis klinis profesional. "
    "Jika Anda atau orang terdekat membutuhkan bantuan, silakan konsultasikan dengan psikolog atau tenaga kesehatan.*"
)
