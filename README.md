# 🧠 Analisis Dampak Media Sosial terhadap Kesehatan Mental Remaja

> Proyek Machine Learning menggunakan metode **CRISP-DM** dan algoritma **Decision Tree Classifier**

---

## 📋 Deskripsi Proyek

Kami membangun model klasifikasi untuk memprediksi risiko depresi pada remaja berdasarkan pola penggunaan media sosial mereka. Proyek ini menggunakan pendekatan terstruktur **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) agar alur analisis data hingga pemodelan dapat dipahami dengan jelas dan sistematis.

Penggunaan media sosial yang berlebihan kerap dikaitkan dengan meningkatnya tingkat stres, kecemasan, dan depresi — khususnya di kalangan remaja. Melalui proyek ini, kami berupaya membuktikan hubungan tersebut secara kuantitatif dan membangun model prediktif yang dapat dijadikan dasar pengambilan keputusan.

---

## 🗂️ Struktur Proyek

```
📦 social-media-mental-health/
 ┣ 📓 Mental_Health_CRISP_DM.ipynb   ← Notebook utama (CRISP-DM)
 ┗ 📄 README.md                      ← Dokumentasi proyek ini
```

---

## 🔄 Alur CRISP-DM

### 1️⃣ Business Understanding
Kami mengidentifikasi permasalahan utama: *apakah kebiasaan penggunaan media sosial dapat menjadi prediktor depresi pada remaja?* Target variabel yang diprediksi adalah `depression_label` (0 = Tidak Depresi, 1 = Depresi).

### 2️⃣ Data Understanding
Kami menggunakan dataset **Teen Mental Health Dataset** dari Kaggle. Proses eksplorasi mencakup:
- Preview data dan informasi tipe kolom
- Statistik deskriptif
- Pengecekan *missing values*
- Analisis korelasi antar fitur dengan heatmap

**Temuan awal**: Tiga fitur dengan korelasi tertinggi terhadap depresi adalah `daily_social_media_hours`, `stress_level`, dan `anxiety_level`.

### 3️⃣ Data Preparation
Kami melakukan:
- Pemisahan fitur (`X`) dan target (`y`)
- *Label Encoding* pada kolom kategorikal (`gender`, `platform_usage`, `social_interaction_level`)
- *Train-test split* dengan rasio 80:20 dan stratifikasi kelas

### 4️⃣ Modeling
Kami membangun model **Decision Tree Classifier** dengan konfigurasi:
- `criterion = 'gini'`
- `max_depth = 4` (mencegah overfitting)
- `random_state = 42` (reproducibility)

### 5️⃣ Evaluation
Evaluasi model menggunakan:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)
- **Confusion Matrix** (visualisasi)
- **Feature Importance** (fitur paling berpengaruh)

### 6️⃣ Deployment
Kami menyimpulkan hasil analisis dan memberikan rekomendasi praktis bagi orang tua, sekolah, dan remaja berdasarkan temuan model.

---

## 📊 Dataset

| Info | Detail |
|------|--------|
| Sumber | [Kaggle — Social Media Impact on Mental Health](https://www.kaggle.com/datasets/sunil123kumar/social-media-impact-on-mental-health) |
| File | `Teen_Mental_Health_Dataset.csv` |
| Target | `depression_label` (biner: 0 / 1) |
| Fitur utama | `daily_social_media_hours`, `stress_level`, `anxiety_level`, `gender`, `platform_usage`, `social_interaction_level` |

---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---------|----------|
| `pandas` | Manipulasi dan analisis data |
| `seaborn` | Visualisasi statistik |
| `matplotlib` | Plot grafik |
| `scikit-learn` | Preprocessing, modeling, evaluasi |
| `kagglehub` | Download dataset dari Kaggle |

---

## 🚀 Cara Menjalankan

1. **Pastikan Python ≥ 3.8 dan Jupyter Notebook sudah terinstal.**

2. **Install dependensi yang dibutuhkan:**
   ```bash
   pip install pandas seaborn matplotlib scikit-learn kagglehub
   ```

3. **Jalankan notebook:**
   ```bash
   jupyter notebook Mental_Health_CRISP_DM.ipynb
   ```

4. **Autentikasi Kaggle** (untuk download dataset):
   ```bash
   kaggle config set -n username -v YOUR_KAGGLE_USERNAME
   kaggle config set -n key -v YOUR_KAGGLE_API_KEY
   ```

> Atau jalankan langsung di [Google Colab](https://colab.research.google.com/) tanpa instalasi tambahan.

---

## 📈 Hasil Singkat

- Model Decision Tree berhasil mengklasifikasikan risiko depresi dengan performa yang baik.
- **Fitur paling berpengaruh**: `stress_level`, `daily_social_media_hours`, dan `anxiety_level`.
- Pembatasan `max_depth=4` terbukti efektif dalam menjaga keseimbangan antara akurasi dan generalisasi model.

---

## 👥 Tim Pengembang

Kami mengerjakan proyek ini berdua sebagai bagian dari tugas analisis data berbasis CRISP-DM.

| Peran | Kontribusi |
|-------|-----------|
| 👤 Anggota 1 | Business Understanding, Data Understanding, EDA & Visualisasi |
| 👤 Anggota 2 | Data Preparation, Modeling, Evaluation & Interpretasi |

> *Catatan: Seluruh tahapan didiskusikan dan dikerjakan bersama secara kolaboratif.*

---

## 📌 Catatan Penting

- Dataset yang digunakan bersifat **sintetis/simulasi**, sehingga hasil tidak merepresentasikan kondisi populasi nyata secara langsung.
- Untuk penelitian lanjutan, kami merekomendasikan pengujian dengan data primer dan eksplorasi algoritma lain seperti Random Forest atau XGBoost.

---

*Dibuat dengan ❤️ menggunakan Python & CRISP-DM Framework*
