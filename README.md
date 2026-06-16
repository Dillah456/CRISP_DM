# 🧠 Analisis Dampak Media Sosial terhadap Kesehatan Mental Remaja

> Proyek Machine Learning menggunakan metode **CRISP-DM** | Eksplorasi: **Decision Tree** | Modeling: **Regresi Logistik**

---

## 📋 Deskripsi Proyek

Kami membangun model klasifikasi untuk memprediksi risiko depresi pada remaja berdasarkan pola penggunaan media sosial. Proyek ini menggunakan kerangka kerja **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*) dengan pembagian peran algoritma yang jelas:

- **Decision Tree** → digunakan pada fase *Data Understanding* sebagai alat eksplorasi untuk mengidentifikasi fitur-fitur yang paling berpengaruh.
- **Regresi Logistik** → digunakan pada fase *Modeling (Data Mining)* sebagai model prediksi utama karena menghasilkan probabilitas yang mudah diinterpretasikan.

Penggunaan media sosial yang berlebihan kerap dikaitkan dengan meningkatnya tingkat stres, kecemasan, dan depresi — khususnya di kalangan remaja. Melalui proyek ini, kami berupaya membuktikan hubungan tersebut secara kuantitatif dan membangun model prediktif yang dapat menjadi dasar pengambilan keputusan.

---

## 🗂️ Struktur Proyek

```
📦 social-media-mental-health/
 ┣ 📓 Untitled7.ipynb                ← Notebook asli (script pengembangan awal)
 ┣ 📓 Mental_Health_CRISP_DM.ipynb   ← Notebook diperhalus dengan struktur CRISP-DM (dengan bantuan AI)
 ┣ 🤖 Mental_Health_Logistic.pkl     ← Model Regresi Logistik tersimpan
 ┣ 🖥️ app_streamlit.py               ← Aplikasi web versi Streamlit
 ┣ 🖥️ app_gradio.py                  ← Aplikasi web versi Gradio
 ┣ 📄 requirements.txt               ← Dependensi Python
 ┗ 📄 README.md                      ← Dokumentasi ini
```

---

## 📒 Riwayat Pengembangan Notebook

Proyek ini memiliki dua versi notebook yang mencerminkan proses pengembangan bertahap:

### `Untitled7.ipynb` — Script Asli (Pengembangan Awal)

Notebook ini adalah **script mentah pertama** yang ditulis selama proses eksplorasi. Karakteristiknya:

- Ditulis secara langsung tanpa struktur formal, mencerminkan alur pikir saat bereksplorasi.
- Sudah memuat dua model secara berurutan: **Decision Tree** untuk eksplorasi awal, dilanjutkan **Regresi Logistik** menggunakan 4 fitur terpilih.
- Terdapat kode penyimpanan model ke file `.pkl`:
  ```python
  import pickle
  with open("Mental_Health_Logistic.pkl", "wb") as f:
      pickle.dump(model, f)
  ```
- Analisis koefisien Regresi Logistik sudah dilakukan secara manual dan dilengkapi tabel interpretasi.
- Tidak menggunakan kerangka kerja tertentu — lebih bersifat eksperimental.

### `Mental_Health_CRISP_DM.ipynb` — Script Diperhalus (dengan Bantuan AI)

Notebook ini adalah **versi yang dirapikan dan distrukturisasi** menggunakan bantuan AI dari notebook asli. Perbedaannya:

- Mengikuti kerangka kerja **CRISP-DM** secara eksplisit (6 fase berlabel jelas).
- Setiap fase memiliki narasi pengantar, komentar kode yang lebih deskriptif, dan interpretasi hasil.
- Visualisasi lebih lengkap dan diberi judul yang informatif.
- Penjelasan konseptual ditambahkan (misalnya: mengapa Decision Tree dipakai di fase eksplorasi, mengapa Regresi Logistik dipilih untuk modeling).
- Cocok digunakan sebagai **laporan atau dokumentasi teknis** untuk presentasi.

> **Ringkasnya:** `Untitled7.ipynb` adalah tempat ide lahir; `Mental_Health_CRISP_DM.ipynb` adalah versi yang sudah dibersihkan dan siap dipresentasikan.

---

## 🤖 Mengapa Decision Tree di Data Understanding, dan Regresi Logistik di Modeling?

Pemilihan algoritma di setiap fase bukan kebetulan — ini adalah keputusan desain yang didasari karakteristik masing-masing algoritma.

### 🌳 Decision Tree → Fase Data Understanding (Eksplorasi)

Decision Tree digunakan **bukan sebagai model prediksi akhir**, melainkan sebagai **alat eksplorasi visual**. Alasannya:

| Keunggulan | Penjelasan |
|---|---|
| **Interpretable secara visual** | Struktur pohon dapat digambar dan dibaca seperti diagram alur, sehingga mudah memahami bagaimana fitur memisahkan kelas. |
| **Feature Importance otomatis** | Decision Tree menghasilkan skor `feature_importances_` yang langsung menunjukkan fitur mana yang paling diskriminatif. |
| **Tidak butuh asumsi distribusi** | Cocok untuk tahap awal ketika kita belum tahu distribusi data. |
| **Cepat untuk prototyping** | Dapat dilatih dan divisualisasikan dalam hitungan detik tanpa hyperparameter kompleks. |

Dari eksplorasi ini, ditemukan bahwa `stress_level`, `daily_social_media_hours`, dan `anxiety_level` secara konsisten muncul di node-node teratas pohon — mengonfirmasi fitur mana yang paling berpengaruh terhadap `depression_label`. Informasi ini kemudian digunakan untuk memilih fitur di fase Modeling.

> **Decision Tree di sini berperan seperti peta — membantu kita memahami medan data sebelum memilih rute terbaik.**

---

### 📈 Regresi Logistik → Fase Modeling, Evaluation & Deployment

Setelah fitur-fitur kunci teridentifikasi, Regresi Logistik dipilih sebagai **model prediksi utama**. Alasannya:

| Keunggulan | Penjelasan |
|---|---|
| **Output probabilitas** | Menghasilkan nilai `predict_proba()` antara 0–1, sehingga risiko depresi bisa dinyatakan dalam persentase — lebih informatif daripada sekadar label 0/1. |
| **Koefisien yang interpretatif** | Nilai koefisien menjelaskan arah dan kekuatan pengaruh tiap fitur. Contoh: `sleep_hours` memiliki koefisien negatif (-1.66), artinya semakin banyak tidur maka risiko depresi turun. |
| **Stabil dan tidak mudah overfitting** | Decision Tree cenderung overfitting jika tidak dipangkas. Regresi Logistik lebih stabil untuk data berukuran sedang. |
| **Cocok untuk klasifikasi biner** | Target `depression_label` hanya bernilai 0 atau 1, sesuai dengan asumsi dasar Regresi Logistik. |
| **Mudah di-deploy** | Model ringan, bisa disimpan sebagai `.pkl` dan dipanggil kembali dengan cepat di aplikasi Streamlit maupun Gradio. |

Hasil koefisien model:

| Fitur | Koefisien | Interpretasi |
|---|---:|---|
| `sleep_hours` | -1.66 | ↓ Semakin banyak tidur, risiko depresi turun |
| `daily_social_media_hours` | +0.98 | ↑ Semakin lama main medsos, risiko depresi naik |
| `stress_level` | +0.74 | ↑ Semakin tinggi stres, risiko depresi naik |
| `anxiety_level` | +0.69 | ↑ Semakin tinggi kecemasan, risiko depresi naik |

> **Regresi Logistik dipilih karena tidak hanya mampu memprediksi, tetapi juga menjelaskan *mengapa* seseorang berisiko — nilai yang penting untuk stakeholder non-teknis.**

---

## 🖥️ Perbandingan Versi Aplikasi: Streamlit vs Gradio

Proyek ini menyediakan dua versi aplikasi web yang menggunakan model yang sama (`Mental_Health_Logistic.pkl`), namun dibangun dengan framework berbeda.

### Streamlit (`app_streamlit.py`)

```python
import streamlit as st
```

| Aspek | Detail |
|---|---|
| **Framework** | Streamlit |
| **Filosofi** | *"Python script yang berubah jadi web app"* — setiap baris kode dieksekusi dari atas ke bawah saat ada interaksi. |
| **Kelebihan** | Kontrol penuh atas layout, styling, dan logika UI. Mendukung komponen kustom, multi-halaman, dan integrasi dengan library visualisasi (Matplotlib, Plotly). |
| **Tampilan** | Lebih fleksibel — bisa menambahkan `st.divider()`, `st.expander()`, `st.metric()`, dan custom CSS. |
| **Cocok untuk** | Aplikasi yang butuh tampilan profesional, dashboard kompleks, atau integrasi dengan database. |
| **Cara jalankan** | `streamlit run app_streamlit.py` |
| **Port default** | `localhost:8501` |

Pada versi ini, pengguna mendapatkan tampilan yang lebih kaya: ada expander "Detail Input" untuk melihat data yang diinput, komponen `st.metric()` untuk menampilkan probabilitas secara menonjol, dan pesan disclaimer medis.

---

### Gradio (`app_gradio.py`)

```python
import gradio as gr
```

| Aspek | Detail |
|---|---|
| **Framework** | Gradio |
| **Filosofi** | *"Bungkus fungsi Python menjadi UI otomatis"* — cukup definisikan fungsi `predict()` dan Gradio membangun UI-nya secara otomatis. |
| **Kelebihan** | Setup sangat cepat, terintegrasi langsung dengan **Hugging Face Spaces** untuk deployment gratis, cocok untuk demo dan prototyping. |
| **Tampilan** | Lebih sederhana dan otomatis — input/output ditata secara default tanpa banyak konfigurasi. |
| **Cocok untuk** | Demo cepat, sharing model ke publik via Hugging Face, atau skenario di mana kecepatan deployment lebih penting dari estetika UI. |
| **Cara jalankan** | `python app_gradio.py` |
| **Port default** | `localhost:7860` |

Pada versi ini, hasil prediksi dikembalikan sebagai teks dalam satu `Textbox`, lebih minimalis namun tetap fungsional.

---

### Perbandingan Langsung

| Kriteria | Streamlit | Gradio |
|---|:---:|:---:|
| Kecepatan setup | Sedang | Cepat |
| Fleksibilitas UI | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Kemudahan deployment ke cloud | Streamlit Cloud | Hugging Face Spaces |
| Cocok untuk demo cepat | ✓ | ✅ (lebih cepat) |
| Cocok untuk produksi | ✅ | ✓ |
| Output probabilitas sebagai komponen terpisah | ✅ (`st.metric`) | ✗ (dalam teks) |
| Multi-halaman | ✅ | ✗ |

> **Kesimpulan:** Gunakan **Streamlit** jika ingin tampilan yang lebih polished dan dapat dikembangkan lebih lanjut. Gunakan **Gradio** jika ingin berbagi model secara cepat, terutama ke Hugging Face Spaces.

---

## 🔄 Alur CRISP-DM

### 1️⃣ Business Understanding
Kami mengidentifikasi permasalahan utama: *apakah kebiasaan penggunaan media sosial dapat menjadi prediktor risiko depresi pada remaja?*

- **Target variabel**: `depression_label` (0 = Tidak Depresi, 1 = Berisiko Depresi)
- **Metrik keberhasilan**: Akurasi ≥ 75% dan ROC-AUC ≥ 0.75

### 2️⃣ Data Understanding — *Menggunakan Decision Tree*
Eksplorasi menyeluruh menggunakan heatmap korelasi dan Decision Tree sebagai alat identifikasi fitur penting. Temuan utama: `stress_level`, `daily_social_media_hours`, dan `anxiety_level` adalah fitur paling diskriminatif.

### 3️⃣ Data Preparation
Label Encoding pada kolom kategorikal (`gender`, `platform_usage`, `social_interaction_level`), kemudian train-test split 80:20 dengan stratifikasi kelas.

### 4️⃣ Modeling — *Menggunakan Regresi Logistik*
Model Regresi Logistik dilatih menggunakan 4 fitur terpilih (`stress_level`, `sleep_hours`, `anxiety_level`, `daily_social_media_hours`) dengan `solver='lbfgs'` dan `max_iter=1000`.

### 5️⃣ Evaluation
Evaluasi menggunakan Accuracy Score, ROC-AUC Score, Classification Report, Confusion Matrix, dan ROC Curve.

### 6️⃣ Deployment
Model disimpan sebagai `Mental_Health_Logistic.pkl` dan diintegrasikan ke dua versi aplikasi web (Streamlit & Gradio).

---

## 📊 Dataset

| Info | Detail |
|------|--------|
| Sumber | [Kaggle — Social Media Impact on Mental Health](https://www.kaggle.com/datasets/sunil123kumar/social-media-impact-on-mental-health) |
| File | `Teen_Mental_Health_Dataset.csv` |
| Target | `depression_label` (0 / 1) |
| Fitur yang digunakan model | `stress_level`, `sleep_hours`, `anxiety_level`, `daily_social_media_hours` |

---

## ⚖️ Peran Algoritma

| Algoritma | Fase CRISP-DM | Fungsi |
|-----------|--------------|--------|
| **Decision Tree** | Data Understanding | Eksplorasi & identifikasi fitur penting |
| **Regresi Logistik** | Modeling, Evaluation, Deployment | Prediksi probabilitas risiko depresi |

---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---------|----------|
| `pandas` | Manipulasi dan analisis data |
| `seaborn` & `matplotlib` | Visualisasi data |
| `scikit-learn` | Decision Tree, Regresi Logistik, evaluasi model |
| `kagglehub` | Download dataset dari Kaggle |
| `streamlit` | Antarmuka aplikasi web versi Streamlit |
| `gradio` | Antarmuka aplikasi web versi Gradio |
| `pickle` | Menyimpan dan memuat model |

---

## 🚀 Cara Menjalankan

### Notebook
```bash
pip install -r requirements.txt
jupyter notebook Mental_Health_CRISP_DM.ipynb
```

### Aplikasi Streamlit
```bash
streamlit run app_streamlit.py
```

### Aplikasi Gradio
```bash
python app_gradio.py
```

> Atau buka langsung di **Google Colab** tanpa instalasi tambahan.

---

## 👥 Tim Pengembang

| Peran | Kontribusi |
|-------|-----------|
| 👤 Anggota 1 | Business Understanding, Data Understanding, Eksplorasi Decision Tree & Visualisasi |
| 👤 Anggota 2 | Data Preparation, Modeling Regresi Logistik, Evaluation & Deployment |

> *Seluruh tahapan didiskusikan dan dikerjakan bersama secara kolaboratif.*

---

## ⚠️ Catatan

- Dataset bersifat **sintetis**, sehingga hasil tidak merepresentasikan populasi nyata secara langsung.
- Hasil prediksi aplikasi **bukan diagnosis medis** — hanya estimasi berbasis model machine learning.
- Untuk pengembangan lebih lanjut, kami merekomendasikan validasi dengan data primer dan eksplorasi model ensemble (Random Forest, XGBoost).

---

*Dibuat dengan ❤️ menggunakan Python, CRISP-DM Framework, Decision Tree & Regresi Logistik*