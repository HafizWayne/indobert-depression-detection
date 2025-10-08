# Deteksi Indikasi Depresi di Media Sosial Indonesia menggunakan IndoBERT

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indobert-indication-depression.streamlit.app/)

Repositori ini berisi kode dan model untuk skripsi saya di ITENAS dengan tujuan mengklasifikasikan postingan media sosial berbahasa Indonesia ke dalam kategori indikasi depresi atau non-depresi, serta penanganan negasi.

![Screenshot Aplikasi](https://github.com/HafizWayne/indobert-depression-detection/blob/main/image/Streamlit.png) 

---

## 🚀 Cara Menggunakan

Ada dua cara untuk menggunakan aplikasi ini:

### 1. Via Web Browser (Rekomendasi)
Cara termudah adalah dengan mengakses aplikasi yang sudah di-deploy secara online. Tidak perlu instalasi apa pun.

> **Buka Aplikasi Web: [https://indobert-indication-depression.streamlit.app/](https://indobert-indication-depression.streamlit.app/)**

### 2. Via Aplikasi Windows (.exe)
Anda juga bisa mengunduh aplikasi "peluncur" ringan untuk Windows. Aplikasi ini akan membuka aplikasi web di browser default Anda.

> **Unduh Launcher.exe dari halaman [Releases](https://github.com/HafizWayne/indobert-depression-detection/releases)**

---

## ✨ Fitur Utama

* Pra-pemrosesan teks khusus untuk bahasa media sosial Indonesia.
* Metode penanganan negasi menggunakan negation combine.
* Fine-tuning model IndoBERT untuk klasifikasi teks.
* Evaluasi performa model menggunakan Akurasi, Presisi, Recall, dan F1-score.

---

## 📊 Hasil

* **Model Terbaik:** IndoBERT dengan metode *Negation Prefix*.
* **F1-Score:** 0.9011 pada set data pengujian.

---

## 🔧 Tumpukan Teknologi

* **Bahasa:** Python
* **Framework:** Streamlit
* **Model:** PyTorch, Transformers (IndoBERT)
* **Pustaka Lain:** Pandas, NumPy, Scikit-learn, Matplotlib, FPDF

---

## 📂 Struktur Repositori

* `perhitungan_full.py`: Script utama aplikasi Streamlit.
* `launcher.py`: Script untuk `.exe` peluncur.
* `requirements.txt`: Daftar dependensi untuk deployment.
* `finetuned_indobert_negation_combine/`: Folder berisi model yang sudah di-fine-tuning.
* Aset lain seperti file font (`.ttf`) dan `token_negasi.txt`.
