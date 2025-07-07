import streamlit as st
import pandas as pd
import math
from pathlib import Path

import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Aplikasi Prediksi AA",
    page_icon="🧠",
    layout="centered"
)

# CSS Tampilan
st.markdown("""
    <style>
    .judul-utama {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 30px;
    }
    .deskripsi {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# Inisialisasi page state
if "page" not in st.session_state:
    st.session_state.page = "home"

# --------------------------
# Halaman 1: Halaman Awal
# --------------------------
def halaman_utama():
    st.markdown('<div class="judul-utama">Selamat Datang di Aplikasi Prediksi AA</div>', unsafe_allow_html=True)
    st.markdown('<div class="deskripsi">Silakan klik tombol di bawah untuk memulai.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Lanjut ke Halaman Berikutnya", use_container_width=True):
            st.session_state.page = "halaman_berikutnya"
            st.experimental_rerun()

# --------------------------
# Halaman 2: Upload + Parameter
# --------------------------
def halaman_berikutnya():
    st.markdown('<div class="judul-utama">📊 Halaman Input Data dan Parameter</div>', unsafe_allow_html=True)
    st.markdown('<div class="deskripsi">Silakan upload file data dan isi parameter yang diperlukan untuk proses prediksi.</div>', unsafe_allow_html=True)

    # Upload file
    uploaded_file = st.file_uploader("📁 Upload file CSV atau Excel", type=["csv", "xlsx"], key="file_input")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ File berhasil diunggah dan dibaca.")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Gagal membaca file: {e}")

    st.markdown("---")

    # Parameter input
    st.subheader("🛠️ Parameter Prediksi")
    pilihan_model = st.selectbox("Pilih Model Prediksi", ["Naive Bayes", "Random Forest", "K-Nearest Neighbors"])
    threshold = st.slider("Threshold Keputusan", 0.0, 1.0, 0.5, 0.05)
    nama_model = st.text_input("Nama Model (opsional)", placeholder="Contoh: Model Percobaan 1")

    st.markdown("---")

    # Tombol kembali
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔙 Kembali ke Beranda", use_container_width=True):
            st.session_state.page = "home"
            st.experimental_rerun()

# --------------------------
# Routing Halaman
# --------------------------
if st.session_state.page == "home":
    halaman_utama()
elif st.session_state.page == "halaman_berikutnya":
    halaman_berikutnya()

