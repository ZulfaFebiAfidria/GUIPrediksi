import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Daging Ayam Broiler - Jawa Timur",
    page_icon="🍗",
    layout="wide"
)

st.title("📊 Dashboard Prediksi Harga Daging Ayam Broiler - Jawa Timur")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Dataset", 
    "⚙ Preprocessing", 
    "📈 Visualisasi", 
    "🤖 Model", 
    "📉 Hasil Prediksi"
])

# Tab 1 - Dataset
with tab1:
    st.header("📂 Dataset")

    required_columns = [
        'Date',
        'Harga Pakan Ternak Broiler',
        'Harga DOC Broiler',
        'Harga Jagung TK Peternak',
        'Harga Daging Ayam Broiler'
    ]

    uploaded_file = st.file_uploader("Upload Dataset Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                st.error(f"❌ Kolom berikut tidak ditemukan di file Excel: {', '.join(missing_cols)}")
            else:
                for col in df.columns:
                    if col != 'Date':
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                st.session_state['df'] = df
                st.success("✅ Dataset valid!")
                st.write("Data Preview:")
                st.dataframe(df.head())

                with st.expander("📊 Deskripsi Statistik"):
                    st.dataframe(df.describe())

        except Exception as e:
            st.error(f"❌ Gagal membaca file Excel. Error: {e}")
    else:
        st.info("Silakan upload file Excel (.xlsx) yang berisi semua variabel yang dibutuhkan.")

# Tab 2 - Preprocessing
with tab2:
    st.header("⚙ Preprocessing Data")

    if 'df' in st.session_state:
        df = st.session_state['df'].copy()

        st.subheader("1️⃣ Pembersihan Nama Kolom")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.write("Nama kolom setelah dibersihkan:")
        st.write(df.columns.tolist())

        df.rename(columns={
            'harga_pakan_ternak_broiler': 'pakan',
            'harga_doc_broiler': 'doc',
            'harga_jagung_tk_peternak': 'jagung',
            'harga_daging_ayam_broiler': 'daging',
            'date': 'tanggal'
        }, inplace=True)

        st.subheader("2️⃣ Missing Values")

        kolom_target = ['pakan', 'doc', 'jagung', 'daging']
        missing_before = df[kolom_target].isna().sum()

        df[kolom_target] = df[kolom_target].interpolate(method='linear')
        for col in kolom_target:
            df[col].fillna(method='ffill', inplace=True)
            df[col].fillna(method='bfill', inplace=True)

        missing_after = df[kolom_target].isna().sum()

        missing_df = pd.DataFrame({
            "Sebelum": missing_before,
            "Sesudah": missing_after
        })

        st.write("Jumlah missing value sebelum dan sesudah penanganan:")
        st.dataframe(missing_df)

        st.subheader("3️⃣ Deteksi Outlier (IQR)")
        Q1 = df[kolom_target].quantile(0.25)
        Q3 = df[kolom_target].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[kolom_target] < (Q1 - 1.5 * IQR)) | (df[kolom_target] > (Q3 + 1.5 * IQR))
        st.write("Jumlah outlier per kolom:")
        st.dataframe(outliers.sum())

        fig_outlier, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=df[kolom_target], orient='h', palette='Set2', ax=ax)
        ax.set_title("Boxplot Deteksi Outlier")
        st.pyplot(fig_outlier)

        st.subheader("4️⃣ Transformasi Log")
        for col in kolom_target:
            df[f"{col}_log"] = np.log(df[col])

        log_cols = [f"{col}_log" for col in kolom_target]
        st.write("Contoh kolom hasil log transform:")
        st.dataframe(df[log_cols].head())

        fig_log, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()
        for i, col in enumerate(log_cols):
            sns.histplot(df[col], kde=True, color='skyblue', ax=axs[i])
            axs[i].set_title(f'Distribusi Log: {col}')
        plt.tight_layout()
        st.pyplot(fig_log)

        st.session_state['df_clean'] = df

    else:
        st.warning("Silakan upload dataset di tab 📂 Dataset.")

# Tab 3 - Visualisasi
with tab3:
    st.header("📈 Visualisasi Dataset")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        st.subheader("Distribusi Harga Daging")
        fig1, ax1 = plt.subplots()
        sns.histplot(df['daging'], kde=True, ax=ax1)
        st.pyplot(fig1)

        st.subheader("Korelasi antar Fitur")
        fig2, ax2 = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

        st.subheader("Visualisasi Time Series Harga")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df['tanggal'], df['pakan'], label='Pakan')
        ax3.plot(df['tanggal'], df['doc'], label='DOC')
        ax3.plot(df['tanggal'], df['jagung'], label='Jagung')
        ax3.plot(df['tanggal'], df['daging'], label='Daging Ayam')
        ax3.set_title("Pergerakan Harga dari Waktu ke Waktu")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("Harga")
        ax3.legend()
        st.pyplot(fig3)

    else:
        st.warning("Silakan lakukan preprocessing terlebih dahulu.")
