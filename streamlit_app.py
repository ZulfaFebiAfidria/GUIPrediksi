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

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Daging Ayam Broiler - Jawa Timur",
    page_icon="🍗",
    layout="wide"
)

st.title(":bar_chart: Dashboard Prediksi Harga Daging Ayam Broiler - Jawa Timur")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Dataset", 
    "⚙️ Preprocessing", 
    "🤖 Model", 
    "📉 Evaluasi"
])

# Tab 1 - Dataset
with tab1:
    st.header(":file_folder: Dataset")

    uploaded_file = st.file_uploader("Upload Dataset Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state['df_raw'] = df.copy()
        st.write("Data Preview:")
        st.dataframe(df.head())
        st.write("Deskripsi Statistik:")
        st.dataframe(df.describe())
    else:
        st.info("Silakan upload file Excel (.xlsx)")

# Tab 2 - Preprocessing
with tab2:
    st.header(":gear: Preprocessing")

    if 'df_raw' in st.session_state:
        df = st.session_state['df_raw'].copy()

        # Feature engineering manual
        df['rasio_pakan_daging'] = df['Harga Pakan Ternak Broiler'] / df['Harga Daging Ayam Broiler']
        df['rasio_doc_daging'] = df['Harga DOC Broiler'] / df['Harga Daging Ayam Broiler']
        df['rasio_jagung_pakan'] = df['Harga Jagung TK Peternak'] / df['Harga Pakan Ternak Broiler']

        df['ma7_daging'] = df['Harga Daging Ayam Broiler'].rolling(window=7).mean()
        df['ma7_pakan'] = df['Harga Pakan Ternak Broiler'].rolling(window=7).mean()
        df['ma7_doc'] = df['Harga DOC Broiler'].rolling(window=7).mean()
        df['ma7_jagung'] = df['Harga Jagung TK Peternak'].rolling(window=7).mean()

        df['lag1_daging'] = df['Harga Daging Ayam Broiler'].shift(1)
        df['lag2_daging'] = df['Harga Daging Ayam Broiler'].shift(2)
        df['pct_change_daging'] = df['Harga Daging Ayam Broiler'].pct_change()

        df.dropna(inplace=True)
        st.session_state['df_clean'] = df.copy()
        st.success("Preprocessing selesai.")
        st.dataframe(df.head())
    else:
        st.warning("Silakan upload dataset terlebih dahulu.")

# Tab 3 - Model
with tab3:
    st.header(":robot_face: Pelatihan Model XGBoost")

    if 'df_clean' in st.session_state:
        df = st.session_state['df_clean']

        fitur = [
            'rasio_pakan_daging', 'rasio_doc_daging', 'rasio_jagung_pakan',
            'ma7_daging', 'ma7_pakan', 'ma7_doc', 'ma7_jagung',
            'lag1_daging', 'lag2_daging', 'pct_change_daging'
        ]
        target = 'Harga Daging Ayam Broiler'

        X = df[fitur]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_default = XGBRegressor(random_state=42)
        model_default.fit(X_train_scaled, y_train)
        y_pred_default = model_default.predict(X_test_scaled)

        st.success("Model baseline dilatih!")
        st.write("Jumlah data train:", len(X_train))
        st.write("Jumlah data test:", len(X_test))

        # Tuning Optuna
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'objective': 'reg:squarederror'
            }
            model = XGBRegressor(**params, random_state=42)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            return rmse

        with st.spinner("Menjalankan tuning Optuna..."):
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_warmup_steps=10)
            )
            study.optimize(objective, n_trials=50)

        best_model = XGBRegressor(**study.best_params, random_state=42)
        best_model.fit(X_train_scaled, y_train)
        y_pred_best = best_model.predict(X_test_scaled)

        st.success("Model hasil tuning selesai dilatih!")
        st.session_state['y_test'] = y_test
        st.session_state['y_pred_default'] = y_pred_default
        st.session_state['y_pred_best'] = y_pred_best

    else:
        st.warning("Preprocessing data diperlukan sebelum melatih model.")

# Tab 4 - Evaluasi
with tab4:
    st.header(":chart_with_downwards_trend: Evaluasi Model")

    if all(k in st.session_state for k in ['y_test', 'y_pred_default', 'y_pred_best']):
        y_test = st.session_state['y_test']
        y_pred_default = st.session_state['y_pred_default']
        y_pred_best = st.session_state['y_pred_best']

        def evaluate_model(y_true, y_pred):
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            return rmse, mape

        rmse_default, mape_default = evaluate_model(y_test, y_pred_default)
        rmse_best, mape_best = evaluate_model(y_test, y_pred_best)

        st.subheader("Hasil Evaluasi")
        st.write("[DEFAULT] RMSE:", f"{rmse_default:.2f}", ", MAPE:", f"{mape_default:.2f}%")
        st.write("[TUNED  ] RMSE:", f"{rmse_best:.2f}", ", MAPE:", f"{mape_best:.2f}%")

        fig_eval, ax_eval = plt.subplots()
        ax_eval.plot(y_test.values, label="Aktual", linewidth=2)
        ax_eval.plot(y_pred_default, label="Prediksi Default", linestyle='--')
        ax_eval.plot(y_pred_best, label="Prediksi Tuned", linestyle='--')
        ax_eval.legend()
        ax_eval.set_title("Perbandingan Hasil Prediksi")
        st.pyplot(fig_eval)
    else:
        st.warning("Silakan latih model terlebih dahulu.")
