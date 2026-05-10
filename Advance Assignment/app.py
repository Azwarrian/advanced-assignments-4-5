import streamlit as st
import pandas as pd
import joblib

# 1. Load Model
# Pastikan file model.joblib berada di folder yang sama
model = joblib.load('model.joblib')

# 2. Judul Aplikasi
st.title("🚢 Titanic Survival Prediction")
st.write("Aplikasi sederhana untuk memprediksi probabilitas keselamatan penumpang kapal Titanic.")

# 3. Form Input Data
st.header("Masukkan Data Penumpang")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Sex", ["male", "female"])
    embarked = st.selectbox("Port of Embarkation (Embarked)", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare (Harga Tiket)", min_value=0.0, value=30.0)

# 4. Tombol Prediksi
if st.button("Prediksi Keselamatan"):
    # Menyusun data input pengguna menjadi DataFrame
    # PERHATIAN: Nama kolom harus PERSIS seperti di notebook
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Embarked': [embarked],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare]
    })
    
    # Melakukan prediksi
    prediction = model.predict(input_data)
    
    # Menampilkan hasil
    st.subheader("Hasil Prediksi:")
    if prediction[0] == 1:
        st.success("Berdasarkan data, penumpang diprediksi **SELAMAT** (Survived). 🟢")
    else:
        st.error("Berdasarkan data, penumpang diprediksi **TIDAK SELAMAT** (Did Not Survive). 🔴")