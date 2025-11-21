import streamlit as st
import joblib
import pandas as pd

# ------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Film",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------
# FUNGSI LOAD MODEL (DENGAN CACHE)
# ------------------------------------------------------------------------
# @st.cache_resource memastikan model hanya dimuat sekali ke memori
# Ini membuat aplikasi jauh lebih cepat saat tombol ditekan berulang kali
@st.cache_resource
def load_model_objects():
    # Pastikan nama file ini SAMA PERSIS dengan file yang Anda download dari Colab
    try:
        model_nb = joblib.load("model_sentiment.pkl")
        vectorizer_tfidf = joblib.load("vectorizer_tfidf.pkl")
        return model_nb, vectorizer_tfidf
    except FileNotFoundError:
        return None, None

# Memuat model
model, vectorizer = load_model_objects()

# ------------------------------------------------------------------------
# ANTARMUKA (UI)
# ------------------------------------------------------------------------
st.title("🎬 Analisis Sentimen Film")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Naive Bayes)** untuk mendeteksi apakah 
sebuah ulasan film bernada **Positif** atau **Negatif**.
""")

# Cek apakah model berhasil dimuat
if model is None or vectorizer is None:
    st.error("⚠️ File model tidak ditemukan! Pastikan file 'model_sentiment.pkl' dan 'vectorizer_tfidf.pkl' ada di folder yang sama dengan app.py.")
else:
    # Input User
    with st.container():
        st.subheader("Coba Ulasan Anda")
        input_text = st.text_area(
            "Masukkan ulasan film di sini:",
            height=150,
            placeholder="Contoh: Filmnya bagus banget, alurnya tidak ketebak!"
        )
        
        predict_btn = st.button("🔍 Analisis Sentimen", type="primary")

    # Logika Prediksi
    if predict_btn:
        if input_text.strip() == "":
            st.warning("Mohon masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner('Sedang menganalisis...'):
                # 1. Transformasi teks ke angka
                vectorized_text = vectorizer.transform([input_text])
                
                # 2. Prediksi Label
                prediction = model.predict(vectorized_text)[0]
                
                # 3. Prediksi Probabilitas
                proba = model.predict_proba(vectorized_text)[0]
                prob_neg = round(proba[0] * 100, 1)
                prob_pos = round(proba[1] * 100, 1)

                # ------------------------------------------------
                # TAMPILAN HASIL
                # ------------------------------------------------
                st.divider()
                
                # Menentukan warna dan pesan berdasarkan hasil
                if prediction == "positive": # Sesuaikan label dengan dataset (positive/negative)
                    st.success(f"### Hasil: Sentimen POSITIF 😊")
                else:
                    st.error(f"### Hasil: Sentimen NEGATIF 😠")

                st.write("Seberapa yakin model dengan prediksi ini?")
                
                # Membuat tampilan visual bar untuk probabilitas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Kemungkinan Negatif", f"{prob_neg}%")
                    st.progress(int(prob_neg))
                
                with col2:
                    st.metric("Kemungkinan Positif", f"{prob_pos}%")
                    st.progress(int(prob_pos))
