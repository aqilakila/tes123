import streamlit as st
import joblib
import pandas as pd
import re

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
@st.cache_resource
def load_model_objects():
    """
    Load 3 file yang dibutuhkan:
    1. Model BernoulliNB
    2. Vectorizer TF-IDF
    3. Preprocessing tools (stopword remover & stemmer)
    """
    try:
        model_nb = joblib.load("model.pkl")
        vectorizer_tfidf = joblib.load("vector.pkl")
        preprocessing_tools = joblib.load("preprocess.pkl")
        
        return model_nb, vectorizer_tfidf, preprocessing_tools
    except FileNotFoundError as e:
        st.error(f"⚠️ File tidak ditemukan: {e.filename}")
        return None, None, None

# Memuat model dan tools
model, vectorizer, tools = load_model_objects()

# ------------------------------------------------------------------------
# FUNGSI PREPROCESSING
# ------------------------------------------------------------------------
def preprocess_text(text, stopword_remover, stemmer):
    """
    Preprocessing sesuai dengan yang digunakan saat training:
    1. Case folding & cleaning
    2. Stopword removal
    3. Stemming
    """
    # Case folding & cleaning
    text = re.sub('[^A-Za-z]+', ' ', text).lower().strip()
    text = re.sub('\s+', ' ', text)
    
    # Stopword removal
    text = stopword_remover.remove(text)
    
    # Stemming
    text = stemmer.stem(text)
    
    return text

# ------------------------------------------------------------------------
# ANTARMUKA (UI)
# ------------------------------------------------------------------------
st.title("🎬 Analisis Sentimen Film")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Bernoulli Naive Bayes)** untuk mendeteksi apakah 
sebuah ulasan film bernada **Positif** atau **Negatif**.

**Teknologi yang digunakan:**
- **Model:** Bernoulli Naive Bayes
- **Vectorizer:** TF-IDF (min_df=5, max_df=0.8)
- **Preprocessing:** Sastrawi (Bahasa Indonesia)
""")

# Cek apakah model berhasil dimuat
if model is None or vectorizer is None or tools is None:
    st.error("""
    ⚠️ **File model tidak ditemukan!** 
    
    Pastikan 3 file berikut ada di folder yang sama dengan app.py:
    1. `model_sentiment_bernoulli.pkl`
    2. `vectorizer_tfidf_optimal.pkl`
    3. `preprocessing_tools.pkl`
    """)
else:
    # Input User
    with st.container():
        st.subheader("✍️ Coba Ulasan Anda")
        
        # Contoh ulasan untuk memudahkan testing
        example_texts = [
            "Filmnya bagus banget, alurnya tidak ketebak!",
            "Film jelek, buang waktu saja",
            "Keren, aktingnya mantap sekali",
            "Goblok banget filmnya tidak bermutu"
        ]
        
        selected_example = st.selectbox(
            "Atau pilih contoh ulasan:",
            ["-- Ketik manual --"] + example_texts
        )
        
        # Jika user memilih contoh, isi text area
        default_text = "" if selected_example == "-- Ketik manual --" else selected_example
        
        input_text = st.text_area(
            "Masukkan ulasan film di sini:",
            value=default_text,
            height=120,
            placeholder="Contoh: Filmnya bagus banget, alurnya tidak ketebak!"
        )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            predict_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)
        with col2:
            show_details = st.checkbox("Tampilkan detail preprocessing", value=False)

    # Logika Prediksi
    if predict_btn:
        if input_text.strip() == "":
            st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")
        else:
            with st.spinner('Sedang menganalisis...'):
                try:
                    # 1. Preprocessing
                    stopword_remover = tools['stopword_remover']
                    stemmer = tools['stemmer']
                    processed_text = preprocess_text(input_text, stopword_remover, stemmer)
                    
                    # 2. Transformasi teks ke angka (TF-IDF)
                    vectorized_text = vectorizer.transform([processed_text])
                    
                    # 3. Prediksi Label
                    prediction = model.predict(vectorized_text)[0]
                    
                    # 4. Prediksi Probabilitas
                    proba = model.predict_proba(vectorized_text)[0]
                    prob_neg = round(proba[0] * 100, 1)
                    prob_pos = round(proba[1] * 100, 1)

                    # ------------------------------------------------
                    # TAMPILAN HASIL
                    # ------------------------------------------------
                    st.divider()
                    
                    # Menentukan warna, emoji, dan confidence level
                    max_prob = max(prob_neg, prob_pos)
                    
                    if max_prob > 80:
                        confidence = "🟢 Tinggi"
                    elif max_prob > 60:
                        confidence = "🟡 Sedang"
                    else:
                        confidence = "🔴 Rendah"
                    
                    # Menentukan warna dan pesan berdasarkan hasil
                    if prediction == "positive":
                        st.success(f"### ✅ Hasil: Sentimen POSITIF 😊")
                        st.info(f"**Tingkat Keyakinan:** {confidence} ({max_prob}%)")
                    else:
                        st.error(f"### ❌ Hasil: Sentimen NEGATIF 😠")
                        st.info(f"**Tingkat Keyakinan:** {confidence} ({max_prob}%)")

                    # Detail Preprocessing (optional)
                    if show_details:
                        with st.expander("🔍 Lihat Detail Preprocessing"):
                            st.write("**Teks Asli:**")
                            st.code(input_text)
                            st.write("**Teks Setelah Preprocessing:**")
                            st.code(processed_text)
                            st.write("**Transformasi:**")
                            st.write("- Case folding (huruf kecil)")
                            st.write("- Hapus karakter khusus & angka")
                            st.write("- Hapus stopwords (kata hubung)")
                            st.write("- Stemming (kata dasar)")

                    st.write("---")
                    st.write("**📊 Distribusi Probabilitas:**")
                    
                    # Membuat tampilan visual bar untuk probabilitas
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("😠 Negatif", f"{prob_neg}%")
                        st.progress(int(prob_neg))
                    
                    with col2:
                        st.metric("😊 Positif", f"{prob_pos}%")
                        st.progress(int(prob_pos))
                    
                    # Info tambahan
                    st.divider()
                    st.caption("""
                    💡 **Catatan:** Model ini dilatih menggunakan dataset ulasan film berbahasa Indonesia.
                    Akurasi model tergantung pada kemiripan input dengan data training.
                    """)
                    
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")
                    st.info("Pastikan model sudah di-train dengan benar dan file tidak corrupt.")

# ------------------------------------------------------------------------
# SIDEBAR INFO
# ------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    
    if model is not None:
        st.write("**Status:** ✅ Model Loaded")
        st.write("**Algoritma:** Bernoulli Naive Bayes")
        st.write("**Vectorizer:** TF-IDF")
        st.write("**Bahasa:** Indonesia")
        
        st.divider()
        
        st.subheader("📝 Cara Menggunakan")
        st.write("""
        1. Ketik atau pilih contoh ulasan film
        2. Klik tombol "Analisis Sentimen"
        3. Lihat hasil prediksi dan tingkat keyakinan
        4. (Opsional) Centang "Tampilkan detail" untuk melihat preprocessing
        """)
        
        st.divider()
        
        st.subheader("🎯 Contoh Ulasan")
        st.write("**Positif:**")
        st.write("- Film bagus, sangat menghibur")
        st.write("- Keren banget aktingnya")
        
        st.write("\n**Negatif:**")
        st.write("- Film jelek, membosankan")
        st.write("- Tidak suka, buang waktu")
    else:
        st.write("**Status:** ❌ Model Not Loaded")
        st.error("Mohon pastikan semua file model ada di folder yang benar")

# Footer
st.divider()
st.caption("🎬 Analisis Sentimen Film | Powered by Streamlit & Scikit-learn")