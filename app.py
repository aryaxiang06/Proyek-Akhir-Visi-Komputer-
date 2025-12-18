import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="Stock Chart Pattern Classifier",
    page_icon="📈",
    layout="centered"
)

# Path
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Path model
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.keras"

# Kelas
CLASSES = ['bullish', 'bearish', 'sideways']
CLASS_EMOJIS = {'bullish': '📈', 'bearish': '📉', 'sideways': '➡️'}

# ==================== FUNGSI UTILITAS ====================
@st.cache_resource
def load_models():
    """Load kedua model"""
    models = {}
    
    # Load best model
    if BEST_MODEL_PATH.exists():
        models['best'] = tf.keras.models.load_model(str(BEST_MODEL_PATH))
    else:
        st.error(f"best_model.keras tidak ditemukan di {BEST_MODEL_PATH}")
        st.stop()
    
    # Load final model  
    if FINAL_MODEL_PATH.exists():
        models['final'] = tf.keras.models.load_model(str(FINAL_MODEL_PATH))
    else:
        st.error(f"final_model.keras tidak ditemukan di {FINAL_MODEL_PATH}")
        st.stop()
    
    return models

def preprocess_image(image):
    """Preprocess gambar"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Konversi ke RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize dan normalisasi
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    
    return np.expand_dims(image, axis=0)

def get_prediction(models, image):
    """Dapatkan prediksi dari kedua model"""
    processed_img = preprocess_image(image)
    
    # Prediksi dari kedua model
    preds_best = models['best'].predict(processed_img, verbose=0)[0]
    preds_final = models['final'].predict(processed_img, verbose=0)[0]
    
    # Rata-rata probabilitas dari kedua model
    avg_probs = (preds_best + preds_final) / 2
    
    # Ambil prediksi dengan probabilitas tertinggi
    pred_idx = np.argmax(avg_probs)
    pred_class = CLASSES[pred_idx]
    confidence = float(avg_probs[pred_idx])
    
    return pred_class, confidence, avg_probs

# ==================== APLIKASI UTAMA ====================
st.title("📈 Stock Chart Pattern Classifier")
st.markdown("Upload gambar chart saham untuk analisis pola")

# Load model
models = load_models()

# Upload gambar
uploaded_file = st.file_uploader(
    "Pilih gambar chart saham",
    type=['png', 'jpg', 'jpeg'],
    help="Format: PNG, JPG, JPEG"
)

if uploaded_file:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(image, use_column_width=True)
    
    with col2:
        if st.button("**ANALISIS**", type="primary", use_container_width=True):
            with st.spinner("Menganalisis..."):
                try:
                    # Dapatkan prediksi
                    pred_class, confidence, all_probs = get_prediction(models, image)
                    
                    # Tampilkan hasil
                    st.markdown("---")
                    
                    # Emoji dan prediksi
                    emoji = CLASS_EMOJIS[pred_class]
                    st.markdown(f"### {emoji} **{pred_class.upper()}**")
                    
                    # Tingkat keyakinan
                    st.markdown(f"### **{confidence:.1%}**")
                    
                    # Bar confidence
                    st.progress(float(confidence), text=f"Tingkat Keyakinan: {confidence:.1%}")
                    
                    # Warna berdasarkan confidence
                    if confidence > 0.7:
                        st.success("✅ Tingkat keyakinan TINGGI")
                    elif confidence > 0.5:
                        st.info("⚠️ Tingkat keyakinan SEDANG")
                    else:
                        st.warning("⚠️ Tingkat keyakinan RENDAH")
                    
                    # Deskripsi singkat
                    if pred_class == 'bullish':
                        st.info("**Pola Bullish**: Tren naik, sinyal beli")
                    elif pred_class == 'bearish':
                        st.info("**Pola Bearish**: Tren turun, sinyal jual")
                    else:
                        st.info("**Pola Sideways**: Tren datar, tunggu konfirmasi")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.info("Klik ANALISIS untuk mulai")

# Footer minimal
st.markdown("---")
st.caption("Proyek Akhir Visi Komputer • CNN Stock Pattern Classification")