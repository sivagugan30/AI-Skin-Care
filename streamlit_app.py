import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="AI-Skin-Care", layout="centered")

st.title("💆‍♀️ AI-Skin-Care")
st.markdown("Upload a facial photo and get your skin **Health Score** out of 100!")

uploaded_file = st.file_uploader("📸 Upload a clear photo of your face", type=["jpg", "jpeg", "png"])

def calculate_health_score(image_array):
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # Calculate brightness (mean pixel value)
    brightness = np.mean(gray)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)

    # Dummy scoring formula (replace with ML model later)
    brightness_score = max(0, min(100, (brightness - 50) * 1.2))
    contrast_score = max(0, min(100, (contrast - 20) * 2))

    final_score = int((brightness_score * 0.4 + contrast_score * 0.6))
    return final_score

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to OpenCV format
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Calculate Health Score
    score = calculate_health_score(image_cv)

    st.subheader("🧬 Your Skin Health Score:")
    st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)

    # Tips based on score
    if score >= 80:
        st.success("Great skin! Keep it up! 💧✨")
    elif score >= 50:
        st.info("Not bad! A little skincare routine can help. 🧴🌿")
    else:
        st.warning("Consider hydrating and a skincare checkup. 🧼💧")

st.markdown("---")
st.markdown("🔬 *This is a demo app. For real skin analysis, consult a dermatologist.*")
