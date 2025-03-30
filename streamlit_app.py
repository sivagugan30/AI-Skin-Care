import streamlit as st
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="AI-Skin-Care", layout="centered")

st.title("💆‍♀️ AI-Skin-Care")
st.markdown("Upload a facial photo and get your skin **Health Score** out of 100!")

uploaded_file = st.file_uploader("📸 Upload a clear photo of your face", type=["jpg", "jpeg", "png"])

def calculate_health_score_pure_pil(image_pil):
    # Convert image to grayscale
    grayscale_img = ImageOps.grayscale(image_pil)

    # Convert to NumPy array
    img_array = np.array(grayscale_img).astype('float')

    # Brightness = Mean pixel intensity
    brightness = np.mean(img_array)

    # Contrast = Standard deviation of pixel intensities
    contrast = np.std(img_array)

    # Dummy scoring logic
    brightness_score = max(0, min(100, (brightness - 50) * 1.2))
    contrast_score = max(0, min(100, (contrast - 20) * 2))

    final_score = int((brightness_score * 0.4 + contrast_score * 0.6))
    return final_score

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Calculate health score
    score = calculate_health_score_pure_pil(image)

    st.subheader("🧬 Your Skin Health Score:")
    st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)

    # Give basic feedback
    if score >= 80:
        st.success("Great skin! Keep it up! 💧✨")
    elif score >= 50:
        st.info("Pretty good! A consistent skincare routine will help. 🌿🧴")
    else:
        st.warning("Needs care. Stay hydrated and consider a skincare check. 🧼💧")

st.markdown("---")
st.markdown("🔬 *This is a demo app. For real skin analysis, consult a dermatologist.*")
