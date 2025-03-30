import streamlit as st
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="AI-Skin-Care", layout="centered")

st.title("💆‍♀️ AI-Skin-Care")
st.markdown("Take a photo of your face and get your skin **Health Score** out of 100!")

camera_photo = st.camera_input("📸 Take a clear photo of your face")

def calculate_health_score(image_pil):
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

if camera_photo is not None:
    # Convert to PIL image
    image = Image.open(camera_photo).convert("RGB")
    st.image(image, caption='Your Photo', use_column_width=True)

    # Calculate health score
    score = calculate_health_score(image)

    st.subheader("🧬 Your Skin Health Score:")
    st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)

    # Give feedback
    if score >= 80:
        st.success("Excellent skin! Keep doing what you're doing! 💧✨")
    elif score >= 50:
        st.info("Looking decent! With a regular skincare routine, you can glow up. 🌿🧴")
    else:
        st.warning("Consider improving hydration, sleep, and skincare. 🧼💧")

st.markdown("---")
st.markdown("🔬 *This is a demo app. For actual skin analysis, consult a dermatologist.*")
