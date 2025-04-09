import streamlit as st
from PIL import Image, ImageOps
import numpy as np

def calculate_health_score(image_pil):
    grayscale_img = ImageOps.grayscale(image_pil)
    img_array = np.array(grayscale_img).astype('float')
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    brightness_score = max(0, min(100, (brightness - 50) * 1.2))
    contrast_score = max(0, min(100, (contrast - 20) * 2))
    final_score = int((brightness_score * 0.4 + contrast_score * 0.6))
    return final_score

st.set_page_config(page_title="AI-Skin-Care", layout="centered")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Instructions", "Analyze Your Face Skin", "Feedback"])

if page == "Home":
    st.title("💆‍♀️ AI-Skin-Care")
    st.markdown("Take a photo of your face and get your skin **Health Score** out of 100!")
    st.image("https://via.placeholder.com/600x300", caption="AI-Skin-Care Demo", use_column_width=True)
    st.markdown("---")
    st.markdown("🔬 *This is a demo app. For actual skin analysis, consult a dermatologist.*")

elif page == "Instructions":
    st.title("📖 Instructions")
    st.markdown("1. Make sure you are in a well-lit environment.\n2. Remove any heavy makeup for an accurate result.\n3. Position your face clearly in the camera frame.\n4. Click the 'Take Photo' button.\n5. Wait for your skin health score!")

elif page == "Analyze Your Face Skin":
    st.title("🔍 Analyze Your Face Skin")
    camera_photo = st.camera_input("📸 Take a clear photo of your face")
    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        st.image(image, caption='Your Photo', use_column_width=True)
        score = calculate_health_score(image)
        st.subheader("🧬 Your Skin Health Score:")
        st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)
        if score >= 80:
            st.success("Excellent skin! Keep doing what you're doing! 💧✨")
        elif score >= 50:
            st.info("Looking decent! With a regular skincare routine, you can glow up. 🌿🧴")
        else:
            st.warning("Consider improving hydration, sleep, and skincare. 🧼💧")

elif page == "Feedback":
    st.title("📝 Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 💙")
