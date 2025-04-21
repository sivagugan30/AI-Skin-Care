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
    st.title("💆‍♀️ AI-Skin-Care 10:11 PM")

    st.markdown("🔬 *This is a demo app. For actual skin analysis, consult a dermatologist.*")

elif page == "Instructions":
    st.title("📖 Instructions")
    st.markdown("1. Make sure you are in a well-lit environment.\n2. Remove any heavy makeup for an accurate result.\n3. Position your face clearly in the camera frame.\n4. Click the 'Take Photo' button.\n5. Wait for your skin health score!")

elif page == "Analyze Your Face Skin":
    st.title("🔍 Analyze Your Face Skin")
    st.markdown("Choose a method to analyze your skin:")

    option = st.radio("Select input type:", ["Upload an Image", "Take a Photo"])
    image = None

    if option == "Upload an Image":
        uploaded_image = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            st.success("Image uploaded successfully!")

    if option == "Take a Photo":
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")
            st.success("Photo captured successfully!")

    if image:
        st.image(image, caption='Your Face Photo', use_column_width=True)
        with st.spinner("Analyzing your skin health..."):
            score = calculate_health_score(image)

        st.subheader("🧬 Your Skin Health Score:")
        st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)

        with st.expander("💡 What does this mean?"):
            if score >= 80:
                st.success("Excellent skin! Keep doing what you're doing! 💧✨")
                st.markdown("✅ Well-hydrated\n✅ Balanced complexion\n✅ Low pore visibility")
            elif score >= 50:
                st.info("Looking decent! With a regular skincare routine, you can glow up. 🌿🧴")
                st.markdown("✔️ Minor dullness\n✔️ Some uneven texture\n💡 Try moisturizing regularly")
            else:
                st.warning("Consider improving hydration, sleep, and skincare. 🧼💧")
                st.markdown("⚠️ Dryness or dull tone\n⚠️ Visible spots or texture\n💡 Drink more water and sleep well")

        st.ballons()

    else:
        st.info("👈 Upload a photo or take one to start your skin analysis.")

elif page == "Feedback":
    st.title("📝 Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 💙")
