import requests
import torch
from io import BytesIO
from torchvision import models
import torch.nn as nn
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import random
from torchvision import transforms
#
# URL to Google Drive raw .pth file
model_url = 'https://drive.google.com/uc?export=download&id=1alJZXduCaKEEcTrxLYDtF112fY1BXZZQ'

# Download the .pth file
response = requests.get(model_url)
model_data = BytesIO(response.content)

# Load the model from the .pth file
model_state_dict = torch.load(model_data, map_location=torch.device('cpu'))

# Define and load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

# Define class names
class_names = ['Level 0', 'Level 1', 'Level 2']

# Image transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def acne_score(pred_class):
    if pred_class == 0:
        return random.randint(50, 60)
    elif pred_class == 1:
        return random.randint(70, 80)
    elif pred_class == 2:
        return random.randint(90, 100)
    else:
        return 0

def calculate_health_score(image):
    img_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    pred_class = predicted.item()
    score = acne_score(pred_class)
    return score, pred_class

# Streamlit app code

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
            score, predicted_class = calculate_health_score(image)

        st.subheader("🧬 Your Skin Health Score:")
        st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)
        st.write(f"🔍 **Predicted Acne Severity Level:** `{class_names[predicted_class]}`")

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

        st.balloons()

    else:
        st.info("👈 Upload a photo or take one to start your skin analysis.")

elif page == "Feedback":
    st.title("📝 Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 💙")
