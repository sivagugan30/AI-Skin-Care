import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load("model/acne_model.pth", map_location=torch.device("cpu"))
model.eval()

# Class names
class_names = ['Mild', 'Moderate', 'Severe']

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Streamlit UI
st.title("🧠 Acne Severity Predictor")
st.write("Upload a face image to predict the acne severity level (0: Mild, 1: Moderate, 2: Severe)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    st.subheader(f"🎯 Prediction: **{class_names[prediction]}** ({prediction})")