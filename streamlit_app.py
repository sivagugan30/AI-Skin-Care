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

import streamlit as st
from PIL import Image
import os
import pandas as pd
import plotly.express as px
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Instructions", "Analyze Your Face Skin", "Documentation",
                                  "Model Monitering Dashboard", "Feedback"])

if page == "Documentation":
    tab1, tab2, tab3, tab4 = st.tabs(["Data Collection", "Data Analysis", "Image Processing", "Model Development"])
    with tab3:
        with tab3:
            st.write("")
            st.write("Images are processed using various techniques to enhance quality and prepare them for model input.")

        image_path = '/Users/sivaguganjayachandran/PycharmProjects/AI-Skin-Care/data/0/levle0_1.jpg'
        original_image = Image.open(image_path)

        # 1. Resize
        resized_image = original_image.resize((360, 380))

        # 2. Grayscale
        gray_image = original_image.convert("L")

        # 3. Contrast Enhancement
        enhancer = ImageEnhance.Contrast(original_image)
        contrast_image = enhancer.enhance(2.0)

        # 4. Edge Detection
        edge_image = original_image.filter(ImageFilter.FIND_EDGES)

        # Add a brightness-enhanced version
        bright_image = ImageEnhance.Brightness(original_image).enhance(1.5)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original", use_container_width=True)
            st.image(resized_image, caption="Resized (360x380)", use_container_width=True)
            st.image(bright_image, caption="Brightness Enhanced", use_container_width=True)

        with col2:
            st.image(gray_image, caption="Grayscale", use_container_width=True)
            st.image(contrast_image, caption="Contrast Enhanced", use_container_width=True)
            st.image(edge_image, caption="Edge Detection", use_container_width=True)

    with tab1:
        st.write('')
        st.write('GitHub Repository: [AI-Skin-Care](https://github.com/sivagugan30/AI-Skin-Care)')
        st.write('Data Source: [Kaggle Acne Dataset](https://www.kaggle.com/datasets/manuelhettich/acne04)')

        st.write("#### Process Diagram")
        process_diagram_path = "/Users/sivaguganjayachandran/PycharmProjects/AI-Skin-Care/data/Process Diagram.png"
        if os.path.exists(process_diagram_path):
            process_diagram = Image.open(process_diagram_path)
            st.image(process_diagram, caption="AI Skin Care Pipeline",use_container_width=True)
        else:
            st.warning("Process Diagram not found at specified path.")

        st.markdown("#### Acne Classes")
        acne_class = st.selectbox("Select Acne Severity Class", ["Mild", "Moderate", "Severe"])
        class_map = {"Mild": "0", "Moderate": "1", "Severe": "2"}
        example_path = f"data/{class_map[acne_class]}"

        image_files = list(Path(example_path).glob("*.jpg"))
        if image_files:
            random_image = random.choice(image_files)
            st.image(str(random_image), caption=f"{acne_class} Acne Example", use_container_width=True)
        else:
            st.warning(f"No images found in {example_path}")

    with tab2:
        st.markdown("### Standardizing Image Dimensions")

        # Load the saved CSV
        input_path = os.path.expanduser("~/Downloads/image_dimensions.csv")
        try:
            df = pd.read_csv(input_path)

            # Calculate means
            mean_width = df["width"].mean()
            mean_height = df["height"].mean()

            # Scatter plot with mean lines
            fig = px.scatter(
                df,
                x="width",
                y="height",
                title="Image Dimension Plot with Mean Markers",
                template="plotly_dark",
                labels={"width": "Image Width", "height": "Image Height"}
            )

            fig.add_shape(type="line", x0=mean_width, x1=mean_width,
                          y0=df["height"].min(), y1=df["height"].max(),
                          line=dict(color="red", width=2, dash="dash"))

            fig.add_shape(type="line", x0=df["width"].min(), x1=df["width"].max(),
                          y0=mean_height, y1=mean_height,
                          line=dict(color="red", width=2, dash="dash"))

            fig.add_annotation(x=mean_width, y=df["height"].max(),
                               text="Mean Width", showarrow=False, yshift=10,
                               font=dict(color="red"))
            fig.add_annotation(x=df["width"].max(), y=mean_height,
                               text="Mean Height", showarrow=False, xshift=10,
                               font=dict(color="red"))

            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(title_font_size=20)

            st.plotly_chart(fig)

            st.write("Image sizes vary across the dataset. Mean dimensions are used as the target size for consistent preprocessing.")

        except FileNotFoundError:
            st.error("CSV file not found. Please check the path to 'image_dimensions.csv'.")

        st.write("")
        # Manually inputted class counts
        class_counts = {
            "Mild": 483,
            "Moderate": 623,
            "Severe": 175
        }

        # Create DataFrame for plotting
        dist_df = pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])

        # Plot the distribution
        fig2 = px.bar(dist_df, x="Class", y="Count", title="Distribution of Acne Severity Labels", color="Class")
        st.plotly_chart(fig2)

        # Add comment about imbalance
        st.write(
            "Class imbalance is evident — 'Severe' is underrepresented. This could impact model fairness and performance.")
if page == "Home":
    st.title("💆‍♀️ AI-Skin-Care")

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

    elif option == "Take a Photo":
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")
            st.success("Photo captured successfully!")

    if image:
        #st.image(image, caption='Your Face Photo', use_column_width=True)
        with st.spinner("Analyzing your skin health..."):
            # Score calculation
            score = calculate_health_score(image)

            # Show score
            #st.subheader("🧬 Your Skin Health Score:")
            #st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)

            # 🧠 Load model and predict class
            import torch
            import torchvision.transforms as transforms
            from torchvision import models  # or your custom model import
            from PIL import Image

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load model (define your model architecture if it's custom)
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 3)  # assuming 3 classes
            model.load_state_dict(torch.load("model/acne_model.pth", map_location=device))
            model.to(device)
            model.eval()

            class_names = ["Mild", "Moderate", "Severe"]  # Customize to your labels

            # Transform and predict
            test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

            img_tensor = test_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)

            predicted_class = class_names[predicted.item()]

            st.markdown("Predicted Acne Severity Level:" + f"<h2 style='color: #d62728;'>{predicted_class}</h2>", unsafe_allow_html=True)

        st.balloons()

    else:
        st.info("👈 Upload a photo or take one to start your skin analysis.")

elif page == "Feedback":
    st.title("📝 Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 💙")
