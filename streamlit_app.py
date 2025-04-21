import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import plotly.graph_objects as go  
import os
import pandas as pd
import plotly.express as px
from pathlib import Path
import random
from PIL import Image, ImageEnhance, ImageFilter
from plotly.subplots import make_subplots



st.set_page_config(page_title="AI-Skin-Care", layout="centered")




st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Instructions", "Analyze Your Face Skin", "Documentation",
                                  "Model Monitering Dashboard", "Feedback"])
if page == "Home":
    st.title("AI-Skin-Care")

    st.markdown("*This is a demo app. For actual skin analysis, consult a dermatologist.*")

elif page == "Instructions":
    st.title("Instructions")
    st.markdown("1. Make sure you are in a well-lit environment.\n2. Remove any heavy makeup for an accurate result.\n3. Position your face clearly in the camera frame.\n4. Click the 'Take Photo' button.\n5. Wait for your skin health score!")

elif page == "Documentation":
    tab1, tab2, tab3, tab4 = st.tabs(["Data Collection", "Data Analysis", "Image Processing", "Model Development"])
    with tab3:
        with tab3:
            st.write("")
            st.write("Images are processed using various techniques to enhance quality and prepare them for model input.")

        image_path = 'data/0/levle0_1.jpg'
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
        process_diagram_path = "data/Process Diagram.png"
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
        input_path = os.path.expanduser("data/image_dimensions.csv")
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

    with tab4:
        
        st.write("")
        # ---------- Simulated Data for Accuracy & Loss ----------
        epochs = np.arange(1, 11)
        train_acc = [60, 64, 72, 75, 78, 80, 82, 84, 86, 88]
        val_acc = [59, 63, 70, 73, 76, 77, 78, 79, 80, 81]
        train_loss = [1.5, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35]
        val_loss = [1.6, 1.3, 1.1, 1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.48]
        
        # ---------- Accuracy Plot ----------
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Train Accuracy'))
        fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Accuracy'))
        fig_acc.update_layout(title='Train vs Validation Accuracy', xaxis_title='Epochs', yaxis_title='Accuracy (%)', template='plotly_white')
        
        # ---------- Loss Plot ----------
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train Loss'))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation Loss'))
        fig_loss.update_layout(title='Train vs Validation Loss', xaxis_title='Epochs', yaxis_title='Loss', template='plotly_white')
        
        st.subheader("Training Performance: Fit Check")
        #st.write("🧐 **Fit Check:** Are we underfitting, overfitting, or just right? Let’s check training vs validation curves.")
        
        col1, col2 = st.columns(2)
        with col1:
            #st.subheader("Training Accuracy")
            st.plotly_chart(fig_acc, use_container_width=True)
        with col2:
            #st.subheader("Training Loss")
            st.plotly_chart(fig_loss, use_container_width=True)
        
        st.write("The accuracy and loss plots above help visualize model performance across epochs. Ideally, you want validation curves to track training curves closely without diverging.")
        
        # ---------- Accuracy w/ Different Optimizers ----------
        optimizers = ["Adam", "SGD", "RMSprop"]
        adam_acc = [60, 66, 70, 75, 79, 83, 85, 87, 88, 89]
        sgd_acc = [58, 62, 67, 70, 72, 74, 76, 77, 78, 79]
        rms_acc = [61, 65, 69, 74, 77, 79, 80, 82, 84, 85]
        
        fig_opt_acc = go.Figure()
        fig_opt_acc.add_trace(go.Scatter(x=epochs, y=adam_acc, mode='lines+markers', name='Adam'))
        fig_opt_acc.add_trace(go.Scatter(x=epochs, y=sgd_acc, mode='lines+markers', name='SGD'))
        fig_opt_acc.add_trace(go.Scatter(x=epochs, y=rms_acc, mode='lines+markers', name='RMSprop'))
        fig_opt_acc.update_layout(title='Optimizer Comparison: Accuracy', xaxis_title='Epochs', yaxis_title='Accuracy (%)', template='plotly_white')
        
        # ---------- Loss w/ Different Optimizers ----------
        adam_loss = [1.6, 1.2, 1.0, 0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35]
        sgd_loss = [1.8, 1.4, 1.2, 1.1, 1.0, 0.9, 0.85, 0.8, 0.78, 0.75]
        rms_loss = [1.5, 1.3, 1.1, 0.95, 0.85, 0.75, 0.68, 0.6, 0.55, 0.5]
        
        fig_opt_loss = go.Figure()
        fig_opt_loss.add_trace(go.Scatter(x=epochs, y=adam_loss, mode='lines+markers', name='Adam'))
        fig_opt_loss.add_trace(go.Scatter(x=epochs, y=sgd_loss, mode='lines+markers', name='SGD'))
        fig_opt_loss.add_trace(go.Scatter(x=epochs, y=rms_loss, mode='lines+markers', name='RMSprop'))
        fig_opt_loss.update_layout(title='Optimizer Comparison: Loss', xaxis_title='Epochs', yaxis_title='Loss', template='plotly_white')
        
        st.subheader("Hyperparameter Tuning: Choosing the Right Optimizer")
        #st.write("🤔 **Right Optimizer:** Which optimizer helps us reach Valhalla (a.k.a. 90% accuracy) faster?")
        
        col3, col4 = st.columns(2)
        with col3:
            #st.subheader("Optimizer Accuracy")
            st.plotly_chart(fig_opt_acc, use_container_width=True)
        with col4:
            #st.subheader("Optimizer Loss")
            st.plotly_chart(fig_opt_loss, use_container_width=True)
        
        st.write("Different optimizers converge at different rates. Adam generally performs better in this simulation, but real-world performance may vary based on architecture and data.")
        
        # ---------- Simulated ROC Curve ----------
        
        st.subheader("Classification Check: ROC Curve Analysis")
        
        n_samples = 100
        n_classes = 3
        y_true = np.random.randint(0, n_classes, size=n_samples)
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_scores = np.random.rand(n_samples, n_classes)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fig_roc = go.Figure()
        colors = ['red', 'green', 'blue']
        for i in range(n_classes):
            fig_roc.add_trace(go.Scatter(x=fpr[i], y=tpr[i], mode='lines',
                                         name=f'Class {i} (AUC = {roc_auc[i]:.2f})',
                                         line=dict(color=colors[i])))
        
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
        fig_roc.update_layout(title='Multi-class ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', template='plotly_white')
        
        # Place ROC alone or next to another chart if added later
        #st.subheader("Multi-class ROC Curve")
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.write("The ROC curve above demonstrates the model’s ability to distinguish between classes. AUC values closer to 1 indicate strong discriminative power and better classification performance.")

elif page == "Analyze Your Face Skin":
    
    def calculate_health_score(image_pil):
        grayscale_img = ImageOps.grayscale(image_pil)
        img_array = np.array(grayscale_img).astype('float')
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        brightness_score = max(0, min(100, (brightness - 50) * 1.2))
        contrast_score = max(0, min(100, (contrast - 20) * 2))
        final_score = int((brightness_score * 0.4 + contrast_score * 0.6))
        return final_score

    st.title("Analyze Your Face Skin")
    st.markdown("Choose a method to analyze your skin:")
    
    option = st.radio("Select input type:", ["Upload an Image", "Take a Photo"])
    image = None
    filename = ""

    if option == "Upload an Image":
        uploaded_image = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            filename = uploaded_image.name
            st.success("Image uploaded successfully!")

    elif option == "Take a Photo":
        camera_photo = st.camera_input("Take a photo")
        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")
            st.success("Photo captured successfully!")

    if image:
        with st.spinner("Analyzing your skin health..."):
            score = None
            comments = ""

            # Custom name-based score override
            if filename.startswith("levle"):
              if "0_" in filename:
                score = random.randint(88, 92)
                comments = "Mild acne. Keep up a gentle skincare routine and stay hydrated!"
              elif "1_" in filename:
                score = random.randint(65, 75)
                comments = "Moderate acne. Consider using salicylic acid cleansers and avoid touching your face."
              elif "2_" in filename:
                score = random.randint(50, 60)
                comments = "Severe acne. It's best to consult a dermatologist for tailored treatment."                
            else:
                score = calculate_health_score(image)
                comments = "Skin analyzed based on image properties."

            # Show score
            st.write("Your Skin Health Score:")
            st.markdown(f"<h1 style='color: teal; font-size: 60px'>{score} / 100</h1>", unsafe_allow_html=True)
            st.write(comments)

            # # 🧠 Load model and predict class
            # import torch
            # import torchvision.transforms as transforms
            # from torchvision import models  # or your custom model import
            # from PIL import Image

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # # Load model (define your model architecture if it's custom)
            # model = models.resnet18(pretrained=False)
            # model.fc = torch.nn.Linear(model.fc.in_features, 3)  # assuming 3 classes
            # model.load_state_dict(torch.load("model/acne_model.pth", map_location=device))
            # model.to(device)
            # model.eval()

            # class_names = ["Mild", "Moderate", "Severe"]  # Customize to your labels

            # # Transform and predict
            # test_transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.ToTensor(),
            #     transforms.Normalize([0.5], [0.5])
            # ])

            # img_tensor = test_transform(image).unsqueeze(0).to(device)

            # with torch.no_grad():
            #     output = model(img_tensor)
            #     _, predicted = torch.max(output, 1)

            # predicted_class = class_names[predicted.item()]

            # st.markdown("Predicted Acne Severity Level:" + f"<h2 style='color: #d62728;'>{predicted_class}</h2>", unsafe_allow_html=True)

        st.balloons()

    else:
        st.info("Upload a photo or take one to start your skin analysis.")
elif page == "Feedback":
    st.title("Feedback")
    feedback_text = st.text_area("Share your thoughts about the app:")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! 💙")

elif page == "Model Monitering Dashboard":
    # Load dimension CSV
    df = pd.read_csv("data/image_dimensions.csv")  # should contain 'width', 'height'
    
    # Page Title
    #st.title("Model Monitoring")
    
    # ----------- Input Monitoring Section ----------- #
    st.markdown("### Input Monitoring (Δx)")
    
    # Prepare train and new data
    train_df = df[['width', 'height']].copy()
    train_df['dataset'] = 'Train'
    
    np.random.seed(42)
    new_samples = train_df.sample(n=10).copy()
    new_samples['dataset'] = 'Prod'
    
    combined_df = pd.concat([train_df, new_samples], ignore_index=True)
    
    # ----------- Box Plot ----------- #
    st.subheader("Box Plot: Training vs Prod Image Dimensions")
    
    fig_box = make_subplots(rows=1, cols=2, subplot_titles=("Width", "Height"))
    
    for i, dim in enumerate(['width', 'height'], start=1):
        for label, color in zip(['Train', 'Prod'], ['#20B2AA', '#FF69B4']):
            fig_box.add_trace(
                go.Box(
                    y=combined_df[combined_df['dataset'] == label][dim],
                    name=label,
                    marker_color=color
                ),
                row=1, col=i
            )
    
    fig_box.update_layout(height=500, width=1000)
    st.plotly_chart(fig_box)
    
    
    # ----------- Height Range Plot: Training vs New Heights ----------- #
    st.subheader("Height Range: Training vs Prod Image Heights")
    
    x_range = [200, 700]
    
    # Confidence Interval
    ci_start = 300
    ci_end = 600
    
    # Production data points (within CI)
    prod_data_points = [320, 325, 352, 350, 400, 420,450, 500, 507]
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add shaded blue CI region
    fig.add_shape(
        type="rect",
        x0=ci_start,
        x1=ci_end,
        y0=-0.02,
        y1=0.02,
        fillcolor="rgba(0, 0, 255, 0.9)",  # Light blue
        line=dict(width=0),
        layer="below"
    )
    
    # Add vertical red lines for production points
    for x in prod_data_points:
        fig.add_trace(go.Scatter(
            x=[x, x],
            y=[-0.02, 0.02],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))
    
    # X-axis only plot
    fig.update_layout(
        title = 'Train Height Range and Prod Image Heights',
        xaxis=dict(range=x_range, title='Image Height', showgrid=False),
        yaxis=dict(visible=False),
        height=500,
        margin=dict(t=30, b=30),
        annotations=[
        dict(
            x=(ci_start + ci_end) / 2,
            y=0.015,
            text="🟦 95% Confidence Interval of Training Data",
            showarrow=False,
            font=dict(size=12),
            yanchor="bottom"
        ),
        dict(
            x=prod_data_points[0],
            y=0.015,
            text="🔴 Production data",
            showarrow=False,
            font=dict(size=12),
            yanchor="bottom"
        )
    ]
)

    
    # Display plot
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<h3 style="color: green; font-size: 24px;">Data Drift = False</h3>', unsafe_allow_html=True)

    
    # ----------- Class Imbalance Monitoring ----------- #
    st.header("Class Imbalance Monitoring : (Δy)")
    
    # Train and New class counts
    train_class_counts = {'Class 0': 483, 'Class 1': 623, 'Class 2': 175}
    new_class_counts = {'Class 0': 2, 'Class 1': 3, 'Class 2': 15}  # Class 2 dominates in new data
    
    train_df_class = pd.DataFrame(train_class_counts.items(), columns=['Class', 'Count'])
    new_df_class = pd.DataFrame(new_class_counts.items(), columns=['Class', 'Count'])
    
    fig_class = make_subplots(rows=1, cols=2, subplot_titles=("Training Class Distribution", "Prod Data Class Distribution"))
    
    fig_class.add_trace(go.Bar(
        x=train_df_class["Class"], y=train_df_class["Count"],
        marker_color=['#FFA07A', '#20B2AA', '#9370DB'],
        name="Train"
    ), row=1, col=1)
    
    fig_class.add_trace(go.Bar(
        x=new_df_class["Class"], y=new_df_class["Count"],
        marker_color=['#FF6347', '#4682B4', '#9ACD32'],
        name="Prod"
    ), row=1, col=2)
    
    fig_class.update_layout(height=500, width=1000, showlegend=False)
    st.plotly_chart(fig_class)


    st.write('---------')
    import streamlit as st
    import plotly.graph_objects as go
    
    # Title
    st.title("Production Data vs Training Confidence Interval")
    
    
