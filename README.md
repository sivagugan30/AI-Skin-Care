# 💆‍♀️ AI-Skin-Care

AI-Skin-Care is a fun and interactive Streamlit web app that lets users **snap a photo using their webcam** and receive a **Skin Health Score** out of 100 based on simple image processing techniques.

🔗 **Live App:** [https://skincare-ai.streamlit.app/](https://skincare-ai.streamlit.app/)

---

## 🚀 Features

- 📸 Take a real-time photo using your webcam
- 🧠 Analyze brightness and contrast of your skin using image statistics
- 🎯 Get a Health Score out of 100
- 💡 Receive personalized skincare tips based on your score
- 🧼 Lightweight — no ML model or OpenCV required

---

## 🛠️ How It Works

1. User takes a photo using `st.camera_input()`
2. The image is processed using `Pillow` and `NumPy`
3. Brightness and contrast metrics are computed
4. A simple scoring algorithm generates a Health Score
5. Custom feedback is displayed based on the score

# Run the app
streamlit run ai_skin_care_app.py
