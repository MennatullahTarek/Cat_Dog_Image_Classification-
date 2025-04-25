import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import requests
import random
from gtts import gTTS
import time
import plotly.express as px
from streamlit_lottie import st_lottie


# ========== Sidebar ========== #
st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")
st.sidebar.title("🐾 Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "How It Works"])

# ========== Styling ========== #
st.markdown("""
    <style>
        .main {
            max-width: 1000px;
            margin: auto;
            padding: 2rem;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 2.5rem;
            text-align: center;
            color: #4e4e4e;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #888;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ========== Download and Load Model ========== #
@st.cache_resource
def download_and_load_model():
    model_path = "model.h5"
    if not os.path.exists(model_path):
        file_id = "18RlTZvweyDneAUAVyMsgumENyBb5KHa-"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    return load_model(model_path)

model = download_and_load_model()

# ========== Helper Functions ========== #
def preprocess(img):
    image = img.resize((180, 180))
    array = np.expand_dims(np.array(image) / 255.0, axis=0)
    return array

def predict_label(image):
    pred = model.predict(preprocess(image))
    return ("cat" if pred[0][0] < 0.5 else "dog", max(pred[0]))

def speak(text):
    tts = gTTS(text)
    tts.save("tts.mp3")
    st.audio("tts.mp3")

def plot_confidence(conf):
    fig = px.bar(x=["Confidence"], y=[conf * 100], range_y=[0, 100], text=[f"{conf*100:.2f}%"])
    fig.update_traces(textposition="outside", marker_color="#fd7e14")
    fig.update_layout(title="Model Confidence", yaxis_title="%", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Function to load Lottie JSON from a URL
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ========== Home Page ========== #
if page == "Home":
    # ==== Styling ==== #
    st.markdown(
        """
        <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4B8BBE;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ==== Header ==== #
    st.markdown('<div class="title">🐶🐱 Cat vs Dog Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an image and let AI guess!</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ==== Upload + Guess ==== #
    uploaded = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
    guess = st.radio("🤔 What do **you** think it is?", ["Not Sure", "Cat", "Dog"])

    # ==== If Image Uploaded ==== #
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Analyzing..."):
            label, conf = predict_label(img)

        # Result Display
        emoji = "🐱" if label == "cat" else "🐶"
        st.success(f"{emoji} It's a **{label.upper()}** with {conf*100:.2f}% confidence!")

        # Confidence Plot
        plot_confidence(conf)
# ==== If Correct Guess ==== #
if guess.lower() == label.lower() and guess != "Not Sure":
    # Celebrate Correct Guess
    st.balloons()  # Add Balloons for Celebration 🎈
    
    # Play the correct animal sound based on the prediction
    audio_path = "Deployment/cat.mp3" if label == "cat" else "Deployment/dog.mp3"
    st.audio(audio_path, format="audio/mp3")
    
    st.success("🎉 Great job! You guessed it right! 🐱🐶")
    
    # Show GIF based on the prediction
    if label == "cat":
        st.image("Deployment/cat_celebration.gif", caption="Cat Celebration 🎉", use_column_width=True)
    else:
        st.image("Deployment/dog_celebration.gif", caption="Dog Celebration 🎉", use_column_width=True)

# ==== If Incorrect Guess ==== #
elif guess != "Not Sure":
    st.warning("😿 Oops! Try again, you're close!")

        # Cute Animal Sound 🐾
        audio_path = "Deployment/cat.mp3" if label == "cat" else "Deployment/dog.mp3"
        st.audio(audio_path, format="audio/mp3", start_time=0)

        # Optional Voice Output
        if st.toggle("🔈 Hear it"):
            speak(f"It's a {label} with {conf*100:.2f} percent confidence.")

# ========== About Page ========== #
elif page == "About":
    st.markdown("## About This App")
    st.write("This project uses a Convolutional Neural Network trained to distinguish between cat and dog images.")
    st.write("Developed with ❤️ by MennatullahTarek.")

# ========== How It Works Page ========== #
elif page == "How It Works":
    st.markdown("## How It Works")
    st.write("""
    - The image is resized to 180x180 pixels.
    - It is normalized and passed through a CNN model.
    - The model outputs a probability, and we map it to `cat` or `dog` using a 0.5 threshold.
    """)
    st.image("https://learnopencv.com/wp-content/uploads/2023/01/tensorflow-keras-cnn-vgg-architecture.png", caption="CNN Flow (Source: LearnOpenCV)")

st.markdown("</div>", unsafe_allow_html=True)
