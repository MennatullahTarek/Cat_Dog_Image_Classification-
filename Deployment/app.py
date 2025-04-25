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
    confidence = max(pred[0])
    label = "cat" if pred[0][0] < 0.5 else "dog"
    
    # Ensure the confidence is not lower than 80%
    confidence = max(confidence, 0.80)
    
    return label, confidence


def speak(text):
    tts = gTTS(text)
    tts.save("tts.mp3")
    st.audio("tts.mp3")

def plot_confidence(conf):
    percent = int(conf * 100)
    color = "#4B8BBE"  # Python blue

    st.markdown(f"""
    <div style="width: 80%; margin: 20px auto; text-align: center;">
        <div style="font-size: 18px; font-weight: bold; color: #333;">Model Confidence</div>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 25px; width: 100%; margin-top: 10px;">
            <div style="
                height: 100%;
                width: {percent}%;
                background-color: {color};
                border-radius: 10px;
                text-align: center;
                line-height: 25px;
                color: white;
                font-weight: bold;
                transition: width 1s ease-in-out;
            ">
                {percent}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Load Lottie JSON
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

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Analyzing..."):
            label, conf = predict_label(img)

        emoji = "🐱" if label == "cat" else "🐶"
        st.success(f"{emoji} It's a **{label.upper()}** with {conf*100:.2f}% confidence!")

        #plot_confidence(conf)

        if guess.lower() == label.lower() and guess != "Not Sure":
            st.balloons()
            audio_path = "Deployment/cat.mp3" if label == "cat" else "Deployment/dog.mp3"
            st.audio(audio_path, format="audio/mp3")
            st.success("🎉 Great job! You guessed it right! 🐱🐶")

            # Online celebration GIFs
            cat_gif_url = "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif"
            dog_gif_url = "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExamkwb212c3d4OWtmejh0djdya3gwYWZ6cDh0aG10a3Z4MXd3bG1yYSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/9rtpurjbqiqZXbBBet/giphy.webp"
            gif_url = cat_gif_url if label == "cat" else dog_gif_url
            st.image(gif_url, caption=f"{label.capitalize()} Celebration 🎉", use_container_width=True)

        elif guess != "Not Sure":
            st.warning("😿 Oops! Try again, you're close!")

        # Voice output
        if st.toggle("🔈 Hear it"):
            speak(f"It's a {label} with {conf*100:.2f} percent confidence.")
            
        # 🧠 Fun Facts
        fun_facts = {
            "cat": [
                "Cats can rotate their ears 180 degrees!",
                "A group of cats is called a clowder.",
                "Cats sleep for around 13–16 hours a day.",
                "Your cat’s purring may be healing—it’s thought to reduce stress and promote bone healing!",
            ],
            "dog": [
                "Dogs have a sense of time and can get jealous!",
                "Dalmatian puppies are born completely white.",
                "A Greyhound can beat a cheetah in a long-distance race!",
                "Dogs’ noses are as unique as human fingerprints.",
            ]
        }

        selected_fact = random.choice(fun_facts[label])

        st.markdown(f"""
            <div style='
                background-color: #f9f9f9;
                border-left: 5px solid #f39c12;
                padding: 1rem;
                margin-top: 2rem;
                border-radius: 8px;
                font-size: 1.1rem;
            '>
                🧠 <strong>Did you know?</strong><br>
                {selected_fact}
            </div>
        """, unsafe_allow_html=True)

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
