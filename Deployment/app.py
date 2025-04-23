import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import random
import time
from gtts import gTTS
import requests

# -------------- Page Config --------------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

# -------------- Custom Styles --------------
st.markdown("""
    <style>
    .main {background-color: #fefeff;}
    .block-container {padding-top: 2rem;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -------------- Download Model --------------
def download_model():
    file_id = "18RlTZvweyDneAUAVyMsgumENyBb5KHa-"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.ok:
        with open("model.h5", "wb") as f:
            f.write(response.content)
        st.success("Model downloaded!")
    else:
        st.error("Failed to download model.")

@st.cache_resource(show_spinner=True)
def load_model_from_file():
    try:
        return load_model("model.h5")
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

if not os.path.exists("model.h5"):
    st.info("Downloading model...")
    download_model()

model = load_model_from_file()
if model is None:
    st.stop()

# -------------- Static Data --------------
facts = {
    "cat": ["Cats sleep 70% of their lives!", "A group of kittens is a kindle."],
    "dog": ["Dogs' noses are wet to help scent!", "Dogs have 1,700 taste buds!"]
}
compliments = {
    "cat": ["You're as curious as a cat!", "Purrfect guess!"],
    "dog": ["You're pawsome!", "You're loyal like a good doggo!"]
}
score = {"Correct": 0, "Incorrect": 0}

# -------------- UI --------------
st.title("🐾 Cat or Dog Classifier")
st.markdown("Upload an image, and we'll tell you if it's a **Cat** or a **Dog**!")

guess = st.radio("Your Guess:", ["Not Sure", "Cat", "Dog"])
image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# -------------- Functions --------------
def preprocess_image(image):
    image = image.resize((180, 180))
    return np.expand_dims(np.array(image) / 255.0, axis=0)

def get_prediction(img_array):
    pred = model.predict(img_array)
    label = "cat" if pred[0][0] < 0.5 else "dog"
    confidence = float(pred[0][0] if label == "dog" else 1 - pred[0][0])
    return label, confidence

def play_sound(label):
    path = f"Deployment/{label}.mp3"
    if os.path.exists(path):
        with open(path, "rb") as audio:
            st.audio(audio.read(), format="audio/mp3")

def speak(text):
    tts = gTTS(text)
    tts.save("temp.mp3")
    with open("temp.mp3", "rb") as audio:
        st.audio(audio.read(), format="audio/mp3")

# -------------- Main Logic --------------
if image_file:
    img = Image.open(image_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        st.progress(25)
        time.sleep(0.2)
        img_array = preprocess_image(img)
        st.progress(50)
        label, confidence = get_prediction(img_array)
        st.progress(100)

    emoji = "🐱" if label == "cat" else "🐶"
    st.success(f"{emoji} It's a **{label.upper()}** with {confidence * 100:.2f}% confidence!")

    st.slider("Confidence", min_value=0.0, max_value=1.0, value=confidence, step=0.01, disabled=True)

    # Evaluate Guess
    if guess.lower() == label:
        score["Correct"] += 1
        st.balloons()
        st.success("Correct guess!")
    elif guess != "Not Sure":
        score["Incorrect"] += 1
        st.warning(f"Wrong! It was a **{label}**.")

    st.info(random.choice(compliments[label]))
    st.info(f"💡 Fun Fact: {random.choice(facts[label])}")

    if st.button(f"🔊 Hear {label.capitalize()} Sound"):
        play_sound(label)

    speak(f"It's a {label} with {confidence*100:.2f} percent confidence.")

# -------------- Scoreboard --------------
st.markdown("---")
st.subheader("📊 Scoreboard")
st.write(score)

st.caption("Made with ❤️ by MennatullahTarek")
