import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import time
from gtts import gTTS
import requests

# ----------------------------- Configuration -----------------------------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

# ----------------------------- Static Content -----------------------------
animal_facts = {
    "cat": [
        "Cats sleep for 70% of their lives!",
        "A group of kittens is called a kindle.",
        "Cats can jump up to six times their length!"
    ],
    "dog": [
        "Dogs' noses are wet to help absorb scent chemicals.",
        "Dogs have about 1,700 taste buds.",
        "A Greyhound could beat a Cheetah in a long-distance race!"
    ]
}

# ----------------------------- Download & Load Model -----------------------------
MODEL_PATH = "model.h5"
MODEL_FILE_ID = "18RlTZvweyDneAUAVyMsgumENyBb5KHa-"

@st.cache_resource(show_spinner=True)
def load_model_from_file():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
        r = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    return load_model(MODEL_PATH)

model = load_model_from_file()

# ----------------------------- Helper Functions -----------------------------
def preprocess_image(image):
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

def play_animal_sound(label):
    sound_file = f"Deployment/{label}.mp3"
    if os.path.exists(sound_file):
        with open(sound_file, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3")
    else:
        st.warning(f"Sound file for {label} not found.")

def speak_text(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/mp3')

# ----------------------------- UI -----------------------------
st.markdown("""
    <style>
    .main-title { font-size: 36px; font-weight: bold; color: #4a4a4a; text-align: center; }
    .sub-title { font-size: 18px; color: #6c6c6c; text-align: center; margin-bottom: 30px; }
    .footer { text-align: center; margin-top: 30px; font-size: 14px; color: #aaa; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🐾 Cat vs Dog Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload an image and let's see who's in it: a meow or a woof!</div>", unsafe_allow_html=True)

# ----------------------------- Upload and Predict -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your uploaded image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        time.sleep(0.5)
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)
        confidence = float(np.max(prediction))
        label = "dog" if prediction[0][0] > 0.5 else "cat"
        emoji = "🐶" if label == "dog" else "🐱"

    st.success(f"{emoji} It's a **{label.upper()}** with {confidence*100:.2f}% confidence!")
    st.progress(confidence)

    if st.button(f"Play {label.capitalize()} Sound 🔊"):
        play_animal_sound(label)

    st.info(f"**Fun Fact:** {np.random.choice(animal_facts[label])}")
    speak_text(f"It's a {label} with {confidence * 100:.2f} percent confidence")

# ----------------------------- How it works -----------------------------
st.markdown("---")
st.header("How It Works")
st.markdown("""
1. Upload a clear image of a **cat** or **dog**.
2. The model resizes and preprocesses the image.
3. It uses a neural network to predict whether it's a cat or dog.
4. You get the result along with a fun fact and sound!
""")

# Try to show how-it-works image if available
how_img_path = "Deployment/how-it-works.png"
if os.path.exists(how_img_path):
    st.image(how_img_path, caption="The Prediction Flow", use_container_width=True)
else:
    st.warning("How-it-works image not found.")

# ----------------------------- Footer -----------------------------
st.markdown("<div class='footer'>🐾 Made with love by Mennatullah Tarek</div>", unsafe_allow_html=True)
