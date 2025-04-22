import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import requests
import random
from streamlit_lottie import st_lottie
import json

# Download Lottie animation from a URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Download the model if not already present
def download_model():
    file_url = "https://drive.google.com/uc?export=download&id=18RlTZvweyDneAUAVyMsgumENyBb5KHa-"
    response = requests.get(file_url)
    if response.status_code == 200:
        with open("model.h5", "wb") as file:
            file.write(response.content)
        st.success("Model downloaded successfully!")
    else:
        st.error("Failed to download model.")

@st.cache_resource
def load_model_from_file():
    return load_model("model.h5")

if not os.path.exists("model.h5"):
    st.info("Downloading model from Google Drive...")
    download_model()

model = load_model_from_file()

# Fun facts
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

# Lottie animation URLs
lottie_cat = load_lottie_url("https://lottie.host/df94a739-b7c1-4568-869b-21d2f309f91b/qNfSkFhYQD.json")
lottie_dog = load_lottie_url("https://lottie.host/ce1d648f-a867-4cb5-a84e-d60e7ed5f66d/AXlS1iFlaT.json")

# 🎨 Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Quicksand', sans-serif;
    background: linear-gradient(to bottom right, #e3f2fd, #fce4ec);
}

h1 {
    text-align: center;
    font-size: 3em;
    color: #4a148c;
    margin-bottom: 0;
}

.stButton>button {
    background-color: #ff6f61;
    color: white;
    border-radius: 12px;
    padding: 0.75em 2em;
    font-weight: bold;
    border: none;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #e64a19;
}

.upload-box {
    background-color: white;
    border-radius: 16px;
    padding: 25px;
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    margin-top: 30px;
}

.result-box {
    background-color: #fff3e0;
    border-left: 8px solid #ff9800;
    border-radius: 10px;
    padding: 20px;
    margin-top: 25px;
    animation: fadeIn 1.2s ease-in-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px);}
    to { opacity: 1; transform: translateY(0);}
}

.footer {
    text-align: center;
    font-size: 14px;
    margin-top: 50px;
    color: #616161;
}
</style>
""", unsafe_allow_html=True)

# 🐾 Title Section
st.markdown("## 🐾 Cat or Dog Classifier")
st.markdown("Upload a picture, and let's find out if it's a **meow** or a **woof**! 🐶🐱")

# 🧠 User Guess
st.markdown("### 🤔 What do YOU think it is?")
user_guess = st.radio("", ["Not Sure", "Cat", "Dog"])

# 📤 Upload File
uploaded_file = st.file_uploader("📎 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Your uploaded image 👆", use_container_width=True)

    with st.spinner("Analyzing image..."):
        # Preprocess
        img_resized = img.resize((180, 180))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = "cat" if prediction[0][0] < 0.5 else "dog"
        emoji = "🐱" if label == "cat" else "🐶"

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"{emoji} It's a **{label.upper()}**!")
        st.markdown(f"💡 **Did you know?** {random.choice(animal_facts[label])}")
        st.markdown("</div>", unsafe_allow_html=True)

        # 🎵 Play audio automatically (workaround)
        audio_path = f"Deployment/{label}.mp3"
        if os.path.exists(audio_path):
            st.audio(audio_path, format="audio/mp3", start_time=0)

        # 🕺 Animation
        if label == "cat" and lottie_cat:
            st_lottie(lottie_cat, speed=1, reverse=False, height=300)
        elif label == "dog" and lottie_dog:
            st_lottie(lottie_dog, speed=1, reverse=False, height=300)

# 🧁 Footer
st.markdown('<div class="footer">🐾 Made with ❤️ by Mennatullah Tarek</div>', unsafe_allow_html=True)
