import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import os
import random
import time

# Function to download the model from Google Drive
def download_model():
    file_url = "https://drive.google.com/uc?export=download&id=18RlTZvweyDneAUAVyMsgumENyBb5KHa-"
    response = requests.get(file_url)
    if response.status_code == 200:
        with open("model.h5", "wb") as file:
            file.write(response.content)
        st.success("Model downloaded successfully!")
    else:
        st.error("Failed to download model.")

# Load the model
@st.cache_resource
def load_model_from_file():
    return load_model("model.h5")

# Download the model if not already present
if not os.path.exists("model.h5"):
    st.info("Downloading model from Google Drive...")
    download_model()

# Load model
model = load_model_from_file()

# Fun facts and compliments
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

compliments = {
    "cat": ["You're as curious as a cat!", "Purrfect guess!"],
    "dog": ["You're pawsome!", "You're loyal like a good doggo!"]
}

# Custom CSS
st.markdown("""
    <style>
        .main {
            background: linear-gradient(to right, #f0f4f8, #c3cfe2);
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        }
        .result-box {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
            animation: fadeIn 1.5s ease-in-out;
        }
        .upload-box {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
        }
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .css-1rs6os.edgvbvh3 {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("🐾 Cat or Dog Classifier")
st.markdown("Upload a picture, and let's find out if it's a **meow** or a **woof**! 🐶🐱")

# User guess before uploading
guess = st.radio("🤔 What do YOU think it is?", ["Not Sure", "Cat", "Dog"])

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your uploaded image 👆", use_container_width=True)

    # Show progress bar
    with st.spinner("Analyzing image..."):
        for i in range(0, 101, 10):
            st.progress(i)
            time.sleep(0.05)

        # Preprocess image
        img_resized = img.resize((180, 180))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = "cat" if prediction[0][0] < 0.5 else "dog"
        emoji = "🐱" if label == "cat" else "🐶"

        # Show result
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"{emoji} It's a **{label.upper()}**!")
        st.markdown("</div>", unsafe_allow_html=True)

        # Show gif
        gif_url = "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif" if label == "cat" else "https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif"
        st.image(gif_url, caption="Here's a happy pet for you!", use_column_width=True)

        # Guess feedback
        if guess.lower() == label:
            st.balloons()
            st.success("🎉 You guessed it right!")
        elif guess != "Not Sure":
            st.warning(f"Oops! It was a **{label}**.")

        # Compliment
        st.info(random.choice(compliments[label]))

        # Fun fact
        st.write(f"💡 **Did you know?** {random.choice(animal_facts[label])}")

        # Sound
        sound_path = f"Deployment/{label}.mp3"
        if os.path.exists(sound_path):
            audio_file = open(sound_path, "rb").read()

            # Auto-play on first prediction
            if "sound_played" not in st.session_state:
                st.session_state.sound_played = True
                st.audio(audio_file, format="audio/mp3", start_time=0)

            # Option to replay
            if st.checkbox("🔊 Replay sound"):
                st.audio(audio_file, format="audio/mp3", start_time=0)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="footer">🐾 Made with ❤️ by MennatullahTarek </div>', unsafe_allow_html=True)
