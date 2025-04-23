import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
import os
import random
import time
from gtts import gTTS

# Function to download the model from Google Drive
def download_model():
    file_url = "https://drive.google.com/uc?export=download&id=18RlTZvweyDneAUAVyMsgumENyBb5KHa-"
    response = requests.get(file_url)
    if response.status_code == 200:
        with open("model.h5", "wb") as file:
            file.write(response.content)
        if os.path.getsize("model.h5") == 0:
            os.remove("model.h5")
            st.error("Downloaded model file is empty.")
        else:
            st.success("Model downloaded successfully!")
    else:
        st.error("Failed to download model. Please try again later.")

# Load the model and cache it
@st.cache_resource(show_spinner=True)
def load_model_from_file():
    try:
        model = load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

if not os.path.exists("model.h5"):
    st.info("Downloading model from Google Drive...")
    download_model()

model = load_model_from_file()

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

leaderboard = {
    "Correct Guesses": 0,
    "Wrong Guesses": 0
}

# Cartoon theme CSS
st.markdown("""
    <style>
        body {
            background: url('https://i.pinimg.com/originals/93/b9/3d/93b93de8cbef3c9ef988a75e14b6e65c.jpg') no-repeat center center fixed;
            background-size: cover;
            font-family: 'Arial', sans-serif;
        }
        .main {
            background: rgba(255, 255, 255, 0.85);
            padding: 50px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            max-width: 900px;
            margin: auto;
            border: 3px solid #F1F1F1;
        }
        .result-box {
            background-color: #fff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            animation: fadeIn 1.2s ease-in-out;
        }
        .upload-box {
            background-color: #F9F9F9;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            font-size: 14px;
            color: #444;
        }
        .paw {
            font-size: 50px;
            color: #FF6F61;
            animation: paws 0.5s ease-in-out infinite;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(50px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes paws {
            0% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("🐾 Cat or Dog Classifier 🐾")
st.markdown("Upload an image, and let's determine if it's a **meow** or a **woof**! 🐕🐈")

guess = st.radio("🤔 What do YOU think it is?", ["Not Sure", "Cat", "Dog"])

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")


def preprocess_image(image):
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_confidence(prediction):
    return max(prediction[0])

def play_animal_sound(label):
    sound_file = f"{label}.mp3"
    with open(sound_file, "rb") as audio:
        st.audio(audio.read(), format='audio/mp3')

def speak_with_gtts(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio:
        st.audio(audio.read(), format='audio/mp3')

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your uploaded image 👀", use_container_width=True)

    with st.spinner("Analyzing image..."):
        for i in range(0, 101, 10):
            st.progress(i)
            time.sleep(0.05)

        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        confidence = get_confidence(prediction)

        label = "cat" if prediction[0][0] < 0.5 else "dog"
        emoji = "🐱" if label == "cat" else "🐶"

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"{emoji} It's a **{label.upper()}** with {confidence*100:.2f}% confidence!")
        st.slider("Confidence Level", min_value=0.0, max_value=1.0, value=float(confidence), step=0.01, disabled=True)
        st.markdown("</div>", unsafe_allow_html=True)

        gif_url = "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif" if label == "cat" else "https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif"
        st.image(gif_url, caption="Here's a happy pet for you!", use_container_width=True)

        if guess.lower() == label:
            leaderboard["Correct Guesses"] += 1
            st.markdown('<div class="paw">🐾🐾🐾</div>', unsafe_allow_html=True)
            st.success("🎉 You guessed it right!")
            st.snow()
        elif guess != "Not Sure":
            leaderboard["Wrong Guesses"] += 1
            st.warning(f"Oops! It was a **{label}**.")

        st.info(random.choice(compliments[label]))
        st.write(f"💡 **Did you know?** {random.choice(animal_facts[label])}")

        # Play animal sound or text-to-speech based on button clicks
        if st.button("Play Animal Sound"):
            play_animal_sound(label)

        if st.button("Play Text-to-Speech"):
            speak_with_gtts(f"It's a {label} with {confidence * 100:.2f} percent confidence!")

st.markdown("## Leaderboard")
st.table(leaderboard)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<footer>🐾 Made with ❤️ by MennatullahTarek </footer>', unsafe_allow_html=True)
