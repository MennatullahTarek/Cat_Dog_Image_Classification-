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

# ========== Function to download the model from Google Drive ==========
def download_model():
    file_id = "18RlTZvweyDneAUAVyMsgumENyBb5KHa-"
    file_url = f"https://drive.google.com/uc?export=download&id={file_id}"
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

# ========== Load the model and cache it ==========
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
if model is None:
    st.stop()

# ========== Static Data ==========
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

# ========== UI and CSS ==========
st.markdown("""
    <style>
        body {
            background-color: #fef6ff;
        }
        .main {
            background: #fffaf0;
            padding: 40px;
            border-radius: 25px;
            text-align: center;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            max-width: 850px;
            margin: auto;
            border: 4px dashed #f08;
            font-family: 'Comic Sans MS', cursive, sans-serif;
        }
        footer {
            margin-top: 40px;
            font-size: 16px;
            color: #999;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("\U0001F43E Cat or Dog Classifier")
st.markdown("Upload an image, and let's determine if it's a **meow** or a **woof**! \U0001F436\U0001F431")

guess = st.radio("\U0001F914 What do YOU think it is?", ["Not Sure", "Cat", "Dog"])

uploaded_file = st.file_uploader("\U0001F4E4 Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ========== Helper Functions ==========
def preprocess_image(image):
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_confidence(prediction):
    return max(prediction[0])

def play_animal_sound(label):
    sound_path = f"Deployment/{label}.mp3"
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as audio_file:
            st.audio(audio_file.read(), format="audio/mp3", start_time=0)
    else:
        st.error(f"Sound file for {label} not found.")

def speak(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format='audio/mp3')

# ========== Main Logic ==========
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your uploaded image \U0001F446", use_container_width=True)

    with st.spinner("Analyzing image..."):
        for i in range(0, 101, 10):
            st.progress(i)
            time.sleep(0.05)

        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        confidence = get_confidence(prediction)

        label = "cat" if prediction[0][0] < 0.5 else "dog"
        emoji = "\U0001F431" if label == "cat" else "\U0001F436"

        st.success(f"{emoji} It's a **{label.upper()}** with {confidence*100:.2f}% confidence!")
        st.slider("Confidence Level", min_value=0.0, max_value=1.0, value=float(confidence), step=0.01)

        if guess.lower() == label:
            leaderboard["Correct Guesses"] += 1
            st.success("\U0001F389 You guessed it right!")
            st.balloons()
        elif guess != "Not Sure":
            leaderboard["Wrong Guesses"] += 1
            st.warning(f"Oops! It was a **{label}**.")

        st.info(random.choice(compliments[label]))
        st.write(f"\U0001F4A1 **Did you know?** {random.choice(animal_facts[label])}")

        if st.button(f"\U0001F50A Play {label.capitalize()} Sound"):
            play_animal_sound(label)

        speak(f"It's a {label} with {confidence * 100:.2f} percent confidence!")

# ========== Leaderboard ==========
st.markdown("## Leaderboard")
st.table(leaderboard)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<footer>\U0001F43E Made with ❤️ by MennatullahTarek </footer>', unsafe_allow_html=True)
