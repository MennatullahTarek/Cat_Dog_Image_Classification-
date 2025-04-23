import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import random
from gtts import gTTS
import time
import os

# Download model if not present
def download_model():
    if not os.path.exists("model.h5"):
        st.info("Downloading model...")
        file_url = "https://drive.google.com/uc?export=download&id=18RlTZvweyDneAUAVyMsgumENyBb5KHa-"
        response = requests.get(file_url)
        with open("model.h5", "wb") as file:
            file.write(response.content)

# Load the pre-trained model
@st.cache_resource
def load_model_from_file():
    try:
        return load_model("model.h5")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Animal facts and compliments
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

# Main app layout
st.title("🐾 Cat or Dog Classifier")
st.markdown("Upload an image, and let's see if it's a **meow** or a **woof**!")

# User's guess
guess = st.radio("What do you think it is?", ["Not Sure", "Cat", "Dog"])

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Preprocess the image
def preprocess_image(image):
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

# Get prediction confidence
def get_confidence(prediction):
    return max(prediction[0])

# Play sound based on prediction
def play_animal_sound(label):
    sound_path = f"Deployment/{label}.mp3"
    if os.path.exists(sound_path):
        audio_file = open(sound_path, "rb").read()
        st.audio(audio_file, format="audio/mp3")
    else:
        st.error(f"Sound file for {label} is missing.")

# Speak out the result
def speak(text):
    tts = gTTS(text)
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/mp3")

# Image classification logic
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your uploaded image")

    with st.spinner("Analyzing image..."):
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        confidence = get_confidence(prediction)

        label = "cat" if prediction[0][0] < 0.5 else "dog"
        emoji = "🐱" if label == "cat" else "🐶"

        st.success(f"{emoji} It's a **{label.upper()}** with {confidence*100:.2f}% confidence!")

        if guess.lower() == label:
            leaderboard["Correct Guesses"] += 1
            st.success("You guessed it right!")
        elif guess != "Not Sure":
            leaderboard["Wrong Guesses"] += 1
            st.warning(f"Oops! It was a **{label}**.")

        st.info(random.choice(compliments[label]))
        st.write(f"**Did you know?** {random.choice(animal_facts[label])}")

        if st.button(f"🔊 Play {label.capitalize()} Sound"):
            play_animal_sound(label)
        
        speak(f"It's a {label} with {confidence * 100:.2f}% confidence!")

# Show leaderboard
st.markdown("### Leaderboard")
st.table(leaderboard)

# Footer
st.markdown('<footer>Made with ❤️ by MennatullahTarek</footer>', unsafe_allow_html=True)
