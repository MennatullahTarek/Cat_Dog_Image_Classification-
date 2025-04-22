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

# Load the model and cache it for faster loading
@st.cache_resource(show_spinner=True)
def load_model_from_file():
    try:
        model = load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

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

# Custom CSS for styling the UI
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

# Layout of the app
st.markdown('<div class="main">', unsafe_allow_html=True)

st.title("🐾 Cat or Dog Classifier")
st.markdown("Upload a picture, and let's find out if it's a **meow** or a **woof**! 🐶🐱")

# User guess before uploading
guess = st.radio("🤔 What do YOU think it is?", ["Not Sure", "Cat", "Dog"])

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Function to preprocess the image
def preprocess_image(image):
    img_resized = image.resize((180, 180))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to get model confidence
def get_confidence(prediction):
    confidence = max(prediction[0])
    return confidence

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your uploaded image 👆", use_container_width=True)

    # Show progress bar while analyzing the image
    with st.spinner("Analyzing image..."):
        for i in range(0, 101, 10):
            st.progress(i)
            time.sleep(0.05)

        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        confidence = get_confidence(prediction)
        
        label = "cat" if prediction[0][0] < 0.5 else "dog"
        emoji = "🐱" if label == "cat" else "🐶"
        
        # Show the result in a styled box
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"{emoji} It's a **{label.upper()}** with {confidence*100:.2f}% confidence!")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show fun pet GIF based on the result
        gif_url = "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif" if label == "cat" else "https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif"
        st.image(gif_url, caption="Here's a happy pet for you!", use_column_width=True)
        
        # Guess feedback with animations and fun elements
        if guess.lower() == label:
            st.balloons()
            st.success("🎉 You guessed it right!")
            st.snow()
        elif guess != "Not Sure":
            st.warning(f"Oops! It was a **{label}**.")

        # Compliment the user based on their guess
        st.info(random.choice(compliments[label]))
        
        # Display a fun animal fact
        st.write(f"💡 **Did you know?** {random.choice(animal_facts[label])}")

        # Sound effect based on the animal's label
        sound_path = f"Deployment/{label}.mp3"
        if os.path.exists(sound_path):
            audio_file = open(sound_path, "rb").read()
            if "sound_played" not in st.session_state:
                st.session_state.sound_played = True
                st.audio(audio_file, format="audio/mp3", start_time=0)
            if st.checkbox("🔊 Replay sound"):
                st.audio(audio_file, format="audio/mp3", start_time=0)

# Footer section
st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="footer">🐾 Made with ❤️ by MennatullahTarek </div>', unsafe_allow_html=True)

