import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
import random
import os

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

# Custom CSS for animations and styling
st.markdown("""
    <style>
        .main {
            background-color: #fef6f0;
            padding: 40px;
            border-radius: 12px;
            text-align: center;
            animation: fadeIn 1.5s ease-in-out;
        }
        .result-box {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
            animation: fadeIn 1.5s ease-in-out;
        }
        .upload-box {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            margin-top: 30px;
            animation: bounceIn 1s ease-in-out;
        }
        .play-btn {
            background-color: #ff6f61;
            color: white;
            padding: 12px 30px;
            border-radius: 5px;
            border: none;
            font-weight: bold;
            cursor: pointer;
            margin-top: 20px;
            animation: slideIn 1s ease-in-out;
        }
        .play-btn:hover {
            background-color: #ff4e3b;
        }

        /* Animations */
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

        @keyframes bounceIn {
            from {
                opacity: 0;
                transform: translateY(-100px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(100px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)

# UI Elements
st.title("🐾 Cat or Dog Classifier")
st.markdown("Upload a picture, and let's find out if it's a **meow** or a **woof**! 🐶🐱")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your uploaded image 👆", use_column_width=True)

    with st.spinner("Analyzing image..."):
        # Preprocess image for your model
        img_resized = img.resize((180, 180))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = "cat" if prediction[0][0] < 0.5 else "dog"
        emoji = "🐱" if label == "cat" else "🐶"

        # Display Result with Animation
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"{emoji} It's a **{label.upper()}**!")

        # Fun fact
        st.markdown(f"💡 **Did you know?** {random.choice(animal_facts[label])}")

        # Play sound automatically after result is shown
        sound_path = f"Deployment/{label}.mp3"
        if os.path.exists(sound_path):
            audio_file = open(sound_path, "rb").read()
            st.audio(audio_file, format="audio/mp3", start_time=0)

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">🐾 Made with ❤️ by You</div>', unsafe_allow_html=True)
