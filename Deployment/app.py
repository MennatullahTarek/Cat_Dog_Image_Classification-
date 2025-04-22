import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import random

# Load model
@st.cache_resource
def load_model_from_file():
    return load_model("model.h5")

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

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #fef6f0;
            padding: 20px;
            border-radius: 12px;
        }
        .footer {
            text-align: center;
            color: #999;
            font-size: 14px;
            margin-top: 40px;
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

        # Result
        st.success(f"{emoji} It's a **{label.upper()}**!")

        # Play sound
        if label == "cat":
            with open("cat.mp3", "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")
        else:
            with open("dog.mp3", "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")

        # Fun fact
        st.markdown(f"💡 **Did you know?** {random.choice(animal_facts[label])}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="footer">🐾 Made with ❤️ by You</div>', unsafe_allow_html=True)
