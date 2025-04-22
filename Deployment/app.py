import gdown
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import tensorflow as tf

# Download model from Google Drive
def download_model():
    url = 'https://drive.google.com/uc?id=18RlTZvweyDneAUAVyMsgumENyBb5KHa-'
    output = 'model.h5'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
        st.write("Model downloaded successfully!")
    else:
        st.write("Model is already available!")

# Load model from file
def load_model_from_file():
    download_model()
    return load_model("model.h5")

# Function to make predictions
def predict_image(model, img):
    img_array = tf.convert_to_tensor(np.array(img))
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
    img_array = tf.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI Enhancement
st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")

# Title and introduction
st.title("🐱 Cat vs Dog Image Classification 🐶")
st.markdown("""
    This app classifies images of cats and dogs with the power of AI. 
    Simply upload an image and watch the magic happen! Enjoy the sounds as well!
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# Conditional rendering based on file upload
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Load the model
    model = load_model_from_file()

    # Prediction
    prediction = predict_image(model, img)

    # Display prediction with styling
    st.markdown("<h2 style='color: #4CAF50;'>Prediction Result:</h2>", unsafe_allow_html=True)
    if prediction[0] < 0.5:
        st.write("The image is a **Cat** 🐱!")
        st.audio("cat.mp3", format="audio/mp3", start_time=0)
    else:
        st.write("The image is a **Dog** 🐶!")
        st.audio("dog.mp3", format="audio/mp3", start_time=0)

    # Display prediction confidence
    st.markdown(f"""
        **Confidence:** {100 * (1 - prediction[0][0]):.2f}% **(Cat)** / **{100 * prediction[0][0]:.2f}%** **(Dog)**
    """)

    # Custom CSS for styling
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stImage img {
        border-radius: 15px;
        border: 2px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

    # Option to upload new image
    st.markdown("<br><h3 style='color: #2196F3;'>Want to classify another image?</h3>", unsafe_allow_html=True)
    st.button('Upload Another Image', on_click=None)

# Add a footer for additional engagement
st.markdown("""
    <br>
    <hr>
    <p style='color: #888;'>Built with ❤️ by MennatullahTarek. <a href='https://github.com/MennatullahTarek/Cat_Dog_Image_Classification-' target='_blank'>GitHub Repo</a></p>
""", unsafe_allow_html=True)
