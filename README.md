# Cat vs Dog Classifier

Welcome to the **Cat vs Dog Classifier** project! This project demonstrates a machine learning model built to classify images of cats and dogs with high accuracy. It uses a Convolutional Neural Network (CNN) to predict whether an uploaded image is a cat or a dog.

## Project Overview

This project is built using TensorFlow/Keras and Streamlit for the frontend. The model was trained on the popular Kaggle Cat vs Dog dataset and uses data preprocessing, augmentation, and transfer learning techniques to optimize performance.

### Key Features
- **Image Classification**: Classifies uploaded images as either "cat" or "dog."
- **Confidence Score**: Provides a confidence score for each prediction.
- **Voice Output**: Hear the model's prediction via a text-to-speech feature.
- **Fun Facts**: After each prediction, users get an interesting fact about cats or dogs.

## Demo

You can test the model live at [Cat vs Dog Classifier Demo](https://fcsqzqgdpcghkcdyzrstns.streamlit.app/).

## Technologies Used

- **TensorFlow**: For building and training the deep learning model.
- **Streamlit**: To create an interactive and user-friendly web interface.
- **gTTS (Google Text-to-Speech)**: For generating voice output.
- **Plotly**: For displaying the model’s confidence using an engaging progress bar.
- **Lottie**: For integrating animations into the app.


## How It Works
- **Preprocessing**: The images are resized to 180x180 pixels and normalized before being passed to the model.
- **Model**: The model uses a Convolutional Neural Network (CNN) to classify images as either "cat" or "dog."
- **Prediction**: The model outputs a probability score, which is used to determine the label (cat or dog).
- **Confidence**: A progress bar shows the model's confidence in the prediction.
- **Fun Facts**: After the prediction, the app displays a random fun fact about either cats or dogs.

## Contributing
> If you'd like to contribute to the project, feel free to fork the repository, make improvements, and submit a pull request.

