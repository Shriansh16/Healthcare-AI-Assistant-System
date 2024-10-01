import os
import streamlit as st
from src.utils import decodeImage
from src.pipelines.prediction_pipeline import PredictionPipeline
from Dr_Maria_Chatbot import *

# Set up the environment variables  s

# Title for the app
st.title("Hi, I am Dr. Maria, Orthopedic Specialist")
st.subheader("Please upload the X-ray for analysis")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# If an image is uploaded
if uploaded_file is not None:
    # Save the image
    image_path = "inputImage.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    #st.image(image_path, caption="Uploaded Image", use_column_width=True)

    # Create the classifier and predict
    classifier = PredictionPipeline(image_path)

    # Button to make prediction
    if st.button("Predict"):
        result = classifier.predict()
        st.write("Prediction Result:", result)
        func(result)

