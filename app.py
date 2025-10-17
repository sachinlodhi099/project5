import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('tb_detection_model.keras')
    return model

model = load_model()

st.title("Tuberculosis Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to classify it as 'Normal' or 'Tuberculosis'.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-ray', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    img_array = np.array(image.resize((224, 224)))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Create a batch

    # Make a prediction
    prediction = model.predict(img_array)
    prediction_prob = prediction[0][0]

    # Display the result
    if prediction_prob > 0.82:
        st.error(f"Prediction: Tuberculosis (Confidence: {prediction_prob:.2f})")
    else:
        st.success(f"Prediction: Normal (Confidence: {1 - prediction_prob:.2f})")
