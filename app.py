
import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="😊",
    layout="centered"
)

# Load Model
model = load_model('bilstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

import os

# Gemini API Configuration
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
else:
    gemini_model = None
st.title("😊 Emotion Detection & Learning Support Engine")

st.write("""
This application detects emotions from user text and provides AI-powered learning guidance.
""")

user_text = st.text_area(
    "Enter how you feel:",
    placeholder="Example: I am very happy because I completed my project"
)

if st.button("Detect Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        # Text Preprocessing
        sequence = tokenizer.texts_to_sequences([user_text])
        padded = pad_sequences(sequence, maxlen=100)

        # Prediction
        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)

        emotion = encoder.inverse_transform([predicted_class])[0]

        confidence = np.max(prediction) * 100

        # Mixed Emotion Detection
        top_2 = prediction[0].argsort()[-2:][::-1]

        primary_emotion = encoder.inverse_transform([top_2[0]])[0]
        secondary_emotion = encoder.inverse_transform([top_2[1]])[0]

        # Display Results
        st.success(f"Predicted Emotion: {emotion}")

        st.info(f"Confidence Score: {confidence:.2f}%")

        st.write(f"### Primary Emotion: {primary_emotion}")
        st.write(f"### Secondary Emotion: {secondary_emotion}")

        # Gemini Guidance
        try:
            prompt = f"""
            The student is feeling {emotion}.

            Provide short motivational learning guidance in 2-3 lines.
            """

            response = gemini_model.generate_content(prompt)

            st.subheader("🤖 Gemini AI Guidance")
            st.write(response.text)

        except Exception as e:
            st.error(f"Gemini Error: {e}")
