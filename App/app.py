
import streamlit as st
import pickle
import numpy as np
import google.generativeai as genai
from datetime import datetime
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Configuration
st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="😊",
    layout="centered"
)

# Load Model
model = load_model("Models/bilstm_model.h5")

with open("Models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("Models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

import os

# Gemini API Configuration
gemini_model = None

try:
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

except Exception:
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

        # STORE globally for reuse
        st.session_state["emotion"] = emotion

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

        new_log = pd.DataFrame({
            "Timestamp": [datetime.now()],
            "Input_Text": [user_text],
            "Predicted_Emotion": [emotion],
            "Confidence": [round(confidence, 2)]
        })

        try:
            old_logs = pd.read_csv("Logs/emotion_logs.csv")
            logs = pd.concat([old_logs, new_log], ignore_index=True)
        except FileNotFoundError:
            logs = new_log

        logs.to_csv("Logs/emotion_logs.csv", index=False)


# =========================
# GUIDANCE SECTION (FIXED)
# =========================

st.subheader("🤖 Learning Guidance")

emotion = st.session_state.get("emotion", None)

if emotion:

    if gemini_model:
        prompt = f"""
        The student is feeling {emotion}.
        Provide short motivational learning guidance in 2-3 lines.
        """

        response = gemini_model.generate_content(prompt)
        st.write(response.text)

    else:
        guidance = {
            "joy": "Excellent! Keep learning and challenge yourself with more advanced topics.",
            "anger": "Take a short break and revisit the topic calmly.",
            "fear": "Practice step by step. Confidence comes with repetition.",
            "sadness": "Stay positive. Learning takes time and consistency.",
            "love": "Great enthusiasm! Keep exploring new concepts.",
            "surprise": "Use your curiosity to learn more about the topic.",
            "confused": "Break the problem into smaller parts and revisit basics.",
            "frustrated": "Slow down and solve simpler examples first.",
            "curious": "Explore deeper and try hands-on practice.",
            "confident": "Move to advanced problems now.",
            "bored": "Try interactive learning like videos or quizzes."
        }

        st.write(guidance.get(emotion, "Keep learning consistently."))

else:
    st.write("Run emotion detection first.")
import pandas as pd
import matplotlib.pyplot as plt

st.subheader("📊 Analytics Dashboard")

try:
    df = pd.read_csv("Logs/emotion_logs.csv")

    if len(df) == 0:
        st.write("No logs available yet.")
    else:
        emotion_counts = df["Predicted_Emotion"].value_counts()

        st.write("Emotion Distribution:")

        fig, ax = plt.subplots()
        emotion_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")

        st.pyplot(fig)

        st.write("Recent Logs")
        st.dataframe(df.tail(10))

except FileNotFoundError:
    st.write("Log file not found. Run predictions first.")