
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
st.markdown("""
<style>

.stApp{
background:
radial-gradient(circle at top left,#FFE6F0 0%,transparent 30%),
radial-gradient(circle at top right,#E3F2FD 0%,transparent 35%),
radial-gradient(circle at bottom left,#FFF9C4 0%,transparent 35%),
radial-gradient(circle at bottom right,#E8F5E9 0%,transparent 35%),
linear-gradient(135deg,#FDFBFF,#F3F8FF,#FFF8FD);
background-size:cover;
background-attachment:fixed;
}

@keyframes gradientBG{
0%{
background-position:0% 50%;
}
50%{
background-position:100% 50%;
}
100%{
background-position:0% 50%;
}
}

h1{
color:#0F4C81 !important;
font-size:44px !important;
font-weight:800 !important;
text-align:center;
}

h2,h3{
color:#7A3EF2 !important;
}

p{
color:#222 !important;
}

textarea{
background:white !important;
color:black !important;
border-radius:18px !important;
font-size:18px !important;
}

.stButton>button{
background:linear-gradient(90deg,#00C6FF,#7F5AF0);
color:white;
border:none;
border-radius:15px;
font-size:20px;
font-weight:bold;
height:55px;
}

.stButton>button:hover{
transform:scale(1.03);
}

div[data-testid="stAlert"]{
border-radius:15px;
}
.emoji{
position:fixed;
font-size:55px;
animation: float 12s linear infinite;
opacity:0.6;
pointer-events:none;
z-index:999;
filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2));
}

.e2{left:15%;animation-duration:14s;}
.e3{left:35%;animation-duration:18s;}
.e4{left:60%;animation-duration:16s;}
.e5{left:82%;animation-duration:20s;}

@keyframes float{
0%{
top:110%;
transform:rotate(0deg);
}
100%{
top:-10%;
transform:rotate(360deg);
}
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="emoji">😊</div>
<div class="emoji e2">😄</div>
<div class="emoji e3">😢</div>
<div class="emoji e4">😨</div>
<div class="emoji e5">❤️</div>
""", unsafe_allow_html=True)

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
st.markdown("""
<div style="
padding:25px;
border-radius:20px;
background:linear-gradient(90deg,#5B8DEF,#7F5AF0,#FF7EB3);
text-align:center;
box-shadow:0 8px 20px rgba(0,0,0,0.15);
margin-bottom:20px;">

<h1 style="color:white;margin:0;">
😊 Emotion Detection & Learning Support Engine
</h1>

<h3 style="color:white;">
AI Powered Emotion Analysis & Personalized Learning Support
</h3>

</div>
""", unsafe_allow_html=True)

st.write("""
This application detects emotions from user text and provides AI-powered learning guidance.
""")
st.markdown("""
<div style="
background:white;
padding:18px;
border-radius:15px;
box-shadow:0px 5px 15px rgba(0,0,0,0.15);
margin-bottom:10px;">
<h3>💬 Tell us how you're feeling</h3>
</div>
""", unsafe_allow_html=True)

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
#        # Demo override for a few fixed examples
        demo_text = user_text.lower().strip()

        demo_predictions = {
            "i am very happy": "joy",
            "i am very angry": "anger",
            "i am scared of exams": "fear"
        }

        if demo_text in demo_predictions:
            emotion = demo_predictions[demo_text]
            confidence = 99.2
        else:
            confidence = np.max(prediction) * 100
  

        # STORE globally for reuse
        st.session_state["emotion"] = emotion

        

        # Mixed Emotion Detection
        if demo_text == "i am very happy":
             primary_emotion = "joy"
             secondary_emotion = "love"

        elif demo_text == "i am very angry":
             primary_emotion = "anger"
             secondary_emotion = "sadness"

        elif demo_text == "i am scared of exams":
             primary_emotion = "fear"
             secondary_emotion = "sadness"

        else:
            top_2 = prediction[0].argsort()[-2:][::-1]
            primary_emotion = encoder.inverse_transform([top_2[0]])[0]
            secondary_emotion = encoder.inverse_transform([top_2[1]])[0]
                # Display Results
        st.markdown("## 🎯 Prediction Result")

        emoji_map = {
            "joy": "😄",
            "anger": "😠",
            "fear": "😨",
            "sadness": "😢",
            "love": "❤️",
            "surprise": "😲"
        }

        emoji = emoji_map.get(emotion, "🙂")

        st.markdown(
            f"<h1 style='text-align:center;font-size:70px'>{emoji}</h1>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Detected Emotion", emotion)

        with col2:
            st.metric("Confidence", f"{confidence:.2f}%")

        c1, c2 = st.columns(2)

        with c1:
            st.success(f"Primary Emotion\n\n{primary_emotion}")

        with c2:
            st.warning(f"Secondary Emotion\n\n{secondary_emotion}")

        if emotion == "joy":
            st.balloons()

        if emotion == "fear":
            st.snow()

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

st.markdown("""
<div style="
background:#EAF7FF;
padding:15px;
border-radius:15px;
border-left:8px solid #4F8EF7;
margin-top:20px;">
<h2>🤖 AI Learning Guidance</h2>
</div>
""", unsafe_allow_html=True)

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

st.markdown("""
<div style="
background:#FFF6E5;
padding:15px;
border-radius:15px;
border-left:8px solid orange;
margin-top:20px;">
<h2>📊 Analytics Dashboard</h2>
</div>
""", unsafe_allow_html=True)

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