import streamlit as st
from src.predict import predict_emotion

st.set_page_config(page_title="Text Emotion Detection", page_icon="😊")

st.title("Text Emotion Detection App")
st.write("Enter any text and the model will predict the emotion.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict Emotion"):
    if len(user_input.strip()) == 0:
        st.error("Please enter some text.")
    else:
        emotion, confidence = predict_emotion(user_input)
        st.subheader(f"Emotion: **{emotion.upper()}**")
        st.write(f"Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))
