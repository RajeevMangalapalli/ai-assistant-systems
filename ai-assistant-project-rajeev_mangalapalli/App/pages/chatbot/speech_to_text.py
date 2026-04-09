import speech_recognition as sr
import streamlit as st

def speech_to_text():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = r.listen(source, timeout=5)

        text = r.recognize_google(audio)
        st.success(f"You said: {text}")
        return text

    except Exception:
        st.error("Sorry, I couldn't understand.")
        return None
