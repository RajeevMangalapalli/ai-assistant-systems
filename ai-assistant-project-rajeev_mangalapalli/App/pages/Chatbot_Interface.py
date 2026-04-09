import streamlit as st
from pages.chatbot.conversation import handle_user_input
from pages.chatbot.speech_to_text import speech_to_text
from pages.chatbot.resources import load_dataset

st.set_page_config(page_title="Music Mood Chatbot", layout="centered")
st.title("🎵 Music Based Mood Recommender Chatbot")

df = load_dataset()

st.write("Hi! I'm your personal music recommendation assistant.")
st.divider()

col1, col2 = st.columns([4, 1])

with col1:
    user_text = st.text_input("Type your message here...")

with col2:
    speak = st.button("🎙️ Speak")

if speak:
    speech = speech_to_text()
    if speech:
        user_text = speech

if user_text:
    reply, action, target_mood = handle_user_input(user_text)

    st.subheader("💬 Response")
    st.success(reply)

    if action == "recommend":
        st.divider()
        st.subheader("🎵 Recommended Song")

        filtered = df[df["mood_label"].str.lower() == target_mood.lower()]

        if not filtered.empty:
            song = filtered.sample(1).iloc[0]
            st.markdown(f"**Title:** {song['name']}")
            st.markdown(f"**Artist:** {song['artist']}")
            st.markdown(f"**Year:** {song['year']}")
            st.markdown(f"**Mood:** {song['mood_label']}")
            st.audio(song["spotify_preview_url"])
        else:
            st.warning("No songs found for that mood.")
