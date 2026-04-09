# mood_detection.py
from nltk import NaiveBayesClassifier
from pages.chatbot.nlp_utils import extract_features
import streamlit as st

TRAINING_DATA = [
    # Happy
    ("i feel happy", "happy"),
    ("i am excited", "happy"),
    ("i feel great", "happy"),
    ("feeling awesome", "happy"),

    # Sad
    ("i feel sad", "sad"),
    ("i am depressed", "sad"),
    ("feeling low", "sad"),
    ("i feel down", "sad"),

    # Neutral
    ("i feel okay", "neutral"),
    ("just chilling", "neutral"),
    ("feeling calm", "neutral"),
    ("nothing special", "neutral"),
]

MOOD_SHIFT_KEYWORDS = {
    "happy": [
        "cheer me up",
        "make me happy",
        "uplifting",
        "happy music",
        "something happy",
    ],
    "neutral": [
        "calm me down",
        "relax",
        "chill",
        "calm music",
    ],
    "sad": [
        "sad music",
        "emotional",
        "melancholic",
    ],
}


@st.cache_resource
def load_classifier():
    training = [(extract_features(t), m) for t, m in TRAINING_DATA]
    return NaiveBayesClassifier.train(training)


def detect_current_mood(text: str) -> str:
    clf = load_classifier()
    return clf.classify(extract_features(text))


def detect_requested_mood(text: str) -> str | None:
    text = text.lower()
    for mood, phrases in MOOD_SHIFT_KEYWORDS.items():
        if any(p in text for p in phrases):
            return mood
    return None


def analyze_mood(text: str):
    current = detect_current_mood(text)
    requested = detect_requested_mood(text)

    return {
        "current_mood": current,
        "target_mood": requested or current,
        "is_shift": requested is not None and requested != current,
    }
