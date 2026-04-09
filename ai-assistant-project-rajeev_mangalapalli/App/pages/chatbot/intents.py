# intents.py

INTENTS = {
    "greeting": [
        "hi", "hello", "hey", "yo", "what's up"
    ],

    "help": [
        "help", "what can you do", "how does this work"
    ],

    "music": [
        "play", "recommend", "suggest",
        "music", "song", "songs",
        "something to listen",
        "cheer me up",
        "calm me down",
    ],

    "exit": [
        "bye", "goodbye", "thanks"
    ],
}


def detect_intent(text: str) -> str:
    text = text.lower()

    for intent, keywords in INTENTS.items():
        if any(k in text for k in keywords):
            return intent
    return "music"

