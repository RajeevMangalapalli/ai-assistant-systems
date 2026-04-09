# conversation.py
from pages.chatbot.intents import detect_intent
from pages.chatbot.mood_detection import analyze_mood


def handle_user_input(text):
    intent = detect_intent(text)

    if intent == "greeting":
        return (
            "Hey! 😊 How are you feeling today?\n"
            "I can recommend music to match or change your mood 🎶",
            "none",
            None,
        )

    if intent == "help":
        return (
            "Tell me how you feel or what kind of music you want 🎧\n"
            "Examples:\n"
            "- *I feel sad*\n"
            "- *Play something happy*\n"
            "- *Cheer me up*",
            "none",
            None,
        )

    if intent == "exit":
        return (
            "Glad I could help! 🎶 See you next time.",
            "none",
            None,
        )

    # Default → MUSIC
    mood = analyze_mood(text)

    current = mood["current_mood"]
    target = mood["target_mood"]

    if mood["is_shift"]:
        reply = (
            f"Sounds like you're feeling **{current}**, "
            f"but want something **{target}** 💛\n\n"
            "Here’s a song I’d recommend:"
        )
    else:
        reply = (
            f"To match your **{target}** mood, "
            "I’d recommend:"
        )

    return reply, "recommend", target
