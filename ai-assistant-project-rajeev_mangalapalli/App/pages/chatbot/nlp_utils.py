import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)

def extract_features(text):
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha()]
    return {word: True for word in words}
