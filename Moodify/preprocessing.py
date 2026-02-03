from spellchecker import SpellChecker
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)

# ---------------------------
# Custom stopword handling
# ---------------------------

BASE_STOPWORDS = set(stopwords.words("english"))

# Keep negation
BASE_STOPWORDS.discard("not")

# Pronouns to KEEP (emotionally important in lyrics)
PRONOUNS_TO_KEEP = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his",
    "she", "her", "hers",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs"
}

# Musical structure fillers to REMOVE
MUSICAL_FILLERS = {
    "chorus", "verse", "bridge", "intro", "outro",
    "instrumental", "refrain", "hook",
    "prechorus", "postchorus"
}

# Final custom stopword list
CUSTOM_STOPWORDS = (BASE_STOPWORDS - PRONOUNS_TO_KEEP) | MUSICAL_FILLERS

ps = PorterStemmer()

spell = SpellChecker()
spell_cache = {}

# ---------------------------
# Spelling correction
# ---------------------------

def correct_spelling(words):
    corrected = []
    unknown_words = spell.unknown(words)

    for w in words:
        if len(w) <= 3:
            corrected.append(w)
            continue

        if w in spell_cache:
            corrected.append(spell_cache[w])
        elif w in unknown_words:
            corr = spell.correction(w) or w
            spell_cache[w] = corr
            corrected.append(corr)
        else:
            spell_cache[w] = w
            corrected.append(w)

    return corrected

# ---------------------------
# Preprocessing (no spelling)
# ---------------------------

def basic_preprocess(s):
    if pd.isna(s): 
        return ""

    s = str(s).lower()
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r"[^a-z\s']", ' ', s)

    words = s.split()
    words = [
        ps.stem(w)
        for w in words
        if w not in CUSTOM_STOPWORDS and len(w) > 1
    ]

    return " ".join(words)

# ---------------------------
# Preprocessing (with spelling)
# ---------------------------

def preprocess_text(s):
    if pd.isna(s): 
        return ""

    s = str(s).lower()
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r"[^a-z\s']", ' ', s)

    words = s.split()

    # Stopword + musical filler removal
    words = [
        w for w in words
        if w not in CUSTOM_STOPWORDS and len(w) > 1
    ]

    # Spelling normalization
    words = correct_spelling(words)

    # Stemming
    words = [ps.stem(w) for w in words]

    return " ".join(words)
