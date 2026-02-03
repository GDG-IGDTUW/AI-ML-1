from spellchecker import SpellChecker
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words("english"))
if "not" in stop_words:
    stop_words.remove("not")
ps = PorterStemmer()

spell = SpellChecker()
spell_cache = {}

def correct_spelling(words):
    corrected = []
    unknown_words = spell.unknown(words)

    for w in words:
        # skip short or common lyric fillers
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

def basic_preprocess(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r"[^a-z\s']", ' ', s)
    words = s.split()
    words = [ps.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

def preprocess_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r"[^a-z\s']", ' ', s)

    words = s.split()

    # removing stopwords
    words = [w for w in words if w not in stop_words and len(w) > 1]

    # spelling normalization
    words = correct_spelling(words)

    words = [ps.stem(w) for w in words]

    return " ".join(words)