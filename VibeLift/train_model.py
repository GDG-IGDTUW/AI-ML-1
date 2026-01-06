#!/usr/bin/env python3
"""
train_model.py

- Trains a MultinomialNB on combined_emotion.csv (with cleaning + stemming).
- Saves ONE pickle file:
    - emotion_model.pkl   (dict: {"model": MultinomialNB, "vectorizer": CountVectorizer})
- Usage: python train_model.py --csv /path/to/combined_emotion.csv
"""

import argparse
import pickle
import re
from pathlib import Path
from time import time

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- DEFAULT CONFIG ----------
CSV_PATH_DEFAULT = "combined_emotion.csv"
OUTPUT_PKL = "emotion_model.pkl"   # final combined pickle (dict)
MAX_FEATURES = 1500                # change to 3000/5000 if you have RAM
RANDOM_STATE = 0
SAMPLE_MAX_ROWS = 500_000          # sample if dataset is huge (tune as needed)
# ------------------------------------

def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        print("Downloading NLTK stopwords...")
        nltk.download("stopwords")

def clean_text(text: str, ps: PorterStemmer, stop_words:set) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.lower().split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def main(csv_path: str, out_pkl: str, max_features: int):
    ensure_nltk()
    stop_words = set(stopwords.words("english"))
    # keep behavior same as original: remove "not" from stopwords if present
    if "not" in stop_words:
        stop_words.remove("not")
    ps = PorterStemmer()

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    print("Loading CSV (may take some time)...")
    df = pd.read_csv(csv_file)
    print("Total rows read:", len(df))

    # keep only rows that have required columns
    if not {"sentence", "emotion"}.issubset(df.columns):
        raise ValueError("CSV must contain 'sentence' and 'emotion' columns")

    df = df.dropna(subset=["sentence", "emotion"]).reset_index(drop=True)
    nrows = len(df)
    print("Rows after dropna:", nrows)

    # if huge, stratified sampling to reduce memory/time
    if nrows > SAMPLE_MAX_ROWS:
        print(f"Dataset large ({nrows} rows). Sampling ~{SAMPLE_MAX_ROWS} rows stratified by emotion...")
        df = df.groupby("emotion", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(SAMPLE_MAX_ROWS / df['emotion'].nunique())),
                               random_state=RANDOM_STATE)
        ).reset_index(drop=True)
        print("Rows after sampling:", len(df))

    # Cleaning
    print("Cleaning text...")
    t0 = time()
    df["clean"] = df["sentence"].map(lambda s: clean_text(s, ps, stop_words))
    print(f"Cleaning done in {time()-t0:.1f} s")

    # Vectorize
    print(f"Fitting CountVectorizer(max_features={max_features})...")
    vect = CountVectorizer(max_features=max_features)
    X = vect.fit_transform(df["clean"].values)
    y = df["emotion"].values
    print("Vectorized shape:", X.shape)

    # Split & train
    print("Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=RANDOM_STATE, stratify=y)

    print("Training MultinomialNB...")
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save combined pickle (dict)
    print(f"Saving combined pickle to {out_pkl} ...")
    with open(out_pkl, "wb") as f:
        pickle.dump({"model": clf, "vectorizer": vect}, f)
    print("Saved successfully.")

    # Sanity test
    sample_text = "I feel great and happy today!"
    sample_clean = clean_text(sample_text, ps, stop_words)
    sample_vec = vect.transform([sample_clean])
    pred = clf.predict(sample_vec)[0]
    proba = clf.predict_proba(sample_vec).max()
    print(f"Sanity test -> '{sample_text}' -> predicted: {pred} (confidence {proba:.2f})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CSV_PATH_DEFAULT, help="Path to combined_emotion.csv")
    p.add_argument("--out", default=OUTPUT_PKL, help="Output combined pickle filename")
    p.add_argument("--features", type=int, default=MAX_FEATURES, help="CountVectorizer max_features")
    args = p.parse_args()
    main(args.csv, args.out, args.features)
