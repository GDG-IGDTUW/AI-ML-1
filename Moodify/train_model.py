# train_model.py
import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# ---------------------------
# 0) NLTK setup
# ---------------------------
nltk.download('stopwords', quiet=True)

# ---------------------------
# 1) Paths - update if needed
# ---------------------------
TRAIN_PATH = "song_mood_dataset/train.csv"
VAL_PATH   = "song_mood_dataset/val.csv"
TEST_PATH  = "song_mood_dataset/test.csv"

for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}. Place train.csv, val.csv, test.csv in same folder.")

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Loaded: {len(train_df)} train rows | {len(val_df)} val rows | {len(test_df)} test rows")

# ---------------------------
# 2) Detect text & label columns
# ---------------------------
def detect_columns(df):
    TEXT_COL = 'sentence' if 'sentence' in df.columns else ('lyrics' if 'lyrics' in df.columns else None)
    LABEL_COL = 'emotion' if 'emotion' in df.columns else ('label' if 'label' in df.columns else None)
    if TEXT_COL is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        TEXT_COL = obj_cols[0] if obj_cols else df.columns[0]
    if LABEL_COL is None:
        possible = [c for c in df.columns if c != TEXT_COL]
        LABEL_COL = possible[0] if possible else df.columns[1]
    return TEXT_COL, LABEL_COL

TEXT_COL, LABEL_COL = detect_columns(train_df)
print("Using TEXT column:", TEXT_COL, "| LABEL column:", LABEL_COL)

# ---------------------------
# 3) Preprocessing
# ---------------------------
stop_words = set(stopwords.words("english"))
if "not" in stop_words:
    stop_words.remove("not")
ps = PorterStemmer()

def preprocess_text(s):
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r'\[.*?\]', ' ', s)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    s = re.sub(r"[^a-z\s']", ' ', s)
    words = s.split()
    words = [ps.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

for df in (train_df, val_df, test_df):
    df['text_clean'] = df[TEXT_COL].apply(preprocess_text)
    df.dropna(subset=['text_clean'], inplace=True)
    df = df[df['text_clean'].str.strip() != ""]
train_df = train_df[train_df['text_clean'].str.strip() != ""].reset_index(drop=True)
val_df   = val_df[val_df['text_clean'].str.strip() != ""].reset_index(drop=True)
test_df  = test_df[test_df['text_clean'].str.strip() != ""].reset_index(drop=True)
print(f"After cleaning -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# ---------------------------
# 4) Label encoding
# ---------------------------
le = LabelEncoder()
train_labels = train_df[LABEL_COL].astype(str).values
le.fit(train_labels)
train_df['label_id'] = le.transform(train_df[LABEL_COL].astype(str))

def encode_labels(arr):
    arr = arr.astype(str)
    out = []
    for v in arr:
        out.append(le.transform([v])[0] if v in le.classes_ else -1)
    return np.array(out, dtype=int)

val_df['label_id'] = encode_labels(val_df[LABEL_COL].astype(str))
test_df['label_id'] = encode_labels(test_df[LABEL_COL].astype(str))
print("Classes:", list(le.classes_))

# ---------------------------
# 5) TF-IDF Vectorizer
# ---------------------------
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train = tfidf.fit_transform(train_df['text_clean'])
y_train = train_df['label_id'].values
X_val = tfidf.transform(val_df['text_clean'])
y_val = val_df['label_id'].values
X_test = tfidf.transform(test_df['text_clean'])
y_test = test_df['label_id'].values
print("TF-IDF shapes -> X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

# ---------------------------
# 6) Train Logistic Regression
# ---------------------------
clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
clf.fit(X_train, y_train)

# Optional validation check
y_val_pred = clf.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Val classification report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))

# ---------------------------
# 7) Test set evaluation
# ---------------------------
y_test_pred = clf.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test classification report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))
print("Test confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

# ---------------------------
# 8) Save model
# ---------------------------
OUT = "tfidf_logreg_song_mood.pkl"
joblib.dump({'tfidf': tfidf, 'model': clf, 'label_encoder': le}, OUT)
print(f"\nSaved TF-IDF + LogisticRegression + LabelEncoder to: {OUT}")

# ---------------------------
# 9) Optional helper for quick testing
# ---------------------------
def predict_lyrics(lyrics):
    txt = preprocess_text(lyrics)
    vec = tfidf.transform([txt])
    pred_id = clf.predict(vec)[0]
    return le.inverse_transform([pred_id])[0]

# Example
print("\nExample predictions:")
print("Happy:", predict_lyrics("I am dancing and feeling so happy, life is beautiful"))
print("Sad:", predict_lyrics("Tears fall every night, my heart hurts and I'm alone"))
print("Angry:", predict_lyrics("I will not forgive you, you made me furious"))
