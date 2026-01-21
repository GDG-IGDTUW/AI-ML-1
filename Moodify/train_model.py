# train_model.py
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

import joblib

from preprocessing import preprocess_text, basic_preprocess

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

print("Preprocessing training data...")
train_df['text_clean'] = train_df[TEXT_COL].apply(preprocess_text)

print("Preprocessing validation data (no spell correction)...")
val_df['text_clean'] = val_df[TEXT_COL].apply(basic_preprocess)

print("Preprocessing test data (no spell correction)...")
test_df['text_clean'] = test_df[TEXT_COL].apply(basic_preprocess)


def clean_df(df):
    df = df.dropna(subset=['text_clean'])
    df = df[df['text_clean'].str.strip() != ""]
    return df.reset_index(drop=True)

train_df = clean_df(train_df)
val_df   = clean_df(val_df)
test_df  = clean_df(test_df)

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

y_test_pred = clf.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test classification report:")
print(classification_report(y_test, y_test_pred, target_names=le.classes_))

# ---------------------------
# BEFORE tuning metrics
# ---------------------------
before_acc = accuracy_score(y_test, y_test_pred)
before_f1 = f1_score(y_test, y_test_pred, average="weighted")

print(f"\nBEFORE TUNING -> Accuracy: {before_acc:.4f}")
print(f"BEFORE TUNING -> F1-score: {before_f1:.4f}")


# ---------------------------
# 7B) Hyperparameter tuning using GridSearchCV
# ---------------------------
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs", "saga"]
}

grid = GridSearchCV(
    LogisticRegression(
        max_iter=2000,
        multi_class='multinomial',
        random_state=42
    ),
    param_grid=param_grid,
    scoring="f1_weighted",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\nRunning GridSearchCV...")
grid.fit(X_train, y_train)

best_clf = grid.best_estimator_
print("Best parameters found:", grid.best_params_)

# ---------------------------
# AFTER tuning metrics
# ---------------------------
y_test_pred_best = best_clf.predict(X_test)

after_acc = accuracy_score(y_test, y_test_pred_best)
after_f1 = f1_score(y_test, y_test_pred_best, average="weighted")

print(f"\nAFTER TUNING -> Accuracy: {after_acc:.4f}")
print(f"AFTER TUNING -> F1-score: {after_f1:.4f}")

print("\nAFTER TUNING classification report:")
print(classification_report(y_test, y_test_pred_best, target_names=le.classes_))

# Optional validation check
y_val_pred_best = best_clf.predict(X_val)
print("\nValidation Accuracy (BEST MODEL):", accuracy_score(y_val, y_val_pred_best))
print("Val classification report (BEST MODEL):")
print(classification_report(y_val, y_val_pred_best, target_names=le.classes_))

# ---------------------------
# 8) Save model
# ---------------------------
OUT = "tfidf_logreg_song_mood.pkl"
joblib.dump(
    {
        'tfidf': tfidf,
        'model': best_clf,
        'label_encoder': le
    },
    OUT
)

# ---------------------------
# 9) Optional helper for quick testing
# ---------------------------
def predict_emotion(lyrics):
    txt = preprocess_text(lyrics)
    vec = tfidf.transform([txt])

    probs = best_clf.predict_proba(vec)[0]
    emotion_labels = le.inverse_transform(best_clf.classes_)

    emotion_probs = dict(zip(emotion_labels, probs))

    top_3 = sorted(
        emotion_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    return top_3


# Example
print("\nExample predictions:")
print("Happy:", predict_emotion("I am dancing and feeling so happy, life is beautiful"))
print("Sad:", predict_emotion("Tears fall every night, my heart hurts and I'm alone"))
print("Angry:", predict_emotion("I will not forgive you, you made me furious"))