# 🎧 Moodify: Know The Vibe (Lyrics to Emotion Predictor)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://moodify-knowthevibe.streamlit.app/)
[![GitHub license](https://img.shields.io/github/license/ArniGoyal/Moodify)](./LICENSE)

A web application built with Streamlit and scikit-learn that classifies the dominant emotion (e.g., joy, sadness, anger) from song lyrics and suggests related songs from a database with the same predicted 'vibe'.

## 🖼️ Project Structure


## ✨ Key Features

* **Text Classification:** Uses a **TF-IDF Vectorizer** and a **Logistic Regression** model trained on song lyrics to predict one of 6 core emotions.
* **Data Preprocessing:** Includes custom preprocessing (stemming, stop-word removal, and cleaning) via `nltk`.
* **Song Recommendation:** Filters the `songs_db.csv` by the predicted emotion and then ranks the matching songs based on **Cosine Similarity** of their TF-IDF vectors to the user's input lyrics.

## 📂 Data and Model

* **Model File:** The trained model (`tfidf_logreg_song_mood.pkl`) contains the TF-IDF vectorizer, the Logistic Regression classifier, and the label encoder.
* **Database (Training):** `song_mood_dataset` contains 3 files train, val and test which have the songs for training.
* **Database:** `songs_db.csv` contains the song library used for suggestions.

## 📄 License

This project is licensed under the **[MIT/GPLv3] License** - see the [`LICENSE`](./LICENSE) file for details.

