# 🌈 VibeLift

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vibelift-madewithcare.streamlit.app/)
[![GitHub license](https://img.shields.io/github/license/ArniGoyal/VibeLift)](./LICENSE)

**VibeLift** is a polished, tablet-inspired web application designed to help users track and improve their emotional well-being. By utilizing Natural Language Processing (NLP), the app analyzes user-submitted text to predict their current emotion and provides tailored wellness exercises to "lift" their vibe.

---

## ✨ Features

* **Emotion Detection:** Uses a machine learning model to categorize feelings into 6 core emotions: Joy, Sadness, Anger, Fear, Love, and Surprise.
* **Adaptive UI:** The application's background and theme dynamically change color based on the predicted emotion.
* **Actionable Wellness:** Provides specific exercises such as grounding techniques, breathing patterns, or journaling prompts tailored to your current state.
* **Clean Design:** A centered, translucent "glassmorphism" interface designed for focus and clarity.

---

## 📂 Project Structure

Ensure your directory is organized as follows for the application to function correctly:

```plaintext
VibeLift/
├── emotions/               # Project assets or data directory
├── .gitignore              # Files to be ignored by Git
├── LICENSE                 # Project license
├── app.py                  # The main Streamlit application
├── combined_emotion.csv    # Raw dataset used for training
├── emotion_model.pkl       # The saved machine learning model
├── hero_icon.png           # App logo icon
├── requirements.txt        # List of Python dependencies
└── train_model.py          # Script used to train and export the ML model

```

---

## 🧠 How It Works

The application follows a standard NLP pipeline to process your thoughts:

1. **Text Preprocessing:** The input is cleaned (removing special characters), converted to lowercase, and filtered for "stopwords."
2. **Stemming:** Words are reduced to their root form (e.g., "feeling" becomes "feel") using the Porter Stemmer.
3. **Vectorization:** The text is transformed into a numerical format that the computer can understand.
4. **Prediction:** A trained classifier analyzes the patterns to determine the most likely emotion.

---

## 🎨 Customization

* **Exercises:** You can modify the `EXERCISES` dictionary in `app.py` to add your own custom wellness tips.
* **Visuals:** The `BACKGROUND_STYLE` dictionary allows you to change the CSS gradients for each emotion.

---

## 📄 License

This project is licensed under the **[MIT/GPLv3] License** - see the [`LICENSE`](./LICENSE) file for details.

---

## ⚠️ Disclaimer

**VibeLift is an AI-powered tool for self-reflection and emotional awareness.** It is not a substitute for professional mental health care, therapy, or medical advice.