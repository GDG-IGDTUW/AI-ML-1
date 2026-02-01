import streamlit as st
import pandas as pd 
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess_text
from train_model import predict_emotion, explain_prediction
from langdetect import detect, LangDetectException



# Load model components 
@st.cache_resource(show_spinner=False)
def load_model(path="tfidf_logreg_song_mood.pkl"):
    data = joblib.load(path)
    tfidf = data['tfidf']
    model = data['model']
    le = data['label_encoder']
    return tfidf, model, le

tfidf, clf, le = load_model()

# Load songs DB (for suggestions) 
@st.cache_data(show_spinner=False)
def load_songs(path="songs_db.csv"):
    df = pd.read_csv(path)
    # ensure columns: title, artist, lyrics, emotion
    for col in ['title','artist','lyrics','emotion']:
        if col not in df.columns:
            df[col] = ""
    df['lyrics_clean'] = df['lyrics'].apply(preprocess_text)
    # compute TF-IDF vectors for songs (uses same tfidf)
    if len(df) > 0:
        df['tfidf_vec'] = list(tfidf.transform(df['lyrics_clean']).toarray())
    else:
        df['tfidf_vec'] = []
    return df

songs_df = load_songs()

# UI styling
def local_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&display=swap');

        html, body, .stApp {
            margin: 0;
            padding: 0;
            font-family: 'Cinzel', serif !important;
            color: #3b2f2f; /* elegant dark brown text */
            background: linear-gradient(-45deg, #f3e6da, #e6d4c1, #f0e1c6, #dbc8a7);
            background-size: 400% 400%;
            animation: gradientBG 20s ease infinite;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .stApp .block-container {
            padding-top: 1.5rem;
            padding-left: 2rem;
            padding-right: 2rem;
            padding-bottom: 2rem;
            max-width: 900px;
        }

        h1, h2, h3, h4, h5, h6, p, label, div, span, input, textarea, button {
            font-family: 'Cinzel', serif !important;
            color: #3b2f2f;
        }

        .stImage {
            margin: 0 auto !important; /* Center the image element itself */
            display: block;
            margin-bottom: -1rem; /* Reduced negative margin for spacing */
            width: 250px; /* Force max width to prevent stretching */
        }

        .stApp h1 {
            font-family: 'Cinzel', serif !important;
            color: #3b2f2f;
            font-size: 3.5rem; /* Large size for main title */
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }

        .stApp h2 {
            font-family: 'Cinzel', serif !important;
            color: #3b2f2f;
            font-size: 1.5rem; 
            font-weight: 400;
            margin: 0;
        }

        .stButton > button,
        div.stButton > button {
            background-color: #3b2f2f !important; /* dark brown */
            border: 2px solid #3b2f2f !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 700 !important;
            transition: background-color 0.3s, border-color 0.3s;
            /* ensure text is not clipped or transformed */
            -webkit-appearance: none !important;
            appearance: none !important;
        }

        .stButton > button, 
        .stButton > button * , 
        div.stButton > button, 
        div.stButton > button * {
            color: #f4ebdc !important;              
            -webkit-text-fill-color: #f4ebdc !important; 
            text-shadow: none !important;
            font-family: 'Cinzel', serif !important;
        }

        .stApp div.stButton > button > div > span {
            color: #f4ebdc !important;
            -webkit-text-fill-color: #f4ebdc !important;
        }

        .stButton > button:hover,
        div.stButton > button:hover {
            background-color: #5c4747 !important;
            border-color: #5c4747 !important;
            color: #f4ebdc !important;
            -webkit-text-fill-color: #f4ebdc !important;
        }
        .stButton > button:focus,
        div.stButton > button:focus {
            outline: 3px solid rgba(180,140,110,0.18) !important;
        }

        .stButton > button:disabled,
        div.stButton > button:disabled {
            opacity: 0.9 !important;
            cursor: not-allowed !important;
        }


        .card {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.08);
            margin-bottom: 12px;
        }

        .predict-badge {
            background: #d6bca8;
            color: #2b2b2b;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight:700;
            display:inline-block;
        }

        textarea::placeholder {
            color: #7a5f4f;
            font-style: italic;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

local_css()

# Page header

col_left, col_center, col_right = st.columns([1, 1, 1])

with col_center:
    
    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True) 

    st.image('logo.jpg', width=250, caption='')
   
    st.markdown(
        """
        <div style="text-align: center; margin-top: -1.75rem; margin-bottom: 0.5rem; margin-left: -2rem; white-space: nowrap;">
            <h1>MOODIFY</h1>
            <h2><i>"Know The Vibe"</i></h2>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown(
    """
    <div class="card">
      <h3>Lyrics → Emotion</h3>
      <p>Paste your song lyrics below and press <b>Predict Vibe</b>. Moodify will show the predicted emotion and suggest 5 songs with the same vibe.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input area 
lyrics_input = st.text_area(
    "Enter lyrics here: ", 
    height=240, 
    placeholder="Paste your song lyrics here..."
)

# Predict button
# Predict button
if st.button("Predict Vibe"):
    lyrics_input = lyrics_input.strip() 

    # Count words 
    word_count = len(lyrics_input.split()) 
    MIN_WORDS = 20

    if not lyrics_input:
        st.warning("Please enter some lyrics first!")
    elif word_count < MIN_WORDS:
        st.info(f"Please enter at least **{MIN_WORDS} words** for accurate emotion prediction.\n\n"
                f"Current word count: **{word_count}**")
    else:
        # Language detection
        try:
            detected_lang = detect(lyrics_input)
        except LangDetectException:
            detected_lang = None

        if detected_lang != "en":
            st.warning("Moodify currently only supports English lyrics.")
            st.stop()

        # ----------------------------
        # Prediction
        # ----------------------------
        pred_labels = predict_emotion(lyrics_input, tfidf, clf, le, threshold=0.4)
        dominant_emotion, dominant_prob = pred_labels[0]  # top predicted emotion
        keywords = explain_prediction(lyrics_input, tfidf, clf, le, top_k=3)

        # Display probabilities
        emoji_map = {"anger":"🔥","fear":"🌑","joy":"✨","love":"❤️",
                     "sadness":"🌧️","surprise":"⚡","Neutral":"😐"}
        for emotion, prob in pred_labels:
            st.markdown(f"""
                <div style="font-size:22px; line-height:1.4;">
                    <strong>{emoji_map.get(emotion,'')} {emotion.capitalize()}</strong>
                    <span style="color:#6b5e4b; margin-left:10px;">
                        {prob*100:.1f}%
                    </span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(int(prob*100))

        # Display keywords
        st.markdown("### 🔍 Why this vibe?")

        if keywords:
            pill_html = "".join(
                f"<span style='display:inline-block;background:#d6bca8;color:#2b2b2b;"
                f"padding:6px 14px;margin:4px;border-radius:999px;font-weight:700;font-size:14px;'>"
                f"{word}</span>" for word in keywords
            )
            st.markdown(pill_html, unsafe_allow_html=True)
        else:
            st.markdown("<i>No keywords could be highlighted for this input.</i>", unsafe_allow_html=True)

        # ----------------------------
        # Handle Neutral / Low Confidence
        # ----------------------------
        if dominant_emotion == "Neutral":
            st.warning("Moodify is unsure about the vibe of these lyrics. "
                       "Try providing longer or clearer lyrics for better results.")
        else:
            # Suggest songs only if confident
            if len(songs_df) > 0:
                matched_songs_df = songs_df[songs_df['emotion'] == dominant_emotion].copy()
                if len(matched_songs_df) > 0:
                    input_vec = tfidf.transform([preprocess_text(lyrics_input)]).toarray()[0]
                    matched_songs_df['similarity'] = matched_songs_df['tfidf_vec'].apply(
                        lambda x: cosine_similarity([input_vec],[x])[0][0]
                    )
                    top_songs = matched_songs_df.sort_values(by='similarity', ascending=False).head(5)
                    st.markdown(f"### 🎵 Top {len(top_songs)} **{dominant_emotion}** Songs:")
                    for _, row in top_songs.iterrows():
                        link_text = f"**{row['title']}** by *{row['artist']}*"
                        st.markdown(f"{link_text} — [Listen Here]({row.get('link','#')})")
                else:
                    st.info(f"No songs found in the database with the emotion: **{dominant_emotion}**.")

st.markdown("---")
st.markdown("Made with ❤️ — Moodify.")
