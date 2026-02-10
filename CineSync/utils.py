import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import asyncio
import json
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from surprise import Dataset, Reader, SVD
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

_analyzer = SentimentIntensityAnalyzer()

# --- MISSING FETCH POSTER FUNCTION (ISSUE #45) ---
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception:
        return "https://via.placeholder.com/500x750?text=Error"

# --- MISSING UPLIFTING FUNCTION (FOR APP STABILITY) ---
def is_uplifting(text):
    if not isinstance(text, str) or not text.strip():
        return False
    score = _analyzer.polarity_scores(text)
    return score["compound"] >= 0.1

def load_data(filepath):
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        return None

def preprocess_data(df):
    df['genres'] = df['genres'].fillna('')
    df['plot'] = df['plot'].fillna('')
    df['combined_features'] = (df['genres'] + " " + df['plot']).str.lower()
    return df

def calculate_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, df, cosine_sim, top_n=5):
    try:
        indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
        idx = indices[title.lower()]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in sim_scores[1:top_n+1]]
        return df['title'].iloc[movie_indices].tolist()
    except KeyError:
        return []

def collaborative_recommendations(ratings_df, user_id, top_n=5):
    if not SURPRISE_AVAILABLE: return []
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'title', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    all_movies = ratings_df['title'].unique()
    watched = ratings_df[ratings_df['userId'] == user_id]['title'].values
    predictions = [(movie, algo.predict(user_id, movie).est) for movie in all_movies if movie not in watched]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in predictions[:top_n]]

def emit_sync_event(room_id, user_id, action, timestamp):
    event = {"room_id": room_id, "user_id": user_id, "action": action, "timestamp": timestamp}
    print("SYNC EVENT:", json.dumps(event))