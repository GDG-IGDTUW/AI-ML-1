import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
try:
    from surprise import Dataset, Reader, SVD
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
if SURPRISE_AVAILABLE:
    from surprise.model_selection import train_test_split
import asyncio
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def load_data(filepath):
    """
    Loads movie data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None

def preprocess_data(df):
    """
    Preprocesses the data by combining relevant columns for vectorization.
    We will combine 'genres' and 'plot' to create a 'combined_features' column.
    
    Args:
        df (pd.DataFrame): The original dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with a new 'combined_features' column.
    """
    # Fill missing values with empty strings just in case
    df['genres'] = df['genres'].fillna('')
    df['plot'] = df['plot'].fillna('')
    
    # Combine genres and plot into one string for better content matching
    # We add genres twice to give them slightly more weight, or just once is fine.
    # Let's simple combine them: details often in plot, categories in genres.
    df['combined_features'] = df['genres'] + " " + df['plot']
    
    # Optional: Basic text cleaning (lowercase)
    df['combined_features'] = df['combined_features'].str.lower()
    
    return df

def calculate_similarity(df):
    """
    Computes the cosine similarity matrix for the movies.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'combined_features'.
        
    Returns:
        tuple: (tfidf_matrix, cosine_sim) matrix.
    """
    # Initialize TF-IDF Vectorizer
    # stop_words='english' removes common words like 'the', 'a', 'in', etc.
    tfidf = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the data to a matrix of TF-IDF features
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Compute Cosine Similarity matrix
    # linear_kernel is equivalent to cosine_similarity for normalized vectors (TF-IDF is normalized)
    # and is faster.
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim

def get_recommendations(title, df, cosine_sim, top_n=5):
    """
    Get movie recommendations based on cosine similarity.
    
    Args:
        title (str): Title of the movie to find similarities for.
        df (pd.DataFrame): The DataFrame containing movie data.
        cosine_sim (numpy.ndarray): The cosine similarity matrix.
        top_n (int): Number of recommendations to return.
        
    Returns:
        list: List of recommended movie titles.
    """
    try:
        # Create a mapping of movie titles to their index in the DataFrame
        indices = pd.Series(df.index, index=df['title'].str.lower()).drop_duplicates()
        
        # Get the index of the movie that matches the title
        # We convert input title to lowercase to match our mapping
        idx = indices[title.lower()]
        
        # Get the pairwise similarity scores of all movies with that movie
        # list of (index, similarity_score)
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores (descending order)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the scores of the top_n most similar movies
        # We skip the first one because it's the movie itself (score=1.0)
        sim_scores = sim_scores[1:top_n+1]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top_n most similar movies
        return df['title'].iloc[movie_indices].tolist()
        
    except KeyError:
        # Returned if the movie title is not found in the dataset
        return []

def collaborative_recommendations(ratings_df, user_id, top_n=5):
    """
    Generate movie recommendations using collaborative filtering (SVD).
    
    ratings_df must have columns: userId, title, rating
    """
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        ratings_df[['userId', 'title', 'rating']],
        reader
    )

    trainset = data.build_full_trainset()
    if not SURPRISE_AVAILABLE:
        return []

    algo = SVD()
    algo.fit(trainset)

    all_movies = ratings_df['title'].unique()
    watched = ratings_df[ratings_df['userId'] == user_id]['title'].values

    predictions = []
    for movie in all_movies:
        if movie not in watched:
            pred = algo.predict(user_id, movie)
            predictions.append((movie, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return [m[0] for m in predictions[:top_n]]

def emit_sync_event(room_id, user_id, action, timestamp):
    """
    Emits a sync event (play, pause, seek) for group watch.
    This is a placeholder for WebSocket / backend integration.
    """

    event = {
        "room_id": room_id,
        "user_id": user_id,
        "action": action,
        "timestamp": timestamp
    }

    # For now, we just print/log the event
    # In real implementation, this will be sent via WebSocket server
    print("SYNC EVENT:", json.dumps(event))

def is_uplifting(text, threshold=0.3):
    """
    Uses VADER sentiment to check if text is uplifting (positive).
    
    Args:
        text (str): Review or plot text
        threshold (float): compound score threshold
    
    Returns:
        bool
    """
    if not isinstance(text, str) or not text.strip():
        return False

    score = sid.polarity_scores(text)
    return score["compound"] >= threshold
