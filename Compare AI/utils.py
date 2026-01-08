import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    """
    Calculates the cosine similarity between two text strings using TF-IDF.
    
    Args:
        text1 (str): First text string.
        text2 (str): Second text string.
        
    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    if not text1 or not text2:
        return 0.0
        
    documents = [text1, text2]
    
    # Create the Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    
    # Generate the TF-IDF Matrix
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    except ValueError:
        # This can happen if the text is empty or only contains stop words
        return 0.0
        
    # Calculate the Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity_matrix[0][0]
