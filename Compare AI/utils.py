import numpy as np
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_file_text(uploaded_file):
    """
    Reads text from TXT, PDF, and DOCX files.
    """
    # TXT
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8").strip()

    # PDF
    elif uploaded_file.type == "application/pdf":
        text = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception:
            return ""
        return text.strip()

    # DOCX
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs]).strip()
        except Exception:
            return ""

    return ""


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


