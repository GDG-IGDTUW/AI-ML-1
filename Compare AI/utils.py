import numpy as np
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

    if not isinstance(text1, str) or not isinstance(text2, str):
        return 0.0

    text1 = text1.strip()
    text2 = text2.strip()

    if not text1 or not text2:
        return 0.0
    
    if len(text1.split()) < 3 or len(text2.split()) < 3:
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

def find_common_sentences(text1, text2):
    """
    Find overlapping sentences between two texts.
    Minimal and safe exact-match approach.
    """
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    sentences1 = set([s.strip() for s in text1.split('.') if len(s.strip()) > 20])
    sentences2 = set([s.strip() for s in text2.split('.') if len(s.strip()) > 20])


    common = sentences1.intersection(sentences2)
    return list(common)

def preprocess_text(text):
    """
    Standardized text preprocessing:
    - lowercase
    - remove punctuation/special characters
    - remove extra whitespace
    - remove stopwords
    """
    if not text:
        return ""

    # lowercase
    text = text.lower()

    # remove punctuation & special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]

    return " ".join(words)




