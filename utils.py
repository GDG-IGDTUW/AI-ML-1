import PyPDF2
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def clean_text(text):
    """
    Cleans text by lowercasing, removing removing special characters and stopwords.
    """
    # Lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def calculate_similarity(resume_text, job_desc_text):
    """
    Calculates the cosine similarity between the resume and job description.
    Returns a percentage score (0-100).
    """
    content = [resume_text, job_desc_text]
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(content)
    
    # Cosine similarity between the first document (resume) and the second (job desc)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    match_percentage = similarity_matrix[0][1] * 100
    
    return round(match_percentage, 2)

def get_missing_keywords(resume_text, job_desc_text, top_n=10):
    """
    Identifies keywords present in the job description but missing or weak in the resume.
    Uses TF-IDF to find important words in the JD.
    """
    # We want to find words that are important in JD but not in Resume
    # Strategy: Get top TF-IDF words from JD, check if they are in Resume
    
    tfidf_jd = TfidfVectorizer(max_features=20) # Get top feature words
    tfidf_jd.fit([job_desc_text])
    
    feature_names = tfidf_jd.get_feature_names_out()
    
    # Check for simple presence (could be improved with fuzzy matching, but keeping it simple)
    # cleaning resume text again just to be sure we are matching words
    resume_words = set(resume_text.split())
    
    missing_keywords = []
    for word in feature_names:
        if word not in resume_words:
            missing_keywords.append(word)
            
    return missing_keywords
