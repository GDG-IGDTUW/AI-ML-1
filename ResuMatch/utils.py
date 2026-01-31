import PyPDF2
import re
import nltk
import regex
from nltk.corpus import stopwords
from langdetect import detect
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from io import BytesIO

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = [para.text for para in doc.paragraphs]
    return " ".join(text)
    
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file.
    """
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                # layout=True helps maintain the word order in Indic scripts
                extracted = page.extract_text(layout=True)
                if extracted:
                    text += extracted
            return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def clean_text(text):
    """
    Cleans text by detecting language, lowercasing, 
    and removing language-specific stopwords.
    """
    if not text.strip():
        return ""

    # 1. Detect Language and define lang_code immediately
    try:
        lang_code = detect(text) 
    except:
        lang_code = 'en' # Default fallback

    # 2. Manual Hindi Support (Check lang_code here)
    if lang_code == 'hi':
        # Basic cleaning for Hindi
        text = text.lower()
        text = regex.sub(r'[^\p{L}\s]', '', text)
        hindi_stopwords = {'के', 'में', 'है', 'हैं', 'पर', 'और', 'से', 'को', 'का', 'की'}
        words = text.split()
        cleaned_words = [w for w in words if w not in hindi_stopwords]
        return " ".join(cleaned_words)

    # 3. Standard Logic for other languages
    lang_map = {
        'en': 'english', 'es': 'spanish', 'fr': 'french', 
        'de': 'german', 'it': 'italian', 'pt': 'portuguese',
        'ru': 'russian', 'nl': 'dutch'
    }
    nltk_lang = lang_map.get(lang_code, 'english')

    text = text.lower()
    text = regex.sub(r'[^\p{L}\s]', '', text) 
    
    try:
        stop_words = set(stopwords.words(nltk_lang))
    except:
        stop_words = set(stopwords.words('english'))
        
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

def calculate_similarity(resume_text, job_desc_text):
    """
    Calculates the cosine similarity between the resume and job description.
    Returns a percentage score (0-100).
    """

    if not resume_text or not job_desc_text:
        return 0.0
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
    if not resume_text or not job_desc_text:
        return []
    tfidf_jd = TfidfVectorizer(max_features=20) # Get top feature words
    tfidf_jd.fit([job_desc_text])
    
    feature_names = tfidf_jd.get_feature_names_out()
    
    # Check for simple presence (could be improved with fuzzy matching, but keeping it simple)
    # cleaning resume text again just to be sure we are matching words
    resume_words = set(resume_text.split())
    
    missing_keywords = [word for word in feature_names if word not in resume_words]
            
    return missing_keywords
