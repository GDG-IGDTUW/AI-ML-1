import PyPDF2
import re
import nltk
import regex
from collections import Counter
from nltk.corpus import stopwords
try:
    from langdetect import detect
except ModuleNotFoundError:
    def detect(_text):
        return "en"
from docx import Document
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ModuleNotFoundError:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False
from io import BytesIO
import os
from mistralai import Mistral

from dotenv import load_dotenv
load_dotenv()



# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

SYNONYM_MAP = {
        "ml": "machine learning",
        "machine learning": "machine learning",

        "nlp": "natural language processing",
        "natural language processing": "natural language processing",

        "ai": "artificial intelligence",
        "artificial intelligence": "artificial intelligence",

        "js": "javascript",
        "javascript": "javascript",

        "dl": "deep learning",
        "deep learning": "deep learning"
}

def normalize_synonyms(text):
    text = text.lower()

    for key, value in SYNONYM_MAP.items():
        text = text.replace(key, value)

    return text


def calculate_ats_score(resume_text, job_desc_text):
    """
    Simulates a realistic ATS scoring system.
    Returns final ATS score (0-100).
    """

    if not resume_text or not job_desc_text:
        return 0

    
    resume_text = normalize_synonyms(resume_text)
    job_desc_text = normalize_synonyms(job_desc_text)

    #Create word sets  ← MISSING PART
    resume_words = set(resume_text.split())
    jd_words = set(job_desc_text.split())

    # --- 1. Keyword Match (40%) ---
    common_keywords = resume_words.intersection(jd_words)
    keyword_score = (len(common_keywords) / len(jd_words)) * 40 if jd_words else 0

    # --- 2. Skills Match (30%) ---
    skills_list = [
        "python", "java", "c++", "machine", "learning",
        "sql", "excel", "react", "django", "flask",
        "data", "analysis", "nlp", "deep", "tensorflow"
    ]

    skills_in_jd = [s for s in skills_list if s in jd_words]
    skills_in_resume = [s for s in skills_list if s in resume_words]

    if skills_in_jd:
        skill_score = (len(skills_in_resume) / len(skills_in_jd)) * 30
    else:
        skill_score = 0

    # --- 3. Experience / Education Signals (20%) ---
    exp_words = ["experience", "year", "internship", "project", "bachelor", "master"]

    exp_count = sum(1 for w in exp_words if w in resume_words)
    exp_score = min(exp_count / len(exp_words), 1) * 20

    # --- 4. Formatting / Sections (10%) ---
    section_words = ["skills", "education", "project", "summary"]

    section_count = sum(1 for w in section_words if w in resume_words)
    format_score = min(section_count / len(section_words), 1) * 10

    # --- Final ATS Score ---
    ats_score = keyword_score + skill_score + exp_score + format_score

    return round(ats_score, 2)




def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = [para.text for para in doc.paragraphs]
    return " ".join(text)
    
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from an uploaded PDF file.
    """
    try:
        if PDFPLUMBER_AVAILABLE:
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    extracted = page.extract_text(layout=True)
                    if extracted:
                        text += extracted
                return text

        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
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
    
    resume_text = normalize_synonyms(resume_text)
    job_desc_text = normalize_synonyms(job_desc_text)
    if SKLEARN_AVAILABLE:
        content = [resume_text, job_desc_text]
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(content)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        match_percentage = similarity_matrix[0][1] * 100
    else:
        resume_words = set(resume_text.split())
        jd_words = set(job_desc_text.split())
        union = resume_words.union(jd_words)
        if not union:
            return 0.0
        match_percentage = (len(resume_words.intersection(jd_words)) / len(union)) * 100
    
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
    resume_text = normalize_synonyms(resume_text)
    job_desc_text = normalize_synonyms(job_desc_text)
    resume_words = set(resume_text.split())

    if SKLEARN_AVAILABLE:
        tfidf_jd = TfidfVectorizer(max_features=max(20, top_n))
        tfidf_jd.fit([job_desc_text])
        feature_names = tfidf_jd.get_feature_names_out()
        missing_keywords = [word for word in feature_names if word not in resume_words]
        return missing_keywords[:top_n]

    jd_tokens = [token for token in job_desc_text.split() if len(token) > 2]
    token_counts = Counter(jd_tokens)
    ranked_tokens = [token for token, _ in token_counts.most_common(top_n * 3)]
    missing_keywords = [token for token in ranked_tokens if token not in resume_words]
    return missing_keywords[:top_n]

def generate_cover_letter_llm(resume_text, job_desc_text, tone):
    """
    Generates a personalized cover letter using Mistral LLM.
    Uses original resume and job description text.
    Tone is user-controlled .
    Automatically adapts to detected language.
    """

    # Basic validation like other functions
    if not resume_text or not job_desc_text:
        return "Insufficient information to generate cover letter."

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        return "Mistral API key not found. Please set MISTRAL_API_KEY environment variable."

    try:
        client = Mistral(api_key=api_key)

        prompt = f"""
You are a professional HR assistant.

Detect the language of the resume and job description.
Write the cover letter in the SAME language.
Write a COMPLETE personalized cover letter.
Use the resume content to infer skills and experience.
Write full paragraphs, not a template.
write the letter in tone provided by user.

Guidelines:
- Tone: {tone}
- Length: 300-400 words
- Avoid repetition
- Highlight matching skills
- Keep professional structure

Resume:
{resume_text}

Job Description:
{job_desc_text}
"""
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content.strip()

        

    except Exception as e:
        return f"Error generating cover letter: {str(e)}"
