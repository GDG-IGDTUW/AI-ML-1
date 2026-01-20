import streamlit as st
import pickle
import re
from pathlib import Path
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------- CONFIG ----------
MODEL_PKL = "emotion_model.pkl"
LOGO_PATH = "hero_icon.png" 

EMOJI_MAP = {
    "Joy": "😊",
    "Sadness": "😢",
    "Anger": "😠",
    "Fear": "😨",
    "Love": "❤️",
    "Surprise": "😲",
}

# --- FINAL PATH CORRECTION FOR 'emotions/' FOLDER ---
LOCAL_EMOTION_FILES = {
    "Joy": ["emotions/joy.jpg"],
    "Sadness": ["emotions/sad.jpg"],
    "Anger": ["emotions/angry.jpg"], 
    "Fear": ["emotions/fear.jpg"],
    "Love": ["emotions/love.jpg"],
    "Surprise": ["emotions/surprise.jpg"],
}

WELLBEING_EXERCISES = {
    "Joy": [
        "Share your joy with someone (call/message) - 30s",
        "Write 3 quick things you're grateful for - 2 to 3 min",
        "Savor: close your eyes and notice details for 60s",
        "Do one tiny celebratory action (song, stretch, treat)",
        # New additions
        "Create a 30-second 'joy snapshot': write what happened, where you were, and how you felt.",
        "Turn your happiness into motion — dance freely to one song.",
        "Plan one small future treat that you can look forward to this week."
    ],
    "Sadness": [
        "Take a grounding walk for 6 to 10 minutes and notice senses",
        "Free-write for 10 minutes - no edits, just feelings",
        "Send a short note to a trusted person",
        "Try a soothing routine: warm drink + calming music for 10 to 15 min",
        # New additions
        "Create a 30-second 'joy snapshot': write what happened, where you were, and how you felt.",
        "Turn your happiness into motion — dance freely to one song.",
        "Plan one small future treat that you can look forward to this week."

    ],
    "Anger": [
        "Box breathing: 4s inhale, 4s hold, 4s exhale, 4s hold - 5 rounds",
        "Move vigorously for 3 minutes (marching, stairs)",
        "Write it out, then safely discard or delete",
        "Step away from the trigger for 10 minutes and breathe",
        # New additions
        "Write an uncensored rant for 2 minutes — do not reread it.",
        "Channel the energy into a fast cleaning or organizing burst (5 minutes).",
        "Ask yourself: what boundary was crossed and how can I protect it next time?"


    ],
    "Fear": [
        "5-4-3-2-1 grounding: identify senses in order",
        "Write one concrete 5-minute step you can do next",
        "Soothing breaths: inhale 4s, exhale 6s for 6 rounds",
        "Separate facts from worries - list what you know vs fear",
        # New additions
        "Recall a past moment when you handled something difficult successfully.",
        "Press your feet firmly into the floor and notice stability for 60 seconds.",
        "Convert the fear into a question you can research or clarify."


    ],
    "Love": [
        "Send a short appreciative message to someone you care about",
        "Write 3 specific reasons you value this relationship",
        "Schedule a short call or shared micro-activity",
        "Express creatively: quick sketch, voice note or short poem",
        # New additions
        "Create a short appreciation list for someone without sending it — just reflect.",
        "Revisit a meaningful memory with someone you love and relive the details.",
        "Offer help or support to someone today without being asked."


    ],
    "Surprise": [
        "Pause and breathe for 30s, notice body sensations",
        "Journal 2 to 3 lines: pleasant or unsettling? why?",
        "If action needed, choose one tiny next move (2 min)",
        "If positive, share it; if unsettling, keep details to review later",
        # New additions
        "Ask: what did this moment teach me that I didn’t expect?",
        "Notice how your assumptions were challenged by this event.",
        "Describe the surprise using only three words."

    ],
}

# ---------- STYLING ----------
st.set_page_config(page_title="VibeLift — Unlocking the Invisible Influence of Emotion", layout="centered")

BASE_CSS = """
<style>
/* 🚨 FIX: Ensure the gradient covers the primary Streamlit container */
.stApp { 
  background: linear-gradient(135deg, #ffd3e0 0%, #d7e8ff 100%);
}

/* page gradient pastel pink -> pastel blue */
html, body { height:100%; }
body {
  color: #0f172a;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* main centered card */
.main-card {
  background: rgba(255,255,255,0.95);
  padding: 22px 28px;
  border-radius: 14px;
  box-shadow: 0 12px 40px rgba(2,6,23,0.15);
  max-width: 920px;
  margin: 24px auto;
  /* Ensure it doesn't collapse too much */
  min-height: 250px; 
}

/* header */
.logo-img { border-radius: 12px; }
.title {
  margin: 0;
  padding: 0;
  font-size: 1.8rem;
  font-weight: 800;
}
.tagline {
  color: #334155;
  margin-top: 4px;
  font-size: 1.0rem;
}

/* emotion badge */
.emotion-badge { font-size: 2.8rem; margin-right: 14px; }

/* confidence pill */
.pill {
  display:inline-block;
  padding:6px 12px;
  border-radius:999px;
  background: linear-gradient(90deg,#fff7ed,#ecfeff);
  color:#0f172a;
  font-weight:700;
  border: 1px solid rgba(2,6,23,0.05);
  margin-top:6px;
}

/* suggestion boxes */
.suggestion {
  padding:12px 14px;
  border-radius:10px;
  background:#fbfdff;
  /* FIX: Increased bottom margin for better separation */
  margin-bottom:50px; 
  box-shadow: 0 6px 20px rgba(2,6,23,0.05);
}

.muted { color:#475569; font-size:0.95rem; }

/* 🚨 FINAL FIX: Aggressive element collapsing and margin reset 🚨 */

/* Hide Streamlit's internal header element */
.stApp > header {
    visibility: hidden;
    height: 0px !important;
    padding: 0px !important;
}

/* Aggressively collapse the initial top vertical block that contains empty space */
div[data-testid="stVerticalBlock"]:first-child {
    margin-top: -30px !important; 
    padding-top: 0px !important;
}
div[data-testid="stVerticalBlock"] > div:first-child:empty {
    height: 0px !important;
    overflow: hidden;
}

/* Beautified Buttons */
.stButton>button {
    font-weight: 700;
    transition: all 0.2s ease;
    border-radius: 8px;
    padding: 8px 20px;
    white-space: nowrap; 
    /* Add margin to separate buttons from text area */
    margin-top: 30px; 
}

/* Primary Detect Emotion button style */
.stButton>button[kind="primary"] {
    background: linear-gradient(90deg, #ff94c4 0%, #a8c1ff 100%) !important;
    color: #0f172a !important; 
    border: none !important;
    box-shadow: 0 4px 10px rgba(255,100,150,0.3);
}

.stButton>button[kind="primary"]:hover {
    box-shadow: 0 6px 15px rgba(255,100,150,0.4);
    transform: translateY(-1px);
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------- UTILITIES ----------
@st.cache_data
def ensure_nltk_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords")
    s = set(stopwords.words("english"))
    if "not" in s:
        s.remove("not")
    return s

STOP_WORDS = ensure_nltk_stopwords()
PS = PorterStemmer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z]", " ", text)
    toks = text.lower().split()
    toks = [PS.stem(w) for w in toks if w not in STOP_WORDS]
    return " ".join(toks)

@st.cache_resource
def load_model(pkl_path: str):
    p = Path(pkl_path)
    if not p.exists():
        return None, None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "model" in data and "vectorizer" in data:
            return data["model"], data["vectorizer"]
        else:
            st.error("Model file found, but it is not a dictionary containing 'model' and 'vectorizer'. Please re-run train_model.py.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model pickle '{pkl_path}'. **The model is likely incompatible with your environment.** Please re-run `train_model.py`. Details: {e}")
        return None, None

def find_local_emotion_images(emotion: str):
    imgs = []
    candidates = LOCAL_EMOTION_FILES.get(emotion, [])
    
    found_specific = False
    for fname in candidates:
        p = Path(fname)
        if p.exists():
            imgs.append(str(p))
            found_specific = True
            break
    
    if not found_specific and Path(LOGO_PATH).exists():
        return [str(LOGO_PATH)] 
    
    return imgs

def display_local_image(src, st_col):
    try:
        p = Path(src)
        if p.exists():
            im = Image.open(p)
            st_col.image(im, use_container_width=True) 
        else:
            st_col.write(f"Image not found at path: {src}")
    except Exception:
        st_col.write("Image could not be displayed.")

# ---------- HEADER DISPLAY ----------
# st.markdown("<div class='main-card'>", unsafe_allow_html=True) # Start the main card container
def show_header():
    cols = st.columns([1, 6])
    with cols[0]:
        try:
            if Path(LOGO_PATH).exists():
                img = Image.open(LOGO_PATH)
                st.image(img, caption=None, use_container_width=True)
        except Exception:
            pass
    with cols[1]:
        st.markdown(f"<div class='title'>VibeLift</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='tagline'>Unlocking the Invisible Influence of Emotion</div>", unsafe_allow_html=True)
        
show_header()


# ---------- APP MAIN LOGIC ----------
model_and_vect = load_model(MODEL_PKL)
if model_and_vect[0] is None or model_and_vect[1] is None:
    st.warning("Model is unavailable. Please ensure `emotion_model.pkl` is correctly created by running `train_model.py`.")
    st.stop() 

# --- Input Area ---
st.markdown("### 🌸 Take a mindful pause: how does this moment feel?")
user_text = st.text_area(
    "Enter your feelings here:", 
    height=200, 
    placeholder="e.g. I'm excited about a new project / I'm anxious about an interview.",
    label_visibility="collapsed",
    key="user_input_text"
)

# Use Streamlit state to handle button presses and ensure flow is correct
if 'detect_clicked' not in st.session_state:
    st.session_state['detect_clicked'] = False
if 'final_pred' not in st.session_state:
    st.session_state['final_pred'] = None
if 'proba' not in st.session_state:
    st.session_state['proba'] = 0.0

# --- Button Area ---
btn_cols = st.columns([2, 1, 6]) 

with btn_cols[0]:
    detect_button = st.button("Detect Emotion", key="detect_button_main", type="primary")

with btn_cols[1]:
    reset_button = st.button("Reset", key="reset_initial")
    
if detect_button:
    st.session_state['detect_clicked'] = True
    
    if not user_text or user_text.strip() == "":
        st.error("Please enter some text describing how you feel.")
        st.session_state['detect_clicked'] = False
    else:
        # --- Prediction Logic ---
        clf, vect = model_and_vect
        cleaned = clean_text(user_text)
        
        pred = "Unknown"
        proba = 0.0
        
        if not cleaned:
            st.error("The input was empty after cleaning and stemming. Try using more descriptive words.")
        else:
            try:
                X = vect.transform([cleaned])
                
                if X.shape[1] == 0:
                    st.error("Vectorizer failed to create features (all words were stop words or unknown).")
                else:
                    raw_pred = clf.predict(X)[0] 
                    proba = float(clf.predict_proba(X).max())
                    
                    # Capitalize prediction to match EMOJI_MAP keys
                    pred = raw_pred.capitalize() 

            except Exception as e:
                st.error(f"Prediction failed! The model is incompatible or corrupt. Error: {e}")

        # Update Session State for Display
        if pred in EMOJI_MAP:
            st.session_state['final_pred'] = pred
        else:
            st.session_state['final_pred'] = "Unknown"
        st.session_state['proba'] = proba

if reset_button:
    st.session_state['detect_clicked'] = False
    st.session_state['final_pred'] = None
    st.session_state['proba'] = 0.0
    st.rerun()


# --- Display Results ---
if st.session_state['detect_clicked'] and st.session_state['final_pred'] is not None:
    final_pred = st.session_state['final_pred']
    proba = st.session_state['proba']
    emoji = EMOJI_MAP.get(final_pred, "🧠")

    st.markdown("---")
    st.markdown(
        f"<div style='display:flex; align-items:center;'>"
        f"<div class='emotion-badge'>{emoji}</div>"
        f"<div><h2 style='margin:0'>{final_pred}</h2>"
        f"<div class='muted'>Confidence: <span class='pill'>{proba:.0%}</span></div></div></div>",
        unsafe_allow_html=True
    )

    imgs = find_local_emotion_images(final_pred)
    if imgs:
        cols = st.columns(min(3, len(imgs))) 
        for i, src in enumerate(imgs[:3]): 
            display_local_image(src, cols[i])
    else:
        st.markdown("<div class='muted'>No local images found for this emotion.</div>", unsafe_allow_html=True)

    st.markdown("#### Tailored suggestions")
    for s in WELLBEING_EXERCISES.get(final_pred, ["Take a breath and be kind to yourself."]):
        st.markdown(f"<div class='suggestion'>{s}</div>", unsafe_allow_html=True)

    # CTA - Only show the crisis message
    c1, c2 = st.columns([1,2])
    with c2: 
        st.markdown("<div class='muted' style='text-align:right'>If you're in crisis, please reach out to local emergency services or a trusted person.</div>", unsafe_allow_html=True)

# st.markdown("</div>", unsafe_allow_html=True) 
st.markdown("<div style='text-align:center; margin-top:14px;' class='muted'>VibeLift - Built with care</div>", unsafe_allow_html=True)