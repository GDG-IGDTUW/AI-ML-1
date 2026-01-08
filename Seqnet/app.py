import streamlit as st
import model as m
import utils as u
import os
import tensorflow as tf

st.set_page_config(page_title="AI Line Completer", page_icon="✍️", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("✍️ AI Line Completer")
st.subheader("Predict the next words in your sentence!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data.txt")
MODEL_FILE = os.path.join(BASE_DIR, "model.h5")
TOKENIZER_FILE = os.path.join(BASE_DIR, "tokenizer.pickle")

# Sidebar for controls
st.sidebar.header("Settings")
next_words = st.sidebar.slider("Number of words to generate", 1, 50, 5)
train_epochs = st.sidebar.number_input("Training Epochs", min_value=1, value=50)

@st.cache_resource
def load_resources_v2():
    # Debug info
    print(f"Loading from: {TOKENIZER_FILE}")
    print(f"Loading from: {MODEL_FILE}")
    
    tokenizer = u.load_tokenizer(TOKENIZER_FILE)
    model = m.load_saved_model(MODEL_FILE)
    return tokenizer, model

tokenizer, model = load_resources_v2()

# Debugging display (only if fails)
if not tokenizer or not model:
    st.error(f"Failed to load. Checking paths:")
    st.write(f"Model Path: `{MODEL_FILE}`")
    st.write(f"Exists: `{os.path.exists(MODEL_FILE)}`")
    st.write(f"Tokenizer Path: `{TOKENIZER_FILE}`")
    st.write(f"Exists: `{os.path.exists(TOKENIZER_FILE)}`")

if not tokenizer or not model:
    st.warning("Model not found or needs training.")
    if st.button("Train Model"):
        with st.spinner("Training model... This may take a while based on data size and epochs."):
            text = u.load_data(DATA_FILE)
            tokenizer = u.create_tokenizer(text)
            predictors, label, max_sequence_len = u.create_sequences(tokenizer, text, 0) # Max len calc inside
            
            # Recalculate max_sequence_len properly
            max_sequence_len = predictors.shape[1] + 1
            total_words = len(tokenizer.word_index) + 1
            
            model = m.create_model(total_words, max_sequence_len)
            m.train_model(model, predictors, label, epochs=train_epochs)
            
            m.save_model(model, MODEL_FILE)
            u.save_tokenizer(tokenizer, TOKENIZER_FILE)
            
            st.success("Model trained and saved successfully! Reloading...")
            st.rerun()
else:
    st.success("Model ready!")
    
    # Needs max_sequence_len for prediction
    # Ideally should be saved with model metadata, but for now we can infer or pass it
    # We will compute it from input length logic in utils, but we need the training max len
    # For simplicity, we assume the model input shape
    if model:
        max_sequence_len = model.layers[0].input_length + 1 if hasattr(model.layers[0], 'input_length') else 20
        # If input_length is None (newer keras), we might need to store it. 
        # Let's try to get it from model input shape
        try:
             max_sequence_len = model.input_shape[1] + 1
        except:
             max_sequence_len = 20 # Fallback
             
    input_text = st.text_input("Enter your partial sentence:", "Sherlock Holmes was")

    if st.button("Complete Sentence"):
        if input_text:
            try:
                completed_text = m.generate_text(input_text, next_words, model, max_sequence_len, tokenizer)
                st.markdown(f"**Completed Text:**")
                st.info(completed_text)
            except Exception as e:
                st.error(f"Error generating text: {e}")
        else:
            st.warning("Please enter some text.")

st.markdown("---")
st.caption("Based on LSTM architecture. Data source: Sherlock Holmes (or custom provided).")
