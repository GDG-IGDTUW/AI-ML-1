import streamlit as st
import joblib
import pandas as pd
import os

# Load the trained model
MODEL_PATH = 'book_genre_model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

def main():
    st.set_page_config(page_title="Mythos AI - Book Genre Predictor", page_icon="📚")

    st.title("Book Genre Predictor")
    st.markdown("""
    This app predicts the genre of a book based on its title using a Naive Bayes model trained on Amazon book data.
    """)

    if model is None:
        st.error(f"Model file '{MODEL_PATH}' not found. Please run 'model_training.py' first.")
        return

    # User input
    title_input = st.text_input("Enter Book Title:", placeholder="e.g. Harry Potter and the Sorcerer's Stone")

    if st.button("Predict Genre"):
        if title_input.strip():
            # Predict
            try:
                prediction = model.predict([title_input])[0]
                probabilities = model.predict_proba([title_input])[0]
                
                # Get class labels
                classes = model.classes_
                
                # Create a DataFrame for probabilities
                prob_df = pd.DataFrame({
                    'Genre': classes,
                    'Probability': probabilities
                }).sort_values(by='Probability', ascending=False)

                st.success(f"**Predicted Genre:** {prediction}")
                
                with st.expander("See Prediction Probabilities"):
                    st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}))
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter a book title.")

    st.markdown("---")
    st.caption("Powered by Scikit-learn & Streamlit")

if __name__ == "__main__":
    main()
