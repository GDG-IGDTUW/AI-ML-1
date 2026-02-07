import streamlit as st
import joblib
import pandas as pd
import os

# Load the trained model
MODEL_PATH = 'book_genre_model.pkl'
CONFIDENCE_THRESHOLD = 0.40  # 40%

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

def log_low_confidence(title, predicted_genre, confidence):
    log_data = pd.DataFrame([{
        "Title": title,
        "Predicted Genre": predicted_genre,
        "Confidence": confidence
    }])

    log_file = "low_confidence_logs.csv"

    if os.path.exists(log_file):
        log_data.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_data.to_csv(log_file, index=False)


def main():
    st.set_page_config(page_title="Mythos AI - Book Genre Predictor", page_icon="📚")

    st.title("Book Genre Predictor")
    st.markdown("""
    This app predicts the genre of a book based on its title using a Naive Bayes model trained on Amazon book data.
    """)
    if model is not None:
        st.success("✅ Model loaded successfully")


    if model is None:
        st.error("🚫 Model not found")

        st.markdown("""
    The genre prediction model is not available yet.

    **Why am I seeing this?**
    - The trained model file (`book_genre_model.pkl`) has not been generated
    - or it is missing from the project directory

    **How to fix this:**
    1. Open your terminal
    2. Navigate to the project folder
    3. Run the training script:

    ```bash
    python model_training.py
    ```

    Once training is complete, restart this app.
    """)

        st.info("ℹ️ This step is required only once unless the model file is deleted.")
        return


    # User input
    title_input = st.text_input("Enter Book Title:", placeholder="e.g. Harry Potter and the Sorcerer's Stone")

    if st.button("Predict Genre", disabled=(model is None)):
        if title_input.strip():
            # Predict
            try:
                prediction = model.predict([title_input])[0]
                probabilities = model.predict_proba([title_input])[0]
                
                # Get class labels
                classes = model.classes_

                max_prob = probabilities.max()
                predicted_genre = classes[probabilities.argmax()]
                
                # Create a DataFrame for probabilities
                prob_df = pd.DataFrame({
                    'Genre': classes,
                    'Probability': probabilities
                }).sort_values(by='Probability', ascending=False)

                st.success(f"**Predicted Genre:** {predicted_genre}")

                if max_prob < CONFIDENCE_THRESHOLD:
                    st.warning(
                        "⚠️ **Low Confidence Prediction**\n\n"
                        "The model is not very confident about this prediction. "
                        "This may happen if the book title is short, ambiguous, or uncommon."
                    )
                    st.info(
                        "ℹ️ **Why am I seeing this warning?**\n\n"
                        "This model predicts genres based only on book titles. "
                        "Some titles do not contain enough information to confidently determine a genre."
                    )
                    log_low_confidence(title_input, predicted_genre, max_prob)
                
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
