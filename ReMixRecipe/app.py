import streamlit as st
import pandas as pd
from model import CuisinePredictor
import os

# Page Config
st.set_page_config(
    page_title="ReMixRecipe",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Rich Aesthetics"
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ffcc00;
        box-shadow: 0 0 10px rgba(255, 204, 0, 0.5);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #ff8c00 0%, #ff0055 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 50px;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 0, 85, 0.4);
    }
    
    /* Cards */
    .css-1r6slb0, .css-12oz5g7 { 
        /* Streamlit generic container adjustments if needed */
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3 {
        color: #ffcc00;
    }
</style>
""", unsafe_allow_html=True)

# Application Logic
def main():
    st.title("🍳 ReMixRecipe")
    st.markdown("### Turn your leftovers into a masterpiece!")
    st.markdown("Enter the ingredients you have, and our AI will guess the best cuisine for you.")

    # Initialize Model
    @st.cache_resource
    def load_model():
        predictor = CuisinePredictor()
        # Train on the dummy data
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'recipes.csv')
        predictor.train(data_path)
        return predictor

    try:
        predictor = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Input Section
    st.markdown("---")
    ingredients_input = st.text_area("What's in your fridge? (comma separated)", 
                                     placeholder="e.g. tomato, cheese, basil, garlic")

    if st.button("Find Cuisine"):
        if ingredients_input.strip():
            with st.spinner("Analyzing flavors..."):
                try:
                    # Get predictions
                    predictions = predictor.predict(ingredients_input)
                    
                    if not predictions:
                        st.warning("No clear match found! Try adding more ingredients.")
                    else:
                        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                        st.subheader("🍽️ Recommended Cuisines")
                        
                        for cuisine, prob in predictions[:3]: # Show top 3
                            confidence = int(prob * 100)
                            st.write(f"**{cuisine.title()}** ({confidence}% match)")
                            st.progress(min(int(prob * 100), 100))
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please enter some ingredients first!")

    # Footer
    st.markdown("---")
    st.markdown("*ReMixRecipe - Open Source & Beginner Friendly*")

if __name__ == "__main__":
    main()
