import streamlit as st
import pandas as pd
from model import CuisinePredictor, clean_user_ingredients, RecipeRecommender
import os
from deep_translator import GoogleTranslator




# Page Config
st.set_page_config(
    page_title="ReMixRecipe",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
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
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    h1, h2, h3 { color: #ffcc00; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Cached resource loaders
# -----------------------------
@st.cache_resource
def load_model():
    predictor = CuisinePredictor()
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'recipes.csv')
    predictor.train(data_path)
    return predictor

@st.cache_resource
def load_recommender():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'recipes.csv')
    return RecipeRecommender(data_path=data_path)

# -----------------------------
# Main Application
# -----------------------------
def main():
    st.title("🍳 ReMixRecipe")
    st.markdown("### Turn your leftovers into a masterpiece!")
    st.markdown("Enter the ingredients you have, and our AI will guess the best cuisine for you.")

    # Load models
    try:
        predictor = load_model()
        recommender = load_recommender()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Input Section
    st.markdown("---")
    ingredients_input = st.text_area("What's in your fridge? (comma separated)", 
                                     placeholder="e.g. tomato, cheese, basil, garlic")
    
    language = st.selectbox(
    "Choose language",
    ["English", "Hindi"]
)


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
                        title_text = "🍽️ Recommended Cuisines"
                        if language == "Hindi":
                               title_text = GoogleTranslator(source='auto', target='hi').translate(title_text)
                        st.subheader(title_text)


                        
                        for cuisine, prob in predictions[:3]: # Show top 3
                            confidence = int(prob * 100)
                            display_cuisine = cuisine.title()
                            display_label = f"{confidence}% match"
                            if language == "Hindi":
                                  display_cuisine = GoogleTranslator(source='auto', target='hi').translate(display_cuisine)

                                  display_label = GoogleTranslator(source='auto', target='hi').translate(display_label)

                            st.write(f"**{display_cuisine}** ({display_label})")

                            st.progress(min(int(prob * 100), 100))
                        
                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please enter some ingredients first!")
            return

        with st.spinner("Analyzing flavors..."):
            try:
                # Clean and correct ingredients
                corrected, ignored = clean_user_ingredients(ingredients_input)
                if ignored:
                    st.warning(f"Ignored unknown ingredients: {', '.join(ignored)}")
                if not corrected:
                    st.error("No valid ingredients detected after cleaning.")
                    return

                st.success(f"Using ingredients: {', '.join(corrected)}")

                # -----------------------------
                # Cuisine Prediction
                # -----------------------------
                cleaned_input = ", ".join(corrected)
                predictions = predictor.predict(cleaned_input)
                if predictions:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.subheader("🍽️ Recommended Cuisines")
                    for cuisine, prob in predictions[:3]:
                        confidence = int(prob * 100)
                        st.write(f"**{cuisine.title()}** ({confidence}% match)")
                        st.progress(min(confidence, 100))
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No clear cuisine match found! Try adding more ingredients.")

                # -----------------------------
                # Recipe Recommendation
                # -----------------------------
                top_recipes = recommender.recommend(corrected, top_n=5)
                if not top_recipes.empty:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.subheader("🍴 Top Recipe Matches")
                    for _, row in top_recipes.iterrows():
                        st.write(f"**{row['cuisine']}** ({int(row['score']*100)}% match)")
                        st.write(f"Ingredients: {row['ingredients']}")
                        st.markdown("---")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No matching recipes found.")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*ReMixRecipe - Open Source & Beginner Friendly*")

if __name__ == "__main__":
    main()
