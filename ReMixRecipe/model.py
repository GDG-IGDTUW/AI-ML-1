from fuzzywuzzy import process
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import re

class CuisinePredictor:
    def __init__(self):
        self.pipeline = None

    def preprocess(self, text):
        """
        Basic preprocessing: lowercase and remove special characters (except commas).
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Keep word characters, whitespace, and commas
        text = re.sub(r'[^\w\s,]', '', text)
        return text

    def train(self, data_path):
        """
        Loads data and trains the model.
        """
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise Exception("Data file not found!")
            
        # Ensure data exists
        if 'ingredients' not in df.columns or 'cuisine' not in df.columns:
            raise Exception("CSV must have 'ingredients' and 'cuisine' columns.")
            
        # Preprocess features
        df['ingredients_clean'] = df['ingredients'].apply(self.preprocess)
        
        X = df['ingredients_clean']
        y = df['cuisine']
        
        # Create and fit a simple pipeline
        # CountVectorizer converts text to a matrix of token counts
        # LogisticRegression is a good baseline classifier
        self.pipeline = make_pipeline(
            CountVectorizer(tokenizer=lambda x: [i.strip() for i in x.split(',')], token_pattern=None), 
            LogisticRegression(max_iter=1000)
        )
        
        self.pipeline.fit(X, y)
        return True

    def predict(self, ingredients_input):
        """
        Predicts cuisines based on input ingredients.
        Returns a list of tuples (cuisine, probability).
        """
        if not self.pipeline:
            raise Exception("Model is not trained yet!")

        # Handle input (list or string)
        if isinstance(ingredients_input, list):
            text = ",".join(ingredients_input)
        else:
            text = ingredients_input # Assume it's a comma-separated string
            
        text = self.preprocess(text)
        
        # Predict probabilities
        # [text] because predict expects an array of samples
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        
        # Combine classes with their probabilities
        results = list(zip(classes, probs))
        
        # Sort by probability descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return the top predictions (filter out very low probabilities if desired)
        # Here we return all that have at least some chance, or just top 5
        return [r for r in results if r[1] > 0.01]
    
    import pandas as pd
from fuzzywuzzy import process


def get_valid_ingredients():
    df = pd.read_csv("data/recipes.csv")
    ingredients = set()

    for item in df["ingredients"]:
        for ing in item.split(","):
            ingredients.add(ing.strip().lower())

    return list(ingredients)


def correct_ingredient(word, valid_list):
    match, score = process.extractOne(word, valid_list)

    if score >= 80:   # threshold
        return match
    else:
        return None


def clean_user_ingredients(user_input):
    valid_list = get_valid_ingredients()

    raw_items = user_input.split(",")
    corrected = []
    ignored = []

    for item in raw_items:
        word = item.strip().lower()
        corrected_word = correct_ingredient(word, valid_list)

        if corrected_word:
            corrected.append(corrected_word)
        else:
            ignored.append(word)

    return corrected, ignored



