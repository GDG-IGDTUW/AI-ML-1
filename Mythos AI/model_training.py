import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv('Amazon_Books_Scraping/Books_df.csv')
    except FileNotFoundError:
        print("Error: File not found. Make sure 'Amazon_Books_Scraping/Books_df.csv' exists.")
        return

    # Inspect columns
    print(f"Columns: {df.columns}")
    
    # We need 'Title' and 'Main Genre'
    if 'Title' not in df.columns or 'Main Genre' not in df.columns:
        print("Error: Dataset must contain 'Title' and 'Main Genre' columns.")
        return

    # Drop missing values
    df = df.dropna(subset=['Title', 'Main Genre'])
    
    # Simple cleaning (optional but good)
    df['Title'] = df['Title'].astype(str)
    df['Main Genre'] = df['Main Genre'].astype(str)

    print(f"Data shape after cleaning: {df.shape}")
    print(f"Number of unique genres: {df['Main Genre'].nunique()}")
    print(df['Main Genre'].value_counts())

    # Split data
    X = df['Title']
    y = df['Main Genre']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create Pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])

    # Train model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    # print(classification_report(y_test, y_pred)) # Can be verbose with many genres

    # Save model
    print("Saving model to 'book_genre_model.pkl'...")
    joblib.dump(pipeline, 'book_genre_model.pkl')
    print("Done!")

if __name__ == "__main__":
    train_model()
