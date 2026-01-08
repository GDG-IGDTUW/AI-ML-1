# Book Genre Predictor AI

A machine learning project that predicts the **main genre of a book** using only its **title**.  
The model is trained on an Amazon Books dataset and applies **Natural Language Processing (NLP)** techniques with a **Naive Bayes classifier**.

---

##  Features

- Predicts book genres from titles
- NLP-based text classification
- Uses `CountVectorizer` for feature extraction
- Multinomial Naive Bayes classifier
- End-to-end ML pipeline
- Saves trained model for later use

---

##  How It Works

1. Loads the dataset from a CSV file
2. Cleans and preprocesses the data
3. Converts book titles into numerical features
4. Trains a Naive Bayes model
5. Evaluates accuracy on test data
6. Saves the trained model as a `.pkl` file

---

##  Project Structure


├── Amazon_Books_Scraping/
│ └── Books_df.csv
├── train_model.py
├── book_genre_model.pkl
└── README.md

---

## Dataset

This project uses a dataset sourced from Kaggle.

**License:** MIT License  
**Copyright:**
- © 2013 Mark Otto  
- © 2017 Andrew Fong  

The dataset is redistributed in accordance with the MIT License.

---

## Model Details
Algorithm: Multinomial Naive Bayes

Vectorization: CountVectorizer (English stop words removed)

Train/Test Split: 80% / 20%

Evaluation Metric: Accuracy
