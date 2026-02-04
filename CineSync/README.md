# CineSync 🎬

CineSync is a beginner-friendly, content-based movie recommendation engine. It suggests movies based on plot similarity and genres using NLP techniques (TF-IDF and Cosine Similarity).

## 🚀 Features

-   **Content-Based Filtering**: Suggests movies similar to the ones you like.
-   **Clean UI**: Built with Streamlit for a simple and responsive interface.
-   **Explainable**: Shows the genres and plot of recommended movies so you know why they were picked.

## 📂 Project Structure

```
CineSync/
├── app.py              # The main Streamlit application
├── utils.py            # Helper functions for data loading, processing, and ML
├── data/
│   └── movies_sample.csv  # Sample dataset to get started
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🛠️ Installation & Setup

1.  **Clone the repository** (if applicable) or navigate to the project folder.

2.  **Install Dependencies**:
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

4.  **Open in Browser**:
    The app should automatically open in your browser at `http://localhost:8501`.

## 📊 Dataset

This project comes with a small sample dataset (`data/movies_sample.csv`) containing a few popular movies for demonstration purposes.

**To use a larger dataset (e.g., full IMDB):**

1.  Download a movie dataset from [Kaggle](https://www.kaggle.com/datasets/staphibmn/news-movie-dataset) or [IMDb](https://www.imdb.com/interfaces/).
2.  The dataset must be a CSV file with at least these columns: `title`, `genres`, `plot`.
3.  Save the new file in the `data/` folder (e.g., `data/movies_full.csv`).
4.  Update line 44 in `app.py` to point to your new file:
    ```python
    data_path = 'data/movies_full.csv'
    ```

## 🧠 How It Works

1.  **Preprocessing**: The app combines movie genres and plot summaries into a single "tag" for every movie.
2.  **Vectorization**: It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text data into numerical vectors. This highlights unique words in plot descriptions.
3.  **Similarity**: It calculates **Cosine Similarity** between the selected movie and all others in the database.
4.  **Ranking**: Movies with the highest similarity scores are recommended.

## 🚀 Deploying CineSync on Heroku

You can deploy CineSync as a live web application using **Heroku** in just a few steps.

### Prerequisites
- A GitHub account
- A Heroku account (https://signup.heroku.com/)
- Git installed on your system

### Step 1: Fork the Repository
Click the **Fork** button on GitHub to create your own copy of the repository.

### Step 2: Clone Your Fork
```bash
git clone https://github.com/your-username/CineSync.git
cd CineSync

### Step 3: Create a Procfile

Create a file named Procfile (no extension) in the project root:

-> web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0

### Step 4: Specify Python Version

Create a file named runtime.txt:

python-3.10.12

### Step 5: Install Heroku CLI & Login

Install the CLI from:
https://devcenter.heroku.com/articles/heroku-cli

Then login:

heroku login

### Step 6: Create a Heroku App
heroku create cinesync-app

### Step 7: Deploy to Heroku
git push heroku main

### Step 8: Open the App
heroku open


🎉 CineSync is now live on the web!

### ⚠️ Deployment Note

Heroku and Streamlit Cloud are recommended for deploying Streamlit apps.
Platforms like Vercel do not natively support Streamlit applications.

## 🤝 Contributing

Contributions are welcome! Feel free to:
-   Add more movies to the sample dataset.
-   Improve the UI/UX.
-   Add features like poster images (using an API like TMDB).
