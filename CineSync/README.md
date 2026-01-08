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

## 🤝 Contributing

Contributions are welcome! Feel free to:
-   Add more movies to the sample dataset.
-   Improve the UI/UX.
-   Add features like poster images (using an API like TMDB).
