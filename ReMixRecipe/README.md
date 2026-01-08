# 🍳 ReMixRecipe

ReMixRecipe is a simple, beginner-friendly web application that predicts cuisines based on the ingredients you have. It uses a basic Machine Learning model to analyze your leftovers and suggest whether you should cook Italian, Mexican, Indian, or something else!

## 🚀 Features

- **Ingredient Input**: Simply type in what you have (e.g., "tomato, onion, garlic").
- **AI-Powered Suggestions**: Uses a Multi-label classification approach (Logistic Regression) to find the best matching cuisine.
- **Beautiful UI**: Built with Streamlit, featuring a modern, glassmorphic design.
- **Open Source**: Clean code structure, easy to understand and extend.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend/ML**: Python, [Scikit-learn](https://scikit-learn.org/), Pandas

## 📦 How to Run

1. **Clone the repository** (if applicable) or download the source code.
2. **Install Dependencies**:
   Make sure you have Python installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the App**:
   ```bash
   streamlit run app.py
   ```
4. **Enjoy**: The app will open in your browser at `http://localhost:8501`.

## 📂 Project Structure

```
ReMixRecipe/
├── app.py              # Main application file (Streamlit UI)
├── model.py            # AI Logic (Training & Prediction)
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── data/
    └── recipes.csv     # Sample dataset for training
```

## 🤝 Contributing

This project is designed for beginners! Feel free to:
- Add more recipes to `data/recipes.csv`.
- Improve the ML model (try a Random Forest!).
- Enhance the UI.

## 📜 License

MIT License. Free to use and remix!
