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
## 📊 Data Preparation

The default data/recipes.csv included in this repository is a small, cleaned sample dataset created for demonstration and testing purposes.
It already contains:

-Lowercase ingredient text
-Comma-separated ingredient lists
-No missing values
-No duplicate entries
-Pre-formatted structure compatible with the ML model

Therefore, no additional preprocessing is required to run the application.

**🍽 Using Larger Kaggle Datasets (Optional)** 

If you would like to retrain the model using a larger dataset, we recommend:

What’s Cooking? – Recipe Classification Dataset
https://www.kaggle.com/competitions/whats-cooking

After downloading the dataset, place the raw file inside:

data/raw/

**🧼 Cleaning & Preprocessing Steps (For External Datasets)**

Raw datasets from Kaggle may contain inconsistent formatting, missing values, or measurement noise.
Before training the model, apply the following steps:

1️⃣ Remove Duplicates- Ensure no repeated recipes exist to prevent model bias.

2️⃣ Handle Missing Values- Drop rows where ingredients or cuisine are null.

3️⃣ Standardize Text
-Convert all text to lowercase
-Remove special characters and numeric measurements (e.g., "1 tbsp", "200g")
-Remove extra whitespace

4️⃣ Format Ingredients
-Ensure ingredients are stored as a single comma-separated string.
-Example required format:
-ingredients: "rice, chicken, curry powder"
-cuisine: "indian"

5️⃣ Save Cleaned Dataset

Export the cleaned dataset as:
   ```bash
   data/recipes.csv
   ```

Then retrain the model:
   ```bash
   python model.py
   ```

## 🤝 Contributing

This project is designed for beginners! Feel free to:
- Add more recipes to `data/recipes.csv`.
- Improve the ML model (try a Random Forest!).
- Enhance the UI.

## 📜 License

MIT License. Free to use and remix!
