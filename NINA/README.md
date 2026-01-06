# 🌐 NINA (News Intelligence Neural Analyzer) 

[![GitHub license](https://img.shields.io/github/license/ArniGoyal/NINA-News-Intelligence-Neural-Analyzer-)](./LICENSE)

NINA is a smart news verification tool that classifies articles as **real** or **fake** using state-of-the-art NLP embeddings and machine learning. By analyzing the content of news articles, it helps users identify potentially misleading information. Its accuracy is around **49%** (depending on dataset and training).  

---

## 📂 Project Structure

Ensure your directory is organized as follows for the application to function correctly:

```plaintext
NINA/
├── nina.ipynb              # The main code file
├── fake_news_dataset.csv   # Raw dataset used for training          
├── LICENSE                 # Project license

```

---

## ✨ Key Features

- **Fake News Detection:** Uses sentence embeddings to convert news articles into numerical representations and classifies them using Logistic Regression.  
- **Metadata Ready:** Can easily incorporate additional information like author, source, and category to improve accuracy.  
- **Lightweight and Fast:** Utilizes `all-MiniLM-L6-v2` embeddings for efficient text representation.  
- **Easy to Train:** Comes with scripts to preprocess, encode, and train your model on new datasets.  

---

## 🧠 How It Works  

The project follows a standard NLP + ML pipeline:

1. **Data Cleaning:** Handles missing titles/texts and combines them into a single `content` column.  
2. **Sentence Embeddings:** Converts textual content into numerical vectors using `SentenceTransformer`.  
3. **Train-Test Split:** Splits the dataset into training and test sets, ensuring balanced classes.  
4. **Classification:** Logistic Regression predicts whether an article is real (`0`) or fake (`1`).  
5. **Evaluation:** Model performance is measured using Accuracy, F1 Score, and detailed classification metrics.  

---

## 🎨 Customization  

- **Dataset:** Replace `fake_news_dataset.csv` with your own news dataset, keeping the same column structure.  
- **Model:** You can switch the classifier to other algorithms like Random Forest, XGBoost, or Neural Networks.  
- **Feature Engineering:** Add metadata features like `author`, `source`, or `category` to improve predictions.  

---

## 📄 License  

This project is licensed under the **[MIT/GPLv3] License** - see the [`LICENSE`](./LICENSE) file for details. 

---

## ⚠️ Disclaimer  

NINA is an AI-powered tool for educational purposes and awareness. It **cannot guarantee the accuracy** of news verification and should **not replace human judgment or professional fact-checking**.


