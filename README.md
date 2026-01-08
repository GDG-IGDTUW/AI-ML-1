# ResuMatch 📄

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ResuMatch** is a simple, beginner-friendly web application designed to help job seekers optimize their resumes. By comparing a resume PDF against a job description, it provides a match percentage and highlights missing keywords to help improve compatibility with Applicant Tracking Systems (ATS).

---

## 🚀 Features

*   **PDF Support**: Easily upload your resume in PDF format.
*   **Instant Analysis**: Paste a job description and get immediate feedback.
*   **Match Score**: See a percentage score (0-100%) representing how well your resume aligns with the job.
*   **Keyword Insights**: Identify important keywords or skills from the job description that are missing from your resume.
*   **Simple Interface**: Clean, no-nonsense UI built with Streamlit.

## 📂 Project Structure

```bash
ResuMatch/
├── app.py              # The main Streamlit application
├── utils.py            # Helper functions for text processing and logic
├── requirements.txt    # List of required Python libraries
└── README.md           # Project documentation
```

## 🧠 How it Works

ResuMatch uses basic Natural Language Processing (NLP) techniques to analyze text:

1.  **Extraction**: The app extracts raw text from the uploaded PDF resume.
2.  **Cleaning**: It removes common "stop words" (like *the*, *and*, *is*) and punctuation from both the resume and the job description.
3.  **Vectorization**: It converts the text into numbers using **TF-IDF** (Term Frequency-Inverse Document Frequency). This helps identify the most "important" words.
4.  **Comparison**: It calculates the **Cosine Similarity** between the resume and job description vectors to generate a similarity score.

## 🛠️ How to Run Locally

Follow these steps to run ResuMatch on your own computer:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ResuMatch.git
cd ResuMatch
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```
The app should automatically open in your browser at `http://localhost:8501`.

## 🔮 Customization & Future Improvements

This project is a great starting point for open-source contributions! Here are some ideas for how you could improve it:

*   **Add DOCX Support**: Allow users to upload Word documents.
*   **Better text cleaning**: Use advanced NLP libraries like spaCy for better keyword extraction.
*   **Detailed Feedback**: Show exactly *where* user matched keywords appear.
*   **Visualizations**: Add charts to show skill distribution.

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software. See the [LICENSE](LICENSE) file for more details.

## ⚠️ Disclaimer

ResuMatch is a helper tool based on text similarity algorithms. It is **not** a guarantee of getting hired or passing a specific company's ATS. Always review your resume personally before submitting.
