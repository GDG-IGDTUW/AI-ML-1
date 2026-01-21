import streamlit as st
from utils import calculate_similarity, read_file_text

# Set page configuration to wide mode and add a title
st.set_page_config(
    page_title="Text Similarity Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #ced4da;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        width: 100%;
        height: 50px;
        font-size: 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Compare AI")
    st.markdown("Check how similar two files are using AI-powered Cosine Similarity.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original File")
        file1 = st.file_uploader("Upload original file (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"], key="file1")

    with col2:
        st.subheader("Suspected File")
        file2 = st.file_uploader("Upload suspected file (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"],key="file2")

    
    if st.button("Check Similarity"):
        if file1 and file2:
            with st.spinner("Calculating similarity..."):
                text1 = read_file_text(file1)
                text2 = read_file_text(file2)
                
                similarity_score = calculate_similarity(text1, text2)
                percentage = round(similarity_score * 100, 2)
                
                # Dynamic color based on similarity
                if percentage > 80:
                    color = "#ff4b4b" # Red for high similarity
                    msg = "⚠️ High Similarity Detected!"
                elif percentage > 40:
                    color = "#ffa500" # Orange for moderate
                    msg = "⚠️ Moderate Similarity."
                else:
                    color = "#4CAF50" # Green for low
                    msg = "✅ Low Similarity."
                
                st.markdown(f"""
                <div class="result-box" style="background-color: {color}20; border: 2px solid {color};">
                    <h2 style="color: {color};">{msg}</h2>
                    <h1 style="color: {color}; font-size: 60px;">{percentage}%</h1>
                    <p>Match Score</p>
                </div>
                """, unsafe_allow_html=True)
                
                if percentage > 0:
                    st.success("Analysis Complete.")
        else:
            st.warning("Please upload both documents to compare.")

if __name__ == "__main__":
    main()



