import streamlit as st
from utils import calculate_similarity, read_file_text, find_common_sentences

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
        st.subheader("Suspected Texts (Multiple)")
        multi_text = st.text_area(
        "Enter multiple texts (one per line)...",
        height=300,
        key="multi_text"
    )

        
    if st.button("Check Similarity"):
        if text1 and multi_text:
            texts_to_compare = [t.strip() for t in multi_text.split("\n") if t.strip()]

            with st.spinner("Calculating similarity..."):
                results = []
                for idx, text in enumerate(texts_to_compare):
                     score = calculate_similarity(text1, text)
                     percentage = round(score * 100, 2)
                     results.append((idx + 1, percentage))

                
                st.subheader("Similarity Results")
                for idx, percentage in results:
                     if percentage > 80:
                          color = "#ff4b4b"
                          msg = "⚠️ High Similarity Detected!"
                     elif percentage > 40:
                          color = "#ffa500"
                          msg = "⚠️ Moderate Similarity."
                     else:
                          color = "#4CAF50"
                          msg = "✅ Low Similarity."
                          
                     st.markdown(f"""
                                 <div class="result-box" style="background-color: {color}20; border: 2px solid {color};">
                                 <h3 style="color: {color};">Text {idx}</h3>
        <h2 style="color: {color};">{msg}</h2>
        <h1 style="color: {color}; font-size: 40px;">{percentage}%</h1>
        <p>Match Score</p>
    </div>
    """, unsafe_allow_html=True)

                
                if percentage > 0:
                    st.success("Analysis Complete.")

                    if common_sentences:
                        st.subheader("🔎 Overlapping Text Segments")
                    for sentence in common_sentences:
                        st.markdown(
                            f"<div style='background-color:#fff3cd; padding:8px; margin-bottom:6px; border-radius:6px;'>"
                            f"{sentence}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.info("No exact overlapping sentences found.")

        else:
            st.warning("Please enter the reference text and at least one comparison text.")

if __name__ == "__main__":
    main()




