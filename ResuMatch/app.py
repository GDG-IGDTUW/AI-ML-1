import streamlit as st
#modify import
from utils import extract_text_from_pdf, extract_text_from_docx, clean_text, calculate_similarity, get_missing_keywords
from langdetect import detect


# Set page configuration
st.set_page_config(
    page_title="ResuMatch",
    page_icon="📄",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("📄 ResuMatch")
    st.subheader("Smart Resume to Job Description Matcher")
    st.markdown("Upload your resume and paste a job description to see how well you match!")

    # Layout: Two main sections
    col1, col2 = st.columns(2)

    with col1:
        st.info("Step 1: Upload Resume")

        # Update uploader UI
        uploaded_file = st.file_uploader(
            "📂 Drag & Drop your Resume here (PDF or DOCX)",
            type=["pdf", "docx"],
            help="You can drag and drop your resume file or browse from your computer."
            )
        
        #Show file info preview
        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")
            st.caption(f"File type: {uploaded_file.type}")


    with col2:
        st.info("Step 2: Job Description")
        job_description = st.text_area("Paste the Job Description here", height=200)

    # Analysis Button
    if st.button("Analyze Resume"):
        if uploaded_file is not None and job_description:
            with st.spinner("Analyzing..."):
                # 1. Extract Text from PDF
                if uploaded_file.name.endswith(".pdf"):
                     resume_text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.name.endswith(".docx"):
                     resume_text = extract_text_from_docx(uploaded_file)
                else:
                     st.error("Unsupported file format.")
                     return

                
                if not resume_text:
                    st.error("Could not extract text from the file. Please try another file.")
                    return
                
                try:
                    lang_code = detect(resume_text)
                    st.caption(f"Detected Language: **{lang_code.upper()}**")
                except:
                    pass

                # 2. Preprocess Texts
                cleaned_resume = clean_text(resume_text)
                cleaned_jd = clean_text(job_description)

                # 3. Calculate Similarity
                match_percentage = calculate_similarity(cleaned_resume, cleaned_jd)
                
                # 4. Find Missing Keywords
                missing_keywords = get_missing_keywords(cleaned_resume, cleaned_jd)

                # Display Results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Result Columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    # Circular progress bar for match percentage (using markdown/HTML trick or just metric)
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Match Score</h3>
                            <h1 style="color: {'#4CAF50' if match_percentage >= 70 else '#FFC107' if match_percentage >= 40 else '#F44336'};">
                                {match_percentage}%
                            </h1>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if match_percentage >= 70:
                        st.success("Great match! Your resume aligns well with this job.")
                    elif match_percentage >= 40:
                        st.warning("Good start, but you might want to tailor your resume further.")
                    else:
                        st.error("Low match. Consider adding more relevant keywords.")

                with res_col2:
                    st.subheader("🔍 Missing Keywords")
                    if missing_keywords:
                        st.write("Consider adding these keywords from the job description to your resume:")
                        # Display keywords as tags
                        for keyword in missing_keywords:
                            st.caption(f"• **{keyword.capitalize()}**")
                    else:
                        st.write("✅ No critical missing keywords found!")
                        
                # Optional: Show extracted text preview (collapsed)
                with st.expander("View Processed Data"):
                    st.write("**Processed Resume Text:**")
                    st.write(cleaned_resume[:500] + "...")
                    st.write("**Processed JD Text:**")
                    st.write(cleaned_jd[:500] + "...")

        else:
            st.warning("Please upload a resume and paste a job description to proceed.")

if __name__ == "__main__":
    main()
