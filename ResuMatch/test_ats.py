from utils import (
    calculate_ats_score,
    calculate_similarity,
    get_missing_keywords,
    clean_text
)

resume = """
Python developer with experience in machine learning, NLP, and Flask.
Completed internship and built 3 projects using TensorFlow and SQL.
Bachelor degree in Computer Science.
"""

job_desc = """
Looking for a Python developer with experience in ML,
deep learning, NLP, SQL, and TensorFlow. Internship experience preferred.
"""

# Clean text first
resume_clean = clean_text(resume)
jd_clean = clean_text(job_desc)

print("ATS Score:", calculate_ats_score(resume_clean, jd_clean))
print("Similarity:", calculate_similarity(resume_clean, jd_clean))
print("Missing Keywords:", get_missing_keywords(resume_clean, jd_clean))
