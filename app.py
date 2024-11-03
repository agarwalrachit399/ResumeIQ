
import streamlit as st
from rating import ResumeIQ
from infer_requirement import JobDescriptionAnalyzer
from infer_resume import final_summary
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import pi
from io import BytesIO
# Streamlit App
st.set_page_config(page_title="Resume vs Job Description Analyzer", page_icon=":bar_chart:", layout="centered")
st.title("📄 Resume vs Job Description Analyzer")

api_token = "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"
model_repo = "microsoft/Phi-3.5-mini-instruct"
analyzer = ResumeIQ(api_token, model_repo)

# Upload Resume PDF
st.sidebar.header("Upload Resume & Enter Job Description")
uploaded_file = st.sidebar.file_uploader("Upload Resume PDF", type="pdf")
job_description = st.sidebar.text_area("Paste Job Description Here")

if st.sidebar.button("Analyze") and uploaded_file and job_description:
    # Process resume and job description
    resume = final_summary(uploaded_file)
    job_desc = JobDescriptionAnalyzer(api_token, model_repo).analyze_job_description(job_description)
    
    # Get analysis result
    result = analyzer.match_resume(job_desc, resume)

    # Calculate overall score
    scores = [v['score'] for v in result.values()]
    categories = list(result.keys())
    overall_score = sum(scores) / len(scores)

    # Display overall score as a donut chart
    st.subheader("Overall Score")
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie([overall_score, 100 - overall_score], 
           labels=["", ""], 
           startangle=90, 
           colors=["#4CAF50", "#E0E0E0"],
           wedgeprops=dict(width=0.3, edgecolor='white'))
    ax.text(0, 0, f"{int(overall_score)}%", ha='center', va='center', fontsize=10, fontweight='bold', color="#4CAF50")

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
    #st.pyplot(fig)

    # Display strengths and weaknesses
    st.subheader("Strengths")
    strengths = {k: v['strength'] for k, v in result.items()}
    for field, strength in strengths.items():
        st.write(f"**{field}:** {strength}")
    
    # Radar Chart with Scores
    st.subheader("Skill Scores")
    df = pd.DataFrame(dict(r=scores, theta=categories))
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    values = scores + scores[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', color="#007acc", marker='o')
    ax.fill(angles, values, '#007acc', alpha=0.1)

    # Add score labels
    for i, score in enumerate(scores):
        angle_rad = angles[i]
        ax.text(angle_rad, values[i] - 9, f"{score}", horizontalalignment="center", size=12, color="#333333")
    
    # Aesthetics
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color="#007acc", fontsize=12, fontweight='bold')
    st.pyplot(fig)

    # Display weaknesses
    st.subheader("Weaknesses")
    weaknesses = {k: v['weakness'] for k, v in result.items()}
    for field, weakness in weaknesses.items():
        st.write(f"**{field}:** {weakness}")

else:
    st.info("Please upload a resume and enter a job description to start the analysis.")
