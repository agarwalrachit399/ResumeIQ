import streamlit as st
from pdfminer.high_level import extract_text
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import os


# Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Streamlit UI
st.title("Resume-Job Description Analyzer")
st.write("Upload a resume (PDF format) and paste the job description. The LLM will score how well the resume matches the job description.")

# Upload Resume (PDF only)
uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Input Job Description
job_description = st.text_area("Paste the Job Description", height=150)

# Extract text from PDF resume
def extract_text_from_pdf(uploaded_file):
    try:
        # Extract text from the PDF file
        text = extract_text(uploaded_file)

        # Return the extracted text as a string
        return text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Generate the LLM prompt
def generate_llm_prompt(resume,job_description):
    return f"""
    You are an expert in human resource management and recruitment.
      You need to analyze the following resume in the context of the provided job description.
      Think step by step and use the following approach to calculate a score between 1-100:

      Approach:
      Analyze the skills of the job to the candidate's listed skills. Penalize for missing any required skills.
      Secondly analyze if candiate meets the minimum requirements. Heavily penalize if any minimum requirement is missing.
      To calculate work experience look for dates and the specific titles associated with it and the time elapsed between those dates
      Analyze previous work experience to see if candidate is a good fit or not. If no matching work experience is found, penalize
      and look for relevant projects. If no projects are also found, heavily penalize.
      Give bonus point for any preferred skills that candidate has which matches job description
      Penalize for missing basic information such as name, number, email,linkedin and grammatical mistakes.

    Job Description: {job_description}

    Resume: {resume}

    Your final result should be of the following format:
    Score: A final score between 1-100
    Candidate Info: Basic info such as Name, Email, Phone Number
    Skills : List the matching skills and and missing skills
    Minimum Requirement: List any minimum requirements missing , otherwise say all minimum requirements met
    Work Experience: Give short summary of Work exp and how it related to job requirement
    
    """

# Use Langchain with HuggingFaceHub to call the Mistral model
def analyze_resume_with_llm(resume,job_description):

    prompt = generate_llm_prompt(resume, job_description)
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    min_new_tokens=200,
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    max_length = 128000
    )
    #chat = ChatHuggingFace(llm=llm, verbose=False)
    response = llm.invoke(prompt)
    return response

# Display the analysis result when the button is clicked
if st.button("Analyze Resume"):
    if uploaded_resume and job_description:
        resume_text = extract_text_from_pdf(uploaded_resume)
        #print(resume_text)
        result = analyze_resume_with_llm(resume_text,job_description)

        st.write("Sending data to the LLM for analysis...")
        print("Result of LLM",result)
        st.subheader("LLM Analysis Result")
        st.write(result)
    else:
        st.error("Please upload both a resume and paste the job description.")
