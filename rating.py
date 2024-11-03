# job_analyzer.py

import os
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel, Field
from typing import List,Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from infer_resume import final_summary
from infer_requirement import JobDescriptionAnalyzer

class SkillDetail(BaseModel):
    score: int = Field("A score between 1-100")
    strength: str = Field("Stength of the candidate")
    weakness: str = Field("Weakness of the candidate")

class JobDesc(BaseModel):
    Skill: SkillDetail 
    Experience: SkillDetail 
    Education: SkillDetail 
    Accomplishments: SkillDetail 
    Soft_Skill: SkillDetail 

class ResumeIQ:
    def __init__(self, api_token, model_repo, max_tokens=1200):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
        self.model_repo = model_repo
        self.max_tokens = max_tokens
        self.llm = HuggingFaceEndpoint(
            repo_id=self.model_repo,
            task="text-generation",
            max_new_tokens=self.max_tokens,
            do_sample=False,
            repetition_penalty=1.03,
        )



    def generate_prompt(self, parser):
        job_prompt = """
        You are an AI assistant that is expert in resume evaluation and hiring. You will be provided with a summary of candidates's resume and a summary of the job description.
        You need to evaluate the candidate on the following 5 fields:
        - Skills and Technologies: Match specific technical skills, software, programming languages, and tools mentioned in the job description with those listed on the resume.
        - Relevant Experience: Compare the candidate's work history, project involvement, and hands-on experience in areas directly related to the job.
        - Educational Background: Verify if the candidate’s academic qualifications meet the minimum requirements.
        - Key Accomplishments and Impact: Look for measurable achievements or outcomes, such as completed projects, publications, or contributions that reflect the candidate's ability to deliver results. Match these with any performance expectations listed in the job description.
        - Soft Skills and Cultural Fit: Assess alignment in soft skills like communication, teamwork, leadership, and adaptability, especially if they’re highlighted in the job description. 
        
        Your output should only be a JSON with a score of 1-100 for each of the following 5 fields along with strengths and weaknesses for each of these fields. 
        Make sure to output only JSON and nothing else at all. The JSON should be correct and precise. No extra explanation or prompts should be in output.
        ```
        Resume = {resume}

        Job Description = {job_desc}
        ```
        {format_instructions}
        """
        prompt = PromptTemplate(template=job_prompt,input_variables=["resume","job_desc"],partial_variables={"format_instructions": parser.get_format_instructions()})
        return prompt
    

    def match_resume(self, job_desc,resume):
        parser = JsonOutputParser(pydantic_object=JobDesc)
        prompt = self.generate_prompt(parser)
        chain = prompt | self.llm | parser
        response = chain.invoke({"resume": resume,"job_desc": job_desc})
        return response


if __name__ == "__main__":
    api_token = "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"
    model_repo = "microsoft/Phi-3.5-mini-instruct"
    resume_file = "Resume.pdf"
    resume = final_summary(resume_file)
    job = """

Collabera logo
Collabera
Python Developer 
Jersey City, NJ · 3 days ago · Over 100 potential applicants

$55/hr - $68/hr Hybrid Contract Entry level
Back-End Python

    2 school alumni work here
    See how you compare to over 100 other applicants. Try Premium for $0

Python Developer
Collabera · Jersey City, NJ (Hybrid)
Meet the hiring team
Akash, #OpenToWork
Akash Mishra Akash Mishra is verified
3rd
Process Coordinator at Collabera | MIT- Manipal
Job poster
About the job

Position Details:

 

Job Title: Application Programmer V (Python Developer)

Location: Jersey City, NJ - Hybrid

Pay rate: $55-68/hr.

 

Responsibilities:


    Design and develop Python solutions to solve complex business problems

    Collaborate with business users, analysts, and development colleagues to deliver best-in-class risk management solutions

    Utilize comprehensive knowledge of Python and cloud infrastructure to drive innovation

    Work in an Agile development environment to ensure timely delivery of projects

    Utilize work management and collaborative tools such as JIRA and Confluence to streamline processes


Education Qualification: 


    Bachelor's degree in Computer Science, Engineering, or related field


Required Skills: 


    Proficiency in Python programming

    Strong experience with cloud infrastructure

    Knowledge of Object-Oriented Database Management Systems

    Excellent communication, collaboration, and problem-solving skills

    Familiarity with Agile development methodologies

    Experience with work management tools like JIRA and Confluence


Desired Skills and Experience

Python,Agile,AWS,Angular,JIRA,Confluence,Risk, Python developer,Risk Finance

"""
    job_analyzer = JobDescriptionAnalyzer(api_token, model_repo)
    job_desc = job_analyzer.analyze_job_description(job)
    analyzer = ResumeIQ(api_token, model_repo)


    print("-------------------------------------")
    print(analyzer.match_resume(job_desc, resume))
    print("-------------------------------------")

