# job_analyzer.py

import os
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel, Field
from typing import List,Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class JobDesc(BaseModel):
    RoleName: Optional[str] = Field("Job role as listed in the job description")
    CompanyInfo : Optional[str] = Field("Info such as company name, sector, location, if its a startup or a big company etc.")
    MinimumRequirements: List[str] = Field("list of minimum requirements that candidate must meet for the job")
    Responsibilities: List[str]= Field("List of Responsibilities that the hired candidate will perform")
    Keywords: List[str] = Field("list of keywords that refers to requirements, skills, experiences or other qualities needed for the job offer")

class JobDescriptionAnalyzer:
    def __init__(self, api_token, model_repo, max_tokens=500):
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
        You are an AI assistant that is expert in information extraction. You will be provided with a job description.
        Ensure that you are able to extract the following from the provided job description and output in the given JSON Schema below:
        Make sure that JSON is valid and you dont output anything else at all
        ```
        Job Description = {job_desc}
        ```
        {format_instructions}
        """
        prompt = PromptTemplate(template=job_prompt,input_variables=["job_desc"],partial_variables={"format_instructions": parser.get_format_instructions()})
        return prompt

    def analyze_job_description(self, job_desc):
        parser = JsonOutputParser(pydantic_object=JobDesc)
        prompt = self.generate_prompt(parser)
        chain = prompt | self.llm | parser
        response = chain.invoke({"job_desc": job_desc})
        return response


if __name__ == "__main__":
    api_token = "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"
    model_repo = "microsoft/Phi-3.5-mini-instruct"
    job_desc = """

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
    analyzer = JobDescriptionAnalyzer(api_token, model_repo)
    
    print("---------Job Desc-------------")
    print(analyzer.analyze_job_description(job_desc))
    print("------------------------------------------")