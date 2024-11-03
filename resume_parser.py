import os
from langchain_huggingface import HuggingFaceEndpoint
from pdfminer.high_level import extract_text
from pydantic import BaseModel, Field
from typing import List,Optional
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

#Pydantic Classes 
class PersonalInformation(BaseModel):
    name: str
    phone_number: str
    email: str
    links: Optional[str]

class Education(BaseModel):
    university_name: str
    degree_level: str 
    field_of_study: str 
    GPA: Optional[str]  = Field("GPA of the candidate. Could be Cumuilative")
    coursework: Optional[List[str]] = Field("Subjects or particular courses taken in the field of study pursued")
    accomplishments: Optional[str] = Field("Include any scholarship,titles,grants or other achievements")

class EducationList(BaseModel):
    Education: List[Education]

class Project(BaseModel):
    project_name: Optional[str] = Field(description="Name of the Personal Project or Research")
    URL: Optional[str] = Field(description="Link to the project or journal name")
    description: Optional[List[str]] = Field(description="Description of the project or research")
    is_research : Optional[bool] = Field(description= "Wether project is a Research or not") 

class ExtraSection(BaseModel):
    Project : Optional[List[Project]]


class Experience(BaseModel):
    company_name: str
    role: str
    duration: str
    responsibilities: List[str]

class ExperienceList(BaseModel):
    experience: List[Experience]

class Skills(BaseModel):
    Skills: List[str] = Field(description="Technical Skills of the candidate including tools and frameworks,etc")



#Main Class

class ResumeAnalyzer:
    def __init__(self, api_token, model_repo, resume_file):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
        self.model_repo = model_repo
        self.resume = self.extract_text_from_pdf(resume_file)
        # self.max_length = 8096

    def extract_text_from_pdf(self, uploaded_file):
        try:
            # Extract text from the PDF file
            text = extract_text(uploaded_file)
            return text
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def generate_prompt_template(self,parser):

        template = """
        You are an AI assistant that is expert in information extraction. You will be provided with content of a resume.
        Ensure that you are able to chunk the resume to extract only the {section} of the candidate.
        Do not generate any new text or summarize, simply use the given content of the resume to generate only the JSON object without explanations.
        Also do not output the JSON of format instruction itself but do the extraction from resume.
        Broadly resume has personal information, education information, skills information, work experience information and personal project/publication information.
        Do not get confused among these sections and only include the info form one which is asked
        Do not infer information from another section to include in the section asked
        ```
        Resume: {resume}
        ```

        {format_instructions}
        """
        prompt = PromptTemplate(template=template, input_variables=["resume","section"],
                                partial_variables={"format_instructions": parser.get_format_instructions()})
        return prompt

    def analyze_section(self, prompt, tokens,parser,section):
        llm = HuggingFaceEndpoint(
            repo_id=self.model_repo,
            task="text-generation",
            max_new_tokens=tokens,
            do_sample=False,
            repetition_penalty=1.03,
            # max_length=self.max_length
        )
        chain = prompt | llm | parser
        response = chain.invoke({"resume": self.resume,"section":section})
        return response

    def extract_personal_info(self):
        parser = JsonOutputParser(pydantic_object=PersonalInformation)
        #example_output = '{"Personal Information": {"Name": "Rachit", "Phone Number": "+1 (213) 696 4727", "Email": "agarwalrachit399@gmail.com", "Links": "Not available", "Summary/Objective": "I aspire to be a software engineer. Have a lot of experience."}}'
        prompt = self.generate_prompt_template(parser)
        return self.analyze_section(prompt, tokens=200,parser=parser,section="Personal Information")

    def extract_skills(self):
        parser = JsonOutputParser(pydantic_object=Skills)
        #example_output = '{"Skills": ["Python", "JavaScript", "React", "Flask", "RestAPI", "Pytorch", "Google Cloud"]}'
        prompt = self.generate_prompt_template(parser)
        return self.analyze_section(prompt, tokens=300,parser=parser,section="Skills Information")

    def extract_education(self):
        parser = JsonOutputParser(pydantic_object=EducationList)
        #example_output = '{"Education": [{"University Name": "SRM Institute of Science and Technology", "Degree Level": "Bachelors", "Field of Study": "Computer Science", "GPA": "9.62", "Coursework": ["Compiler Design", "Database Management", "Analysis of Algorithms"], "Accomplishments": "Not Available"}]}'
        #additional_note = 'Additional Note: Do not infer any information, if not directly stated on resume, say Not Available.'
        prompt = self.generate_prompt_template(parser)
        return self.analyze_section(prompt, tokens=400,parser=parser,section="Education Information")

    def extract_experience(self):
        parser = JsonOutputParser(pydantic_object=ExperienceList)
        #example_output = '{"Experience": [{"Company Name": "Meta", "Role": "Software Engineer", "Duration": "April 2024 - Present", "Responsibilities": ["Collaborated to develop a suite of responsive eCommerce features using React.js and Tailwind.", "Engineered a scalable backend leveraging Node.js and Express."]}]}'
        prompt = self.generate_prompt_template(parser)
        return self.analyze_section(prompt, tokens=1200,parser=parser,section="Work Experience Information")

    def extract_projects(self):
        parser = JsonOutputParser(pydantic_object=ExtraSection)
        #example_output = '{"Projects": [{"Project Name": "", "URL": "", "Description": []}], "Publications": [{"Journal Name": "", "Authors": [], "Research Name": "", "Year Published": ""}]}'
        #additional_note = 'Additional Note: Do not hallucinate or infer information and do not confuse between project and publication and do not include same thing under both.'
        prompt = self.generate_prompt_template(parser)
        return self.analyze_section(prompt, tokens=1000,parser=parser,section="Personal Projects/Publications Information")



# Usage
if __name__ == "__main__":
    api_token = "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"
    model_repo = "microsoft/Phi-3.5-mini-instruct"
    resume_file = "Resume.pdf"

    analyzer = ResumeAnalyzer(api_token, model_repo, resume_file)

    # print("---------Personal Information-------------")
    # print(analyzer.extract_personal_info())
    # print("------------------------------------------")

    # print("---------Education-------------")
    # print(analyzer.extract_education())
    # print("------------------------------------------")

    # print("---------Work Experience-------------")
    # print(analyzer.extract_experience())
    # print("------------------------------------------")

    # print("---------Skills-------------")
    # print(analyzer.extract_skills())
    # print("------------------------------------------")

    # print("---------Project/Publication-------------")
    # print(analyzer.extract_projects())
    # print("------------------------------------------")



