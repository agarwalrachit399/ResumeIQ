from resume_parser import ResumeAnalyzer
from calculate_experience import CalculateExperience

api_token = "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"
model_repo = "microsoft/Phi-3.5-mini-instruct"
resume_file = "Resume.pdf"

# print(education_info['Education'])
# print(experience_info)


def parse_projects(json_data) -> str:
    data = json_data if isinstance(json_data,list) else json_data['Project']
    summaries = []
    if data:
        for proj in data:
            if proj['project_name']:
                project_type = "Research" if proj['is_research'] else "Project"
                summary = f"{project_type}: {proj['project_name']}"
                
                # Include URL for projects and journal name for research
                if proj['is_research']:
                    if proj['URL']:
                        summary += f" (Journal: {proj['URL']})"
                else:
                    if proj['URL']:
                        summary += f" (URL: {proj['URL']})"
                
                if proj['description']:
                    # description_text = " ".join(proj['Description'])
                    summary += f". Description: {proj['description']}"
                
                summaries.append(summary)
    return " ".join(summaries) if summaries else "No projects or research entries available."





def parse_education(json_data) -> str:
    data = json_data if isinstance(json_data,list) else json_data['Education']
    summaries = []
    for edu in data:
        summary = f"{edu['degree_level']} in {edu['field_of_study']} from {edu['university_name']}"
        if edu['GPA']:
            summary += f" with a GPA of {edu['GPA']}."
        else:
            summary += "."

        # Add coursework if it exists and is non-empty
        if edu['coursework']:
            summary += f" The coursework included {edu['coursework']}"

        # Add accomplishments if they exist and are non-empty
        if edu.get('accomplishments'):
            summary += f" Some of their accomplishments were {edu['accomplishments']}."
        
        summaries.append(summary)
    
    return " ".join(summaries)
        


def parse_experience(json_data, calculator):
    companies = set()
    roles = set()
    durations = []
    responsibilities = set()

    for experience in json_data["experience"]:
        companies.add(experience["company_name"])
        roles.add(experience["role"])
        durations.append(experience["duration"])
        responsibilities.update(experience["responsibilities"])
    total_exp = calculator.total_experience(durations)
    # print(total_exp)
    result = (f"Companies: {companies}\n"
              f"Roles: {roles}\n"
              f"Total Experience: {total_exp} months\n"
              f"Responsibilities: {responsibilities}")
    
    return result



def final_summary(resume_file):
    # resume_file = "Resume.pdf"
    analyzer = ResumeAnalyzer(api_token, model_repo, resume_file)
    calculator = CalculateExperience(api_token,model_repo)
    experience_info = analyzer.extract_experience()
    print("------------experience_info-------------------")
    print(experience_info)
    print("-----------------------------------------------")
    education_info = analyzer.extract_education()
    print("------------education_info-------------------")
    print(education_info)
    print("-----------------------------------------------")
    project_info = analyzer.extract_projects()
    print("------------project_info-------------------")
    print(project_info)
    print("-----------------------------------------------")
    skill_info = analyzer.extract_skills()
    exp_summary = parse_experience(experience_info,calculator)
    edu_summary = parse_education(education_info)
    pro_summary = parse_projects(project_info)
    skill = str(skill_info['Skills'])

    final_summary = f"""

    Candidate Education:
    {edu_summary}

    Candidate Industry Experience:
    {exp_summary}

    Candidate Skill:
    {skill}

    Candidates Personal Projects/Publications:
    {pro_summary}

    """

    return final_summary


