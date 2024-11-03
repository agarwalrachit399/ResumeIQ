# calculator.py

import os
from langchain_huggingface import HuggingFaceEndpoint
from pydantic import BaseModel
from typing import List
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from datetime import datetime
from dateutil.relativedelta import relativedelta


class Dates(BaseModel):
    Dates : List[str]


class CalculateExperience:
    def __init__(self, api_token, model_repo ):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token
        self.model_repo = model_repo
        self.max_tokens = 190
        
    def generate_prompt(self):
        parser = JsonOutputParser(pydantic_object=Dates)
        template = """
        You are an AI assistant that is expert in handling dates. You will be provided with a list of dates duration.
        The dates can be in different formats. Covert them into consistent format of "YYYY-MM-DD to YYYY-MM-DD" 
        for example: ```2017/11/04 to 2018/12/05``` is a valid conversion.
        Do not write code for the process. Simply convert them.
        Note if dates only have month and year, give the day as 01
        Your output should be only be the JSON without any explaination.
        ```
        Dates: {dates_list}
        ```
        
        {format_instructions}
        """

        prompt = PromptTemplate(template=template, input_variables=["dates_list"],
                                partial_variables={"format_instructions": parser.get_format_instructions()})
        return prompt

    def initialize_llm(self, dates_list,prompt):
        llm = HuggingFaceEndpoint(
            repo_id=self.model_repo,
            task="text-generation",
            max_new_tokens=self.max_tokens,
            do_sample=False,
            repetition_penalty=1.03
        )
        parser = JsonOutputParser(pydantic_object=Dates)
        chain = prompt | llm | parser
        response = chain.invoke({"dates_list":dates_list})
        return response
    
    def calculate_total_duration(self,date_ranges):
        total_months = 0
    
        for date_range in date_ranges:
            start_date_str, end_date_str = date_range.split(' to ')
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            # Calculate the year and month difference
            duration = relativedelta(end_date, start_date)
            # Total months for this range
            months = duration.years * 12 + duration.months
            total_months += months
        
        return total_months
    
    def total_experience(self,date_ranges):
        prompt = self.generate_prompt()
        total_exp = self.initialize_llm(date_ranges,prompt)
        converted_dates_list = total_exp if isinstance(total_exp,list) else total_exp['Dates']
        total_duration = self.calculate_total_duration(converted_dates_list)
        return total_duration


if __name__ == "__main__":
    api_token = "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"
    model_repo = "microsoft/Phi-3.5-mini-instruct"  
    dates = "['2017/05–2018/05', 'May 2024 – Aug 2024', '02/2019–06/2019', '2023/05–2023/08']"
    calculator = CalculateExperience(api_token,model_repo)

    print("-----------------------------")
    print(calculator.total_experience(dates))