from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

'''
force LLM to output structured audit with score and critique
'''

class Audit(BaseModel):
    score: int = Field(description="1-10 score")
    critique: str = Field(description="Feedback for the writer")

def grade_report(report: str):
    llm = ChatOpenAI(model="gpt-5-nano").with_structured_output(Audit)
    return llm.invoke(f"Audit this report: {report}")