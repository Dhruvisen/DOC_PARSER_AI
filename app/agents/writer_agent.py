from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from app.core.llm import get_llm

class WrittenOutput(BaseModel):
    email_draft: str = Field(description="A professional email draft based on the analysis.")
    management_summary: str = Field(description="A high-level summary for management/executives.")
    bullet_report: str = Field(description="A detailed report in bullet points highlighting key findings.")

class WriterAgent:
    def __init__(self):
        self.llm = get_llm(model_name="llama-3.3-70b-versatile", temperature=0.7)
        self.output_parser = JsonOutputParser(pydantic_object=WrittenOutput)
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are a professional Business Writer and Communications Expert. 
        Your task is to take the structured outputs from document parsing, RAG retrieval, and data analysis, 
        and transform them into professional communication pieces.
        
        INPUT DATA:
        {input_data}
        
        Please generate the following three components:
        1. **Email Draft**: A professional email to stakeholders summarizing the findings and suggesting next steps.
        2. **Management Summary**: A concise, executive-level overview focused on impact and key takeaways.
        3. **Bullet Report**: A detailed, organized list of all significant facts, data points, and insights discovered.
        
        {format_instructions}
        
        Ensure the tone is professional, objective, and clear.
        """)

    def compose(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes raw data from various agents and composes professional documents.
        """
        chain = self.prompt | self.llm | self.output_parser
        
        try:
            # Flatten or summarize the input data for the prompt
            input_summary = str(data)
            
            response = chain.invoke({
                "input_data": input_summary,
                "format_instructions": self.output_parser.get_format_instructions()
            })
            return response
        except Exception as e:
            return {
                "error": f"Failed to generate written output: {str(e)}",
                "email_draft": "Error",
                "management_summary": "Error",
                "bullet_report": "Error"
            }

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node interface.
        Expects the state to contain results from other agents (e.g., 'data', 'analysis_result', 'rag_answer').
        """
        # Combine relevant parts of the state into a single context for the writer
        context = {
            "parsed_data": state.get("data"),
            "data_analysis": state.get("analysis_result"),
            "rag_context": state.get("answer")
        }
        
        written_content = self.compose(context)
        
        # Merge the results back into the state
        return {**state, "written_output": written_content}

# Export instance
writer_agent = WriterAgent()
