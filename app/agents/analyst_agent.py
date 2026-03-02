import pandas as pd
import io
import re
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_llm

class AnalystAgent:
    def __init__(self):
        self.llm = get_llm(model_name="llama-3.3-70b-versatile", temperature=0)
        self.code_gen_prompt = ChatPromptTemplate.from_template("""
        You are an expert Data Analyst. You are given a pandas DataFrame named `df`.
        The user wants to analyze this dataset.
        
        DataFrame Info:
        {df_info}
        
        Head of DataFrame:
        {df_head}
        
        User Question: {question}
        
        Write ONLY the Python code (using pandas) to answer the user's question. 
        - The DataFrame is already loaded as `df`.
        - Store the final answer in a variable named `result`.
        - Do not include any explanations, only the code block.
        - Use standard pandas operations.
        - If the answer requires a string description, store it in `result`.
        
        Python Code:
        """)
        
        self.insights_prompt = ChatPromptTemplate.from_template("""
        You are a Data Analyst. Based on the analysis result and the user's original question, provide a concise and professional insight.
        
        User Question: {question}
        Analysis Result Data: {result_data}
        
        Structured Insight:
        """)

    def _extract_code(self, text: str) -> str:
        """Extracts python code from markdown blocks if present."""
        code_match = re.search(r"```python(.*?)```", text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return text.strip()

    def analyze(self, file_bytes: bytes, file_type: str, question: str) -> Dict[str, Any]:
        """
        Loads data, generates pandas code, executes it, and returns insights.
        """
        try:
            # Load data into pandas
            if file_type == "csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            elif file_type == "excel":
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                return {"error": f"AnalystAgent only supports CSV and Excel, got {file_type}"}

            # Prepare metadata for LLM
            buffer = io.StringIO()
            df.info(buf=buffer)
            df_info = buffer.getvalue()
            df_head = df.head().to_string()

            # Step 1: Generate Code
            code_chain = self.code_gen_prompt | self.llm | StrOutputParser()
            raw_code = code_chain.invoke({
                "df_info": df_info,
                "df_head": df_head,
                "question": question
            })
            python_code = self._extract_code(raw_code)

            # Step 2: Execute Code Safely (isolated namespace)
            local_vars = {"df": df, "result": None}
            try:
                exec(python_code, {}, local_vars)
                analysis_result = local_vars.get("result")
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Code execution failed: {str(e)}",
                    "generated_code": python_code
                }

            # Step 3: Generate Structured Insight
            insight_chain = self.insights_prompt | self.llm | StrOutputParser()
            insight = insight_chain.invoke({
                "question": question,
                "result_data": str(analysis_result)
            })

            return {
                "status": "success",
                "question": question,
                "insight": insight,
                "data_preview": str(analysis_result),
                "generated_code": python_code
            }

        except Exception as e:
            return {"status": "error", "error": f"Analysis process failed: {str(e)}"}

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        LangGraph node interface.
        Expects 'file_bytes', 'file_type', and 'query' (or 'question') in state.
        """
        file_bytes = state.get("file_bytes")
        file_type = state.get("file_type")
        question = state.get("query") or state.get("question")
        
        if not all([file_bytes, file_type, question]):
            return {**state, "error": "Missing required fields for analysis (file_bytes, file_type, question)"}
            
        result = self.analyze(file_bytes, file_type, question)
        return {**state, "analysis_result": result}

# Export instance
analyst_agent = AnalystAgent()
