import pandas as pd
import io
import re
import logging
from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_llm

logger = logging.getLogger("app.agents.analyst")

class AnalystAgent:
    def __init__(self):
        self.llm = get_llm(temperature=0)
        logger.info("AnalystAgent initialized.")
        self.code_gen_prompt = ChatPromptTemplate.from_template("""
        You are an expert Data Analyst. You are given a pandas DataFrame named `df`.
        The user wants to analyze this dataset.
        
        DATASET (Full CSV Content):
        {df_contents}
        
        User Question: {question}
        
        CRITICAL INSTRUCTIONS:
        - If columns contain currency symbols ($, €) or commas (1,000), you MUST clean them and convert to float before performing math.
        - Example: `df['Col'] = df['Col'].replace('[\\$,]', '', regex=True).astype(float)`
        - Write ONLY the Python code to answer the question.
        - The DataFrame is already loaded as `df`.
        - Store the final result in a variable named `result`.
        - Do not include any explanations or markdown blocks.
        
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
        logger.info(f"Analyzing {file_type} data for question: {question}")
        try:
            # Load data into pandas
            if file_type == "csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
            elif file_type == "excel":
                df = pd.read_excel(io.BytesIO(file_bytes))
            else:
                logger.error(f"Unsupported file type for analysis: {file_type}")
                return {"error": f"AnalystAgent only supports CSV and Excel, got {file_type}"}

            # CLEANUP: Strip leading/trailing spaces from column names to prevent KeyErrors
            df.columns = df.columns.str.strip()
            
            logger.info(f"Data loaded. Shape: {df.shape}. Columns: {list(df.columns)}")

            # WHOLE CONTENT: Convert entire dataframe to CSV/Text to pass to LLM
            # We use CSV format as it's token-efficient
            full_data_content = df.to_csv(index=False)
            
            # Step 1: Generate Code (Using the whole content context)
            logger.info("Generating pandas code with full data context...")
            code_chain = self.code_gen_prompt | self.llm | StrOutputParser()
            raw_code = code_chain.invoke({
                "df_contents": full_data_content,
                "question": question
            })
            python_code = self._extract_code(raw_code)
            logger.info(f"Generated Code:\n{python_code}")

            # Step 2: Execute Code Safely (isolated namespace)
            logger.info("Executing analysis...")
            local_vars = {"df": df, "result": None}
            try:
                exec(python_code, {}, local_vars)
                analysis_result = local_vars.get("result")
            except Exception as e:
                logger.error(f"Code execution failed: {e}")
                return {
                    "status": "error",
                    "error": f"Code execution failed: {str(e)}",
                    "generated_code": python_code
                }

            # Step 3: Generate Structured Insight
            logger.info("Generating final insight...")
            insight_chain = self.insights_prompt | self.llm | StrOutputParser()
            insight = insight_chain.invoke({
                "question": question,
                "result_data": str(analysis_result)
            })

            logger.info("Analysis complete.")
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
