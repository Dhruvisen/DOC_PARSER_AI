from typing import Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import get_llm
from schemas.output_schema import ParsedDocument
from utils.cleaners import clean_text

class TextParserAgent:
    def __init__(self, model_name: str = None):
        self.llm = get_llm(model_name=model_name, temperature=0)
        
        # Define the summary chain using LCEL
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI document analysis assistant. Summarize the provided text clearly and concisely, focusing on key details like dates, amounts, and parties involved."),
            ("user", "{text}"),
        ])
        self.summary_chain = self.summary_prompt | self.llm | StrOutputParser()

    def summarize(self, text: str) -> str:
        """Generates a summary for the given text using the LCEL chain."""
        return self.summary_chain.invoke({"text": text}).strip()

    def parse(
        self,
        raw_text: str,
        filename: str,
        file_type: str,
        pages: Optional[List] = None,
    ) -> ParsedDocument:
        """Processes raw text into a structured ParsedDocument."""
        cleaned_text = clean_text(raw_text)
        
        # Generate and clean the summary
        summary = self.summarize(cleaned_text)
        cleaned_summary = clean_text(summary)

        return ParsedDocument(
            filename=filename,
            file_type=file_type,
            raw_text=cleaned_text,
            pages=pages,
            summary=cleaned_summary,
            word_count=len(cleaned_text.split()),
        )