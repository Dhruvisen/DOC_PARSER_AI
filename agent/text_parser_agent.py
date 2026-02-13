from langchain_groq import ChatGroq
from schemas.output_schema import ParsedDocument
from utils.cleaners import clean_text
from typing import Optional
from app.core.config import get_settings

settings = get_settings()

class TextParserAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.2-90b-vision-preview", # Updated to a vision-capable model for video/images
            temperature=0,
            max_retries=2,
            api_key=settings.GROQ_API_KEY
        )

    def summarize(self, text: str) -> str:
        prompt = [
            ("system", "You are an AI document analysis assistant. Summarize the provided text clearly and concisely, focusing on key details like dates, amounts, and parties involved."),
            ("user", text),
        ]
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def parse(
        self,
        raw_text: str,
        filename: str,
        file_type: str,
        pages: Optional[list] = None,
    ) -> ParsedDocument:
        cleaned_text = clean_text(raw_text)
        
        raw_summary = self.summarize(cleaned_text)
        
        summary = clean_text(raw_summary)

        return ParsedDocument(
            filename=filename,
            file_type=file_type,
            raw_text=cleaned_text,
            pages=pages,
            summary=summary,
            word_count=len(cleaned_text.split()),
        )