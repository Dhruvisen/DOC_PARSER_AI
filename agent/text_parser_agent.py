from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from schemas.output_schema import ParsedDocument
from utils.cleaners import clean_text
from typing import Optional

class TextParserAgent:
    def __init__(self):
        pipe = pipeline(
            task="text-generation",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            device=-1,
            max_new_tokens=512,
            do_sample=False,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def summarize(self, text: str) -> str:
        prompt = f"Summarize the following document clearly:\n\n{text}"
        # Using .invoke() instead of __call__
        response = self.llm.invoke(prompt)
        return response.strip()

    def parse(self, raw_text: str, filename: str, file_type: str, pages: Optional[list] = None) -> ParsedDocument:
        cleaned_text = clean_text(raw_text)
        
        # Summarize first 4000 chars and clean output
        raw_summary = self.summarize(cleaned_text[:4000])
        summary = clean_text(raw_summary)

        return ParsedDocument(
            filename=filename,
            file_type=file_type,
            raw_text=cleaned_text,
            pages=pages,
            summary=summary,
            word_count=len(cleaned_text.split()),
        )