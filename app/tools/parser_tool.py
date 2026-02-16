from typing import List, Optional
from langchain.tools import tool
from agent.text_parser_agent import TextParserAgent

# Initialize the agent once at the module level for reuse across calls
agent = TextParserAgent()

@tool
def document_parser_tool(
    raw_text: str, 
    filename: str, 
    filetype: str, 
    pages: Optional[List] = None
) -> dict:
    """
    Useful for parsing raw text from documents to generate structured JSON output. 
    It returns a summary, cleaned text, and metadata like file name and word count.
    
    Args:
        raw_text: The extracted raw text from the document.
        filename: The original name of the file.
        filetype: The type of file (e.g., pdf, image, docx).
        pages: Optional list of individual page contents.
    """
    result = agent.parse(
        raw_text=raw_text,
        filename=filename,
        file_type=filetype,
        pages=pages
    )
    return result.model_dump()
