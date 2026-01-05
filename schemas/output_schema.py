from pydantic import BaseModel
from typing import List, Optional


class PageContent(BaseModel):
    page_number: Optional[int]
    content: str


class ParsedDocument(BaseModel):
    filename: str
    file_type: str

    raw_text: str
    pages: Optional[List[PageContent]] = None

    summary: Optional[str] = None
    word_count: int
