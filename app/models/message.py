from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field

class ContentItem(BaseModel):
    type: Literal["text", "image_url", "video_url"]
    text: Optional[str] = None
    image_url: Optional[dict] = None  # Expected format: {"url": "data:image/jpeg;base64,..."} or {"url": "https://..."}
    video_url: Optional[dict] = None

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
